package models;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import utility.FuncUtils;

/**
 * STTM: A Java package for short text topic models
 *
 * Implementation of the one-topic-per-document Dirichlet Multinomial Mixture
 * model, using collapsed Gibbs sampling, as described in:
 * 
 * Jianhua Yin and Jianyong Wang. 2014. A Dirichlet Multinomial Mixture
 * Model-based Approach for Short Text Clustering. In Proceedings of the 20th
 * ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,
 * pages 233–242.
 * 
 * @author: Jipeng Qiang
 */

public class DMM
{

	/**
	 * vocabulary size
	 */
	int V;

	/**
	 * number of topics
	 */
	int K;

	/**
	 * Dirichlet parameter (document--topic associations)
	 */
	double alpha=.1;

	/**
	 * Dirichlet parameter (topic--term associations)
	 */
	double beta=.1;

	/**
	 * topic assignments for each document.
	 */
	int z[];

	//int au
	/**
	 * number of documents in cluster z.
	 */
	int m_z[];

	/**
	 * number of words in cluster z.
	 */
	int n_z[];

	/**
	 * number of occurrences of word w in cluster z.
	 */
	int n_w_z[][];

	/**
	 * number of words in document d.
	 */
	int N_d[];

	public int topWords; // Number of most probable words for each topic
	public List<List<Integer>> corpus; // Word ID-based corpus

	public int numDocuments; // Number of documents in the corpus
	public int numWordsInCorpus; // Number of words in the corpus

	public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
														// given a word
	public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
														// given an ID
    // Given a document, number of times its i^{th} word appearing from
    // the first index to the i^{th}-index in the document
	// Example: given a document of "a a b a b c d c". We have: 1 2 1 3 2 1 1 2
	public List<List<Integer>> occurenceToIndexCount;
	// Double array used to sample a topic
	public double[] multiPros;

	// Path to the directory containing the corpus
	public String folderPath;
	// Path to the topic modeling corpus
	public String corpusPath;

	/**
	 * max iterations
	 */
	private static int ITERATIONS = 500;

	public String expName = "DMMmodel";
	public String orgExpName = "DMMmodel";

	public double initTime = 0;
	public double iterTime = 0;


	public DMM(String pathToCorpus, int inNumTopics,
		double inAlpha, double inBeta, int inNumIterations, int inTopWords)
		throws Exception
	{
		this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
			inTopWords, "DMMmodel");
	}

	public DMM(String pathToCorpus, int inNumTopics,
		double inAlpha, double inBeta, int inNumIterations, int inTopWords,
		String inExpName)
		throws Exception

	{
		alpha = inAlpha;
		beta = inBeta;
		K = inNumTopics;
		ITERATIONS = inNumIterations;
		topWords = inTopWords;

		expName = inExpName;
		orgExpName = expName;
		corpusPath = pathToCorpus;
		folderPath = "results/";
		File dir = new File(folderPath);
		if (!dir.exists())
			dir.mkdir();

		System.out.println("Reading topic modeling corpus: " + pathToCorpus);

		word2IdVocabulary = new HashMap<String, Integer>();
		id2WordVocabulary = new HashMap<Integer, String>();
		corpus = new ArrayList<List<Integer>>();
		occurenceToIndexCount = new ArrayList<List<Integer>>();

		numDocuments = 0;
		numWordsInCorpus = 0;

		BufferedReader br = null;
		try {
			int indexWord = -1;
			br = new BufferedReader(new FileReader(pathToCorpus));
			for (String doc; (doc = br.readLine()) != null;) {
				if (doc.trim().length() == 0)
					continue;

				String[] words = doc.trim().split("\\s+");
				List<Integer> document = new ArrayList<Integer>();

				List<Integer> wordOccurenceToIndexInDoc = new ArrayList<Integer>();
				HashMap<Integer, Integer> wordOccurenceToIndexInDocCount = new HashMap<Integer, Integer>();

				for (String word : words) {
					if (word2IdVocabulary.containsKey(word)) {
						document.add(word2IdVocabulary.get(word));
					}
					else {
						indexWord += 1;
						word2IdVocabulary.put(word, indexWord);
						id2WordVocabulary.put(indexWord, word);
						document.add(indexWord);
					}

					int times = 0;
					if (wordOccurenceToIndexInDocCount.containsKey(indexWord)) {
						times = wordOccurenceToIndexInDocCount.get(indexWord);
					}
					times += 1;
					wordOccurenceToIndexInDocCount.put(indexWord, times);
					wordOccurenceToIndexInDoc.add(times);
				}
				numDocuments++;
				numWordsInCorpus += document.size();
				corpus.add(document);
				occurenceToIndexCount.add(wordOccurenceToIndexInDoc);

			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}

		V = word2IdVocabulary.size();
		// initialise count variables.
		m_z = new int[K];
		n_z = new int[K];
		n_w_z = new int[K][V];
		N_d = new int[numDocuments];



		multiPros = new double[K];
		for (int i = 0; i < K; i++) {
			multiPros[i] = 1.0 / K;
		}

		System.out.println("Corpus size: " + numDocuments + " docs, "
			+ numWordsInCorpus + " words");
		System.out.println("Vocabuary size: " + V);
		System.out.println("Number of topics: " + K);
		System.out.println("alpha: " + alpha);
		System.out.println("beta: " + beta);
		System.out.println("Number of sampling iterations: " + ITERATIONS);
		System.out.println("Number of top topical words: " + topWords);

		initialize();
	}

	/**
	 * Randomly initialize topic assignments
	 */
	public void initialize()
		throws IOException
	{
		System.out.println("Randomly initialzing topic assignments ...");

		long startTime = System.currentTimeMillis();

		z = new int[numDocuments];
		for (int i = 0; i < numDocuments; i++) {
			int topic = FuncUtils.nextDiscrete(multiPros); // Sample a topic
			z[i] = topic;
			m_z[topic]++;

			int docLen = 0;
			for(int n=0; n<corpus.get(i).size(); n++)
			{
				n_w_z[topic][corpus.get(i).get(n)] += 1;

				// N_w_d[m][sData.get(m).get(n)] = sDataNum.get(m).get(n);
				docLen++;
			}

			n_z[topic] += docLen;//documents[m].length;

			N_d[i] = docLen; //documents[m].length;
		}

		initTime =System.currentTimeMillis()-startTime;
	}



	public void inference()
		throws IOException
	{

		writeDictionary();

		System.out.println("Running Gibbs sampling inference: ");

		long startTime = System.currentTimeMillis();

		for (int iter = 1; iter <= ITERATIONS; iter++) {
			if(iter%50==0)
				System.out.print(" " + (iter));
			// System.out.println("\t\tPerplexity: " + computePerplexity());

			for (int m = 0; m < numDocuments; m++)
			{
				//System.out.println("m : " + m);
				int topic = sampleFullConditional(m);
				z[m] = topic;
			}
		}

		iterTime =System.currentTimeMillis()-startTime;

		expName = orgExpName;
		System.out.println();

		System.out.println("Writing output from the last sample ...");
		write();

		System.out.println("Sampling completed!");

		System.out.println("the initiation tims: " + initTime);
		System.out.println("the iteration tims: " + iterTime);

	}

	/**
	 * Sample a topic z_i from the full conditional distribution: p(z_i = j |
	 * z_-i, w) = (n_-i,j(w_i) + beta)/(n_-i,j(.) + W * beta) * (n_-i,j(d_i) +
	 * alpha)/(n_-i,.(d_i) + K * alpha)
	 *
	 * @param m
	 *            document
	 */
	private int sampleFullConditional(int m) {
		// remove z_i from the count variables
		//System.out.println("m: " + m);
		int topic = z[m];
		m_z[topic]--;
		n_z[topic] -= N_d[m];

		for(int n=0; n<corpus.get(m).size(); n++)
			n_w_z[topic][corpus.get(m).get(n)]--;

		// Sample a topic
		for (int tIndex = 0; tIndex < K; tIndex++) {
			multiPros[tIndex] = (m_z[tIndex] + alpha);
			for (int wIndex = 0; wIndex < corpus.get(m).size(); wIndex++) {
				int word = corpus.get(m).get(wIndex);
				multiPros[tIndex] *= (n_w_z[tIndex][word] + beta
						+ occurenceToIndexCount.get(m).get(wIndex) - 1)
						/ (n_z[tIndex] + beta*V + wIndex);
			}
		}
		topic = FuncUtils.nextDiscrete(multiPros);

		// System.out.println("topic: " + topic);
		m_z[topic]++;
		n_z[topic] += N_d[m];


		for(int n=0; n<corpus.get(m).size(); n++)
			n_w_z[topic][corpus.get(m).get(n)]++;


		return topic;
	}

	public void writeParameters()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".paras"));
		writer.write("-model" + "\t" + "DMM");
		writer.write("\n-corpus" + "\t" + corpusPath);
		writer.write("\n-ntopics" + "\t" + K);
		writer.write("\n-alpha" + "\t" + alpha);
		writer.write("\n-beta" + "\t" + beta);
		writer.write("\n-niters" + "\t" + ITERATIONS);
		writer.write("\n-twords" + "\t" + topWords);
		writer.write("\n-name" + "\t" + expName);

		writer.write("\n-initiation time" + "\t" + initTime);
		writer.write("\n-one iteration time" + "\t" + iterTime/ITERATIONS);
		writer.write("\n-total time" + "\t" + (initTime+iterTime));


		writer.close();
	}

	public void writeDictionary()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".vocabulary"));
		for (int id = 0; id < V; id++)
			writer.write(id2WordVocabulary.get(id) + " " + id + "\n");
		writer.close();
	}



	public void writeTopicAssignments()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".topicAssignments"));
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			int topic = z[dIndex];
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				writer.write(topic + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopTopicalWords()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".topWords"));

		for (int tIndex = 0; tIndex < K; tIndex++) {
			//writer.write("Topic" + new Integer(tIndex) + ":");

			Map<Integer, Integer> wordCount = new TreeMap<Integer, Integer>();
			for (int wIndex = 0; wIndex < V; wIndex++) {
				wordCount.put(wIndex, n_w_z[tIndex][wIndex]);
			}
			wordCount = FuncUtils.sortByValueDescending(wordCount);

			Set<Integer> mostLikelyWords = wordCount.keySet();
			int count = 0;
			for (Integer index : mostLikelyWords) {
				if (count < topWords) {
					double pro = (n_w_z[tIndex][index] + beta)
						/ (n_z[tIndex] + beta*V);
					pro = Math.round(pro * 1000000.0) / 1000000.0;
					writer.write( id2WordVocabulary.get(index)+ " " );
					count += 1;
				}
				else {
					writer.write("\n");
					break;
				}
			}
		}
		writer.close();
	}

	public void writeTopicWordPros()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".phi"));
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < V; j++) {
				double pro = (n_w_z[i][j] + beta)
					/ (n_z[i] + beta*V);
				writer.write(pro + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeDocTopicPros()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".theta"));

		for (int i = 0; i < numDocuments; i++) {
			int docSize = corpus.get(i).size();
			double sum = 0.0;
			for (int tIndex = 0; tIndex < K; tIndex++) {
				multiPros[tIndex] = (m_z[tIndex] + alpha);
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = corpus.get(i).get(wIndex);
					multiPros[tIndex] *= (n_w_z[tIndex][word] + beta)
						/ (n_z[tIndex] + beta*V);
				}
				sum += multiPros[tIndex];
			}
			for (int tIndex = 0; tIndex < K; tIndex++) {
				writer.write((multiPros[tIndex] / sum) + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void write()
		throws IOException
	{
		writeParameters();
		writeTopTopicalWords();
		writeDocTopicPros();
		writeTopicAssignments();
		writeTopicWordPros();
	}

	public static void main(String args[])
		throws Exception
	{
		DMM dmm = new DMM("dataset/GoogleNews.txt", 200, 0.1,
			0.1, 50, 10, "GoogleNewsDMM");
		dmm.inference();
	}
}
