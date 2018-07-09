package models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import utility.FuncUtils;

/**
 * jLDADMM: A Java package for the LDA and DMM topic models
 * 
 * Implementation of the one-topic-per-document Dirichlet Multinomial Mixture
 * model, using collapsed Gibbs sampling, as described in:
 * 
 * Jianhua Yin and Jianyong Wang. 2014. A Dirichlet Multinomial Mixture
 * Model-based Approach for Short Text Clustering. In Proceedings of the 20th
 * ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,
 * pages 233–242.
 * 
 * @author: Dat Quoc Nguyen
 */

public class DMM
{
	public double alpha; // Hyper-parameter alpha
	public double beta; // Hyper-parameter alpha
	public int numTopics; // Number of topics
	public int numIterations; // Number of Gibbs sampling iterations
	public int topWords; // Number of most probable words for each topic

	public double alphaSum; // alpha * numTopics
	public double betaSum; // beta * vocabularySize

	public List<List<Integer>> corpus; // Word ID-based corpus
	public List<Integer> topicAssignments; // Topics assignments for documents
	public int numDocuments; // Number of documents in the corpus
	public int numWordsInCorpus; // Number of words in the corpus

	public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
														// given a word
	public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
														// given an ID
	public int vocabularySize; // The number of word types in the corpus

	// Number of documents assigned to a topic
	public int[] docTopicCount;
	// numTopics * vocabularySize matrix
	// Given a topic: number of times a word type assigned to the topic
	public int[][] topicWordCount;
	// Total number of words assigned to a topic
	public int[] sumTopicWordCount;

	// Double array used to sample a topic
	public double[] multiPros;

	// Path to the directory containing the corpus
	public String folderPath;
	// Path to the topic modeling corpus
	public String corpusPath;

	// Given a document, number of times its i^{th} word appearing from
	// the first index to the i^{th}-index in the document
	// Example: given a document of "a a b a b c d c". We have: 1 2 1 3 2 1 1 2
	public List<List<Integer>> occurenceToIndexCount;

	public String expName = "DMMmodel";
	public String orgExpName = "DMMmodel";
	public String tAssignsFilePath = "";
	public int savestep = 0;

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
		this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
			inTopWords, inExpName, "", 0);

	}

	public DMM(String pathToCorpus, int inNumTopics,
		double inAlpha, double inBeta, int inNumIterations, int inTopWords,
		String inExpName, String pathToTAfile)
		throws Exception
	{
		this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
			inTopWords, inExpName, pathToTAfile, 0);

	}

	public DMM(String pathToCorpus, int inNumTopics,
		double inAlpha, double inBeta, int inNumIterations, int inTopWords,
		String inExpName, int inSaveStep)
		throws Exception
	{
		this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
			inTopWords, inExpName, "", inSaveStep);

	}

	public DMM(String pathToCorpus, int inNumTopics,
		double inAlpha, double inBeta, int inNumIterations, int inTopWords,
		String inExpName, String pathToTAfile, int inSaveStep)
		throws IOException
	{
		alpha = inAlpha;
		beta = inBeta;
		numTopics = inNumTopics;
		numIterations = inNumIterations;
		topWords = inTopWords;
		savestep = inSaveStep;
		expName = inExpName;
		orgExpName = expName;
		corpusPath = pathToCorpus;
		folderPath = pathToCorpus.substring(
			0,
			Math.max(pathToCorpus.lastIndexOf("/"),
				pathToCorpus.lastIndexOf("\\")) + 1);

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
				HashMap<String, Integer> wordOccurenceToIndexInDocCount = new HashMap<String, Integer>();

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
					if (wordOccurenceToIndexInDocCount.containsKey(word)) {
						times = wordOccurenceToIndexInDocCount.get(word);
					}
					times += 1;
					wordOccurenceToIndexInDocCount.put(word, times);
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

		vocabularySize = word2IdVocabulary.size();
		docTopicCount = new int[numTopics];
		topicWordCount = new int[numTopics][vocabularySize];
		sumTopicWordCount = new int[numTopics];

		multiPros = new double[numTopics];
		for (int i = 0; i < numTopics; i++) {
			multiPros[i] = 1.0 / numTopics;
		}

		alphaSum = numTopics * alpha;
		betaSum = vocabularySize * beta;

		System.out.println("Corpus size: " + numDocuments + " docs, "
			+ numWordsInCorpus + " words");
		System.out.println("Vocabuary size: " + vocabularySize);
		System.out.println("Number of topics: " + numTopics);
		System.out.println("alpha: " + alpha);
		System.out.println("beta: " + beta);
		System.out.println("Number of sampling iterations: " + numIterations);
		System.out.println("Number of top topical words: " + topWords);

		tAssignsFilePath = pathToTAfile;
		if (tAssignsFilePath.length() > 0)
			initialize(tAssignsFilePath);
		else
			initialize();
	}

	/**
	 * Randomly initialize topic assignments
	 */
	public void initialize()
		throws IOException
	{
		System.out.println("Randomly initialzing topic assignments ...");
		topicAssignments = new ArrayList<Integer>();
		for (int i = 0; i < numDocuments; i++) {
			int topic = FuncUtils.nextDiscrete(multiPros); // Sample a topic
			docTopicCount[topic] += 1;
			int docSize = corpus.get(i).size();
			for (int j = 0; j < docSize; j++) {
				topicWordCount[topic][corpus.get(i).get(j)] += 1;
				sumTopicWordCount[topic] += 1;
			}
			topicAssignments.add(topic);
		}
	}

	/**
	 * Initialize topic assignments from a given file
	 */
	public void initialize(String pathToTopicAssignmentFile)
	{
		System.out.println("Reading topic-assigment file: "
			+ pathToTopicAssignmentFile);

		topicAssignments = new ArrayList<Integer>();

		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(pathToTopicAssignmentFile));
			int docID = 0;
			int numWords = 0;
			for (String line; (line = br.readLine()) != null;) {
				String[] strTopics = line.trim().split("\\s+");
				int topic = new Integer(strTopics[0]) % numTopics;
				docTopicCount[topic] += 1;
				for (int j = 0; j < strTopics.length; j++) {
					// Increase counts
					topicWordCount[topic][corpus.get(docID).get(j)] += 1;
					sumTopicWordCount[topic] += 1;

					numWords++;
				}
				topicAssignments.add(topic);
				docID++;
			}

			if ((docID != numDocuments) || (numWords != numWordsInCorpus)) {
				System.out
					.println("The topic modeling corpus and topic assignment file are not consistent!!!");
				throw new Exception();
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void inference()
		throws IOException
	{
		writeParameters();
		writeDictionary();

		System.out.println("Running Gibbs sampling inference: ");

		for (int iter = 1; iter <= numIterations; iter++) {

			System.out.println("\tSampling iteration: " + (iter));
			// System.out.println("\t\tPerplexity: " + computePerplexity());

			sampleInSingleIteration();

			if ((savestep > 0) && (iter % savestep == 0)
				&& (iter < numIterations)) {
				System.out.println("\t\tSaving the output from the " + iter
					+ "^{th} sample");
				expName = orgExpName + "-" + iter;
				write();
			}
		}
		expName = orgExpName;

		System.out.println("Writing output from the last sample ...");
		write();

		System.out.println("Sampling completed!");

	}

	public void sampleInSingleIteration()
	{
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int topic = topicAssignments.get(dIndex);
			List<Integer> document = corpus.get(dIndex);
			int docSize = document.size();

			// Decrease counts
			docTopicCount[topic] -= 1;
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				int word = document.get(wIndex);
				topicWordCount[topic][word] -= 1;
				sumTopicWordCount[topic] -= 1;
			}

			// Sample a topic
			for (int tIndex = 0; tIndex < numTopics; tIndex++) {
				multiPros[tIndex] = (docTopicCount[tIndex] + alpha);
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = document.get(wIndex);
					multiPros[tIndex] *= (topicWordCount[tIndex][word] + beta
						+ occurenceToIndexCount.get(dIndex).get(wIndex) - 1)
						/ (sumTopicWordCount[tIndex] + betaSum + wIndex);
				}
			}
			topic = FuncUtils.nextDiscrete(multiPros);

			// Increase counts
			docTopicCount[topic] += 1;
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				int word = document.get(wIndex);
				topicWordCount[topic][word] += 1;
				sumTopicWordCount[topic] += 1;
			}
			// Update topic assignments
			topicAssignments.set(dIndex, topic);
		}
	}

	public void writeParameters()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".paras"));
		writer.write("-model" + "\t" + "DMM");
		writer.write("\n-corpus" + "\t" + corpusPath);
		writer.write("\n-ntopics" + "\t" + numTopics);
		writer.write("\n-alpha" + "\t" + alpha);
		writer.write("\n-beta" + "\t" + beta);
		writer.write("\n-niters" + "\t" + numIterations);
		writer.write("\n-twords" + "\t" + topWords);
		writer.write("\n-name" + "\t" + expName);
		if (tAssignsFilePath.length() > 0)
			writer.write("\n-initFile" + "\t" + tAssignsFilePath);
		if (savestep > 0)
			writer.write("\n-sstep" + "\t" + savestep);

		writer.close();
	}

	public void writeDictionary()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".vocabulary"));
		for (int id = 0; id < vocabularySize; id++)
			writer.write(id2WordVocabulary.get(id) + " " + id + "\n");
		writer.close();
	}

	public void writeIDbasedCorpus()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".IDcorpus"));
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				writer.write(corpus.get(dIndex).get(wIndex) + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopicAssignments()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".topicAssignments"));
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			int topic = topicAssignments.get(dIndex);
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

		for (int tIndex = 0; tIndex < numTopics; tIndex++) {
			writer.write("Topic" + new Integer(tIndex) + ":");

			Map<Integer, Integer> wordCount = new TreeMap<Integer, Integer>();
			for (int wIndex = 0; wIndex < vocabularySize; wIndex++) {
				wordCount.put(wIndex, topicWordCount[tIndex][wIndex]);
			}
			wordCount = FuncUtils.sortByValueDescending(wordCount);

			Set<Integer> mostLikelyWords = wordCount.keySet();
			int count = 0;
			for (Integer index : mostLikelyWords) {
				if (count < topWords) {
					double pro = (topicWordCount[tIndex][index] + beta)
						/ (sumTopicWordCount[tIndex] + betaSum);
					pro = Math.round(pro * 1000000.0) / 1000000.0;
					writer.write(" " + id2WordVocabulary.get(index) + "(" + pro
						+ ")");
					count += 1;
				}
				else {
					writer.write("\n\n");
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
		for (int i = 0; i < numTopics; i++) {
			for (int j = 0; j < vocabularySize; j++) {
				double pro = (topicWordCount[i][j] + beta)
					/ (sumTopicWordCount[i] + betaSum);
				writer.write(pro + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopicWordCount()
		throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
			+ expName + ".WTcount"));
		for (int i = 0; i < numTopics; i++) {
			for (int j = 0; j < vocabularySize; j++) {
				writer.write(topicWordCount[i][j] + " ");
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
			for (int tIndex = 0; tIndex < numTopics; tIndex++) {
				multiPros[tIndex] = (docTopicCount[tIndex] + alpha);
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = corpus.get(i).get(wIndex);
					multiPros[tIndex] *= (topicWordCount[tIndex][word] + beta)
						/ (sumTopicWordCount[tIndex] + betaSum);
				}
				sum += multiPros[tIndex];
			}
			for (int tIndex = 0; tIndex < numTopics; tIndex++) {
				writer.write((multiPros[tIndex] / sum) + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void write()
		throws IOException
	{
		writeTopTopicalWords();
		writeDocTopicPros();
		writeTopicAssignments();
		writeTopicWordPros();
	}

	public static void main(String args[])
		throws Exception
	{
		DMM dmm = new DMM("test/corpus.txt", 7, 0.1,
			0.1, 2000, 20, "testDMM");
		dmm.inference();
	}
}
