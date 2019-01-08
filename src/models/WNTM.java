package models;

import utility.FuncUtils;

import java.io.*;
import java.util.*;
import java.text.DecimalFormat;

/**
 * * WNTM: A Java package for the short text topic models
 *
 * Implementation of the WNTM topic modeling, using collapsed Gibbs sampling, as described in:
 *
 * Yuan Zuo, Jichang Zhao, Ke Xu. Word network topic model: a simple but general solution
 *for short and imbalanced texts. In Knowledge And Information System, 2016.
 *
 * @author: Jipeng Qiang on 18/6/6.
 */


public class WNTM
{

    public double alpha; // Hyper-parameter alpha
    public double beta; // Hyper-parameter alpha
    public int numTopics; // Number of topics
    public int numIterations; // Number of Gibbs sampling iterations
    public int topWords; // Number of most probable words for each topic

    public double alphaSum; // alpha * numTopics
    public double betaSum; // beta * vocabularySize

    public List<List<Integer>> corpus; // Word ID-based corpus

    //Map<Integer, Map<Integer, Double>> wordGraph = new HashMap<Integer, Map<Integer, Double>>();
    //Map<Integer,Integer> wordDegree = new HashMap<Integer,Integer>();

    int wordNeighbor[][];
    public List<List<Integer>> pseudo_corpus; // Word ID-based corpus
    public int z[][]; // Topics assignments for words
    // in the corpus
    public int numDocuments; // Number of documents in the corpus
    public int numPseudoDocuments;
    public int numWordsInCorpus; // Number of words in the corpus

    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
    // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
    // given an ID
    public int vocabularySize; // The number of word types in the corpus

    // numDocuments * numTopics matrix
    // Given a document: number of its words assigned to each topic
    public int[][] pseudocTopicCount;
    // Number of words in every document
    //public int[] sumPseudocTopicCount;
    // numTopics * vocabularySize matrix
    // Given a topic: number of times a word type assigned to the topic
    public int[][] topicWordCount;
    // Total number of words assigned to a topic
    public int[] sumTopicWordCount;

    // Double array used to sample a topic
    public double[] multiPros;
    public double[][] phi;

    public int windowSize;
    // Path to the directory containing the corpus
    public String folderPath;
    // Path to the topic modeling corpus
    public String corpusPath;

    public String expName = "WNTMmodel";
    public String orgExpName = "WNTMmodel";
    public String tAssignsFilePath = "";
    public int savestep = 0;

    public double initTime = 0;
    public double iterTime = 0;

    DecimalFormat df = new DecimalFormat("#.000");

    public WNTM(String pathToCorpus, int inNumTopics,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords, int inWindowSize)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, inWindowSize, "WNTMmodel");
    }

    public WNTM(String pathToCorpus, int inNumTopics,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords, int inWindowSize,
               String inExpName)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, inWindowSize, inExpName, "", 0);
    }

    public WNTM(String pathToCorpus, int inNumTopics,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords, int inWindowSize,
               String inExpName, String pathToTAfile)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, inWindowSize, inExpName, pathToTAfile, 0);
    }

    public WNTM(String pathToCorpus, int inNumTopics,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords,int inWindowSize,
               String inExpName, int inSaveStep)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, inWindowSize, inExpName, "", inSaveStep);
    }

    public WNTM(String pathToCorpus, int inNumTopics,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords,int inWindowSize,
               String inExpName, String pathToTAfile, int inSaveStep)
            throws Exception
    {

        alpha = inAlpha;
        beta = inBeta;
        numTopics = inNumTopics;
        windowSize = inWindowSize;
        numIterations = inNumIterations;
        topWords = inTopWords;
        savestep = inSaveStep;
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
        pseudo_corpus = new ArrayList<List<Integer>>();
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
                }

                numDocuments++;
                numWordsInCorpus += document.size();
                corpus.add(document);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        vocabularySize = word2IdVocabulary.size(); // vocabularySize = indexWord

        wordNeighbor = new int[vocabularySize][vocabularySize];

        long startTime = System.currentTimeMillis();

        constructWordGraph();
        constructPseudoCorpus();
        numPseudoDocuments = vocabularySize;
        pseudocTopicCount = new int[numPseudoDocuments][numTopics];
        topicWordCount = new int[numTopics][vocabularySize];
        //sumPseudocTopicCount= new int[numPseudoDocuments];
        sumTopicWordCount = new int[numTopics];

        phi = new double[numTopics][vocabularySize];

        multiPros = new double[numTopics];
        for (int i = 0; i < numTopics; i++) {
            multiPros[i] = 1.0 / numTopics;
        }

        alphaSum = numTopics * alpha;
        betaSum = vocabularySize * beta;



        tAssignsFilePath = pathToTAfile;
        if (tAssignsFilePath.length() > 0)
            initialize();
        else
            initialize();

        initTime =System.currentTimeMillis()-startTime;

        System.out.println("Corpus size: " + numDocuments + " docs, "
                + numWordsInCorpus + " words");
        System.out.println("Vocabuary size: " + vocabularySize);
        System.out.println("Number of topics: " + numTopics);
        System.out.println("alpha: " + alpha);
        System.out.println("beta: " + beta);
        System.out.println("Number of sampling iterations: " + numIterations);
        System.out.println("Number of top topical words: " + topWords);
    }

    public void constructWordGraph()
    {
        for (int i = 0; i < numDocuments; i++) {

            int docSize = corpus.get(i).size();

            if(docSize<=windowSize){
                for (int k = 0;  k<docSize-1; k++) {
                    int wordId = corpus.get(i).get(k);

                    for (int m = k + 1;  m<docSize; m++) {
                        int nextId = corpus.get(i).get(m);

                        //addEdge(wordId,nextId);

                        if(wordId!=nextId) {
                            wordNeighbor[wordId][nextId] += 1;
                            wordNeighbor[nextId][wordId] += 1;
                        }
                    }
                }
            }else {
                for (int j = 0; j < docSize - windowSize + 1; j++) {

                    for (int k = j; k < j + windowSize - 1; k++) {
                        int wordId = corpus.get(i).get(k);

                        for (int m = k + 1; m < j + windowSize; m++) {
                            int nextId = corpus.get(i).get(m);

                            //addEdge(wordId, nextId);
                            if(wordId!=nextId) {
                                wordNeighbor[wordId][nextId] += 1;
                                wordNeighbor[nextId][wordId] += 1;
                            }
                        }
                    }
                }
            }
        }
        //show();
    }

    public void constructPseudoCorpus()
    {

        for(int i=0; i<vocabularySize; i++)
        {
            ArrayList<Integer> onePseudo = new ArrayList<Integer>();

            for(int j=0; j<vocabularySize && j!=i; j++)
            {
                if(wordNeighbor[i][j]==0)
                    continue;

                for(int n=0; n<wordNeighbor[i][j]; n++)
                    onePseudo.add(j);
            }
            pseudo_corpus.add(onePseudo);
        }

        //show();
    }

    /**
     * Randomly initialize topic assignments
     */
    public void initialize()
            throws IOException
    {
        System.out.println("Randomly initializing topic assignments ...");

        z = new int[numPseudoDocuments][];

        for (int i = 0; i < numPseudoDocuments; i++) {

            int docSize = pseudo_corpus.get(i).size();
            z[i] = new int[docSize];
            for (int j = 0; j < docSize; j++) {
                int topic = FuncUtils.nextDiscrete(multiPros); // Sample a topic
                // Increase counts
                pseudocTopicCount[i][topic] += 1;
                topicWordCount[topic][pseudo_corpus.get(i).get(j)] += 1;
                //sumPseudocTopicCount[i] += 1;
                sumTopicWordCount[topic] += 1;

                z[i][j] = topic;
            }
        }
    }

    public void inference()
            throws IOException
    {

        writeDictionary();

        System.out.println("Running Gibbs sampling inference: ");

        long startTime = System.currentTimeMillis();

        for (int iter = 1; iter <= numIterations; iter++) {

            if(iter%50 == 0)
                System.out.print(" " + (iter));
            // System.out.println("\t\tPerplexity: " + computePerplexity());

            sampleInSingleIteration();


        }
        expName = orgExpName;

        iterTime =System.currentTimeMillis()-startTime;
        System.out.println();
        System.out.println("Writing output from the last sample ...");
        write();

        System.out.println("Sampling completed!");

    }

    /**
     * Sample a topic z_i from the full conditional distribution: p(z_i = j |
     * z_-i, w) = (n_-i,j(w_i) + beta)/(n_-i,j(.) + W * beta) * (n_-i,j(d_i) +
     * alpha)/(n_-i,.(d_i) + K * alpha)
     *
     * @param m
     *            document
     * @param n
     *            word
     */
    private int sampleFullConditional(int m, int n) {

        // remove z_i from the count variables
        int topic = z[m][n];

        int wordId = pseudo_corpus.get(m).get(n);

        topicWordCount[topic][wordId]--;
        pseudocTopicCount[m][topic]--;
        sumTopicWordCount[topic]--;


        //int maxK = -1;
        // double value = 0;
        for (int k = 0; k < numTopics; k++) {
            multiPros[k] = (topicWordCount[k][wordId] + beta) / (sumTopicWordCount[k] + betaSum)
                    * (pseudocTopicCount[m][k] + alpha);// / (ndsum[m] + K * alpha);
        }

        topic = FuncUtils.nextDiscrete(multiPros);
        // topic = maxK;
        // add newly estimated z_i to count variables
        topicWordCount[topic][wordId]++;
        pseudocTopicCount[m][topic]++;
        sumTopicWordCount[topic]++;
        // ndsum[m]++;

        return topic;
    }

    public void sampleInSingleIteration()
    {
        for (int dIndex = 0; dIndex < numPseudoDocuments; dIndex++) {
            int docSize = pseudo_corpus.get(dIndex).size();
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                // Get current word and its topic
                int topic = sampleFullConditional(dIndex,wIndex);
                // Update topic assignments
                z[dIndex][wIndex] = topic;

            }
        }
    }

    public void writeParameters()
            throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".paras"));
        writer.write("-model" + "\t" + "WNTM");
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

        writer.write("\n-initiation time" + "\t" + initTime);
        writer.write("\n-one iteration time" + "\t" + iterTime/numIterations);
        writer.write("\n-total time" + "\t" + (initTime+iterTime));
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

    public void writeTopTopicalWords()
            throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".topWords"));

        for (int tIndex = 0; tIndex < numTopics; tIndex++) {
            //writer.write("Topic" + new Integer(tIndex) + ":");

            Map<Integer, Double> wordCount = new TreeMap<Integer, Double>();
            for (int wIndex = 0; wIndex < vocabularySize; wIndex++) {
                wordCount.put(wIndex, phi[tIndex][wIndex]);
            }
            wordCount = FuncUtils.sortByValueDescending(wordCount);

            Set<Integer> mostLikelyWords = wordCount.keySet();
            int count = 0;
            for (Integer index : mostLikelyWords) {
                if (count < topWords){
                    writer.write(id2WordVocabulary.get(index) + " ");
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

        double phiSum[] = new double[numTopics];
        for (int k = 0; k < numTopics; k++) {
            for (int j = 0; j < vocabularySize; j++) {
                double pro = (pseudocTopicCount[j][k] + alpha)
                        / (pseudo_corpus.get(j).size() + alphaSum);
                phi[k][j] = pro;
                phiSum[k] += pro;
              //  writer.write(df.format(pro) + " ");
            }
            //writer.write("\n");
        }
        //writer.close();

        for (int k = 0; k < numTopics; k++) {
            for (int j = 0; j < vocabularySize; j++) {
               // double pro = (pseudocTopicCount[j][k] + alpha)
                 //       / (pseudo_corpus.get(j).size() + alphaSum);
                phi[k][j] /= phiSum[k];
                //phiSum[k] += pro;
                writer.write(phi[k][j] + " ");
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

            int len = corpus.get(i).size();
            for (int j = 0; j < numTopics; j++) {
                double pro = 0;
                for(int wIndex=0; wIndex<len; wIndex++)
                {
                    int wordId = corpus.get(i).get(wIndex);
                    pro += phi[j][wordId]/len;
                }

                writer.write(pro + " ");
            }
            writer.write("\n");
        }


        writer.close();
    }

    public void write()
            throws IOException
    {

        writeTopicWordPros();


        writeDocTopicPros();

        writeTopTopicalWords();

        writeParameters();
    }

    public static void main(String args[])
            throws Exception
    {
        WNTM wntm = new WNTM("dataset/GoogleNews.txt", 152, 0.1,
                0.01, 1000, 10, 10, "GoogleNewsWNTM");
        wntm.inference();
    }
}
