package models;

import utility.FuncUtils;

import java.io.*;
import java.util.*;

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
public class WNTM {

    public double alpha; // Hyper-parameter alpha
    public double beta; // Hyper-parameter alpha
    public int numTopics; // Number of topics
    public int numIterations; // Number of Gibbs sampling iterations
    public int topWords; // Number of most probable words for each topic

    public double alphaSum; // alpha * numTopics
    public double betaSum; // beta * vocabularySize

    public List<List<Integer>> corpus; // Word ID-based corpus

    Map<Integer, Map<Integer, Double>> wordGraph = new HashMap<Integer, Map<Integer, Double>>();
    Map<Integer,Integer> wordDegree = new HashMap<Integer,Integer>();
    public List<List<Integer>> pseudo_corpus; // Word ID-based corpus
    public List<List<Integer>> topicAssignments; // Topics assignments for words
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
    public int[] sumPseudocTopicCount;
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

        long startTime = System.currentTimeMillis();

        constructWordGraph();
        constructPseudoCorpus();
        numPseudoDocuments = pseudo_corpus.size();
        pseudocTopicCount = new int[numPseudoDocuments][numTopics];
        topicWordCount = new int[numTopics][vocabularySize];
        sumPseudocTopicCount= new int[numPseudoDocuments];
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

    public boolean containsEdge(int p1, int p2) {
        if (!wordGraph.containsKey(p1)) {
            return false;
        }
        if (!wordGraph.containsKey(p2)) {
            return false;
        }
        if (!wordGraph.get(p1).containsKey(p2)) {
            return false;
        }
        if (!wordGraph.get(p2).containsKey(p1)) {
            return false;
        }
        return true;
    }


    public void addEdge(int p1, int p2) {
        if (this.containsEdge(p1, p2)) {
            wordGraph.get(p1).put(p2, wordGraph.get(p1).get(p2)+1);
            wordGraph.get(p2).put(p1, wordGraph.get(p2).get(p1)+1);
            wordDegree.put(p1,wordDegree.get(p1)+1);
            wordDegree.put(p2,wordDegree.get(p2)+1);
            return;
        }
        if (!wordGraph.containsKey(p1)) {
            wordGraph.put(p1, new HashMap<Integer, Double>());
            wordDegree.put(p1,0);
        }
        if (!wordGraph.containsKey(p2)) {
            wordGraph.put(p2, new HashMap<Integer, Double>());
            wordDegree.put(p2,0);
        }
        wordGraph.get(p1).put(p2, 1.0);
        wordGraph.get(p2).put(p1, 1.0);
        wordDegree.put(p1,wordDegree.get(p1)+1);
        wordDegree.put(p2,wordDegree.get(p2)+1);
    }


    public void show() {
        Set<Map.Entry<Integer, Map<Integer, Double>>> set = wordGraph.entrySet();
        for (Map.Entry<Integer, Map<Integer, Double>> e : set) {
            Set<Map.Entry<Integer, Double>> temp = e.getValue().entrySet();
            if (temp.size() > 0) {
                System.out.print(id2WordVocabulary.get(e.getKey()) + " -> ");
                for (Map.Entry<Integer, Double> e1 : temp) {
                    System.out.print(id2WordVocabulary.get(e1.getKey()) + "(" + e1.getValue() + ") ");
                }
                System.out.println();
            }
        }

    }
    public void constructWordGraph()
    {
        for (int i = 0; i < numDocuments; i++) {

            int docSize = corpus.get(i).size();
            for (int j = 0; j < docSize - windowSize+1; j++) {

                for (int k = j;  k<j+windowSize-1; k++) {
                    int wordId = corpus.get(i).get(k);

                    for (int m = k + 1;  m<j+windowSize; m++) {
                        int nextId = corpus.get(i).get(m);

                        addEdge(wordId,nextId);
                    }
                }
            }
        }
        //show();
    }

    public void constructPseudoCorpus()
    {
        Iterator<Map.Entry<Integer, Integer>> it = wordDegree.entrySet().iterator();

        while(it.hasNext()){

            Map.Entry<Integer, Integer> entry = it.next();

            System.out.println("key= "+entry.getKey()+" and value= "+entry.getValue());

        }

        Set<Map.Entry<Integer, Map<Integer, Double>>> set = wordGraph.entrySet();
        for (Map.Entry<Integer, Map<Integer, Double>> e : set) {
            Set<Map.Entry<Integer, Double>> temp = e.getValue().entrySet();

            ArrayList<Integer> onePseudo = new ArrayList<Integer>();
            if (temp.size() > 0) {
               // System.out.print(id2WordVocabulary.get(e.getKey()) + " -> ");
                int degree1 = wordDegree.get(e.getKey());
                double activity = (double)degree1/temp.size();
                for (Map.Entry<Integer, Double> e1 : temp) {
                    int degree2 = wordDegree.get(e1.getKey());

                    double activity2 = (double)degree2/wordGraph.get(e1.getKey()).size();

                    if(activity>activity2)
                        activity = activity2;

                    int reweight = (int)Math.ceil(wordGraph.get(e.getKey()).get(e1.getKey())/activity);

                    for(int i=0; i<reweight; i++)
                        onePseudo.add(e1.getKey());
                    wordGraph.get(e.getKey()).put(e1.getKey(), (double)reweight);

                    //System.out.print(id2WordVocabulary.get(e1.getKey()) + "(" + e1.getValue() + ") ");
                }
                pseudo_corpus.add(onePseudo);
                //System.out.println();
            }
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

        topicAssignments = new ArrayList<List<Integer>>();

        for (int i = 0; i < numPseudoDocuments; i++) {
            List<Integer> topics = new ArrayList<Integer>();
            int docSize = pseudo_corpus.get(i).size();
            for (int j = 0; j < docSize; j++) {
                int topic = FuncUtils.nextDiscrete(multiPros); // Sample a topic
                // Increase counts
                pseudocTopicCount[i][topic] += 1;
                topicWordCount[topic][pseudo_corpus.get(i).get(j)] += 1;
                sumPseudocTopicCount[i] += 1;
                sumTopicWordCount[topic] += 1;

                topics.add(topic);
            }
            topicAssignments.add(topics);
        }
    }

    public void inference()
            throws IOException
    {

        writeDictionary();

        System.out.println("Running Gibbs sampling inference: ");

        long startTime = System.currentTimeMillis();

        for (int iter = 1; iter <= numIterations; iter++) {

            System.out.println("\tSampling iteration: " + (iter));
            // System.out.println("\t\tPerplexity: " + computePerplexity());

            sampleInSingleIteration();

            if ((savestep > 0) && (iter % savestep == 0)
                    && (iter < numIterations)) {
                System.out.println("\t\tSaving the output from the " + iter
                        + "^{th} sample");
                expName = orgExpName + "-" + iter;
               // write();
            }
        }
        expName = orgExpName;

        iterTime =System.currentTimeMillis()-startTime;

        System.out.println("Writing output from the last sample ...");
        write();

        System.out.println("Sampling completed!");

    }

    public void sampleInSingleIteration()
    {
        for (int dIndex = 0; dIndex < numPseudoDocuments; dIndex++) {
            int docSize = pseudo_corpus.get(dIndex).size();
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                // Get current word and its topic
                int topic = topicAssignments.get(dIndex).get(wIndex);
                int word = pseudo_corpus.get(dIndex).get(wIndex);

                // Decrease counts
                pseudocTopicCount[dIndex][topic] -= 1;
                // docTopicSum[dIndex] -= 1;
                topicWordCount[topic][word] -= 1;
                sumTopicWordCount[topic] -= 1;

                // Sample a topic
                for (int tIndex = 0; tIndex < numTopics; tIndex++) {
                    multiPros[tIndex] = (pseudocTopicCount[dIndex][tIndex] + alpha)
                            * ((topicWordCount[tIndex][word] + beta) / (sumTopicWordCount[tIndex] + betaSum));
                    // multiPros[tIndex] = ((docTopicCount[dIndex][tIndex] +
                    // alpha) /
                    // (docTopicSum[dIndex] + alphaSum))
                    // * ((topicWordCount[tIndex][word] + beta) /
                    // (topicWordSum[tIndex] + betaSum));
                }
                topic = FuncUtils.nextDiscrete(multiPros);

                // Increase counts
                pseudocTopicCount[dIndex][topic] += 1;
                // docTopicSum[dIndex] += 1;
                topicWordCount[topic][word] += 1;
                sumTopicWordCount[topic] += 1;

                // Update topic assignments
                topicAssignments.get(dIndex).set(wIndex, topic);
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
        for (int i = 0; i < numTopics; i++) {
            for (int j = 0; j < vocabularySize; j++) {
                double pro = (topicWordCount[i][j] + beta)
                        / (sumTopicWordCount[i] + betaSum);
                phi[i][j] = pro;
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

            int len = corpus.get(i).size();
            for (int j = 0; j < numTopics; j++) {
                double pro = 0;
                for(int wIndex=0; wIndex<len; wIndex++)
                {
                    int wordId = corpus.get(i).get(wIndex);
                    pro += phi[j][wIndex]/len;
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
        writeTopTopicalWords();
        writeTopicWordPros();
        writeDocTopicPros();

        writeParameters();
    }

    public static void main(String args[])
            throws Exception
    {
        WNTM wntm = new WNTM("test/corpus.txt", 7, 0.1,
                0.01, 100, 10, 20, "testWNTM");
        wntm.inference();
    }
}
