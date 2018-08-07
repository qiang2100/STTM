package models;

import utility.FuncUtils;

import java.io.*;
import java.util.*;

/**
 * STTM: A Java package for short text topic models
 *
 * Implementation of PTM, using collapsed Gibbs sampling, as described in:
 *
 * Yuan Zuo, Junjie Wu, Hui Zhang, Hao Lin, Fei Wang, Ke Xu, and Hui Xiong. 2016. Topic modeling of short texts: A
 *pseudo-document view. In KDD. ACM, 2105–2114.
 *
 * @author: Jipeng Qiang
 */
public class PTM {

    public double alpha; // Hyper-parameter alpha
    public double beta; // Hyper-parameter beta

    public int numTopics; // Number of topics
    public int numIterations; // Number of Gibbs sampling iterations
    public int topWords; // Number of most probable words for each topic

    public double alphaSum; // alpha * numTopics
    public double betaSum; // beta * vocabularySize

    ArrayList<int[]> Corpus = new ArrayList<>(); // Word ID-based corpus

    public int numShorDoc; // Number of documents in the corpus
    public int numWordsInCorpus; // Number of words in the corpus

    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
    // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
    // given an ID
    public int vocabularySize; // The number of word types in the corpus



    public int numLongDoc;

    //public ArrayList<ArrayList<int[]>> assignmentList  = new ArrayList<ArrayList<int[]>>();// [topic][pseudo-doc]
    private int[][] U; // [topic][pseudo-doc];
    private int[] longDocShortCnts; // U_{.j};  the number of short texts belonging to each long document
    private int[] longDocLength; //the number of words in each long document;
    private int[] docToLong; // pre-long document for each short text
    ArrayList<int[]> wordToTopic = new ArrayList<>(); // topic for each word
    private int[][] V; // [word][topic];
    private int[] topicCnts; // V_{.k}
    private int[][] longDocWordCnts; // [pseudo-doc][word]
    private int[][] shortDocTopicCnts; //[short][topic]
    //private int tokenSize;

    // Double array used to sample a topic
    public double[] multiLongPros;
    public double[] multiTopicPros;

    // Path to the directory containing the corpus
    public String folderPath;
    // Path to the topic modeling corpus
    public String corpusPath;


    public String expName = "PTMmodel";
    public String orgExpName = "PTMmodel";
    public String tAssignsFilePath = "";
    public int savestep = 0;


    public PTM(String pathToCorpus, int inNumTopics, int num_longDoc,
                double inAlpha, double inBeta, int inNumIterations, int inTopWords)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc, inAlpha, inBeta, inNumIterations,
                inTopWords, "PTMmodel");
    }

    public PTM(String pathToCorpus, int inNumTopics, int num_longDoc,
                double inAlpha, double inBeta, int inNumIterations, int inTopWords,
                String inExpName)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc, inAlpha, inBeta, inNumIterations,
                inTopWords, inExpName, "", 0);

    }

    public PTM(String pathToCorpus, int inNumTopics, int num_longDoc,
                double inAlpha, double inBeta, int inNumIterations, int inTopWords,
                String inExpName, String pathToTAfile)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc, inAlpha, inBeta, inNumIterations,
                inTopWords, inExpName, pathToTAfile, 0);

    }

    public PTM(String pathToCorpus, int inNumTopics, int num_longDoc,
                double inAlpha, double inBeta,  int inNumIterations, int inTopWords,
                String inExpName, int inSaveStep)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc,  inAlpha, inBeta,  inNumIterations,
                inTopWords, inExpName, "", inSaveStep);

    }

    public PTM(String pathToCorpus, int inNumTopics, int num_longDoc,
                double inAlpha, double inBeta,  int inNumIterations, int inTopWords,
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
        numLongDoc = num_longDoc;


        folderPath = "results/";
        File dir = new File(folderPath);
        if (!dir.exists())
            dir.mkdir();

        System.out.println("Reading topic modeling corpus: " + pathToCorpus);

        word2IdVocabulary = new HashMap<String, Integer>();
        id2WordVocabulary = new HashMap<Integer, String>();


        numShorDoc = 0;
        numWordsInCorpus = 0;

        BufferedReader br = null;
        try {
            int indexWord = -1;
            br = new BufferedReader(new FileReader(pathToCorpus));
            for (String doc; (doc = br.readLine()) != null;) {

                if (doc.trim().length() == 0)
                    continue;

                String[] words = doc.trim().split("\\s+");
                //List<Integer> document = new ArrayList<Integer>();
                int [] document = new int[words.length];

                int ind = 0;
                for (String word : words) {
                    if (word2IdVocabulary.containsKey(word)) {
                        document[ind++] = word2IdVocabulary.get(word);

                    }
                    else {
                        indexWord += 1;
                        word2IdVocabulary.put(word, indexWord);
                        id2WordVocabulary.put(indexWord, word);

                        document[ind++] = indexWord;
                    }
                }

                numShorDoc++;
                numWordsInCorpus += document.length;
                Corpus.add(document);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        vocabularySize = word2IdVocabulary.size(); // vocabularySize = indexWord
        docToLong = new int[numShorDoc];

        multiTopicPros = new double[numTopics];
        for (int i = 0; i < numTopics; i++) {
            multiTopicPros[i] = 1.0 / numTopics;
        }

        multiLongPros = new double[numLongDoc];
        for (int i = 0; i < numLongDoc; i++) {
            multiLongPros[i] = 1.0 / numLongDoc;
        }

        alphaSum = numTopics * alpha;
        betaSum = vocabularySize * beta;

        // init the counter
        U = new int[numTopics][numLongDoc];
        longDocShortCnts = new int[numLongDoc]; // U_{.j};
        longDocLength = new int[numLongDoc];
        V = new int[vocabularySize][numTopics]; // [word][topic];
        topicCnts = new int[numTopics]; // V_{.k}
        longDocWordCnts = new int[numLongDoc][vocabularySize];
        shortDocTopicCnts = new int[numShorDoc][numTopics];
        System.out.println("Corpus size: " + numShorDoc + " docs, "
                + numWordsInCorpus + " words");
        System.out.println("Vocabuary size: " + vocabularySize);
        System.out.println("Number of topics: " + numTopics);
        System.out.println("alpha: " + alpha);
        System.out.println("beta: " + beta);
        System.out.println("Number of sampling iterations: " + numIterations);
        System.out.println("Number of top topical words: " + topWords);

        tAssignsFilePath = pathToTAfile;
        if (tAssignsFilePath.length() > 0)
            initialize();
        else
            initialize();
    }

    /**
     * Randomly initialize topic assignments
     */
    public void initialize()
            throws IOException
    {
        System.out.println("Randomly initializing topic assignments ...");

        for (int d = 0; d < numShorDoc; d++) {
            int termIDArray[] = Corpus.get(d);

            // int[] termIDArray = docToWordIDList.get(d);
            int longDoc = FuncUtils.nextDiscrete(multiLongPros);
            docToLong[d] = longDoc;
            longDocShortCnts[longDoc]++;
            longDocLength[longDoc] += termIDArray.length;
            //ArrayList<int[]> d_assignment_list = new ArrayList<int[]>();
            int []prewordToTopic = new int[termIDArray.length];
            for (int t = 0; t < termIDArray.length; t++) {
                int termID = termIDArray[t];

                int topic = FuncUtils.nextDiscrete(multiTopicPros); // Sample a topic

                //int[] assignment = new int[2];
                //assignment[0] = topic;
                //assignment[1] = longDoc;

                U[topic][longDoc]++;
                V[termID][topic]++;
                longDocWordCnts[longDoc][termID]++;
                shortDocTopicCnts[d][topic]++;
                topicCnts[topic]++;
                //tokenSize++;
                prewordToTopic[t] = topic;

                //d_assignment_list.add(assignment);
            }
            wordToTopic.add(prewordToTopic);
            //assignmentList.add(d_assignment_list);
        }
        System.out.println("finish init_PTM!");

    }


    public void inference()
            throws IOException
    {

        writeParameters();
        writeDictionary();

        System.out.println("Running Gibbs sampling inference: ");

        for (int iter = 0; iter < this.numIterations; iter++) {
            long startTime = System.currentTimeMillis();


            for (int s = 0; s < Corpus.size(); s++) {

                int[] termIDArray = Corpus.get(s);
                int[] prewordToTopic = wordToTopic.get(s);
                int preLong = docToLong[s];

                longDocShortCnts[preLong]--;
                longDocLength[preLong] -= termIDArray.length;
                for (int t = 0; t < termIDArray.length; t++) {

                    int termID = termIDArray[t];
                    int preTopic = prewordToTopic[t];
                    // update the counter
                    U[preTopic][preLong]--;
                    V[termID][preTopic]--;
                    longDocWordCnts[preLong][termID]--;
                    topicCnts[preTopic]--;
                }

                // Sample a prelong for each short text
                for (int lIndex = 0; lIndex < numLongDoc; lIndex++) {
                    multiLongPros[lIndex] = longDocShortCnts[lIndex];



                    for (int wIndex = 0; wIndex < termIDArray.length; wIndex++) {

                        int topic = prewordToTopic[wIndex];
                        multiLongPros[lIndex] *= (U[topic][lIndex] + alpha
                                + shortDocTopicCnts[s][topic] - 1)
                        /(longDocLength[lIndex] + alphaSum + wIndex-1);

                    }
                }
                preLong = FuncUtils.nextDiscrete(multiLongPros);

                longDocShortCnts[preLong]++;
                longDocLength[preLong] += termIDArray.length;

                for (int wIndex = 0; wIndex < termIDArray.length; wIndex++) {

                    int topic = prewordToTopic[wIndex];
                    shortDocTopicCnts[s][topic]--;
                    for (int tIndex = 0; tIndex < numTopics; tIndex++) {
                        multiTopicPros[tIndex] = (shortDocTopicCnts[s][tIndex] + alpha);


                        multiTopicPros[tIndex] *= (V[wIndex][tIndex] + beta)
                                / (topicCnts[tIndex] + betaSum);

                    }
                    topic = FuncUtils.nextDiscrete(multiTopicPros);
                    shortDocTopicCnts[s][topic]++;
                    U[topic][preLong]++;
                    V[wIndex][topic]++;
                    longDocWordCnts[preLong][wIndex]++;
                    topicCnts[topic]++;
                }
            }

            System.out.println("finished iter :" + iter + "\tcost time:" + ((double) System.currentTimeMillis() - startTime) / 1000);

        }

        expName = orgExpName;

        System.out.println("Writing output from the last sample ...");
        write();

        System.out.println("Sampling completed for PTM!");

    }

    public void write()
            throws IOException
    {
        writeTopTopicalWords();
        writeDocTopicPros();

        writeTopicWordPros();
    }

    public void writeDocTopicPros()
            throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".theta"));

        for (int s = 0; s < Corpus.size(); s++) {

            int[] termIDArray = Corpus.get(s);

            for (int t = 0; t < numTopics; t++) {
                double pro = (shortDocTopicCnts[s][t] + alpha)
                        / (termIDArray.length + alphaSum);
                writer.write(pro + " ");
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
           // writer.write("Topic" + new Integer(tIndex) + ":");

            Map<Integer, Integer> wordCount = new TreeMap<Integer, Integer>();
            for (int wIndex = 0; wIndex < vocabularySize; wIndex++) {
                wordCount.put(wIndex, V[wIndex][tIndex]);
            }
            wordCount = FuncUtils.sortByValueDescending(wordCount);

            Set<Integer> mostLikelyWords = wordCount.keySet();
            int count = 0;
            for (Integer index : mostLikelyWords) {
                if (count < topWords) {
                    double pro = (V[index][tIndex] + beta)
                            / (topicCnts[tIndex] + betaSum);
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
                double pro = (V[j][i] + beta) / (topicCnts[i] + betaSum);
                writer.write(pro + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void writeParameters()
            throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".paras"));
        writer.write("-model" + "\t" + "PTM");
        writer.write("\n-corpus" + "\t" + corpusPath);
        writer.write("\n-ntopics" + "\t" + numTopics);
        writer.write("\n-nlongdoc" + "\t" + numLongDoc);

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


    public static void main(String args[])
            throws Exception
    {
        PTM ptm = new PTM("test/corpus.txt", 7, 500, 0.001,
                0.1, 1000, 20, "testPTM");
        ptm.inference();
    }

}
