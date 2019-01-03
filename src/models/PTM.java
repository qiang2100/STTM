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
    public double gama;

    public int numTopics; // Number of topics
    public int numIterations; // Number of Gibbs sampling iterations
    public int topWords; // Number of most probable words for each topic

    public double alphaSum; // alpha * numTopics
    public double betaSum; // beta * vocabularySize
    public double gamaSum; //gama * numLongDoc

    ArrayList<int[]> Corpus = new ArrayList<>(); // Word ID-based corpus

    public int numShorDoc; // Number of documents in the corpus
    public int numWordsInCorpus; // Number of words in the corpus

    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
    // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
    // given an ID
    public int vocabularySize; // The number of word types in the corpus

    // Example: given a document of "a a b a b c d c". We have: 1 2 1 3 2 1 1 2
    //public List<List<Integer>> occurenceToIndexCount;

    public int numLongDoc;

    //public ArrayList<ArrayList<int[]>> assignmentList  = new ArrayList<ArrayList<int[]>>();// [topic][pseudo-doc]
    private int[][] U; // [topic][pseudo-doc];
    private int[] m_z; // U_{.j};  the number of short texts belonging to each long document
    private int[] n_z; //the number of words in each long document;
    private int[] z; // pre-long document for each short text
    private int[][] z_w ; // topic for each word
    private int[][] V; // [word][topic];
    private int[] topicCnts; // V_{.k}
    //private int[][] n_w_z; // [pseudo-doc][word]
    private int[][] shortDocTopicCnts; //[short][topic]
    /**
     * number of words in document d.
     */
    int N_d[];
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

    public double initTime = 0;
    public double iterTime = 0;


    public PTM(String pathToCorpus, int inNumTopics, int num_longDoc,
                double inAlpha, double inBeta, double inGama, int inNumIterations, int inTopWords)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc, inAlpha, inBeta, inGama, inNumIterations,
                inTopWords, "PTMmodel");
    }

    public PTM(String pathToCorpus, int inNumTopics, int num_longDoc,
                double inAlpha, double inBeta, double inGama, int inNumIterations, int inTopWords,
                String inExpName)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc, inAlpha, inBeta, inGama, inNumIterations,
                inTopWords, inExpName, "", 0);

    }

    public PTM(String pathToCorpus, int inNumTopics, int num_longDoc,
                double inAlpha, double inBeta, double inGama, int inNumIterations, int inTopWords,
                String inExpName, String pathToTAfile)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc, inAlpha, inBeta, inGama, inNumIterations,
                inTopWords, inExpName, pathToTAfile, 0);

    }

    public PTM(String pathToCorpus, int inNumTopics, int num_longDoc,
                double inAlpha, double inBeta, double inGama,  int inNumIterations, int inTopWords,
                String inExpName, int inSaveStep)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc,  inAlpha, inBeta, inGama,  inNumIterations,
                inTopWords, inExpName, "", inSaveStep);

    }

    public PTM(String pathToCorpus, int inNumTopics, int num_longDoc,
                double inAlpha, double inBeta, double inGama,  int inNumIterations, int inTopWords,
                String inExpName, String pathToTAfile, int inSaveStep)
            throws IOException
    {
        alpha = inAlpha;
        beta = inBeta;
        gama = inGama;

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

       // occurenceToIndexCount = new ArrayList<List<Integer>>();

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

                //List<Integer> wordOccurenceToIndexInDoc = new ArrayList<Integer>();
                //HashMap<Integer, Integer> wordOccurenceToIndexInDocCount = new HashMap<Integer, Integer>();

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

                   // int times = 0;
                   // if (wordOccurenceToIndexInDocCount.containsKey(indexWord)) {
                    //    times = wordOccurenceToIndexInDocCount.get(indexWord);
                    //}
                    //times += 1;
                    //wordOccurenceToIndexInDocCount.put(indexWord, times);
                    //wordOccurenceToIndexInDoc.add(times);
                }

                numShorDoc++;
                numWordsInCorpus += document.length;
                Corpus.add(document);

               // occurenceToIndexCount.add(wordOccurenceToIndexInDoc);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        vocabularySize = word2IdVocabulary.size(); // vocabularySize = indexWord
        z = new int[numShorDoc];
        z_w = new int[numShorDoc][];

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
        gamaSum = gama * vocabularySize;

        // init the counter
        U = new int[numTopics][numLongDoc];
        m_z = new int[numLongDoc]; // U_{.j};
        n_z = new int[numLongDoc];
        V = new int[vocabularySize][numTopics]; // [word][topic];
        topicCnts = new int[numTopics]; // V_{.k}
        //n_w_z = new int[numLongDoc][vocabularySize];
        shortDocTopicCnts = new int[numShorDoc][numTopics];
        N_d = new int[numShorDoc];
        System.out.println("Corpus size: " + numShorDoc + " docs, "
                + numWordsInCorpus + " words");
        System.out.println("Vocabuary size: " + vocabularySize);
        System.out.println("Number of topics: " + numTopics);
        System.out.println("alpha: " + alpha);
        System.out.println("beta: " + beta);
        System.out.println("Number of sampling iterations: " + numIterations);
        System.out.println("Number of top topical words: " + topWords);

        initialize();
    }

    /**
     * Randomly initialize topic assignments
     */
    public void initialize()
            throws IOException
    {
        System.out.println("Randomly initializing topic assignments ...");

        long startTime = System.currentTimeMillis();

        for (int d = 0; d < numShorDoc; d++) {
            int termIDArray[] = Corpus.get(d);

            // int[] termIDArray = docToWordIDList.get(d);
            int longDoc = FuncUtils.nextDiscrete(multiLongPros);
            z[d] = longDoc;
            m_z[longDoc]++;
            n_z[longDoc] += termIDArray.length;
            //ArrayList<int[]> d_assignment_list = new ArrayList<int[]>();
            z_w[d] = new int[termIDArray.length];
            N_d[d] = termIDArray.length;

            for (int t = 0; t < termIDArray.length; t++) {
                int termID = termIDArray[t];
                int topic = FuncUtils.nextDiscrete(multiTopicPros); // Sample a topic
                //n_w_z[longDoc][termID]++;
                shortDocTopicCnts[d][topic]++;
                topicCnts[topic]++;
                //tokenSize++;
                z_w[d][t] = topic;
                U[topic][longDoc]++;
                V[termID][topic]++;


            }
           // wordToTopic.add(prewordToTopic);
            //assignmentList.add(d_assignment_list);
        }

        initTime =System.currentTimeMillis()-startTime;

        System.out.println("finish init_PTM!");
    }


    /**
     * Sample a topic z_i from the full conditional distribution: p(z_i = j |
     * z_-i, w) = (n_-i,j(w_i) + beta)/(n_-i,j(.) + W * beta) * (n_-i,j(d_i) +
     * alpha)/(n_-i,.(d_i) + K * alpha)
     *
     * @param m
     *            document
     */
    private int sampleFullConditionalForLong(int m) {
        // remove z_i from the count variables
        //System.out.println("m: " + m);
        int pre_long = z[m];
        m_z[pre_long]--;
        n_z[pre_long] -= N_d[m];

        for(int n=0; n<Corpus.get(m).length; n++) {
            int topic = z_w[m][n];
            U[topic][pre_long]--;
        }
        // Sample a topic
        for (int tIndex = 0; tIndex < numLongDoc; tIndex++) {
            multiLongPros[tIndex] = m_z[tIndex];

            for(int k=0; k<numTopics; k++){
                if(shortDocTopicCnts[m][k]==0)
                    continue;

                for(int i=1; i<=shortDocTopicCnts[m][k]; i++)
                {
                    multiLongPros[tIndex] *= (U[k][tIndex] + alpha + i - 1);
                }
            }
            double prob = 1;
            for(int wIndex=0; wIndex<N_d[m]; wIndex++)
               prob  *= (n_z[tIndex]+alphaSum+wIndex);

            multiLongPros[tIndex] /= prob;
        }

        pre_long = FuncUtils.nextDiscrete(multiLongPros);
        // System.out.println("topic: " + topic);
        m_z[pre_long]++;
        n_z[pre_long] += N_d[m];

        for(int n=0; n<Corpus.get(m).length; n++) {
            int topic = z_w[m][n];
            U[topic][pre_long]++;
        }

        //for(int n=0; n<Corpus.get(m).length; n++)
         //   n_w_z[topic][Corpus.get(m)[n]]++;
        return pre_long;
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
    private int sampleFullConditionalForWordTopic(int preLong, int m, int n) {

        // remove z_i from the count variables
        int topic = z_w[m][n];

        int wordId = Corpus.get(m)[n];
        V[wordId][topic]--;
        U[topic][preLong]--;
        topicCnts[topic]--;

        shortDocTopicCnts[m][topic]--;

        //ndsum[m]--;
        //int maxK = -1;
        // double value = 0;
        for (int k = 0; k < numTopics; k++) {
            multiTopicPros[k] = (V[wordId][k] + beta) / (topicCnts[k] + betaSum) * (U[k][preLong] + alpha);// / (ndsum[m] + K * alpha);
        }

        topic = FuncUtils.nextDiscrete(multiTopicPros);

        V[wordId][topic]++;
        U[topic][preLong]++;
        topicCnts[topic]++;
        shortDocTopicCnts[m][topic]++;

        return topic;
    }

    public void inference()
            throws IOException
    {
        writeDictionary();

        System.out.println("Running Gibbs sampling inference: ");
        long startTime = System.currentTimeMillis();

        for (int iter = 0; iter < this.numIterations; iter++) {

            if(iter%50==0)
                System.out.print(" " + (iter));

            for (int s = 0; s < Corpus.size(); s++) {

                int preLong = sampleFullConditionalForLong(s);

                z[s] = preLong;

                for (int wIndex = 0; wIndex < Corpus.get(s).length; wIndex++) {
                    int topic = sampleFullConditionalForWordTopic(preLong, s, wIndex);
                    z_w[s][wIndex] = topic;
                }

            }

        }

        iterTime =System.currentTimeMillis()-startTime;

        expName = orgExpName;
        System.out.println();

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
        writeParameters();
    }

    public void writeDocTopicPros()
            throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".theta"));

        for (int i = 0; i < numShorDoc; i++) {
            //int docSize = Corpus.get(i).length;
            //double sum = 0.0;
            for (int tIndex = 0; tIndex < numTopics; tIndex++) {
                double pros = (shortDocTopicCnts[i][tIndex] + alpha)/(N_d[i]+numTopics*alpha);
                writer.write(pros + " ");
            }
            writer.write("\n");
        }
        writer.close();
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

    public static void main(String args[])
            throws Exception
    {
        PTM ptm = new PTM("dataset/Tweet.txt", 100, 1000, 0.1,
                0.1, 0.01, 2000, 10, "TweetPTM");
        ptm.inference();
    }

}
