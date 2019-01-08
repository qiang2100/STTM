package models;

import utility.FuncUtils;

import java.io.*;
import java.util.*;

/**
 * * SATM: A Java package for the short text topic models
 *
 * Implementation of the Short and Sparse Text Topic Modeling via Self-Aggregation, using collapsed Gibbs sampling, as described in:
 *
 * Xiaojun Quan, Chunyu Kit, Yong Ge, and Sinno Jialin Pan. Short and Sparse Text Topic Modeling via Self-Aggregation.
 * IJCAI, 2015.
 ** Created by jipengqiang on 18/7/1.
 */
public class SATM {

    public double alpha; // Hyper-parameter alpha
    public double beta; // Hyper-parameter alpha
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

    public double threshold;

    public int numLongDoc;

    private Random rg;

    public double[][] psd;
    private double[] pz;
    public ArrayList<ArrayList<int[]>> assignmentList  = new ArrayList<ArrayList<int[]>>();// [topic][pseudo-doc]
    private int[][] U; // [topic][pseudo-doc];
    private int[] longDocCnts; // U_{.j};
    private int[][] V; // [word][topic];
    private int[] topicCnts; // V_{.k}
    private int[][] longDocWordCnts; // [pseudo-doc][word]
    private int tokenSize;

    // Path to the directory containing the corpus
    public String folderPath;
    // Path to the topic modeling corpus
    public String corpusPath;
    private final static double ZERO_SMOOTH = 0.0000000001;

    public String expName = "SATMmodel";
    public String orgExpName = "SATMmodel";
    public String tAssignsFilePath = "";
    public int savestep = 0;

    public double initTime = 0;
    public double iterTime = 0;


    public SATM(String pathToCorpus, int inNumTopics, int num_longDoc, double threshold,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc, threshold, inAlpha, inBeta, inNumIterations,
                inTopWords, "SATMmodel");
    }

    public SATM(String pathToCorpus, int inNumTopics, int num_longDoc, double threshold,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords,
               String inExpName)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc, threshold, inAlpha, inBeta, inNumIterations,
                inTopWords, inExpName, "", 0);

    }

    public SATM(String pathToCorpus, int inNumTopics, int num_longDoc, double threshold,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords,
               String inExpName, String pathToTAfile)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc, threshold, inAlpha, inBeta, inNumIterations,
                inTopWords, inExpName, pathToTAfile, 0);

    }

    public SATM(String pathToCorpus, int inNumTopics, int num_longDoc, double threshold,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords,
               String inExpName, int inSaveStep)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, num_longDoc, threshold, inAlpha, inBeta, inNumIterations,
                inTopWords, inExpName, "", inSaveStep);

    }

    public SATM(String pathToCorpus, int inNumTopics, int num_longDoc, double threshold,
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
        numLongDoc = num_longDoc;
        rg = new Random();

        this.threshold = threshold;
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

        psd = new double[numShorDoc][numLongDoc];

        pz = new double[numTopics];
        //topicWordCount = new int[numTopics][vocabularySize];

        //sumTopicWordCount = new int[numTopics];



        alphaSum = numTopics * alpha;
        betaSum = vocabularySize * beta;

        // init the counter
        U = new int[numTopics][numLongDoc];
        longDocCnts = new int[numLongDoc]; // U_{.j};
        V = new int[vocabularySize][numTopics]; // [word][topic];
        topicCnts = new int[numTopics]; // V_{.k}
        longDocWordCnts = new int[numLongDoc][vocabularySize];

        System.out.println("Corpus size: " + numShorDoc + " docs, "
                + numWordsInCorpus + " words");
        System.out.println("Vocabuary size: " + vocabularySize);
        System.out.println("Number of topics: " + numTopics);
        System.out.println("alpha: " + alpha);
        System.out.println("beta: " + beta);
        System.out.println("Number of sampling iterations: " + numIterations);
        System.out.println("Number of top topical words: " + topWords);

        tAssignsFilePath = pathToTAfile;

        long startTime = System.currentTimeMillis();

        initialize();

        initTime =System.currentTimeMillis()-startTime;
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
            ArrayList<int[]> d_assignment_list = new ArrayList<int[]>();
            for (int t = 0; t < termIDArray.length; t++) {
                int termID = termIDArray[t];

                int topic = rg.nextInt(numTopics); // Sample a topic
                int longDoc = rg.nextInt(numLongDoc);
                int[] assignment = new int[2];
                assignment[0] = topic;
                assignment[1] = longDoc;

                U[topic][longDoc]++;
                V[termID][topic]++;
                longDocWordCnts[longDoc][termID]++;
                longDocCnts[longDoc]++;
                topicCnts[topic]++;
                tokenSize++;

                d_assignment_list.add(assignment);
            }
            assignmentList.add(d_assignment_list);
        }
        System.out.println("finish init_SATM!");

    }

    public void computePds() {
        // calculate pds
        for (int s = 0; s < numShorDoc; s++) {
            int[] termIDArray = Corpus.get(s);
            for (int l = 0; l < numLongDoc; l++) {

                if(longDocCnts[l] ==0)
                {
                    psd[s][l] = ZERO_SMOOTH;
                    continue;
                }
                double pd = 1.0 * longDocCnts[l] / tokenSize;
                double _score = pd;
                for (int t = 0; t < termIDArray.length; t++) {

                    double pdw = 1.0 * longDocWordCnts[l][termIDArray[t]]
                            / longDocCnts[l];
                    if (Double.compare(pdw, 0.0) == 0) {
                        pdw = ZERO_SMOOTH;
                    }
                    _score *= pdw;
                }
                psd[s][l] = _score;
            }

            psd[s] = FuncUtils.L1NormWithReusable(psd[s]);
            if (psd[s] == null) {
                psd[s] = new double[numLongDoc]; // all zero values
            }
        }
        //System.out.println("finish calculate pds!");
    }

    public int[] joint_sample(double[][] dist, double sum) {

        // scaled sample because of unnormalized p[]
        double u = rg.nextDouble() * sum;
        double temp = 0.0;
        int[] sample = new int[2];
        for (int l = 0; l < dist.length; l++) {
            for (int z = 0; z < dist[l].length; z++) {
                temp += dist[l][z];
                if (Double.compare(temp, u) >= 0) {
                    sample[0] = z;
                    sample[1] = l;
                    return sample;
                }
            }
        }

        return sample;
    }

    public void inference()
            throws IOException
    {
        writeDictionary();

        System.out.println("Running Gibbs sampling inference: ");

        long startTime = System.currentTimeMillis();

        for (int iter = 1; iter < this.numIterations; iter++) {

            if(iter%50==0)
                System.out.print(" " + (iter));

            computePds();

            double pdz = 0;
            double pzw = 0;
            double distSum = 0;
            List<Integer> validLongDocIDList = new ArrayList<Integer>();
            for (int s = 0; s < Corpus.size(); s++) {
                validLongDocIDList.clear();
                int[] termIDArray = Corpus.get(s);
                ArrayList<int[]> s_assignment = assignmentList.get(s);

                for (int d = 0; d < psd[s].length; d++) {
                    if(Double.isNaN(psd[s][d]))
                        continue;
                    if (Double.compare(psd[s][d], threshold) > 0) {
                       // System.out.println(d + " " + psd[s][d]);
                        validLongDocIDList.add(d);
                    }
                }

                if(validLongDocIDList.isEmpty()){
                    continue;
                }

                // create such a big matrix is time-consuming,
                // so we must reuse big matrices.
                double[][] pdzMat = new double[validLongDocIDList.size()][numTopics];
                for (int t = 0; t < termIDArray.length; t++) {
                    distSum = 0;
                    int termID = termIDArray[t];
                    int[] _assignment = s_assignment.get(t);

                    // get the previous assigned values
                    int preTopic = _assignment[0];
                    int preLongDoc = _assignment[1];

                    // update the counter
                    U[preTopic][preLongDoc]--;
                    V[termID][preTopic]--;
                    longDocWordCnts[preLongDoc][termID]--;
                    longDocCnts[preLongDoc]--;
                    topicCnts[preTopic]--;

                    for (int d = 0; d < validLongDocIDList.size(); d++) {
                        int longDocID = validLongDocIDList.get(d);
                        for (int z = 0; z < numTopics; z++) {
                            pdz = 1.0
                                    * (U[z][longDocID] + alpha)
                                    / (longDocCnts[longDocID] + alphaSum);
                            pzw = 1.0 * (V[termID][z] + beta) / (topicCnts[z] +  betaSum);
                            pdzMat[d][z] = psd[s][longDocID] * pdz * pzw;
                            distSum += pdzMat[d][z];
                        }
                    }

                    // sample a new longDoc and topic accroding dist;
                    int[] topicAndLongDoc = joint_sample(pdzMat, distSum);
                    int newTopic = topicAndLongDoc[0];
                    int newLongDocIndex = topicAndLongDoc[1];

                    // update the counter
                    int newLongDoc = validLongDocIDList.get(newLongDocIndex);
                    U[newTopic][newLongDoc]++;
                    V[termID][newTopic]++;
                    longDocWordCnts[newLongDoc][termID]++;
                    longDocCnts[newLongDoc]++;
                    topicCnts[newTopic]++;
                    _assignment[0] = newTopic;
                    _assignment[1] = newLongDoc;
                }
            }

            //System.out.println("finished iter :" + iter + "\tcost time:" + ((double) System.currentTimeMillis() - startTime) / 1000);

        }

        iterTime =System.currentTimeMillis()-startTime;

        expName = orgExpName;
        System.out.println();
        System.out.println("Writing output from the last sample ...");
        write();

        System.out.println("Sampling completed for SATM!");

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

        for (int d = 0; d < numShorDoc; d++) {

            double multiTopic[] = new double[numTopics];
            for (int k = 0; k < numTopics; k++) {

                multiTopic[k] = 1;
                for (int wIndex = 0; wIndex < Corpus.get(d).length; wIndex++) {
                    int word = Corpus.get(d)[wIndex];
                    multiTopic[k] *= (V[word][k] + beta)
                            / (topicCnts[k] + betaSum);
                }

               // double pros = (shortDocTopicCnts[i][tIndex] + alpha)/(N_d[i]+numTopics*alpha);
            }
            multiTopic = FuncUtils.L1NormWithReusable(multiTopic);

            for(int k=0; k<numTopics; k++)
                writer.write(multiTopic[k] + " ");
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
            //writer.write("Topic" + new Integer(tIndex) + ":");

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
        writer.write("-model" + "\t" + "SATM");
        writer.write("\n-corpus" + "\t" + corpusPath);
        writer.write("\n-ntopics" + "\t" + numTopics);
        writer.write("\n-nlongdoc" + "\t" + numLongDoc);
        writer.write("\n-threshold" + "\t" + threshold);
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
        SATM satm = new SATM("dataset/GoogleNews.txt", 152, 500, 0.001, 0.1,
                0.1, 1000, 10, "GoogleNewsSATM");
        satm.inference();
    }
}
