package models;

import utility.FuncUtils;

import java.io.*;
import java.math.BigDecimal;
import java.util.*;

/**
 *  * * GPUDMM: From A Java package STTM for the short text topic models
 *
 * Implementation of GPUDMM, using collapsed Gibbs sampling, as described in:
 *
 * Chenliang Li, Yu Duan, Haoran Wang, Zhiqian Zhang, Aixin Sun, and Zongyang Ma. 2017. Enhancing Topic
 Modeling for Short Texts with Auxiliary Word Embeddings. ACM Trans. Inf. Syst. 36, 2, Article 11 (August
 2017), 30 pages..
 *
 * Created by jipengqiang on 18/7/2.
 */
public class GPUDMM {


    public double alpha, beta;
    public int numTopics; // Number of topics
    public int numIterations; // Number of Gibbs sampling iterations
    public int topWords; // Number of most probable words for each topic
    public double alphaSum; // alpha * numTopics
    public double betaSum; // beta * vocabularySize

    public ArrayList<int[]> Corpus = new ArrayList<>(); // Word ID-based corpus


    private Random rg;
    public double threshold;
    public double weight;

    public int filterSize;


    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
    // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
    // given an ID
    public int vocabularySize; // The number of word types in the corpus


    public Map<Integer, Double> wordIDFMap;
    public Map<Integer, Map<Integer, Double>> docUsefulWords;
    public ArrayList<ArrayList<Integer>> topWordIDList;

    public int numDocuments; // Number of documents in the corpus
    public int numWordsInCorpus; // Number of words in the corpus

    //private double[][] schema;

    //	public String initialFileName;
    public double[][] phi;
    private double[] pz;
    private double[][] pdz;
    private double[][] topicProbabilityGivenWord;

    public ArrayList<ArrayList<Boolean>> wordGPUFlag; // wordGPUFlag.get(doc).get(wordIndex)
    public int[] assignmentList; // topic assignment for every document
    public ArrayList<ArrayList<Map<Integer, Double>>> wordGPUInfo;

    // Number of documents assigned to a topic
    public int[] docTopicCount;
    // numTopics * vocabularySize matrix
    // Given a topic: number of times a word type assigned to the topic
    public int[][] topicWordCount;
    // Total number of words assigned to a topic
    public int[] sumTopicWordCount;

    private Map<Integer, Map<Integer, Double>> schemaMap;

    // Double array used to sample a topic
    public double[] multiPros;

    // Path to the directory containing the corpus
    public String folderPath;
    // Path to the topic modeling corpus
    public String corpusPath;
    //Path to the word2vec
    public String pathToVector;

    public String expName = "GPUDMMmodel";
    public String orgExpName = "GPUDMMmodel";
    public String tAssignsFilePath = "";
    public int savestep = 0;

    public double initTime = 0;
    public double iterTime = 0;

    public GPUDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU, int inFilterSize, int inNumTopics,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords)
            throws Exception
    {
        this(pathToCorpus, pathToVector, inWeight, threshold_GPU, inFilterSize, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, "GPUDMMmodel");
    }

    public GPUDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU,  int inFilterSize, int inNumTopics,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords,
               String inExpName)
            throws Exception
    {
        this(pathToCorpus, pathToVector, inWeight, threshold_GPU, inFilterSize, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, inExpName, "", 0);
    }

    public GPUDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU,int inFilterSize,  int inNumTopics,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords,
               String inExpName, String pathToTAfile)
            throws Exception
    {
        this(pathToCorpus, pathToVector, inWeight, threshold_GPU, inFilterSize, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, inExpName, pathToTAfile, 0);
    }

    public GPUDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU, int inFilterSize,  int inNumTopics,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords,
               String inExpName, int inSaveStep)
            throws Exception
    {
        this(pathToCorpus, pathToVector, inWeight, threshold_GPU, inFilterSize, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, inExpName, "", inSaveStep);
    }

    public GPUDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU, int inFilterSize, int inNumTopics,
               double inAlpha, double inBeta, int inNumIterations, int inTopWords,
               String inExpName, String pathToTAfile, int inSaveStep)
            throws Exception
    {

        alpha = inAlpha;
        beta = inBeta;
        numTopics = inNumTopics;
        numIterations = inNumIterations;
        topWords = inTopWords;
        savestep = inSaveStep;
        expName = inExpName;
        orgExpName = expName;
        weight = inWeight;
        filterSize = inFilterSize;
        threshold = threshold_GPU;
        corpusPath = pathToCorpus;
        this.pathToVector = pathToVector;
        folderPath = "results/";
        File dir = new File(folderPath);
        if (!dir.exists())
            dir.mkdir();

        System.out.println("Reading topic modeling corpus: " + pathToCorpus);

        word2IdVocabulary = new HashMap<String, Integer>();
        id2WordVocabulary = new HashMap<Integer, String>();

        wordGPUFlag = new ArrayList<>();


        numDocuments = 0;
        numWordsInCorpus = 0;

        rg = new Random();

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

                numDocuments++;
                numWordsInCorpus += document.length;
                Corpus.add(document);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }



        vocabularySize = word2IdVocabulary.size(); // vocabularySize = indexWord
        docTopicCount = new int[numTopics];
        topicWordCount = new int[numTopics][vocabularySize];
        sumTopicWordCount = new int[numTopics];

        phi = new double[numTopics][vocabularySize];
        pz = new double[numTopics];


       // schema = new double[vocabularySize][vocabularySize];
        topicProbabilityGivenWord = new double[vocabularySize][numTopics];

        pdz = new double[numDocuments][numTopics];
        multiPros = new double[numTopics];
        for (int i = 0; i < numTopics; i++) {
            multiPros[i] = 1.0 / numTopics;
        }

        alphaSum = numTopics * alpha;
        betaSum = vocabularySize * beta;

        assignmentList = new int[numDocuments];
        wordGPUInfo = new ArrayList<>();
        rg = new Random();

        long startTime = System.currentTimeMillis();

        schemaMap = computSchema(pathToVector);

        initialize();

        initTime =System.currentTimeMillis()-startTime;

        System.out.println("Corpus size: " + numDocuments + " docs, "
                + numWordsInCorpus + " words");
        System.out.println("Vocabuary size: " + vocabularySize);
        System.out.println("Number of topics: " + numTopics);
        System.out.println("alpha: " + alpha);
        System.out.println("beta: " + beta);
        System.out.println("weight: " + weight);
        System.out.println("filterSize: " + filterSize);
        System.out.println("Number of sampling iterations: " + numIterations);
        System.out.println("Number of top topical words: " + topWords);

    }

    public double computeSis(HashMap<Integer, float[]> wordMap, int i, int j)
    {

        if(i==j)
            return 1.0;
        if(!wordMap.containsKey(i) || !wordMap.containsKey(j))
            return 0.0;


        float sis = FuncUtils.ComputeCosineSimilarity(wordMap.get(i),wordMap.get(j));
        return sis;
    }

    public Map<Integer, Map<Integer, Double>> computSchema(String pathToVector) {

        Map<Integer, Map<Integer, Double>> schemaMap = new HashMap<Integer, Map<Integer, Double>>();
        HashMap<Integer, float[]> wordMap = new HashMap<Integer, float[]>();
        try {



            BufferedReader br1 = new BufferedReader(new FileReader(pathToVector));
            String line = "";
            float vector = 0;
            while ((line = br1.readLine()) != null)
            {
                //System.out.println(line);
                String word[] = line.split(" ");

                String word1 = word[0];
                int id = -1;
                if(word2IdVocabulary.containsKey(word1))
                    id = word2IdVocabulary.get(word1);
                else
                    continue;
                float []vec = new float[word.length-1];
                for(int i=1; i<word.length; i++)
                {
                    vector = Float.parseFloat(word[i]);///(word.length-1);
                    vec[i-1] = vector;
                }
                wordMap.put(id, vec);
            }
            br1.close();



            double count = 0.0;
            for (int i = 0; i < vocabularySize; i++) {
                Map<Integer, Double> tmpMap = new HashMap<Integer, Double>();
                for (int j = 0; j < vocabularySize; j++) {
                    double v = computeSis(wordMap,i,j);
                    if (Double.compare(v, threshold) > 0) {
                        tmpMap.put(j, v);
                    }
                }
                if (tmpMap.size() > filterSize) {
                    tmpMap.clear();
                }
                tmpMap.remove(i);
                if (tmpMap.size() == 0) {
                    continue;
                }
                count += tmpMap.size();
                schemaMap.put(i, tmpMap);
            }
            wordMap.clear();
            System.out.println("finish read schema, the avrage number of value is " + count / schemaMap.size());
            return schemaMap;
        } catch (Exception e) {
            System.out.println("Error while reading other file:" + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }


    /**
     * Randomly initialize topic assignments
     */
    public void initialize()
            throws IOException
    {
        System.out.println("Randomly initializing topic assignments ...");



        for (int d = 0; d < numDocuments; d++) {
            int termIDArray[] = Corpus.get(d);

            ArrayList<Boolean> docWordGPUFlag = new ArrayList<>();
            ArrayList<Map<Integer, Double>> docWordGPUInfo = new ArrayList<>();
            // int[] termIDArray = docToWordIDList.get(d);
            ArrayList<int[]> d_assignment_list = new ArrayList<int[]>();

            int topic = FuncUtils.nextDiscrete(multiPros); // Sample a topic
            assignmentList[d] = topic;
            docTopicCount[topic] += 1;
            for (int t = 0; t < termIDArray.length; t++) {
                int termID = termIDArray[t];

                topicWordCount[topic][termID] += 1;
                sumTopicWordCount[topic] += 1;
                docTopicCount[topic] += 1;

                docWordGPUFlag.add(false); // initial for False for every word
                docWordGPUInfo.add(new HashMap<Integer, Double>());

            }
            wordGPUFlag.add(docWordGPUFlag);
            wordGPUInfo.add(docWordGPUInfo);

        }
        System.out.println("finish init_GPU!");

    }

    public void compute_phi() {
        for (int i = 0; i < numTopics; i++) {
            double sum = 0.0;
            for (int j = 0; j < vocabularySize; j++) {
                sum += topicWordCount[i][j];
            }
            for (int j = 0; j < vocabularySize; j++) {
                phi[i][j] = (topicWordCount[i][j] + beta) / (sum + betaSum);
            }
        }
    }

    public void compute_pz() {
        double sum = 0.0;
        for (int i = 0; i < numTopics; i++) {
            sum += sumTopicWordCount[i];
        }
        for (int i = 0; i < numTopics; i++) {
            pz[i] = 1.0 * (sumTopicWordCount[i] + alpha) / (sum + alphaSum);
        }
    }

    /**
     * update the p(z|w) for every iteration
     */
    public void updateTopicProbabilityGivenWord() {
        // TODO we should update pz and phi information before
        compute_pz();
        compute_phi();  //update p(w|z)
        for (int i = 0; i < vocabularySize; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < numTopics; j++) {
                topicProbabilityGivenWord[i][j] = pz[j] * phi[j][i];
                row_sum += topicProbabilityGivenWord[i][j];
            }
            for (int j = 0; j < numTopics; j++) {
                topicProbabilityGivenWord[i][j] = topicProbabilityGivenWord[i][j] / row_sum;  //This is p(z|w)
            }
        }
    }

    public double findTopicMaxProbabilityGivenWord(int wordID) {
        double max = -1.0;
        for (int i = 0; i < numTopics; i++) {
            double tmp = topicProbabilityGivenWord[wordID][i];
            if (Double.compare(tmp, max) > 0) {
                max = tmp;
            }
        }
        return max;
    }

    public double getTopicProbabilityGivenWord(int topic, int wordID) {
        return topicProbabilityGivenWord[wordID][topic];
    }

    /**
     * update GPU flag, which decide whether do GPU operation or not
     * @param docID
     * @param newTopic
     */
    public void updateWordGPUFlag(int docID, int newTopic) {
        // we calculate the p(t|w) and p_max(t|w) and use the ratio to decide we
        // use gpu for the word under this topic or not
        int[] termIDArray = Corpus.get(docID);
        ArrayList<Boolean> docWordGPUFlag = new ArrayList<>();
        for (int t = 0; t < termIDArray.length; t++) {

            int termID = termIDArray[t];
            double maxProbability = findTopicMaxProbabilityGivenWord(termID);
            double ratio = getTopicProbabilityGivenWord(newTopic, termID) / maxProbability;

            double a = rg.nextDouble();
            docWordGPUFlag.add(Double.compare(ratio, a) > 0);
        }
        wordGPUFlag.set(docID, docWordGPUFlag);
    }

    public void ratioCount(Integer topic, Integer docID, int[] termIDArray, int flag) {
       // System.out.println("Topic:"+ topic);


        docTopicCount[topic] += flag;
        for (int t = 0; t < termIDArray.length; t++) {
            int wordID = termIDArray[t];
            topicWordCount[topic][wordID] += flag;
            sumTopicWordCount[topic] += flag;
        }
        // we update gpu flag for every document before it change the counter
        // when adding numbers
        if (flag > 0) {
            updateWordGPUFlag(docID, topic);
            for (int t = 0; t < termIDArray.length; t++) {
                int wordID = termIDArray[t];
                boolean gpuFlag = wordGPUFlag.get(docID).get(t);
                Map<Integer, Double> gpuInfo = new HashMap<>();
                if (gpuFlag) { // do gpu count
                    if (schemaMap.containsKey(wordID)) {
                        Map<Integer, Double> valueMap = schemaMap.get(wordID);
                        // update the counter
                        for (Map.Entry<Integer, Double> entry : valueMap.entrySet()) {
                            Integer k = entry.getKey();
                            double v = weight;
                            topicWordCount[topic][k] += v;
                            sumTopicWordCount[topic] += v;
                            gpuInfo.put(k, v);
                        } // end loop for similar words
                    } else { // schemaMap don't contain the word

                        // the word doesn't have similar words, the infoMap is empty
                        // we do nothing
                    }
                } else { // the gpuFlag is False
                    // it means we don't do gpu, so the gouInfo map is empty
                }
                wordGPUInfo.get(docID).set(t, gpuInfo); // we update the gpuinfo
                // map
            }
        } else { // we do subtraction according to last iteration information
            for (int t = 0; t < termIDArray.length; t++) {
                Map<Integer, Double> gpuInfo = wordGPUInfo.get(docID).get(t);
                int wordID = termIDArray[t];
                // boolean gpuFlag = wordGPUFlag.get(docID).get(t);
                if (gpuInfo.size() > 0) {
                    for (int similarWordID : gpuInfo.keySet()) {
                        // double v = gpuInfo.get(similarWordID);
                        double v = weight;
                        topicWordCount[topic][similarWordID] -= v;
                        sumTopicWordCount[topic] -= v;
                        // if(Double.compare(0, nzw[topic][wordID]) > 0){
                        // System.out.println( nzw[topic][wordID]);
                        // }
                    }
                }
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

            if(iter%50==0)
                System.out.print(" " + (iter));
            // System.out.println("\t\tPerplexity: " + computePerplexity());

            // getTopWordsUnderEachTopicGivenCurrentMarkovStatus();
            updateTopicProbabilityGivenWord();
            for (int s = 0; s < Corpus.size(); s++) {

                int[] termIDArray = Corpus.get(s);
                int preTopic = assignmentList[s];

                ratioCount(preTopic, s, termIDArray, -1);

                //double[] pzDist = new double[numTopics];
                for (int topic = 0; topic < numTopics; topic++) {
                    double pz = 1.0 * (docTopicCount[topic] + alpha) / (numDocuments - 1 + alphaSum);
                    double value = 1.0;
                    double logSum = 0.0;
                    for (int t = 0; t < termIDArray.length; t++) {
                        int termID = termIDArray[t];
                        value *= (topicWordCount[topic][termID] + beta) / (sumTopicWordCount[topic] + betaSum + t);
                        // we do not use log, it is a little slow
                        // logSum += Math.log(1.0 * (nzw[topic][termID] + beta) / (nz[topic] + vocSize * beta + t));
                    }
//					value = pz * Math.exp(logSum);
                    value = pz * value;
                    multiPros[topic] = value;
                }

                int newTopic = FuncUtils.nextDiscrete(multiPros);


                // update
                assignmentList[s] = newTopic;
                ratioCount(newTopic, s, termIDArray, +1);

            }
        }
        expName = orgExpName;

        iterTime =System.currentTimeMillis()-startTime;

        System.out.println();

        System.out.println("Writing output from the last sample ...");
        write();

        System.out.println("Sampling completed!");

    }

    public void writeParameters()
            throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".paras"));
        writer.write("-model" + "\t" + "GPUDMM");
        writer.write("\n-corpus" + "\t" + corpusPath);
        writer.write("\n-ntopics" + "\t" + numTopics);
        writer.write("\n-alpha" + "\t" + alpha);
        writer.write("\n-beta" + "\t" + beta);
        writer.write("\n-threshold" + "\t" + threshold);
        writer.write("\n-weight" + "\t" + weight);
        writer.write("\n-filterSize" + "\t" + filterSize);
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

    public void writeDocTopicPros()
            throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".theta"));

        for (int i = 0; i < numDocuments; i++) {
            int docSize = Corpus.get(i).length;
            double sum = 0.0;
            for (int tIndex = 0; tIndex < numTopics; tIndex++) {
                multiPros[tIndex] = (docTopicCount[tIndex] + alpha);
                for (int wIndex = 0; wIndex < docSize; wIndex++) {
                    int word = Corpus.get(i)[wIndex];
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

    public void writeTopicAssignments()
            throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".topicAssignments"));
        for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
            int docSize = Corpus.get(dIndex).length;
            int topic = assignmentList[dIndex];
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                writer.write(topic + " ");
            }
            writer.write("\n");
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

    public void write()
            throws IOException
    {
        writeTopTopicalWords();
        writeDocTopicPros();
        writeTopicAssignments();
        writeTopicWordPros();
        writeParameters();
    }

    public static void main(String args[])
            throws Exception
    {
        GPUDMM gpudmm = new GPUDMM("dataset/Pascal_Flickr.txt","dataset/glove.6B.200d.txt", 0.7, 0.1, 10, 20, 0.1,
                0.1, 500, 10, "Pascal_FlickrGPUDMM");
        gpudmm.inference();
    }


}
