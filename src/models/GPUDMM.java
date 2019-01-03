package models;

import utility.FuncUtils;

import java.io.*;
import java.math.BigDecimal;
import java.util.*;

/**
 * Created by jipengqiang on 18/7/24.
 */
public class GPU_PDMM {


    public int numTopics;
    public double alpha, beta, lambda;
    public int numIterations;



    public ArrayList<int[]> Corpus = new ArrayList<>(); // Word ID-based corpus


    private Random rg;
    public double threshold;
    public double weight;
    public int topWords;
    public int filterSize;


    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
    // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
    // given an ID
    public int vocabularySize; // The number of word types in the corpus



    public Map<Integer, Set<Integer>> ZdMap;
    public int[] TdArray;



    public int numDocuments; // Number of documents in the corpus
    public int numWordsInCorpus; // Number of words in the corpus
    public int maxTd ; // the maximum number of topics within a document
    public int searchTopK;
     // private double[][] schema;
    public Map<Integer, int[]> docToWordIDListMap;

    public double[][] phi;
    private double[] pz;
    private double[][] pdz;
    private double[][] topicProbabilityGivenWord;
    private double[][] pwz;

    public ArrayList<ArrayList<Boolean>> wordGPUFlag; // wordGPUFlag.get(doc).get(wordIndex)
    public Map<Integer, int[]> assignmentListMap; // topic assignment for every document
    public ArrayList<ArrayList<Map<Integer, Double>>> wordGPUInfo;

    private double[] nz; // [topic]; nums of words in every topic
    private double[][] nzw; // V_{.k}

    private int[] Ck; // Ck[topic]
    private int CkSum;


    private Map<Integer, Map<Integer, Double>> schemaMap;



    // Path to the directory containing the corpus
    public String folderPath;
    // Path to the topic modeling corpus
    public String corpusPath;
    //Path to the word2vec
    public String pathToVector;

    public String expName = "GPU_PDMMmodel";
    public String orgExpName = "GPU_PDMMmodel";
    public String tAssignsFilePath = "";
    public int savestep = 0;

    public double initTime = 0;
    public double iterTime = 0;

    public GPU_PDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU, int inFilterSize, int inNumTopics,
                  double inAlpha, double inBeta, double inlambda, int inNumIterations, int inTopWords)
            throws Exception
    {
        this(pathToCorpus, pathToVector, inWeight, threshold_GPU, inFilterSize, inNumTopics, inAlpha, inBeta, inlambda, inNumIterations,
                inTopWords, 3);
    }
    public GPU_PDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU, int inFilterSize, int inNumTopics,
                    double inAlpha, double inBeta, double inlambda, int inNumIterations, int inTopWords, int inMaxTd)
            throws Exception
    {
        this(pathToCorpus, pathToVector, inWeight, threshold_GPU, inFilterSize, inNumTopics, inAlpha, inBeta, inlambda, inNumIterations,
                inTopWords, inMaxTd, 10);
    }
    public GPU_PDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU, int inFilterSize, int inNumTopics,
                    double inAlpha, double inBeta, double inlambda, int inNumIterations, int inTopWords, int inMaxTd, int inSearchTopK)
            throws Exception
    {
        this(pathToCorpus, pathToVector, inWeight, threshold_GPU, inFilterSize, inNumTopics, inAlpha, inBeta, inlambda, inNumIterations,
                inTopWords, inMaxTd, inSearchTopK, "GPU_PDMMmodel");
    }
    public GPU_PDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU, int inFilterSize, int inNumTopics,
                    double inAlpha, double inBeta, double inlambda, int inNumIterations, int inTopWords,
                    int inMaxTd, int inSearchTopK,String inExpName)
            throws Exception
    {
        this(pathToCorpus, pathToVector, inWeight, threshold_GPU, inFilterSize, inNumTopics, inAlpha, inBeta, inlambda, inNumIterations,
                inTopWords,  inMaxTd, inSearchTopK,inExpName, "", 0);
    }

    public GPU_PDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU,int inFilterSize,  int inNumTopics,
                  double inAlpha, double inBeta, double inlambda,  int inNumIterations, int inTopWords,int inMaxTd, int inSearchTopK,
                  String inExpName, String pathToTAfile)
            throws Exception
    {
        this(pathToCorpus, pathToVector, inWeight, threshold_GPU, inFilterSize, inNumTopics, inAlpha, inBeta, inlambda, inNumIterations,
                inTopWords,inMaxTd, inSearchTopK, inExpName, pathToTAfile, 0);
    }

    public GPU_PDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU, int inFilterSize,  int inNumTopics,
                  double inAlpha, double inBeta,double inlambda,  int inNumIterations, int inTopWords,int inMaxTd, int inSearchTopK,
                  String inExpName, int inSaveStep)
            throws Exception
    {
        this(pathToCorpus, pathToVector, inWeight, threshold_GPU, inFilterSize, inNumTopics, inAlpha, inBeta, inlambda, inNumIterations,
                inTopWords,inMaxTd, inSearchTopK, inExpName, "", inSaveStep);
    }

    public GPU_PDMM(String pathToCorpus, String pathToVector, double inWeight, double threshold_GPU, int inFilterSize, int inNumTopics,
                  double inAlpha, double inBeta, double inlambda,  int inNumIterations, int inTopWords, int inMaxTd, int inSearchTopK,
                  String inExpName, String pathToTAfile, int inSaveStep)
            throws Exception
    {

        alpha = inAlpha;
        beta = inBeta;
        lambda = inlambda;
        numTopics = inNumTopics;
        numIterations = inNumIterations;
        topWords = inTopWords;
        weight = inWeight;
        filterSize = inFilterSize;
        maxTd = inMaxTd;
        searchTopK = inSearchTopK;
        savestep = inSaveStep;
        expName = inExpName;
        orgExpName = expName;
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

        ZdMap = new HashMap<Integer, Set<Integer>>();
        assignmentListMap = new HashMap<Integer, int[]>();
        docToWordIDListMap = new HashMap<Integer, int[]>();

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
            br.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        vocabularySize = word2IdVocabulary.size(); // vocabularySize = indexWord

        phi = new double[numTopics][vocabularySize];
        pz = new double[numTopics];
        pwz = new double[vocabularySize][numTopics];
        TdArray = new int[Corpus.size()];
       // schema = new double[vocabularySize][vocabularySize];
        topicProbabilityGivenWord = new double[vocabularySize][numTopics];

        pdz = new double[numDocuments][numTopics];

        wordGPUInfo = new ArrayList<>();
        rg = new Random();

        // init the counter
        nz = new double[numTopics];
        nzw = new double[numTopics][vocabularySize];
        Ck = new int[numTopics];
        CkSum = 0;

        long startTime = System.currentTimeMillis();
        schemaMap = computSchema(pathToVector,threshold_GPU);



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



    /**
     * Randomly initialize topic assignments
     */
    public void initialize()
            throws IOException
    {
        System.out.println("Randomly initializing topic assignments ...");

        for (int d = 0; d < numDocuments; d++) {
            int termIDArray[] = Corpus.get(d);

            assignmentListMap.put(d, new int[termIDArray.length]);

            ArrayList<Boolean> docWordGPUFlag = new ArrayList<>();
            ArrayList<Map<Integer, Double>> docWordGPUInfo = new ArrayList<>();


            for (int t = 0; t < termIDArray.length; t++) {

                docWordGPUFlag.add(false); // initial for False for every word
                docWordGPUInfo.add(new HashMap<Integer, Double>());

            }
            wordGPUFlag.add(docWordGPUFlag);
            wordGPUInfo.add(docWordGPUInfo);
            docToWordIDListMap.put(d, termIDArray);

        }

        init_GSDMM();
        System.out.println("finish init_GPU-PDMM!");

    }

    public void normalCountWord(Integer topic, int word, Integer flag) {
        nzw[topic][word] += flag;
        nz[topic] += flag;
    }

    public void normalCountZd(Set<Integer> Zd, Integer flag){
        for (int topic : Zd){
            Ck[topic] += flag;
            CkSum += flag;
        }
    }

    public void init_GSDMM() {

        double[] ptd = new double[maxTd];
        double temp_factorial = 1.0;
        for ( int i = 0; i < maxTd; i++ ){
            temp_factorial *= (i+1);
            ptd[i] = Math.pow(lambda, (double)(i+1)) * Math.exp(-lambda)/temp_factorial;
        }

        for (int i = 1; i < maxTd; i++) {
            ptd[i] += ptd[i - 1];
        }

        for (int d = 0; d < numDocuments; d++) {

            double u = rg.nextDouble() * ptd[ptd.length-1];
            int td = -1;
            for (int i = 0, length_ptd = ptd.length; i < length_ptd; i++){
                if(Double.compare(ptd[i], u) >= 0){
                    td = i+1;
                    break;
                }
            }

            assert(td>=1);
            TdArray[d] = td;

            Set<Integer> Zd = new HashSet<Integer>();
            while ( Zd.size() != td ){
                int u_z = rg.nextInt(numTopics);
                Zd.add(u_z);
            }
            ZdMap.put(d, Zd);
            normalCountZd(Zd, +1);

            Object[] ZdList = new Object[td];
            ZdList =  Zd.toArray();
            int[] termIDArray = docToWordIDListMap.get(d);
            for (int w = 0, num_word = termIDArray.length; w < num_word; w++){
                int topic_index = rg.nextInt(td);
                int topic = (int) ZdList[topic_index];
                int word = termIDArray[w];
                assignmentListMap.get(d)[w] = topic;
                normalCountWord(topic, word, +1);
            }
        }
        System.out.println("finish init_GPU_PDMM!");
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

    public Map<Integer, Map<Integer, Double>> computSchema(String pathToVector, double threshold) {

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


           // System.out.println("vocabularySize:" + vocabularySize);
            double count = 0.0;
            for (int i = 0; i < vocabularySize; i++) {

                if(!wordMap.containsKey(i) )
                    continue;
               // System.out.println("------"+i+"---------");
                Map<Integer, Double> tmpMap = new HashMap<Integer, Double>();
                for (int j = 0; j < vocabularySize; j++) {
                    if(!wordMap.containsKey(j) )
                        continue;
                    double v = computeSis(wordMap, i,j);
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

    private static int factorial(int n){
        int value = 1;
        while ( n > 0 ){
            value *= n;
            n--;
        }

        return value;
    }
    private int[][] ZdSearchSize(){
        int count = 0;
        int[] boundary = new int[maxTd];
        for ( int i = 0; i < maxTd; i++ ){
            int temp = 1;
            int factorial = factorial(i+1);
            for ( int j = 0; j < i+1; j++ ){
                temp *= (searchTopK - j);
            }

            count += temp/factorial;
            boundary[i] = count;
        }

        int[][] array = new int[count][];
        for ( int i = 0; i < count; i++ ){
            for ( int j = 0; j < boundary.length; j++ ){
                if ( i < boundary[j] ){
                    array[i] = new int[j+1];
                    break;
                }
            }
        }

        return array;
    }

    public void compute_phi() {
        for (int i = 0; i < numTopics; i++) {
            double sum = 0.0;
            for (int j = 0; j < vocabularySize; j++) {
                sum += nzw[i][j];
            }

            for (int j = 0; j < vocabularySize; j++) {
                phi[i][j] = (nzw[i][j] + beta) / (sum + vocabularySize * beta);
            }
        }
    }

    public void compute_pz() {
        double sum = 0.0;
        for (int i = 0; i < numTopics; i++) {
            sum += nz[i];
        }

        for (int i = 0; i < numTopics; i++) {
            pz[i] = 1.0 * (nz[i] + alpha) / (sum + numTopics * alpha);
        }
    }

    public void compute_pzd() {
        /** calculate P(z|w) **/
        for (int i = 0; i < vocabularySize; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < numTopics; j++) {
                pwz[i][j] = pz[j] * phi[j][i];
                row_sum += pwz[i][j];
            }

            for (int j = 0; j < numTopics; j++) {
                pwz[i][j] = pwz[i][j] / row_sum;
            }
        }

        for (int i = 0; i < numDocuments; i++) {
            int[] doc_word_id = docToWordIDListMap.get(i);
            double row_sum = 0.0;
            for (int j = 0; j < numTopics; j++) {
                pdz[i][j] = 0;
                for (int wordID : doc_word_id) {
                    pdz[i][j] += pwz[wordID][j];
                }
                row_sum += pdz[i][j];
            }

            for (int j = 0; j < numTopics; j++) {
                pdz[i][j] = pdz[i][j] / row_sum;
            }
        }
    }

    public int[][] getTopKTopics(int[][] docTopKTopics){
        Set<Integer> topKTopics = new HashSet<Integer>();
        int minIndex = -1;
        double minValue = 2;
        for(int d = 0; d < numDocuments; d++){
            minValue = 2;
            minIndex = -1;
            topKTopics.clear();

            for(int k = 0; k < numTopics; k++){
                if ( topKTopics.size() < searchTopK ){
                    topKTopics.add(k);
                    if ( Double.compare(minValue, pdz[d][k]) > 0 ){
                        minValue = pdz[d][k];
                        minIndex = k;
                    }
                } else {
                    if (Double.compare(minValue, pdz[d][k]) < 0) {
                        topKTopics.remove(minIndex);
                        topKTopics.add(k);
                        minIndex = minPDZTopicIndex(d, topKTopics);
                        minValue = pdz[d][minIndex];
                    }
                }
            }

            int index = 0;
            for ( int topic : topKTopics ){
                docTopKTopics[d][index++] = topic;
            }
        }

        return docTopKTopics;
    }

    private int minPDZTopicIndex(int doc, Set<Integer> topics){
        double min = 2;
        int minIndex = -1;
        for ( int topic : topics ){
            if ( Double.compare(min, pdz[doc][topic]) > 0 ){
                min = pdz[doc][topic];
                minIndex = topic;
            }
        }

        return minIndex;
    }

    public double findTopicMaxProbabilityGivenWord(int wordID) {
        double max = -1.0;
        double tmp = -1.0;
        for (int i = 0; i < numTopics; i++) {
            tmp = topicProbabilityGivenWord[wordID][i];
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
    public void updateWordGPUFlag(int docID, int word, int index, int newTopic) {
        // we calculate the p(t|w) and p_max(t|w) and use the ratio to decide we
        // use gpu for the word under this topic or not
        double maxProbability = findTopicMaxProbabilityGivenWord(word);
        double ratio = getTopicProbabilityGivenWord(newTopic, word) / maxProbability;
        double a = rg.nextDouble();
        if(Double.compare(ratio, a) > 0){
            wordGPUFlag.get(docID).set(index, true);
        }
        else{
            wordGPUFlag.get(docID).set(index, false);
        }

    }

    public void ratioCount(Integer topic, Integer docID, int word, int index, int flag) {
        nzw[topic][word] += flag;
        nz[topic] += flag;

        // we update gpu flag for every document before it change the counter
        // when adding numbers
        if (flag > 0) {
            updateWordGPUFlag(docID, word, index, topic);
            boolean gpuFlag = wordGPUFlag.get(docID).get(index);
            Map<Integer, Double> gpuInfo = new HashMap<Integer, Double>();
            if (gpuFlag) { // do gpu count
                if (schemaMap.containsKey(word)) {
                    Map<Integer, Double> valueMap = schemaMap.get(word);
                    // update the counter
                    for (Map.Entry<Integer, Double> entry : valueMap.entrySet()) {
                        int k = entry.getKey();
                        double v = weight;
                        nzw[topic][k] += v;
                        nz[topic] += v;
                        gpuInfo.put(k, v);
                    } // end loop for similar words
                    //		valueMap.clear();
                } else { // schemaMap don't contain the word

                    // the word doesn't have similar words, the infoMap is empty
                    // we do nothing
                }
            } else { // the gpuFlag is False
                // it means we don't do gpu, so the gouInfo map is empty
            }
            //		wordGPUInfo.get(docID).get(index).clear();
            wordGPUInfo.get(docID).set(index, gpuInfo); // we update the gpuinfo
            // map
        } else { // we do subtraction according to last iteration information
            Map<Integer, Double> gpuInfo = wordGPUInfo.get(docID).get(index);
            // boolean gpuFlag = wordGPUFlag.get(docID).get(t);
            if (gpuInfo.size() > 0) {
                for (int similarWordID : gpuInfo.keySet()) {
                    // double v = gpuInfo.get(similarWordID);
                    double v = weight;
                    nzw[topic][similarWordID] -= v;
                    nz[topic] -= v;
                    // if(Double.compare(0, nzw[topic][wordID]) > 0){
                    // System.out.println( nzw[topic][wordID]);
                    // }
                }
            }
        }
    }

    private static int enumerateOneTopicSetting(int[][] topicSettingArray,
                                                int[] topKTopics, int index){
        for ( int i = 0; i < topKTopics.length; i++ ){
            topicSettingArray[index++][0] = topKTopics[i];
        }

        return index;
    }

    private static int enumerateTwoTopicSetting(int[][] topicSettingArray,
                                                int[] topKTopics, int index){
        for ( int i = 0; i < topKTopics.length; i++ ){
            for ( int j = i+1; j < topKTopics.length; j++ ){
                topicSettingArray[index][0] = topKTopics[i];
                topicSettingArray[index++][1] = topKTopics[j];
            }
        }

        return index;
    }

    private static int enumerateThreeTopicSetting(int[][] topicSettingArray,
                                                  int[] topKTopics, int index){
        for ( int i = 0; i < topKTopics.length; i++ ){
            for ( int j = i+1; j < topKTopics.length; j++ ){
                for (int n = j + 1; n < topKTopics.length; n++) {
                    topicSettingArray[index][0] = topKTopics[i];
                    topicSettingArray[index][1] = topKTopics[j];
                    topicSettingArray[index++][2] = topKTopics[n];
                }
            }
        }

        return index;
    }

    private static int enumerateFourTopicSetting(int[][] topicSettingArray,
                                                 int[] topKTopics, int index){
        for ( int i = 0; i < topKTopics.length; i++ ){
            for ( int j = i+1; j < topKTopics.length; j++ ){
                for (int n = j + 1; n < topKTopics.length; n++) {
                    for (int m = n +1; m < topKTopics.length; m++){
                        topicSettingArray[index][0] = topKTopics[i];
                        topicSettingArray[index][1] = topKTopics[j];
                        topicSettingArray[index][2] = topKTopics[n];
                        topicSettingArray[index++][3] = topKTopics[m];
                    }
                }
            }
        }

        return index;
    }

    private static int[][] enumerateTopicSetting(int[][] topicSettingArray,
                                                 int[] topKTopics, int maxTd) {
        // TODO Auto-generated method stub
        int index = 0;
        if ( maxTd > 0)
            index = enumerateOneTopicSetting(topicSettingArray, topKTopics, index);

        if ( maxTd > 1)
            index = enumerateTwoTopicSetting(topicSettingArray, topKTopics, index);

        if ( maxTd > 2)
            index = enumerateThreeTopicSetting(topicSettingArray, topKTopics, index);

        if ( maxTd > 3)
            index = enumerateFourTopicSetting(topicSettingArray, topKTopics, index);

        return topicSettingArray;
    }

    private static long getCurrTime() {
        return System.currentTimeMillis();
    }

    /* Create a new memory block like two dimensional array is very
		 * expensive in Java. We need to reuse the memory block instead of
		 * creating a new one every time*/
    public void inference()
            throws IOException
    {

        writeDictionary();

        long _s = getCurrTime();

        int[][] topicSettingArray = ZdSearchSize();
        int[][] docTopKTopics = new int[numDocuments][searchTopK];
        double[] Ptd_Zd = new double[topicSettingArray.length];
        int[] termIDArray = null;
        int[][] mediateSamples = null;

        Map<Integer, int[][]> mediateSampleMap =
                new HashMap<Integer, int[][]>();

        for ( int i = 0; i < numDocuments; i++ ){
            termIDArray = docToWordIDListMap.get(i);
            mediateSamples =
                    new int[topicSettingArray.length][termIDArray.length];
            mediateSampleMap.put(i, mediateSamples);
        }

        System.out.println("Running Gibbs sampling inference: ");

        for (int iter = 1; iter <= numIterations; iter++) {

            if(iter%50==0)
                System.out.print(" " + (iter));



            //		if don't use heu strategy,please don't Use below three code line
            compute_phi();
            compute_pz();
            compute_pzd();

            docTopKTopics = getTopKTopics(docTopKTopics);

            for (int s = 0; s < numDocuments; s++) {
                termIDArray = docToWordIDListMap.get(s);
                int num_word = termIDArray.length;
                Set<Integer> preZd = ZdMap.get(s);
                normalCountZd(preZd, -1);
                mediateSamples = mediateSampleMap.get(s);

                for (int w = 0; w < num_word; w++){
                    //		normalCountWord(assignmentListMap.get(s)[w], termIDArray[w], -1);
                    //		ratioCountNofilter(assignmentListMap.get(s)[w], s, termIDArray[w], w, -1);
                    ratioCount(assignmentListMap.get(s)[w], s, termIDArray[w], w, -1);
                }

                topicSettingArray = enumerateTopicSetting(
                        topicSettingArray, docTopKTopics[s], maxTd);
                int length_topicSettingArray = topicSettingArray.length;

                for(int round = 0; round < length_topicSettingArray; round++){
                    int[] topicSetting = topicSettingArray[round];
                    int length_topicSetting = topicSetting.length;

                    for (int w = 0; w < num_word; w++){
                        int wordID = termIDArray[w];
                        double[] pzDist = new double[length_topicSetting];
                        for (int index = 0; index < length_topicSetting; index++) {
                            int topic = (int) topicSetting[index];
                            //		System.out.println(nzw[topic][wordID]);
                            double pz = 1.0 * (nzw[topic][wordID] + beta) / (nz[topic] +  beta*vocabularySize);
                            pzDist[index] = pz;
                        }

                        for (int i = 1; i < length_topicSetting; i++) {
                            pzDist[i] += pzDist[i - 1];
                        }

                        double u = rg.nextDouble() * pzDist[length_topicSetting - 1];
                        int newTopic = -1;
                        for (int i = 0; i < length_topicSetting; i++) {
                            if (Double.compare(pzDist[i], u) >= 0) {
                                newTopic = topicSetting[i];
                                break;
                            }
                        }
                        // update
                        mediateSamples[round][w] = newTopic;
                        //	normalCountWord(newTopic, wordID, +1);
                        //	ratioCountNofilter(newTopic, s, wordID, w, +1);
                        ratioCount(newTopic, s, wordID, w, +1);
                    }

                    for (int w = 0; w < num_word; w++){
                        int wordID = termIDArray[w];
                        //	normalCountWord(mediateSamples[round][w], wordID, -1);
                        //	ratioCountNofilter(mediateSamples[round][w], s, wordID, w, -1);
                        ratioCount(mediateSamples[round][w], s, wordID, w, -1);
                    }
                }

                for (int round = 0; round < length_topicSettingArray; round++){
                    int[] topicSetting = topicSettingArray[round];
                    int length_topicSetting = topicSetting.length;
                    double p1 = Math.pow(lambda, topicSetting.length) * Math.exp(-lambda);
                    double p21 = 1.0;
                    double p22 = 1.0;

                    for(int k : topicSetting){
                        p21*= (alpha + Ck[k]);
                    }

                    for(int i = 0; i < length_topicSetting; i++){
                        p22 *= (CkSum + numTopics*alpha - i);
                    }
                    double p2 = p21/p22;
                    double p31 = 1.0;
                    double p32 = 1.0;
                    Map<Integer, Map<Integer, Integer>> Ndkt =
                            getNdkt_Zd(s, topicSetting, mediateSamples[round]);
                    Map<Integer, Integer> Ndk =
                            getNdk_Zd(s, topicSetting, mediateSamples[round]);
                    for(int k: topicSetting){
                        Set<Integer> dk =
                                getdk_Zd(s, mediateSamples[round], k);
                        //	System.out.println(dk);
                        for(int t: dk){
                            for (int i = 0; i < Ndkt.get(k).get(t); i++){
                                p31 *= (beta+nzw[k][t]+Ndkt.get(k).get(t)-i);
                            }
                        }
                        for(int j = 0; j < Ndk.get(k); j++){
                            p32 *= (nz[k]+beta*vocabularySize+Ndk.get(k)-j);
                        }
                        dk.clear();

                    }
                    Ndkt.clear();
                    Ndk.clear();
                    double p3 = p31/p32;
                    Ptd_Zd[round] = p1*p2*p3;
                }

                for(int i = 1; i < length_topicSettingArray; i++){
                    Ptd_Zd[i]+=Ptd_Zd[i-1];
                }

                double u_ptdzd = rg.nextDouble()*Ptd_Zd[length_topicSettingArray-1];
                int new_index = -1;
                for (int i = 0; i < length_topicSettingArray; i++) {
                    if (Double.compare(Ptd_Zd[i], u_ptdzd) >= 0) {
                        new_index = i;
                        break;
                    }
                }

                TdArray[s] = topicSettingArray[new_index].length;
                preZd.clear();
                for(int k: topicSettingArray[new_index]){
                    preZd.add(k);
                }

                normalCountZd(preZd, +1);
                System.arraycopy(
                        mediateSamples[new_index], 0,
                        assignmentListMap.get(s), 0, mediateSamples[new_index].length);
                for(int w = 0; w < termIDArray.length; w++){
                    //	normalCountWord(mediateSamples[new_index][w], termIDArray[w], +1);
                    //	ratioCountNofilter(mediateSamples[new_index][w], s, termIDArray[w], w, +1);
                    ratioCount(mediateSamples[new_index][w], s, termIDArray[w], w, +1);
                }
            }


        }
        expName = orgExpName;

        long _e = getCurrTime();
        iterTime = (_e - _s) ;

        System.out.println();
        System.out.println("Writing output from the last sample ...");
        write();

        System.out.println("Sampling completed!");

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
                wordCount.put(wIndex, nzw[tIndex][wIndex]);
            }
            wordCount = FuncUtils.sortByValueDescending(wordCount);

            Set<Integer> mostLikelyWords = wordCount.keySet();
            int count = 0;
            for (Integer index : mostLikelyWords) {
                if (count < topWords) {
                    double pro = (nzw[tIndex][index] + beta)
                            / (nz[tIndex] + beta*vocabularySize);
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

        /** calculate P(z|w) **/
        for (int i = 0; i < vocabularySize; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < numTopics; j++) {
                pwz[i][j] = pz[j] * phi[j][i];
                row_sum += pwz[i][j];
            }

            for (int j = 0; j < numTopics; j++) {
                pwz[i][j] = pwz[i][j] / row_sum;
            }
        }

        for (int i = 0; i < numDocuments; i++) {
            int[] doc_word_id = docToWordIDListMap.get(i);
            double row_sum = 0.0;
            for (int j = 0; j < numTopics; j++) {
                pdz[i][j] = 0;
                for (int wordID : doc_word_id) {
                    pdz[i][j] += pwz[wordID][j];
                }
                row_sum += pdz[i][j];
            }

            for (int j = 0; j < numTopics; j++) {
                pdz[i][j] = pdz[i][j] / row_sum;
                writer.write( pdz[i][j] + " ");
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
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                writer.write(assignmentListMap.get(dIndex)[wIndex] + " ");
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
                double pro = (nzw[i][j] + beta) / (nz[i] + vocabularySize * beta);
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


    public Set<Integer> getdk_Zd(
            int docID, int[] assignment, int topic){
        Set<Integer> dk = new HashSet<Integer>();
        int[] termIDArray = docToWordIDListMap.get(docID);
        for(int i = 0, length = assignment.length; i < length; i++){
            int z = assignment[i];
            if (z==topic){
                dk.add(termIDArray[i]);
            }
        }
        return dk;
    }

    public Map<Integer, Map<Integer, Integer>> getNdkt_Zd(
            int docID, int[] ZdList, int[] assignment){
        Map<Integer, Map<Integer, Integer>> Ndkt = new HashMap<Integer,Map<Integer, Integer>>();
        for(int k : ZdList){
            Ndkt.put(k, new HashMap<Integer, Integer>());
            //		System.out.println(ZdList.length);
        }
        int[] termIDArray = docToWordIDListMap.get(docID);
        for(int i = 0, length = termIDArray.length; i < length; i++){
            int word = termIDArray[i];
            int topic = assignment[i];
            //	System.out.println(topic);
            if (Ndkt.get(topic).containsKey(word)){
                Ndkt.get(topic).put(word, Ndkt.get(topic).get(word)+1);
            }
            else{
                Ndkt.get(topic).put(word, 1);
            }
        }
        return Ndkt;
    }

    public Map<Integer, Integer> getNdk_Zd(
            int docID, int[] ZdList, int[] assignment){
        Map<Integer, Integer> Ndk = new HashMap<Integer,Integer>();
        for(int k : ZdList){
            Ndk.put(k,0);
            //		System.out.println(ZdList.length);
        }
        int[] termIDArray = docToWordIDListMap.get(docID);
        for(int i = 0, length = termIDArray.length; i < length; i++){
            int word = termIDArray[i];
            int topic = assignment[i];
            //	System.out.println(topic);
            Ndk.put(topic, Ndk.get(topic)+1);
        }
        return Ndk;
    }

    public void writeParameters()
            throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".paras"));
        writer.write("-model" + "\t" + "GPU_PDMM");
        writer.write("\n-corpus" + "\t" + corpusPath);
        writer.write("\n-ntopics" + "\t" + numTopics);
        writer.write("\n-alpha" + "\t" + alpha);
        writer.write("\n-beta" + "\t" + beta);
        writer.write("\n-lambda" + "\t" + lambda);
        writer.write("\n-threshold" + "\t" + threshold);
        writer.write("\n-weight" + "\t" + weight);
        writer.write("\n-filterSize" + "\t" + filterSize);
        writer.write("\n-niters" + "\t" + numIterations);
        writer.write("\n-twords" + "\t" + topWords);
        writer.write("\n-" + "\t" + topWords);
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
        GPU_PDMM gpupdmm = new GPU_PDMM("dataset/Tweet.txt","dataset/glove.6B.200d.txt", 0.7, 0.1, 20, 100, 0.1,
                0.1, 1.5, 500, 10, 3, 10, "TweetGPU_PDMM");

        gpupdmm.inference();
    }
}
