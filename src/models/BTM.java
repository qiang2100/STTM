package models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.io.File;

import utility.FuncUtils;

/**
 * * BTM: A Java package for the short text topic models
 *
 * Implementation of the Biterm topic modeling, using collapsed Gibbs sampling, as described in:
 *
 * Xueqi Cheng, Xiaohui Yan, Yanyan Lan, and Jiafeng Guo. BTM: Topic Modeling over Short Texts.
 * In IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, 2014.
 *
 * @author: Jipeng Qiang on 18/6/6.
 */
public class BTM {

    public double alpha; // Hyper-parameter alpha
    public double beta; // Hyper-parameter alpha
    public int numTopics; // Number of topics
    public int numIterations; // Number of Gibbs sampling iterations

    public int topWords; // Number of most probable words for each topic

    public double alphaSum; // alpha * numTopics
    public double betaSum; // beta * vocabularySize



    public int numDocuments; // Number of documents in the corpus
    public int numWordsInCorpus; // Number of words in the corpus

    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
    // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
    // given an ID
    public int vocabularySize; // The number of word types in the corpus

    int[][] wordId_of_corpus = null;
    public ArrayList<HashMap<Long,Integer>> biterm_of_corpus = new ArrayList<>();
    int[] doc_biterm_num ;
    ArrayList<Long> biterms = new ArrayList<>();


    int[] topic_of_biterms;
    int[][] topic_word_num;
    int[] num_of_topic_of_biterm;

    private HashMap<Long, Double> bitermSum = new HashMap<>();


    //public int[][] topicWordCount;
    // Total number of words assigned to a topic
    //public int[] sumTopicWordCount;

    // Double array used to sample a topic
    public double[] multiPros;

    // Path to the directory containing the corpus
    public String folderPath;
    // Path to the topic modeling corpus
    public String corpusPath;

    public String expName = "BTMmodel";
    public String orgExpName = "BTMmodel";
    public String tAssignsFilePath = "";
    public int savestep = 0;

    public BTM(String pathToCorpus, int inNumTopics,
                            double inAlpha, double inBeta, int inNumIterations, int inTopWords)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, "BTMmodel");
    }

    public BTM(String pathToCorpus, int inNumTopics,
                            double inAlpha, double inBeta, int inNumIterations, int inTopWords,
                            String inExpName)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, inExpName, "", 0);

    }

    public BTM(String pathToCorpus, int inNumTopics,
                            double inAlpha, double inBeta, int inNumIterations, int inTopWords,
                            String inExpName, String pathToTAfile)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, inExpName, pathToTAfile, 0);

    }

    public BTM(String pathToCorpus, int inNumTopics,
                            double inAlpha, double inBeta, int inNumIterations, int inTopWords,
                            String inExpName, int inSaveStep)
            throws Exception
    {
        this(pathToCorpus, inNumTopics, inAlpha, inBeta, inNumIterations,
                inTopWords, inExpName, "", inSaveStep);

    }

    public BTM(String pathToCorpus, int inNumTopics,
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
        System.out.println("Reading topic modeling corpus: " + pathToCorpus);

        folderPath = "results/";
        File dir = new File(folderPath);
        if (!dir.exists())
            dir.mkdir();


        word2IdVocabulary = new HashMap<String, Integer>();
        id2WordVocabulary = new HashMap<Integer, String>();
        //corpus = new ArrayList<List<Integer>>();
        ArrayList<int[]> tmpCorpus = new ArrayList<>();

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
                tmpCorpus.add(document);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        vocabularySize = word2IdVocabulary.size(); // vocabularySize = indexWord

        //topicWordCount = new int[numTopics][vocabularySize];

        //sumTopicWordCount = new int[numTopics];

        multiPros = new double[numTopics];
        for (int i = 0; i < numTopics; i++) {
            multiPros[i] = 1.0 / numTopics;
        }

        alphaSum = numTopics * alpha;
        betaSum = vocabularySize * beta;

        this.doc_biterm_num = new int[tmpCorpus.size()];
        this.wordId_of_corpus = new int[tmpCorpus.size()][];
        for (int docIndex = 0; docIndex < this.wordId_of_corpus.length; docIndex++) {
            this.wordId_of_corpus[docIndex] = tmpCorpus.get(docIndex);
        }

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
        System.out.println("Randomly initializing topic assignments using BTM");

        int docIndex = 0;
        for(int[] doc:this.wordId_of_corpus){
            HashMap<Long, Integer> oneCop = new HashMap<>();
            for(int word1:doc){
                for(int word2:doc){
                    if(word1<word2){
                        Long itmeNum = (long)word1*1000000+word2;
                        if(!oneCop.containsKey(itmeNum)){
                            oneCop.put(itmeNum,0);
                        }
                        oneCop.put(itmeNum,oneCop.get(itmeNum)+1);
                        this.biterms.add(itmeNum);
                        this.doc_biterm_num[docIndex] += 1;
                    }
                }
            }
            docIndex++;
            this.biterm_of_corpus.add(oneCop);
        }



        this.topic_of_biterms = new int[this.biterms.size()];
        this.topic_word_num = new int[this.vocabularySize][this.numTopics];
        this.num_of_topic_of_biterm = new int[this.numTopics];

        for(int bitermIndex=0;bitermIndex<this.biterms.size();bitermIndex++){
            int topicId = FuncUtils.nextDiscrete(multiPros); // Sample a topic;
            this.topic_word_num[(int)(this.biterms.get(bitermIndex)%1000000)][topicId] += 1;
            this.topic_word_num[(int)(this.biterms.get(bitermIndex)/1000000)][topicId] += 1;
            this.num_of_topic_of_biterm[topicId] += 1;
            this.topic_of_biterms[bitermIndex] = topicId;
        }

    }

    public void inference()
            throws IOException
    {

        writeParameters();
        writeDictionary();

        System.out.println("Running Gibbs sampling inference: ");

        for (int iter = 0; iter < this.numIterations; iter++) {
            long startTime = System.currentTimeMillis();

            for(int bitermIndex = 0;bitermIndex<this.topic_of_biterms.length;bitermIndex++) {
                int oldTopicId = this.topic_of_biterms[bitermIndex];
                int word1 = (int)(this.biterms.get(bitermIndex)/1000000);
                int word2 = (int)(this.biterms.get(bitermIndex)%1000000);
                this.topic_word_num[word1][oldTopicId] -= 1;
                this.topic_word_num[word2][oldTopicId] -= 1;
                this.num_of_topic_of_biterm[oldTopicId] -= 1;

                int newTopicId = -1;

               // double[] p = new double[this.topic_num];
                for (int k = 0; k < this.numTopics; k++) {
                    multiPros[k] = (this.num_of_topic_of_biterm[k] + alpha)
                            * (this.topic_word_num[word1][k] + beta)
                            * (this.topic_word_num[word2][k] + beta)
                            / Math.pow(this.num_of_topic_of_biterm[k]*2 + this.vocabularySize * beta, 2);
                }

                newTopicId = FuncUtils.nextDiscrete(multiPros);
                this.topic_word_num[word1][newTopicId] += 1;
                this.topic_word_num[word2][newTopicId] += 1;
                this.num_of_topic_of_biterm[newTopicId] += 1;

                this.topic_of_biterms[bitermIndex] = newTopicId;
            }

            System.out.println("finished iter :" + iter + "\tcost time:" + ((double) System.currentTimeMillis() - startTime) / 1000);

        }

        expName = orgExpName;

        System.out.println("Writing output from the last sample ...");
        write();

        System.out.println("Sampling completed for BTM!");

    }

    public void writeParameters()
            throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".paras"));
        writer.write("-model" + "\t" + "BTM");
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

    private void writeTopTopicalWords() throws IOException {

        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".topWords"));

        for (int topic_id = 0; topic_id < this.numTopics; topic_id++) {
            //writer.write("Topic" + new Integer(topic_id) + ":");
            HashMap<Integer, Double> oneLine = new HashMap<>();
            for (int word_id = 0; word_id < this.vocabularySize; word_id++) {
                oneLine.put(word_id, ((double) this.topic_word_num[word_id][topic_id]) / this.num_of_topic_of_biterm[topic_id] / 2);
            }
            List<Map.Entry<Integer, Double>> maplist = new ArrayList<>(oneLine.entrySet());

            Collections.sort(maplist, (o1, o2) -> o2.getValue().compareTo(o1.getValue()));

            //writer.write("Topic:" + topic_id + "\n");
            int count = 0;
            for (Map.Entry<Integer, Double> o1 : maplist) {
                writer.write( this.id2WordVocabulary.get(o1.getKey()) + " ") ;
                count++;
                if (count >= topWords) {
                    break;
                }
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


        int topic_index = 0;
        for (int topic_id = 0; topic_id < this.numTopics; topic_id++) {
            for (int word_id = 0; word_id < vocabularySize; word_id++) {
                writer.write(((this.topic_word_num[word_id][topic_id] + beta) / (this.num_of_topic_of_biterm[topic_id] * 2 + vocabularySize * beta))+" ");
            }

            writer.write("\n");
        }

        writer.close();
    }

    private double getSum(Long biterm){
        if(!bitermSum.containsKey(biterm)) {
            double sum = 0;
            int word1 = (int)(biterm/1000000);
            int word2 = (int)(biterm%1000000);
            for (int topic_id = 0; topic_id < this.numTopics; topic_id++) {
                sum += (this.num_of_topic_of_biterm[topic_id] + alpha)
                        * (this.topic_word_num[word1][topic_id] + beta)
                        * (this.topic_word_num[word2][topic_id] + beta)
                        / Math.pow(this.num_of_topic_of_biterm[topic_id] * 2 + this.vocabularySize * beta, 2);
            }
            bitermSum.put(biterm,sum);
        }
        return bitermSum.get(biterm);
    }

    public void writeDocTopicPros()
            throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                + expName + ".theta"));

        int docIndex = 0;
        for (HashMap<Long,Integer> line : this.biterm_of_corpus) {
            double[] oneTheta = new double[this.numTopics];
            for(int topic_id = 0; topic_id<this.numTopics;topic_id++) {
                double oneSum=0;
                for (Long biterm : line.keySet()) {
                    int word1 = (int)(biterm/1000000);
                    int word2 = (int)(biterm%1000000);
                    oneSum+=(((double)line.get(biterm))/this.doc_biterm_num[docIndex])
                            *((
                            (this.num_of_topic_of_biterm[topic_id] + alpha)
                                    * (this.topic_word_num[word1][topic_id] + beta)
                                    * (this.topic_word_num[word2][topic_id] + beta)
                                    / Math.pow(this.num_of_topic_of_biterm[topic_id]*2 + this.vocabularySize * beta, 2)
                    )/(getSum(biterm)));

                }
                writer.write(oneSum + " ");
            }
            writer.write("\n");
            docIndex++;
        }
        writer.close();

    }



    public void write()
            throws IOException
    {
        writeTopTopicalWords();
        writeDocTopicPros();

        writeTopicWordPros();
    }

    public static void main(String args[])
            throws Exception
    {
        BTM btm = new BTM("dataset/SearchSnippets.txt", 8, 0.1, 0.1, 2000, 20, "SearchSnippetsBTM");
        btm.inference();
    }
}
