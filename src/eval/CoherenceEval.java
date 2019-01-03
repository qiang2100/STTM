package eval;

import utility.FuncUtils;

import java.io.*;
import java.util.*;

/**
 *  Implementation of PMI for topic coherence evaluation scores, as
 * described in Section 4.2 in:
 *
 * David Newman, et al. 2010.
 * Automatic Evaluation of Topic Coherence. NAACL.
 *
 *
 * Created by jipengqiang on 18/7/31.
 */
public class CoherenceEval {

    //Map<Integer, Map<Integer, Integer>> wordGraph;
    //Map<Integer,Integer> wordDegree;

    int [][]ww;
    public int numDocuments; // Number of documents in the corpus

    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
    // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
    // given an ID
 //   public int vocabularySize; // The number of word types in the corpus

    public int windowSize = 100;

    public int windowNum = 0;

    int indexWord = 0;

    public CoherenceEval()
            throws Exception
    {

        word2IdVocabulary = new HashMap<String,Integer>();
        id2WordVocabulary = new HashMap<Integer,String>();
        //wordGraph = new HashMap<Integer, Map<Integer, Integer>>();
       // wordDegree = new HashMap<Integer,Integer>();
       // topTopicWordList = new ArrayList<ArrayList<Integer>>();
        windowNum = 0;
    }

    public void readTopTopicWordForVocabulary(String pathTopTopicFile)
    {
        BufferedReader br = null;
        try {

            br = new BufferedReader(new FileReader(pathTopTopicFile));
            for (String doc; (doc = br.readLine()) != null;) {

                if (doc.trim().length() == 0)
                    continue;

                String[] words = doc.trim().split(" ");
               // ArrayList<Integer> oneTopic = new ArrayList<Integer>();

                for (String word : words) {
                    if (!word2IdVocabulary.containsKey(word) ){

                        word2IdVocabulary.put(word, indexWord);
                        id2WordVocabulary.put(indexWord, word);
                        indexWord++;
                    }
                }
            }
            br.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }


    public ArrayList<ArrayList<Integer>>  readTopTopicWord(String pathTopTopicFile)
    {
        BufferedReader br = null;

        ArrayList<ArrayList<Integer>>  topTopicWordList = new ArrayList<ArrayList<Integer>>();
        try {

            br = new BufferedReader(new FileReader(pathTopTopicFile));
            for (String doc; (doc = br.readLine()) != null;) {

                if (doc.trim().length() == 0)
                    continue;

                String[] words = doc.trim().split(" ");
                ArrayList<Integer> oneTopic = new ArrayList<Integer>();

                for (String word : words) {

                    oneTopic.add(word2IdVocabulary.get(word));

                }

                topTopicWordList.add(oneTopic);
            }
            br.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return topTopicWordList;
    }
    public void readWikipedia(String pathWikipediaFile)
    {
        BufferedReader br = null;
        try {
            int indexWord = -1;
            br = new BufferedReader(new FileReader(pathWikipediaFile));
            int senId = 0;
            for (String doc; (doc = br.readLine()) != null;) {

                if (doc.trim().length() == 0)
                    continue;

                if(senId%1000==0)
                    System.out.print(senId + " ");
                senId++;
                String[] words = doc.trim().split("\\s+");

                int docSize = words.length;

                if(docSize<=windowSize){

                    for (int k = 0; k<docSize; k++) {
                        if (!word2IdVocabulary.containsKey(words[k]))
                            continue;
                        else{
                            int wordId = word2IdVocabulary.get(words[k]);
                           // wordDegree.put(wordId,wordDegree.get(wordId)+1);
                        }

                        for (int m = k + 1; m<docSize; m++)
                            if (word2IdVocabulary.containsKey(words[m]))
                                System.out.println();
                                //addEdge(word2IdVocabulary.get(words[k]),word2IdVocabulary.get(words[m]));
                    }
                    windowNum++;
                }else {
                    for (int j = 0; j < docSize - windowSize + 1; j++) {

                        //for (int k = j; k < j + windowSize; k++)


                        for (int k = j; k < j + windowSize ; k++) {
                            if (!word2IdVocabulary.containsKey(words[k]))
                                continue;
                            else{
                                int wordId = word2IdVocabulary.get(words[k]);
                               // wordDegree.put(wordId, wordDegree.get(wordId) + 1);

                            }

                            for (int m = k + 1; m < j + windowSize; m++)
                                if (word2IdVocabulary.containsKey(words[m]))
                                    System.out.println();// addEdge(word2IdVocabulary.get(words[k]), word2IdVocabulary.get(words[m]));
                        }
                        windowNum++;
                    }
                }
               // corpus.add(document);
            }
            br.close();
           // show();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void readWikipediaWhole(String pathWikipediaFile)
    {
        BufferedReader br = null;
        int senId = 0;
        try {
            int indexWord = -1;
            br = new BufferedReader(new FileReader(pathWikipediaFile));

            System.out.println("began to read file from wikepedia");
            for (String doc; (doc = br.readLine()) != null;) {

                if (doc.trim().length() == 0)
                    continue;



                if(senId%10000==0)
                    System.out.print("| ");
                if(senId%200000==0)
                    System.out.println();
                senId++;
                String[] words = doc.trim().split("\\s+");

                int docSize = words.length;

                ArrayList<Integer> arr = new ArrayList<Integer>();

                for (int k = 0; k < docSize; k++) {

                    int wordId = -1;
                    if (!word2IdVocabulary.containsKey(words[k]))
                        continue;
                    else {
                        wordId = word2IdVocabulary.get(words[k]);

                        if (arr.contains(wordId))
                            continue;
                        else
                        {
                            arr.add(wordId);
                            ww[wordId][wordId] += 1;
                        }

                    }
                }
                if(arr.size()==0)
                    continue;
                for(int k=0; k<arr.size()-1; k++)
                    for (int m = k + 1; m <arr.size(); m++){
                            ww[arr.get(k)][arr.get(m)] += 1;
                            ww[arr.get(m)][arr.get(k)] += 1;
                        }


                // corpus.add(document);
            }
            br.close();
            // show();
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        windowNum = senId;

        if(windowNum==1000000)
            System.out.println("finish!");
        else
            System.out.println("the number of wikipedia documents: " + windowNum);
    }



    public double computOneTopicPMI(ArrayList<Integer> topW)
    {
        double coh = 0;
        // double coh2 = 0;

        for(int i=0; i<topW.size(); i++)
        {

            for(int j=i+1; j<topW.size(); j++)
            {
                //coh = 0;
                //coh2 = 0;
                int id1 = topW.get(i);
                int id2 = topW.get(j);

                int comC = ww[id1][id2];

                if(comC==0)
                    continue;

                double comP = (double)comC;
                double id1P = (double) ww[id1][id1];
                // System.out.println(id2);
                double id2P = (double) ww[id2][id2];

                if(id1P==0 || id2P==0)
                    System.out.println("function \"computOneTopicCoh\" Exception");

                if(comP==0)
                    coh += 0;
                else
                {
                    //coh +=  Math.log((comC+1)/id2P);
                    coh += Math.log(comP*windowNum/(id1P*id2P));
                }
                //System.out.println("Data: " + data.size() + " common:" + comC + " id1P:" + id2fre.get(id1).size() + " id2P:" + id2fre.get(id2).size() + " "+ coh + " "+ coh2);
            }
        }
        coh = coh * (2.0/(topW.size()*(topW.size()-1)));
        // System.out.println(coh);
        return coh;
    }

    public double computeCoherence(String path)
    {

        ArrayList<ArrayList<Integer>> topTopicWordList = readTopTopicWord(path);
        double coh = 0;

        //double sis[][] = computSis(comList);
        for (int i=0; i<topTopicWordList.size(); i++)
        {
            coh += computOneTopicPMI(topTopicWordList.get(i));//sis[i][i]/getSum(sis,i);
        }
        return coh/topTopicWordList.size();
    }

    public  void evaluate(String patToWikipediaFileFile,
                                String pathToTopTopicFiles, String suffix)
            throws Exception
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(
                pathToTopTopicFiles + "/" + suffix + ".Coherence"));
        writer.write("Wikipedia file in: " + patToWikipediaFileFile + "\n\n");
        File[] files = new File(pathToTopTopicFiles).listFiles();

        //List<Double> coherence = new ArrayList<Double>();
        for (File file : files) {
            if (!file.getName().endsWith(suffix))
                continue;

            readTopTopicWordForVocabulary(file.getAbsolutePath());
        }

        ww = new int[word2IdVocabulary.size()][word2IdVocabulary.size()];

        readWikipediaWhole(patToWikipediaFileFile);

        List<Double> coherence = new ArrayList<Double>();
        for (File file : files) {
            if (!file.getName().endsWith(suffix))
                continue;
            writer.write("Results for: " + file.getAbsolutePath() + "\n");

            double value = computeCoherence(file.getAbsolutePath() );
            writer.write("\tCoherence: " + value + "\n");
            coherence.add(value);

            System.out.println(file+ "--coherence---"+value);

        }
        if (coherence.size() == 0) {
            System.out.println("Error: There is no file ending with " + suffix);
            throw new Exception();
        }

        double[] coherenceValues = new double[coherence.size()];


        for (int i = 0; i < coherence.size(); i++)
            coherenceValues[i] = coherence.get(i).doubleValue();

        writer.write("\n---\nMean Coherence: " + FuncUtils.mean(coherenceValues)
                + ", standard deviation: " + FuncUtils.stddev(coherenceValues));



        System.out.println("---\nMean Coherence: " + FuncUtils.mean(coherenceValues)
                + ", standard deviation: " + FuncUtils.stddev(coherenceValues));

        writer.close();
    }

    public static void main(String[] args)
            throws Exception
    {
        CoherenceEval ce = new CoherenceEval();
        ce.evaluate("dataset/wiki.en.text1000000", "results/", "topWords");
    }

}
