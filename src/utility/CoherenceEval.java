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

    String pathTopTopicFile;

    String pathWikipediaFile;

    Map<Integer, Map<Integer, Integer>> wordGraph;
    Map<Integer,Integer> wordDegree;

    public int numDocuments; // Number of documents in the corpus

    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
    // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
    // given an ID
 //   public int vocabularySize; // The number of word types in the corpus

    public int windowSize = 10;

    public int windowNum = 0;

    ArrayList<ArrayList<Integer>> topTopicWordList;

    public CoherenceEval(String inPathWikipediaFile, String inPathTopTopicFile)
            throws Exception
    {
        pathTopTopicFile = inPathTopTopicFile;
        pathWikipediaFile = inPathWikipediaFile;
        word2IdVocabulary = new HashMap<String,Integer>();
        id2WordVocabulary = new HashMap<Integer,String>();
        wordGraph = new HashMap<Integer, Map<Integer, Integer>>();
        wordDegree = new HashMap<Integer,Integer>();
        topTopicWordList = new ArrayList<ArrayList<Integer>>();
        windowNum = 0;

        readTopTopicWord();

        readWikipedia();
    }

    public void readTopTopicWord()
    {
        BufferedReader br = null;
        try {
            int indexWord = 0;
            br = new BufferedReader(new FileReader(pathTopTopicFile));
            for (String doc; (doc = br.readLine()) != null;) {

                if (doc.trim().length() == 0)
                    continue;

                String[] words = doc.trim().split(" ");
                ArrayList<Integer> oneTopic = new ArrayList<Integer>();

                for (String word : words) {
                    if (word2IdVocabulary.containsKey(word)) {
                        oneTopic.add(word2IdVocabulary.get(word));
                    }
                    else {
                        oneTopic.add(indexWord);
                        word2IdVocabulary.put(word, indexWord);
                        id2WordVocabulary.put(indexWord, word);
                        wordDegree.put(indexWord,0);
                        indexWord += 1;

                    }
                }

                topTopicWordList.add(oneTopic);
            }
            br.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
    public void readWikipedia()
    {
        BufferedReader br = null;
        try {
            int indexWord = -1;
            br = new BufferedReader(new FileReader(pathWikipediaFile));
            for (String doc; (doc = br.readLine()) != null;) {

                if (doc.trim().length() == 0)
                    continue;

                String[] words = doc.trim().split("\\s+");

                int docSize = words.length;
                for (int j = 0; j < docSize - windowSize+1; j++) {

                    for (int k = j; k<j+windowSize; k++)
                        if (word2IdVocabulary.containsKey(words[k])) {
                            int wordId = word2IdVocabulary.get(words[k]);
                            wordDegree.put(wordId,wordDegree.get(wordId)+1);

                        }

                    for (int k = j; k<j+windowSize-1; k++) {
                        if (!word2IdVocabulary.containsKey(words[k]))
                            continue;

                        for (int m = k + 1; m<j+windowSize; m++)
                            if (word2IdVocabulary.containsKey(words[m]))
                                addEdge(word2IdVocabulary.get(words[k]),word2IdVocabulary.get(words[m]));
                    }
                    windowNum++;
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

        if(p1==p2)
            return;

        if (this.containsEdge(p1, p2)) {
            wordGraph.get(p1).put(p2, wordGraph.get(p1).get(p2)+1);
            wordGraph.get(p2).put(p1, wordGraph.get(p2).get(p1)+1);
            //wordDegree.put(p1,wordDegree.get(p1)+1);
            //wordDegree.put(p2,wordDegree.get(p2)+1);
            return;
        }
        if (!wordGraph.containsKey(p1)) {
            wordGraph.put(p1, new HashMap<Integer, Integer>());
            //wordDegree.put(p1,0);
        }
        if (!wordGraph.containsKey(p2)) {
            wordGraph.put(p2, new HashMap<Integer, Integer>());
            //wordDegree.put(p2,0);
        }
        wordGraph.get(p1).put(p2, 1);
        wordGraph.get(p2).put(p1, 1);
        //wordDegree.put(p1,wordDegree.get(p1)+1);
       // wordDegree.put(p2,wordDegree.get(p2)+1);
    }

    public void show() {
        Set<Map.Entry<Integer, Map<Integer, Integer>>> set = wordGraph.entrySet();
        for (Map.Entry<Integer, Map<Integer, Integer>> e : set) {
            Set<Map.Entry<Integer, Integer>> temp = e.getValue().entrySet();
            if (temp.size() > 0) {
                System.out.print(id2WordVocabulary.get(e.getKey()) + " -> ");
                for (Map.Entry<Integer, Integer> e1 : temp) {
                    System.out.print(id2WordVocabulary.get(e1.getKey()) + "(" + e1.getValue() + ") ");
                }
                System.out.println();
            }
        }

    }

    public double computOneTopicPMI(ArrayList<Integer> topW)
    {
        double coh = 0;
        // double coh2 = 0;


        for(int i=0; i<topW.size(); i++)
        {
            if(!wordGraph.containsKey(topW.get(i)))
                continue;

            for(int j=i+1; j<topW.size(); j++)
            {
                //coh = 0;
                //coh2 = 0;
                int id1 = topW.get(i);
                int id2 = topW.get(j);

                int comC = 0;

                if(wordGraph.get(id1).containsKey(id2))
                    comC = wordGraph.get(id1).get(id2);

                if(comC==0)
                    continue;

                double comP = (double)comC;
                double id1P = (double) wordDegree.get(id1);
                // System.out.println(id2);
                double id2P = (double) wordDegree.get(id2);

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

    public double computeCoherence()
    {
        double coh = 0;

        //double sis[][] = computSis(comList);
        for (int i=0; i<topTopicWordList.size(); i++)
        {
            coh += computOneTopicPMI(topTopicWordList.get(i));//sis[i][i]/getSum(sis,i);
        }
        return coh/topTopicWordList.size();
    }

    public static void evaluate(String patToWikipediaFileFile,
                                String pathToTopTopicFiles, String suffix)
            throws Exception
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(
                pathToTopTopicFiles + "/" + suffix + ".Coherence"));
        writer.write("Wikipedia file in: " + patToWikipediaFileFile + "\n\n");
        File[] files = new File(pathToTopTopicFiles).listFiles();

        List<Double> coherence = new ArrayList<Double>();
        for (File file : files) {
            if (!file.getName().endsWith(suffix))
                continue;
            writer.write("Results for: " + file.getAbsolutePath() + "\n");
            CoherenceEval dce = new CoherenceEval(patToWikipediaFileFile,
                    file.getAbsolutePath());
            double value = dce.computeCoherence();
            writer.write("\tCoherence: " + value + "\n");
            coherence.add(value);

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
        CoherenceEval.evaluate("dataset/wiki.en.text1000000", "results/", "topWords");
    }

}
