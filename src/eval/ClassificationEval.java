package eval;

import de.bwaldvogel.liblinear.Train;
import utility.FuncUtils;

import java.io.*;
import java.util.*;

/**
 * Created by jipengqiang on 18/7/31.
 */
public class ClassificationEval {


    String pathDocTopicProsFile;

    String pathGoldenLabelsFile;


    ArrayList<String> goldenLabel;
    HashMap<String, Integer> goldenClusers;
    ArrayList<ArrayList<String>> docTopicProb;


    int numDocs;

    public ClassificationEval(String inPathGoldenLabelsFile,
                          String inPathDocTopicProsFile)
            throws Exception
    {
        pathDocTopicProsFile = inPathDocTopicProsFile;
        pathGoldenLabelsFile = inPathGoldenLabelsFile;

        goldenLabel = new ArrayList<String>();
        goldenClusers = new HashMap<String, Integer>();
        docTopicProb = new ArrayList<ArrayList<String>>();

        readGoldenLabelsFile();
        readDocTopicProsFile();
        numDocs = 0;
    }

    public void readGoldenLabelsFile()
            throws Exception
    {
        System.out
                .println("Reading golden labels file " + pathGoldenLabelsFile);

        BufferedReader br = null;
        try {
            int ind = -1;
            br = new BufferedReader(new FileReader(pathGoldenLabelsFile));
            for (String label; (label = br.readLine()) != null;) {
                label = label.trim();
                if(goldenClusers.containsKey(label))
                    ind = goldenClusers.get(label);
                else {
                    ind++;
                    goldenClusers.put(label,ind);
                }
                goldenLabel.add(label);
                numDocs++;
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void readDocTopicProsFile()
            throws Exception
    {
        System.out.println("Reading document-to-topic distribution file "
                + pathDocTopicProsFile);

        HashMap<Integer, String> docLabelOutput = new HashMap<Integer, String>();

        int docIndex = 0;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathDocTopicProsFile));

            for (String docTopicProbs; (docTopicProbs = br.readLine()) != null;) {
                String[] pros = docTopicProbs.trim().split("\\s+");
                ArrayList<String> oneText = new ArrayList<String>();
                for (int topicIndex = 0; topicIndex < pros.length; topicIndex++) {
                    oneText.add(pros[topicIndex]);
                }
                docTopicProb.add(oneText);

                docIndex++;
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        if (numDocs != docIndex) {
            System.out
                    .println("Error: the number of documents is different to the number of labels!");
            throw new Exception();
        }

    }

    public void writeNewDocTopicPros()
            throws IOException
    {
        String folderPath = "results/";
        File dir = new File(folderPath);
        if (!dir.exists())
            dir.mkdir();

        BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath
                 + "theta.temp"));
        for (int i = 0; i < docTopicProb.size(); i++) {
           // System.out.println(goldenLabel.get(i));
            writer.write(goldenClusers.get(goldenLabel.get(i)) + " ");
            for (int j = 0; j < docTopicProb.get(i).size(); j++) {

                int index = j+1;
                writer.write(index + ":" + docTopicProb.get(i).get(j) + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public static void evaluate(String pathGoldenLabelsFile,
                                String pathToFolderOfDocTopicProsFiles, String suffix)
            throws Exception
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(
                pathToFolderOfDocTopicProsFiles + "/" + suffix + ".accuracy"));
        writer.write("Golden-labels in: " + pathGoldenLabelsFile + "\n\n");
        File[] files = new File(pathToFolderOfDocTopicProsFiles).listFiles();

        List<Double> accuracy = new ArrayList<Double>();
        for (File file : files) {
            if (!file.getName().endsWith(suffix))
                continue;
            writer.write("Results for: " + file.getAbsolutePath() + "\n");
            ClassificationEval dce = new ClassificationEval(pathGoldenLabelsFile,
                    file.getAbsolutePath());
            dce.writeNewDocTopicPros();

            String []param = new String[5];
            param[0] = "-v";
            param[1] = "5";
            param[2] = "-s";
            param[3] = "2";
            param[4] = "results/theta.temp";



            double value = new Train().run(param); // dce.computeCoherence();
            writer.write("\tAccuracy: " + value + "\n");
            accuracy.add(value);

        }
        if (accuracy.size() == 0) {
            System.out.println("Error: There is no file ending with " + suffix);
            throw new Exception();
        }

        double[] coherenceValues = new double[accuracy.size()];


        for (int i = 0; i < accuracy.size(); i++)
            coherenceValues[i] = accuracy.get(i).doubleValue();

        writer.write("\n---\nMean accuracy: " + FuncUtils.mean(coherenceValues)
                + ", standard deviation: " + FuncUtils.stddev(coherenceValues));



        System.out.println("---\nMean accuracy: " + FuncUtils.mean(coherenceValues)
                + ", standard deviation: " + FuncUtils.stddev(coherenceValues));

        writer.close();
    }

    public static void main(String[] args)
            throws Exception
    {
        ClassificationEval.evaluate("dataset/SearchSnippets_LABEL.txt", "results", "theta");
    }
}
