package utility;

import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class FuncUtils
{
    public static <K, V extends Comparable<? super V>> Map<K, V> sortByValueDescending(Map<K, V> map)
    {
        List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<K, V>>()
        {
            @Override
            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2)
            {
                int compare = (o1.getValue()).compareTo(o2.getValue());
                return -compare;
            }
        });

        Map<K, V> result = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }

    public static <K, V extends Comparable<? super V>> Map<K, V> sortByValueAscending(Map<K, V> map)
    {
        List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<K, V>>()
        {
            @Override
            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2)
            {
                int compare = (o1.getValue()).compareTo(o2.getValue());
                return compare;
            }
        });

        Map<K, V> result = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }

    /**
     * Sample a value from a double array
     * 
     * @param probs
     * @return
     */
    public static int nextDiscrete(double[] probs)
    {
        double sum = 0.0;
        for (int i = 0; i < probs.length; i++)
            sum += probs[i];

        double r = MTRandom.nextDouble() * sum;

        sum = 0.0;
        for (int i = 0; i < probs.length; i++) {
            sum += probs[i];
            if (sum > r)
                return i;
        }
        return probs.length - 1;
    }

    public static double mean(double[] m)
    {
        double sum = 0;
        for (int i = 0; i < m.length; i++)
            sum += m[i];
        return sum / m.length;
    }

    public static double stddev(double[] m)
    {
        double mean = mean(m);
        double s = 0;
        for (int i = 0; i < m.length; i++)
            s += (m[i] - mean) * (m[i] - mean);
        return Math.sqrt(s / m.length);
    }

    public static double sum(double[] vector){
        double sum = 0.0;
        for ( int i = 0; i < vector.length; i++ )
            sum += vector[i];
        return sum;
    }

    public static double[] L1NormWithReusable(double[] vector){
        double sum = 0.0;

        for (double value : vector){
            sum += value;
        }

        if ( Double.compare(sum, 0.0) > 0 ){
            for ( int i = 0; i < vector.length; i++ ){
                vector[i] = vector[i] / sum;
            }
        } else {
            return null;
        }

        return vector;
    }

    public static float ComputeCosineSimilarity(float[] vector1, float[] vector2)
    {
        if (vector1.length != vector2.length)
            try {
                throw new Exception("DIFER LENGTH");
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }


        float denom = (VectorLength(vector1) * VectorLength(vector2));
        if (denom == 0f)
            return 0f;
        else
            return (InnerProduct(vector1, vector2) / denom);

    }

    public static float computDist(float[] vector1, float[] vector2)
    {

        // return (float)Math.exp((1-ComputeCosineSimilarity(vector1,vector2))/0.3f);
        return 1-ComputeCosineSimilarity(vector1,vector2);

    }


    public static float InnerProduct(float[] vector1, float[] vector2)
    {

        if (vector1.length != vector2.length)
            try {
                throw new Exception("DIFFER LENGTH ARE NOT ALLOWED");
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }


        float result = 0f;
        for (int i = 0; i < vector1.length; i++)
            result += vector1[i] * vector2[i];

        return result;
    }

    public static float VectorLength(float[] vector)
    {
        float sum = 0.0f;
        for (int i = 0; i < vector.length; i++)
            sum = sum + (vector[i] * vector[i]);

        return (float)Math.sqrt(sum);
    }
}
