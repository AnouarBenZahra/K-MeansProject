using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace kMeans.KMeans.Helpers
{
    public class Classifications
    {
        public double[][] Unified(double[][] input)
        {
            double[][] result = new double[input.Length][];
            for (int i = 0; i < input.Length; ++i)
            {
                result[i] = new double[input[i].Length];
                Array.Copy(input[i], result[i], input[i].Length);
            }

            for (int j = 0; j < result[0].Length; ++j)
            {
                double colSum = 0.0;
                for (int i = 0; i < result.Length; ++i)
                    colSum += result[i][j];
                double mean = colSum / result.Length;
                double sum = 0.0;
                for (int i = 0; i < result.Length; ++i)
                    sum += (result[i][j] - mean) * (result[i][j] - mean);
                double sd = sum / result.Length;
                for (int i = 0; i < result.Length; ++i)
                    result[i][j] = (result[i][j] - mean) / sd;
            }
            return result;
        }

        public int[] PreProcessing(int numTuples, int ClustersNumber, int IntRandom)
        {
            int[] meansGroup = new int[numTuples];
            Random alea = new Random(IntRandom);
            for (int counter = 0; counter < ClustersNumber; ++counter)
                meansGroup[counter] = counter;
            for (int counter = ClustersNumber; counter < meansGroup.Length; ++counter)
                meansGroup[counter] = alea.Next(0, ClustersNumber);
            return meansGroup;
        }

        public double[][] Prepare(int ClustersNumber, int columnsNumbers)
        {

            double[][] toReturn = new double[ClustersNumber][];
            for (int mK = 0; mK < ClustersNumber; ++mK)
                toReturn[mK] = new double[columnsNumbers];
            return toReturn;
        }

        public bool MeansUpdate(double[][] data, int[] clustering, double[][] process)
        {
            int ClustersNumber = process.Length;
            int[] clusterIterator = new int[ClustersNumber];
            for (int counter = 0; counter < data.Length; ++counter)
            {
                int cluster = clustering[counter];
                ++clusterIterator[cluster];
            }

            for (int i = 0; i < ClustersNumber; ++i)
                if (clusterIterator[i] == 0)
                    return false;
            for (int i= 0; i < process.Length; ++i)
                for (int j = 0; j < process[i].Length; ++j)
                    process[i][j] = 0.0;

            for (int i = 0; i < data.Length; ++i)
            {
                int cluster = clustering[i];
                for (int j = 0; j < data[i].Length; ++j)
                    process[cluster][j] += data[i][j];
            }

            for (int i = 0; i < process.Length; ++i)
                for (int j = 0; j < process[i].Length; ++j)
                    process[i][j] /= clusterIterator[i];
            return true;
        }

        public bool UpdateClustering(double[][] data, int[] clustering, double[][] process)
        {

            int ClustersNumber = process.Length;
            bool isChanged = false;
            int[] generatedCluster = new int[clustering.Length];
            Array.Copy(clustering, generatedCluster, clustering.Length);

            double[] ecart = new double[ClustersNumber];

            for (int counter = 0; counter < data.Length; ++counter)
            {
                for (int k = 0; k < ClustersNumber; ++k)
                    ecart[k] = Ecart(data[counter], process[k]);

                int generatedClusterID = LowIndex(ecart);
                if (generatedClusterID != generatedCluster[counter])
                {
                    isChanged = true;
                    generatedCluster[counter] = generatedClusterID;
                }
            }

            if (!isChanged)
                return false;
            int[] clusterCounts = new int[ClustersNumber];
            for (int counter = 0; counter < data.Length; ++counter)
            {
                int cluster = generatedCluster[counter];
                ++clusterCounts[cluster];
            }

            for (int k = 0; k < ClustersNumber; ++k)
                if (clusterCounts[k] == 0)
                    return false;

            Array.Copy(generatedCluster, clustering, generatedCluster.Length);
            return true;
        }

        private static double Ecart(double[] data, double[] mean)
        {
            double spacialDiff = 0.0;
            for (int counter = 0; counter < data.Length; ++counter)
                spacialDiff += Math.Pow((data[counter] - mean[counter]), 2);
            return Math.Sqrt(spacialDiff);
        }

        private static int LowIndex(double[] ecart)
        {

            int minIndex = 0;
            double minDistance = ecart[0];
            for (int counter = 0; counter < ecart.Length; ++counter)
            {
                if (ecart[counter] < minDistance)
                {
                    minDistance = ecart[counter];
                    minIndex = counter;
                }
            }
            return minIndex;
        }

    }
}
