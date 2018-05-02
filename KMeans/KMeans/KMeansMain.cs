using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using kMeans.KMeans.Helpers;

namespace kMeans.KMeans
{
    public class KMeansMain
    {
        public int[] KMeans(double[][] input, int ClustersNumber)
        {
            Classifications clas = new Classifications();
            double[][] data =clas.Unified(input);
            bool isModified = true;
            bool isTrue = true;
            int[] clusteringGroups = clas.PreProcessing(data.Length, ClustersNumber, 0);
            double[][] means = clas.Prepare(ClustersNumber, data[0].Length);
            int maxCount = data.Length * 10;
            int compter = 0;
            while (isModified == true && isTrue == true && compter < maxCount)
            {
                ++compter;
                isTrue =clas.MeansUpdate(data, clusteringGroups, means);
                isModified = clas.UpdateClustering(data, clusteringGroups, means);
            }
            return clusteringGroups;
        }
    }
}
