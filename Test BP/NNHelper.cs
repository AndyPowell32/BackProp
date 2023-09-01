using System.Text;

namespace Test_BP
{
    public static class NNHelper
    {
        //private static readonly Random _random = new Random();
         
        public static double HyperTan(double v)
        {
            if (v < -20.0)
                return -1.0;
            else if (v > 20.0)
                return 1.0;
            else return Math.Tanh(v);
        }

        public static void Softmax(double[] sums, double[] outputs)
        {
            double max = sums[0];
            for (int i = 1; i < sums.Length; i++)
                if (sums[i] > max)
                    max = sums[i];
            double scale = 0;
            for (int i = 0; i < sums.Length; i++)
                scale += Math.Exp(sums[i] - max);
            //double[] result = new double[sums.Length]; //?? new
            for (int i = 0; i < sums.Length; ++i)
                outputs[i] = Math.Exp(sums[i] - max) / scale;
            //return result;
        }

        public static string VectorString(double[] vector)
        {
            var sb = new StringBuilder("[");
            for (int i = 0; i < vector.Length - 1; i++)
                sb.Append($"{vector[i]:F0},");
            sb.Append($"{vector[vector.Length - 1]:F0}");
            sb.Append("]");
            return sb.ToString(); ;
        }

        //public static int[] RandomIndices(int n)
        //{
        //    var randomIndices = new int[n];
        //    for (int i = 0; i < n; i++)
        //        randomIndices[i] = i;

        //    for (int j = 0; j < randomIndices.Length; j++)
        //    {
        //        int k = _random.Next(randomIndices.Length);
        //        int temp = randomIndices[j];
        //        randomIndices[j] = randomIndices[k];
        //        randomIndices[k] = temp;
        //    }
        //    return randomIndices;
        //}
    }
}
