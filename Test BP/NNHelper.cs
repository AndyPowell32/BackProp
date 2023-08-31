using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test_BP
{
    public static class NNHelper
    {
        public static double HyperTan(double v)
        {
            if (v < -20.0)
                return -1.0;
            else if (v > 20.0)
                return 1.0;
            else return Math.Tanh(v);
        }

        public static double[] Softmax(double[] oSums)
        {
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max)
                    max = oSums[i];
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);
            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;
            return result;
        }
    }
}
