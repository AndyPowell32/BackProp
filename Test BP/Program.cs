namespace Test_BP
{
    class BackPropProgram
    {
        private const int numInput = 3;
        private const int numHidden = 4;
        private const int numOutput = 2;
        private const double startWeight = 0.5;
        private const double learnRate = 0.05;
        private const double momentum = 0.01;
        private const int maxEpochs = 1000;

        private static readonly double[] xValues = new double[3] { 1.0, 2.0, 3.0 }; // Inputs.
        private static readonly double[] tValues = new double[2] { 0.2500, 0.7500 }; // Target outputs.

        static void Main(string[] args)
        {

            Console.WriteLine("Begin back-propagation demo\n");
            Console.WriteLine($"Creating a {numInput}-{numHidden}-{numOutput} neural network\n");

            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

            double[] weights = new double[26];
            Array.Fill(weights, startWeight);

            Console.WriteLine("Setting dummy initial weights to:");
            ShowVector(weights, 8, 2, true);
            //nn.SetWeights(weights);

            Console.WriteLine("\nSetting fixed inputs = ");
            ShowVector(xValues, 3, 1, true);
            Console.WriteLine("Setting fixed target outputs = ");
            ShowVector(tValues, 2, 4, true);

            Console.WriteLine("\nSetting learning rate = " + learnRate.ToString("F2"));
            Console.WriteLine("Setting momentum = " + momentum.ToString("F2"));
            Console.WriteLine("Setting max epochs = " + maxEpochs + "\n");

            //nn.FindWeights(tValues, xValues, learnRate, momentum, maxEpochs);

            int epoch = 0;
            while (epoch <= maxEpochs)
            {
                nn.UpdateWeights(xValues, tValues, learnRate, momentum);
                if (epoch % 100 == 0)
                {
                    Console.Write("epoch = " + epoch.ToString().PadLeft(5) + "   curr outputs = ");
                    BackPropProgram.ShowVector(nn.outputs, 2, 4, true);
                }
                ++epoch;
            }


            double[] bestWeights = nn.GetWeights();
            Console.WriteLine("\nBest weights found:");
            ShowVector(bestWeights, 8, 4, true);
            Console.WriteLine();

            Console.WriteLine("_________________________________________________");

            var nn2 = new NeuralNetwork2(numInput, numHidden, numOutput);
            for (int i = 0; i <= 1000; i++)
            {
                nn2.UpdateWeights(xValues, tValues);
                if (i % 100 == 0)
                Console.WriteLine($"{i.ToString().PadLeft(5)} {nn2.Outputs[0]:F4} {nn2.Outputs[1]:F4}");

            }
            Console.WriteLine();
            nn2.ComputeOutputs(xValues);
            Console.WriteLine($"{nn2.Outputs[0]:F4} {nn2.Outputs[1]:F4}");

            Console.ReadLine();
        }

        public static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % valsPerRow == 0)
                    Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine == true)
                Console.WriteLine("");
        }

        public static void ShowMatrix(double[][] matrix, int decimals)
        {
            int cols = matrix[0].Length;
            for (int i = 0; i < matrix.Length; ++i) // Each row.
                ShowVector(matrix[i], cols, decimals, true);
        }
    }
}