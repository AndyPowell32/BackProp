namespace Test_BP
{
    class BackPropProgram
    {
        private const int numInput = 3;
        private const int numHidden = 4;
        private const int numOutput = 2;

        private static readonly double[] xValues = new double[3] { 1.0, 2.0, 3.0 }; // Inputs.
        private static readonly double[] tValues = new double[2] { 0.2500, 0.7500 }; // Target outputs.

        static void Main(string[] args)
        {

            Console.WriteLine("Back-propagation demo\n");
            Console.WriteLine($"Create {numInput}-{numHidden}-{numOutput} neural network\n");

            var nn2 = new NeuralNetwork(numInput, numHidden, numOutput);
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
    }
}