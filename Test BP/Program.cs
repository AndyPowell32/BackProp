using System.Text;

namespace Test_BP
{
    class BackPropProgram
    {
        private static readonly int[] _NeuronCounts = { 3, 4, 5, 2 };

        private static readonly double[] xValues = new double[3] { 1.0, 2.0, 3.0 }; // Inputs.
        private static readonly double[] tValues = new double[2] { 0.2500, 0.7500 }; // Target outputs.

        static void Main(string[] args)
        {

            Console.WriteLine("Back-propagation demo\n");
            var sb = new StringBuilder($"Create {_NeuronCounts[0]}");
            for (int i = 1; i < _NeuronCounts.Length - 1; i++)
                sb.Append($"-{_NeuronCounts[i]}");
            sb.Append($"-{_NeuronCounts[_NeuronCounts.Length-1]} neural network\n");

            Console.WriteLine(sb);

            var nn2 = new NeuralNetwork(_NeuronCounts);
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