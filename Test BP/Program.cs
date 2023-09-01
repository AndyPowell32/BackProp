using System.Text;

namespace Test_BP
{
    class BackPropProgram
    {
        private const int _Epochs = 2000;
        private const double _learnRate = 0.05;
        private const double _momentum = 0.01;

        private static readonly int[] _HiddenLayerNeuronCounts = {3, 3};

        private static readonly Tuple<double[], double[]>[] _TrainingSet0 =
        {
            new Tuple<double[], double[]>(new double[]{ 1.0, 2.0, 3.0 }, new double[]{ 0.2500, 0.7500 })
        };

        private static readonly Tuple<double[], double[]>[] _TrainingSet1 =
{
            new Tuple<double[], double[]>(new double[]{1}, new double[]{ 1, 0 }),
            new Tuple<double[], double[]>(new double[]{2}, new double[]{ 1, 0 }),
            new Tuple<double[], double[]>(new double[]{3}, new double[]{ 1, 0 }),
            new Tuple<double[], double[]>(new double[]{4}, new double[]{ 1, 0 }),
            new Tuple<double[], double[]>(new double[]{5}, new double[]{ 1, 0 }),
            new Tuple<double[], double[]>(new double[]{6}, new double[]{ 0, 1 }),
            new Tuple<double[], double[]>(new double[]{7}, new double[]{ 0, 1 }),
            new Tuple<double[], double[]>(new double[]{8}, new double[]{ 0, 1 }),
            new Tuple<double[], double[]>(new double[]{9}, new double[]{ 0, 1 }),
            new Tuple<double[], double[]>(new double[]{10}, new double[]{ 0, 1 })
        };

        private static readonly Random _random = new Random();

        static void Main(string[] args)
        {
            var trainingSet = _TrainingSet1;
            var neuronCounts = new int[_HiddenLayerNeuronCounts.Length + 2];
            neuronCounts[0] = trainingSet[0].Item1.Count();
            for (int i = 0; i < _HiddenLayerNeuronCounts.Length; i++)
                neuronCounts[i + 1] = _HiddenLayerNeuronCounts[i];
            neuronCounts[neuronCounts.Length - 1] = trainingSet[0].Item2.Count();

            Console.WriteLine("Back-propagation\n");

            var sb = new StringBuilder($"{neuronCounts[0]}");
            for (int i = 1; i < neuronCounts.Length - 1; i++)
                sb.Append($"-{neuronCounts[i]}");
            sb.Append($"-{neuronCounts[neuronCounts.Length - 1]} neural network\n");
            Console.WriteLine(sb);
            Console.WriteLine($"Training Epochs: {_Epochs}");
            Console.WriteLine();


            var nn2 = new NeuralNetwork(_learnRate, _momentum, neuronCounts);
            var randomIndices = new int[trainingSet.Length];
            for (int j = 0; j < trainingSet.Length; j++)
                randomIndices[j] = j;

            for (int i = 0; i <= _Epochs; i++)
            {
                for (int j = 0; j < randomIndices.Length; j++)
                {
                    int k = _random.Next(randomIndices.Length);
                    int temp = randomIndices[j];
                    randomIndices[j] = randomIndices[k];
                    randomIndices[k] = temp;
                }
                for (int j = 0; j < trainingSet.Length; j++)
                {
                    int index = randomIndices[j];
                    nn2.UpdateWeights(trainingSet[index].Item1, trainingSet[index].Item2);
                }
            }
            for (int j = 0; j < trainingSet.Length; j++)
            {
                nn2.ComputeOutputs(trainingSet[j].Item1);
                Console.WriteLine($"{(j + 1).ToString().PadLeft(5)} Input: {NNHelper.VectorString(trainingSet[j].Item1)} Expected: {NNHelper.VectorString(trainingSet[j].Item2)} Actual: {NNHelper.VectorString(nn2.Outputs)}");
            }

            Console.ReadLine();
        }
    }
}