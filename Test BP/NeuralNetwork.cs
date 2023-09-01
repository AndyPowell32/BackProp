namespace Test_BP
{
    public class NeuralNetwork
    {
        private InputLayer _inputLayer;
        private HiddenLayer[] _hiddenLayers;
        private OutputLayer _outputLayer;

        public double[] Outputs => _outputLayer.Outputs;

        public NeuralNetwork(double learnRate, double momentum, params int[] neuronCounts)
        {
            _inputLayer = new InputLayer(neuronCounts[0]);
            _hiddenLayers = new HiddenLayer[neuronCounts.Length - 2];
            ILayer prevLayer = _inputLayer;
            for (int i = 0; i < _hiddenLayers.Length; i++)
            {
                _hiddenLayers[i] = new HiddenLayer(neuronCounts[i + 1], prevLayer, ActivationFunction.TanH, learnRate, momentum);
                prevLayer = _hiddenLayers[i];
            }
            _outputLayer = new OutputLayer(neuronCounts[neuronCounts.Length - 1], prevLayer, ActivationFunction.SoftMax, learnRate, momentum);
            for (int i = 0; i < _hiddenLayers.Length - 1; i++)
                _hiddenLayers[i].SetNextLayer(_hiddenLayers[i + 1]);
            _hiddenLayers[_hiddenLayers.Length - 1].SetNextLayer(_outputLayer);
        }

        public void ComputeOutputs(double[] inputs)
        {
            _inputLayer.SetInputs(inputs);
            for (int i = 0; i < _hiddenLayers.Length; i++)
                _hiddenLayers[i].FeedForward();
            _outputLayer.FeedForward();
        }

        public void UpdateWeights(double[] inputs, double[] targetOutputs)
        {
            ComputeOutputs(inputs);
            _outputLayer.ComputeSoftMaxGradients(targetOutputs);
            for (int i = _hiddenLayers.Length - 1; i >= 0; i--)
                _hiddenLayers[i].ComputeTanHGradients();
            for (int i = 0; i < _hiddenLayers.Length; i++)
                _hiddenLayers[i].UpdateWeightsAndBiases();
            _outputLayer.UpdateWeightsAndBiases();
        }
    }
}
