namespace Test_BP
{
    public class NeuralNetwork
    {
        private InputLayer _inputLayer;
        private HiddenLayer _hiddenLayer;
        private OutputLayer _outputLayer;

        public double[] Outputs => _outputLayer.Outputs;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            _inputLayer = new InputLayer(numInput);
            _hiddenLayer = new HiddenLayer(numHidden, _inputLayer, ActivationFunction.TanH);
            _outputLayer = new OutputLayer(numOutput, _hiddenLayer, ActivationFunction.SoftMax);
            _hiddenLayer.SetNextLayer(_outputLayer);
        }

        public void ComputeOutputs(double[] inputs)
        {
            _inputLayer.SetInputs(inputs);
            _hiddenLayer.FeedForward();
            _outputLayer.FeedForward();
        }

        public void UpdateWeights(double[] inputs, double[] targetOutputs)
        {
            ComputeOutputs(inputs);
            _outputLayer.ComputeSoftMaxGradients(targetOutputs);
            _hiddenLayer.ComputeTanHGradients(); //??
            _hiddenLayer.UpdateWeightsAndBiases();
            _outputLayer.UpdateWeightsAndBiases();
        }
    }
}
