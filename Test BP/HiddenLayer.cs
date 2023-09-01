namespace Test_BP
{
    public class HiddenLayer: INextLayer
    {
        private const double _defaultWeight = 0.5;
        private const double _defaultBias = 0.5;

        private ActivationFunction _activationFunction;
        private double _learnRate;
        private double _momentum;
        private ILayer _prevLayer;
        private INextLayer? _nextLayer;
        private double[] _biases;
        private double[][] _prevWeightsDelta;
        private double[] _prevBiasesDelta;
        private double[] _sums;

        public int Count { get; }
        public double[] Outputs { get; }
        public double[][] Weights { get; }
        public double[] Grads { get; }

        public HiddenLayer(int count, ILayer prevLayer, ActivationFunction activationFunction, double learnRate, double momentum )
        {
            Count = count;
            _prevLayer = prevLayer;
            _activationFunction = activationFunction;
            _learnRate = learnRate;
            _momentum = momentum;
            Outputs = new double[count];
            Weights = new double[count][];
            _biases = new double[count];
            Grads = new double[count];
            _prevBiasesDelta = new double[count];
            _prevWeightsDelta = new double[count][];
            _sums = new double[Count];
            for (int i = 0; i < Count; i++)
            {
                Weights[i] = new double[_prevLayer.Count];
                Array.Fill(Weights[i], _defaultWeight);
                _prevWeightsDelta[i] = new double[_prevLayer.Count];
            }
            Array.Fill(_biases, _defaultBias);
        }

        public void SetNextLayer(INextLayer nextLayer) 
        {
            _nextLayer = nextLayer;
        }

        public void FeedForward()
        {
            Array.Fill(_sums, 0);
            for (int j = 0; j < Count; ++j)
                for (int i = 0; i < _prevLayer.Count; ++i)
                    _sums[j] += _prevLayer.Outputs[i] * Weights[j][i];

            for (int i = 0; i < Count; ++i)
                _sums[i] += _biases[i];

            if (_activationFunction == ActivationFunction.TanH)
                for (int i = 0; i < Count; ++i)
                    Outputs[i] = NNHelper.HyperTan(_sums[i]);
            else
            {
                double[] softOut = NNHelper.Softmax(_sums);
                for (int i = 0; i < Count; ++i)
                    Outputs[i] = softOut[i];
            }
        }

        //public void ComputeSoftMaxGradients(double[] targets)
        //{
        //    for (int i = 0; i < Count; ++i)
        //    {
        //        double derivative = (1 - Outputs[i]) * Outputs[i]; // Derivative of softmax is y(1-y).
        //        Grads[i] = derivative * (targets[i] - Outputs[i]); // oGrad = (1 - O)(O) * (T-O)
        //    }
        //}

        public void ComputeTanHGradients()
        {
            for (int i = 0; i < Count; ++i)
            {
                double derivative = (1 - Outputs[i]) * (1 + Outputs[i]); // f' of tanh is (1-y)(1+y).
                double sum = 0.0;
                for (int j = 0; j < _nextLayer.Count; ++j)
                    sum += _nextLayer.Grads[j] * _nextLayer.Weights[j][i];
                Grads[i] = derivative * sum; // hGrad = (1-O)(1+O) * Sum(oGrads*oWts)
            }
        }

        public void UpdateWeightsAndBiases()
        {
            for (int i = 0; i < Count; ++i)
                for (int j = 0; j < _prevLayer.Count; ++j)
                {
                    double delta = _learnRate * Grads[i] * _prevLayer.Outputs[j];
                    Weights[i][j] += delta + _momentum * _prevWeightsDelta[i][j];
                    _prevWeightsDelta[i][j] = delta;
                }

            for (int i = 0; i < Count; ++i)
            {
                double biasDelta = _learnRate * Grads[i] * 1.0;
                _biases[i] += biasDelta + _momentum * _prevBiasesDelta[i];
                _prevBiasesDelta[i] = biasDelta;
            }
        }
    }
}
