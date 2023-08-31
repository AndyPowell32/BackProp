﻿namespace Test_BP
{
    public class OutputLayer: INextLayer
    {
        private const double _learnRate = 0.05;
        private const double _momentum = 0.01;
        private const double _defaultWeight = 0.5;
        private const double _defaultBias = 0.5;

        private ActivationFunction _activationFunction;
        private ILayer _prevLayer;
        private double[] _biases;
        private double[][] _prevWeightsDelta;
        private double[] _prevBiasesDelta;

        public int Count { get; }
        public double[] Outputs { get; }
        public double[][] Weights { get; }
        public double[] Grads { get; }

        public OutputLayer(int count, ILayer prevLayer, ActivationFunction activationFunction)
        {
            Count = count;
            _prevLayer = prevLayer;
            _activationFunction = activationFunction;
            Outputs = new double[count];
            Weights = new double[count][];
            _biases = new double[count];
            Grads = new double[count];
            _prevBiasesDelta = new double[count];
            _prevWeightsDelta = new double[count][];
            for (int i = 0; i < Count; i++)
            {
                Weights[i] = new double[_prevLayer.Count];
                Array.Fill(Weights[i], _defaultWeight);
                _prevWeightsDelta[i] = new double[_prevLayer.Count];
            }
            Array.Fill(_biases, _defaultBias);
            _activationFunction = activationFunction;
        }

        public void FeedForward()
        {
            double[] sums = new double[Count];
            for (int j = 0; j < Count; ++j)
                for (int i = 0; i < _prevLayer.Count; ++i)
                    sums[j] += _prevLayer.Outputs[i] * Weights[j][i];

            for (int i = 0; i < Count; ++i)
                sums[i] += _biases[i];

            if (_activationFunction == ActivationFunction.TanH)
                for (int i = 0; i < Count; ++i)
                    Outputs[i] = NNHelper.HyperTan(sums[i]);
            else
            {
                double[] softOut = NNHelper.Softmax(sums);
                for (int i = 0; i < Count; ++i)
                    Outputs[i] = softOut[i];
            }
        }

        public void ComputeSoftMaxGradients(double[] targets)
        {
            for (int i = 0; i < Count; ++i)
            {
                double derivative = (1 - Outputs[i]) * Outputs[i]; // Derivative of softmax is y(1-y).
                Grads[i] = derivative * (targets[i] - Outputs[i]); // oGrad = (1 - O)(O) * (T-O)
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
