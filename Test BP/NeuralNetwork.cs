namespace Test_BP
{
    public class NeuralNetwork
    {
        private const double _initialWeight = 0.5;
        private const double _initialBias = 0.5;
        private const double _initialBiasDelta = 0.011;
        private const double _initialWeightDelta = 0.011;

        private int _numInput;
        private int _numHidden;
        private int _numOutput;

        private double[] inputs;
        private double[] hOutputs;
        public double[] outputs;

        private double[][] ihWeights;
        private double[][] hoWeights;

        private double[] hBiases;
        private double[] oBiases;

        private double[] oGrads;
        private double[] hGrads;

        private double[][] ihPrevWeightsDelta;
        private double[] hPrevBiasesDelta;
        private double[][] hoPrevWeightsDelta;
        private double[] oPrevBiasesDelta;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            _numInput = numInput;
            _numHidden = numHidden;
            _numOutput = numOutput;

            inputs = new double[numInput];
            hOutputs = new double[numHidden];
            outputs = new double[numOutput];

            ihWeights = MakeMatrix(numInput, numHidden);
            hoWeights = MakeMatrix(numHidden, numOutput);

            hBiases = new double[numHidden];
            oBiases = new double[numOutput];

            oGrads = new double[numOutput];
            hGrads = new double[numHidden];

            ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
            hPrevBiasesDelta = new double[numHidden];
            hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
            oPrevBiasesDelta = new double[numOutput];

            InitMatrix(ihWeights, _initialWeight);
            InitMatrix(hoWeights, _initialWeight);
            Array.Fill(hBiases, _initialBias);
            Array.Fill(oBiases, _initialBias);

            InitMatrix(ihPrevWeightsDelta, _initialWeightDelta);
            Array.Fill(hPrevBiasesDelta, _initialBiasDelta);
            InitMatrix(hoPrevWeightsDelta, _initialWeightDelta);
            Array.Fill(oPrevBiasesDelta, _initialBiasDelta);
        }

        private static double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }

        private static void InitMatrix(double[][] matrix, double value)
        {
            for (int i = 0; i < matrix.Length; ++i)
                Array.Fill(matrix[i], value);
        }

        public double[] GetWeights()
        {
            int numWeights = (_numInput * _numHidden) + _numHidden +
              (_numHidden * _numOutput) + _numOutput;

            double[] result = new double[numWeights];
            int k = 0;

            for (int i = 0; i < _numInput; ++i)
                for (int j = 0; j < _numHidden; ++j)
                    result[k++] = ihWeights[i][j];

            for (int i = 0; i < _numHidden; ++i)
                result[k++] = hBiases[i];

            for (int i = 0; i < _numHidden; ++i)
                for (int j = 0; j < _numOutput; ++j)
                    result[k++] = hoWeights[i][j];

            for (int i = 0; i < _numOutput; ++i)
                result[k++] = oBiases[i];

            return result;
        }

        private static void FeedForwardHyperTan(double[] prevOutputs, double[][] weights, double[] biases, double[] outputs)
        {
            int count = outputs.Length;
            int prevCount = prevOutputs.Length;
            double[] sums = new double[count];
            for (int j = 0; j < count; ++j)
                for (int i = 0; i < prevCount; ++i)
                    sums[j] += prevOutputs[i] * weights[i][j];

            for (int i = 0; i < count; ++i)
                sums[i] += biases[i];

            for (int i = 0; i < count; ++i)
                outputs[i] = HyperTan(sums[i]);
        }

        private static void FeedForwardSoftMax(double[] prevOutputs, double[][] weights, double[] biases, double[] outputs)
        {
            int count = outputs.Length;
            int prevCount = prevOutputs.Length;
            double[] sums = new double[count];
            for (int j = 0; j < count; ++j)
                for (int i = 0; i < prevCount; ++i)
                    sums[j] += prevOutputs[i] * weights[i][j];

            for (int i = 0; i < count; ++i)
                sums[i] += biases[i];

            double[] softOut = Softmax(sums);

            for (int i = 0; i < outputs.Length; ++i)
                outputs[i] = softOut[i];
        }

        private void ComputeOutputs(double[] xValues)
        {
            FeedForwardHyperTan(xValues, ihWeights, hBiases, hOutputs);
            FeedForwardSoftMax(hOutputs, hoWeights, oBiases, outputs);
        }

        private static double HyperTan(double v)
        {
            if (v < -20.0)
                return -1.0;
            else if (v > 20.0)
                return 1.0;
            else return Math.Tanh(v);
        }

        private static double[] Softmax(double[] oSums)
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
            return result; // xi sum to 1.0.
        }

        private static void ComputeSoftMaxGradients(double[] outputs, double[] targets, double[] grads)
        {
            for (int i = 0; i < grads.Length; ++i)
            {
                double derivative = (1 - outputs[i]) * outputs[i]; // Derivative of softmax is y(1-y).
                grads[i] = derivative * (targets[i] - outputs[i]); // oGrad = (1 - O)(O) * (T-O)
            }
        }

        private static void ComputeTanHGradients(double[] outputs, double[] downstreamGrads, double[][] downstreamWeights, double[] grads)
        {
            for (int i = 0; i < grads.Length; ++i)
            {
                double derivative = (1 - outputs[i]) * (1 + outputs[i]); // f' of tanh is (1-y)(1+y).
                double sum = 0.0;
                for (int j = 0; j < downstreamGrads.Length; ++j) // Each hidden delta is the sum of numOutput terms.
                    sum += downstreamGrads[j] * downstreamWeights[i][j]; // Each downstream gradient * outgoing weight.
                grads[i] = derivative * sum; // hGrad = (1-O)(1+O) * Sum(oGrads*oWts)
            }
        }

        private static void UpdateWeightsAndBiases(double[][] weights, double[][] prevWeightsDelta, double[] biases, double[] prevBiasesDelta, double[] grads, double[] inputs, double learnRate, double momentum)
        {
            for (int i = 0; i < weights.Length; ++i)
            {
                for (int j = 0; j < weights[i].Length; ++j)
                {
                    double delta = learnRate * grads[j] * inputs[i];
                    weights[i][j] += delta + momentum * prevWeightsDelta[i][j];
                    prevWeightsDelta[i][j] = delta;
                }
            }

            for (int i = 0; i < biases.Length; ++i)
            {
                double biasDelta = learnRate * grads[i] * 1.0;
                biases[i] += biasDelta + momentum * prevBiasesDelta[i];
                prevBiasesDelta[i] = biasDelta;
            }
        }

        public void UpdateWeights(double[] xValues, double[] tValues, double learnRate, double momentum)
        {
            ComputeOutputs(xValues);

            // 1. Compute output gradients. Assumes softmax.
            //for (int i = 0; i < oGrads.Length; ++i)
            //{
            //    double derivative = (1 - outputs[i]) * outputs[i]; // Derivative of softmax is y(1-y).
            //    oGrads[i] = derivative * (tValues[i] - outputs[i]); // oGrad = (1 - O)(O) * (T-O)
            //}
            ComputeSoftMaxGradients(outputs, tValues, oGrads);

            // 2. Compute hidden gradients. Assumes tanh!
            //for (int i = 0; i < hGrads.Length; ++i)
            //{
            //    double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]); // f' of tanh is (1-y)(1+y).
            //    double sum = 0.0;
            //    for (int j = 0; j < _numOutput; ++j) // Each hidden delta is the sum of numOutput terms.
            //        sum += oGrads[j] * hoWeights[i][j]; // Each downstream gradient * outgoing weight.
            //    hGrads[i] = derivative * sum; // hGrad = (1-O)(1+O) * Sum(oGrads*oWts)
            //}
            ComputeTanHGradients(hOutputs, oGrads, hoWeights, hGrads);

            //// 3. Update input to hidden weights.
            //for (int i = 0; i < ihWeights.Length; ++i)
            //{
            //    for (int j = 0; j < ihWeights[i].Length; ++j)
            //    {
            //        double delta = learnRate * hGrads[j] * inputs[i];
            //        ihWeights[i][j] += delta +  momentum * ihPrevWeightsDelta[i][j];
            //        ihPrevWeightsDelta[i][j] = delta;
            //    }
            //}

            //// 4. Update hidden biases.
            //for (int i = 0; i < hBiases.Length; ++i)
            //{
            //    double biasDelta = learnRate * hGrads[i] * 1.0;
            //    hBiases[i] += biasDelta + momentum * hPrevBiasesDelta[i];
            //    hPrevBiasesDelta[i] = biasDelta;
            //}

            UpdateWeightsAndBiases(ihWeights, ihPrevWeightsDelta, hBiases, hPrevBiasesDelta, hGrads, inputs, learnRate, momentum);

            //// 5. Update hidden to output weights.
            //for (int i = 0; i < hoWeights.Length; ++i)
            //{
            //    for (int j = 0; j < hoWeights[i].Length; ++j)
            //    {
            //        double delta = learnRate * oGrads[j] * hOutputs[i];
            //        hoWeights[i][j] += delta + momentum * hoPrevWeightsDelta[i][j];
            //        hoPrevWeightsDelta[i][j] = delta;
            //    }
            //}

            //// 6.Update output biases.
            //for (int i = 0; i < oBiases.Length; ++i)
            //{
            //    double biasDelta = learnRate * oGrads[i] * 1.0;
            //    oBiases[i] += biasDelta + momentum * oPrevBiasesDelta[i];
            //    oPrevBiasesDelta[i] = biasDelta;
            //}

            UpdateWeightsAndBiases(hoWeights, hoPrevWeightsDelta, oBiases, oPrevBiasesDelta, oGrads, hOutputs, learnRate, momentum);
        }
    }
}
