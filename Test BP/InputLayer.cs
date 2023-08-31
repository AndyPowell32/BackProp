namespace Test_BP
{
    public class InputLayer: ILayer
    {
        public int Count { get; }
        public double[] Outputs { get; }

        public InputLayer(int count)
        {
            Count = count;
            Outputs = new double[count];
        }

        public void SetInputs(double[] inputs)
        {
            Array.Copy(inputs, Outputs, Count);
        }
    }
}
