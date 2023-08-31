namespace Test_BP
{
    public interface INextLayer: ILayer
    {
        double[][] Weights { get; }
        double[] Grads { get; }
    }
}
