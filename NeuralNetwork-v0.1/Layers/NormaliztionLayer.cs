using MathNet.Numerics.LinearAlgebra.Complex;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Data;
using NeuralNetwork.Struct;
using NeuralNetwork.Extensions;

namespace NeuralNetwork.Layers
{
    [Serializable]
    class NormaliztionLayer : LayerBase
    {
        private bool locked = false;
        public override bool Locked { get => locked; set => locked = value; }
        /// <summary>
        /// 训练数据的均值
        /// </summary>
        Tensor average;
        /// <summary>
        /// 训练数据的标准差
        /// </summary>
        Tensor standardDeviation;
        public NormaliztionLayer(IEnumerable<ProcessData> data,NormalizationMode mode)
        {
            switch (mode)
            {
                case (NormalizationMode.Standardization):
                {
                        Tensor avg = TensorBuilder.AllZeros(data.ElementAt(0).Data);//均值
                        Tensor variance = TensorBuilder.AllZeros(data.ElementAt(0).Data);//方差
                        foreach (var item in data)
                        {
                            avg = avg + item.Data;
                        }
                        avg = avg / data.Count();
                        foreach(var item in data)
                        {
                            variance = variance + (item.Data - avg).Map(r => r * r);
                        }
                        variance = variance / data.Count();
                        average = avg;
                        standardDeviation = variance.Map(r => Math.Sqrt(r)).ForceZerosToValue(1);
                        break;
                }
            }
        }

        public override Tensor Weight => average.Clone();
        public override Tensor Bias => average.Clone();
        public override LayerSign Sign => LayerSign.NormalizationLayer;

        public override Tensor Push(Tensor preOutput)
        {
            return (preOutput - average).PointDivide(standardDeviation);
        }

        public override Tensor PushWithoutActivation(Tensor preOutput)
        {
            return Push(preOutput);
        }

        public override Tensor ComputeLoss(Tensor nextLoss, Tensor preOutput, Tensor nextWeight, LayerSign nextType)
        {
            return average.Clone();
        }

        public override Tuple<Tensor, Tensor> GetGradient(Tensor nextLoss, Tensor preOutput, Tensor nextWeight, LayerSign nextType)
        {
            return new Tuple<Tensor, Tensor>(average.Clone(), average.Clone());
        }

        public override void BPRefresh(Tensor gradient, Tensor loss, double theta)
        {
            
        }
    }
}

namespace NeuralNetwork
{
    public enum NormalizationMode
    {
        Standardization=0
    }
}
