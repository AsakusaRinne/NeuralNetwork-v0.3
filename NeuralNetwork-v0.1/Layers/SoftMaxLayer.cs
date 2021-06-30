using MathNet.Numerics.Distributions;
using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Functions.Activation;

namespace NeuralNetwork.Layers
{
    [Serializable]
    class SoftMaxLayer:LayerBase
    {
        public Tensor weight;
        public Tensor bias;
        private bool locked = false;
        ActivationFunction ActivationFunc = new Direct(1);

        public override bool Locked { get => locked; set => locked = value; }
        public override Tensor Weight
        {
            get
            {
                return this.weight;
            }
        }
        public override Tensor Bias
        {
            get
            {
                return this.bias;
            }
        }
        public override LayerSign Sign => LayerSign.SoftMaxLayer;
        public SoftMaxLayer(int preCount,int thisCount,IContinuousDistribution distribution)
        {
            weight = TensorBuilder.FromMatrix(Matrix<double>.Build.Random(thisCount, preCount, distribution));
            bias = TensorBuilder.FromMatrix(Matrix<double>.Build.Dense(thisCount, 1, 0));
        }

        public SoftMaxLayer(Matrix<double> w)
        {
            weight = TensorBuilder.FromMatrix(w);
            bias = TensorBuilder.FromMatrix(Matrix<double>.Build.Dense(weight[0,0].RowCount, 1, 0));
        }

        public override Tensor Push(Tensor preOutput)
        {
            //Tensor temp = (weight * preOutput+bias).Map(r => Math.Exp(r));
            Tensor temp = (weight * preOutput).Map(r => Math.Exp(r));
            //Tensor oneC = TensorBuilder.FromMatrix(Matrix<double>.Build.Dense(1, temp[0, 0].RowCount, 1));
            //return temp / (oneC * temp);
            return temp / (temp.SumAll());
        }

        public override Tensor PushWithoutActivation(Tensor preOutput)
        {
            //Tensor temp = (weight * preOutput+bias).Map(r => Math.Exp(r));
            Tensor temp = (weight * preOutput ).Map(r => Math.Exp(r));
            //Tensor oneC = TensorBuilder.FromMatrix(Matrix<double>.Build.Dense(1, temp[0, 0].RowCount, 1));
            //return temp / (oneC * temp);
            return temp / (temp.SumAll());
        }
        /// <summary>
        /// 计算softmax层的损失函数，这里的nextLoss入参应为实际值-预测值，因为softmax固定使用交叉熵损失函数
        /// </summary>
        /// <param name="nextLoss"></param>
        /// <param name="preOutput"></param>
        /// <param name="nextWeight"></param>
        /// <returns></returns>
        public override Tensor ComputeLoss(Tensor nextLoss, Tensor preOutput, Tensor nextWeight, LayerSign nextType)
        {
            return -nextLoss;
        }

        public override Tuple< Tensor,Tensor> GetGradient(Tensor nextLoss, Tensor preOutput, Tensor nextWeight, LayerSign nextType)
        {
            var loss = -nextLoss;
            return new Tuple<Tensor, Tensor>(loss  *preOutput.InnerTranspose(), loss);
        }

        public override void BPRefresh(Tensor gradient, Tensor loss, double theta)
        {
            if (!locked)
            {
                weight = weight - theta * gradient;
                //bias = bias - theta * loss;
            }
        }
    }
}
