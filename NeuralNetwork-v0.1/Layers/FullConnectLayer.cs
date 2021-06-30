using MathNet.Numerics.Distributions;
using NeuralNetwork.Functions.Activation;
using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Runtime.Remoting.Messaging;

namespace NeuralNetwork.Layers
{
    [Serializable]
    class FullConnectLayer:LayerBase
    {
        ActivationFunction ActivationFunc;
        public Tensor weight;
        Tensor bias;
        private bool locked = false;
        bool inputLayer;
        public override bool Locked { get => locked; set => locked=value; }
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
        public override LayerSign Sign => LayerSign.FullConnectLayer;
        /// <summary>
        /// 创建全连接层
        /// </summary>
        /// <param name="f">激活函数</param>
        /// <param name="preCount">上一层神经元个数</param>
        /// <param name="thisCount">本层神经元个数</param>
        /// <param name="inputLayer">是否是输入层</param>
        /// <param name="distribution">随机分布，如果是输入层则可用null入参</param>
        /// /// <param name="name">名称</param>
        public FullConnectLayer(ActivationFunction f,int preCount,int thisCount,bool inputLayer,IContinuousDistribution distribution,string name=null)
        {
            this.ActivationFunc = f;
            if (inputLayer == true)
            {
                if (preCount != thisCount)
                {
                    throw new Exception("输入层权重必须为方阵");
                }
                weight = TensorBuilder.FromMatrix(Matrix<double>.Build.DenseIdentity(thisCount));
                bias = TensorBuilder.FromMatrix(Matrix<double>.Build.Dense(thisCount,1, 0));
                locked = true;
            }
            else
            {
                if (distribution == null)
                {
                    distribution = new MathNet.Numerics.Distributions.Normal();
                }
                weight = TensorBuilder.FromMatrix(Matrix<double>.Build.Random(thisCount, preCount, distribution));
                bias= TensorBuilder.FromMatrix(Matrix<double>.Build.Random(thisCount, 1, distribution));
            }
            this.inputLayer = inputLayer;
        }
        public FullConnectLayer(ActivationFunction f,bool inputLayer, Matrix<double> w,Matrix<double> b, string name = null)
        {
            this.ActivationFunc = f;
            weight = TensorBuilder.FromMatrix(w);
            bias = TensorBuilder.FromMatrix(b);
            this.inputLayer = inputLayer;
        }
        public override Tensor Push(Tensor preOutput)
        {
            if (inputLayer)
            {
                return preOutput.Map(ActivationFunc.Activate);
            }
            else
            {
                return (weight * preOutput + bias).Map(ActivationFunc.Activate);
            }
            //return (weight * preOutput + bias).Map(ActivationFunc.Activate);
        }

        public override Tensor PushWithoutActivation(Tensor preOutput)
        {
            if (inputLayer)
            {
                return preOutput.Clone();
            }
            else
            {
                return (weight * preOutput + bias);
            }
            //return (weight * preOutput + bias);
        }

        public override Tensor ComputeLoss(Tensor nextLoss, Tensor preOutput, Tensor nextWeight, LayerSign nextType)
        {
            if(inputLayer)
            {
                return nextWeight.InnerTransposeAndMultiply(nextLoss);
            }
            else
            {
                Tensor pureInput = PushWithoutActivation(preOutput);
                //return pureInput.Map(ActivationFunc.Derivate).InnerDiagonal() * (nextWeight.InnerTranspose()) * nextLoss;
                return pureInput.Map(ActivationFunc.Derivate).PointMutiply((nextWeight.InnerTransposeAndMultiply(nextLoss)));
            }
        }

        public override Tuple<Tensor,Tensor> GetGradient(Tensor nextLoss, Tensor preOutput, Tensor nextWeight, LayerSign nextType)
        {
            var loss = ComputeLoss(nextLoss, preOutput, nextWeight,nextType);
            if (inputLayer)
            {
                //如果是全连接输入层，不需要输出梯度，后续计算理应不涉及，直接设为本层weight
                return new Tuple<Tensor, Tensor>(weight, loss);
            }
            else
            {
                return new Tuple<Tensor, Tensor>(loss * preOutput.InnerTranspose(), loss);
            }
        }

        public override void BPRefresh(Tensor gradient, Tensor loss, double theta)
        {
            if (locked)
            {

            }
            else
            {
                weight = weight - theta * gradient;
                bias = bias - theta * loss;
            }
        }
    }
}
