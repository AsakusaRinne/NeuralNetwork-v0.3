using MathNet.Numerics.Distributions;
using NeuralNetwork.Functions.Activation;
using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Layers
{
    class FullConnectLayer:LayerBase
    {
        ActivationFunction ActivationFunc;
        public Tensor weight;
        Tensor bias;
        private bool Locked = false;
        bool inputLayer;
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
                weight = TensorBuilder.FromMatrix(Matrix<double>.Build.DenseIdentity(thisCount),name==null?null:(name+".weight"));
                bias = TensorBuilder.FromMatrix(Matrix<double>.Build.Dense(thisCount,1, 0),name == null ? null : (name + ".bias"));
                Locked = true;
            }
            else
            {
                if (distribution == null)
                {
                    distribution = new MathNet.Numerics.Distributions.Normal();
                }
                weight = TensorBuilder.FromMatrix(Matrix<double>.Build.Random(thisCount, preCount, distribution), name == null ? null : (name + ".weight"));
                bias= TensorBuilder.FromMatrix(Matrix<double>.Build.Random(thisCount, 1, distribution), name == null ? null : (name + ".bias"));
            }
            this.inputLayer = inputLayer;
        }
        public FullConnectLayer(ActivationFunction f,bool inputLayer, Matrix<double> w,Matrix<double> b, string name = null)
        {
            this.ActivationFunc = f;
                weight = TensorBuilder.FromMatrix(w, name == null ? null : (name + ".weight"));
                bias = TensorBuilder.FromMatrix(b, name == null ? null : (name + ".bias"));
            this.inputLayer = inputLayer;
        }
        public override Tensor Push(Tensor preOutput)
        {
            return (weight*preOutput + bias).Map(ActivationFunc.Activate);
        }

        public override Tensor PushWithoutActivation(Tensor preOutput)
        {
            return weight * preOutput + bias;
        }

        public override Tensor ComputeLoss(Tensor nextLoss, Tensor preOutput, Tensor nextWeight)
        {
            if(inputLayer)
            {
                return (nextWeight.InnerTranspose()) * nextLoss;
            }
            else
            {
                Tensor pureInput = PushWithoutActivation(preOutput);
                return pureInput.Map(ActivationFunc.Derivate).InnerDiagonal() * (nextWeight.InnerTranspose()) * nextLoss;
            }
        }

        public override Tuple<Tensor,Tensor> GetGradient(Tensor nextLoss, Tensor preOutput, Tensor nextWeight)
        {
            var loss = ComputeLoss(nextLoss, preOutput, nextWeight);
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
            if (Locked)
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
