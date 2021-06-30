using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Functions.Activation;
using NeuralNetwork.Struct;

namespace NeuralNetwork.Layers
{
    [Serializable]
    class PoolingLayer:LayerBase
    {
        ActivationFunction ActivationFunc = new Direct(1);
        Tensor weight;
        Tensor bias;
        PoolingMode mode;
        int poolSize;
        private bool locked = false;
        bool final;
        Tuple<int, int> inputSize;
        int inputFatureCount;
        public override bool Locked { get => locked; set => locked = value; }
        public override Tensor Weight => this.weight;
        public override LayerSign Sign => LayerSign.PoolingLayer;
        public override Tensor Bias => this.bias;

        public PoolingLayer(int poolSize,bool final,int inputFatureCount,Tuple<int,int> inputSize,PoolingMode mode)
        {
            this.mode = mode;
            this.poolSize = poolSize;
            this.final = final;
            this.inputSize = inputSize;
            this.inputFatureCount = inputFatureCount;
            System.Diagnostics.Debug.Assert(inputSize.Item1 % poolSize == 0 && inputSize.Item2 % poolSize == 0);
            this.weight = TensorBuilder.FromMatrix(1,inputFatureCount,Matrix<double>.Build.Dense(inputSize.Item1/poolSize, inputSize.Item2/poolSize,0));
            this.bias = TensorBuilder.FromMatrix(1,inputFatureCount,Matrix<double>.Build.Dense((int)Math.Ceiling((double)inputSize.Item1 / poolSize), (int)Math.Ceiling((double)inputSize.Item2 / poolSize), 0));
        }

        public override Tensor Push(Tensor preOutput)
        {
            if (final)
            {
                switch (mode)
                {
                    case (PoolingMode.Max):
                        {
                            var temp = preOutput.MaxPool(poolSize);
                            weight = temp.Item2;
                            return temp.Item1.Stretch();
                        }
                    case (PoolingMode.Average):
                        {
                            var temp = preOutput.AveragePool(poolSize);
                            weight = temp.Map(r => 1.0);
                            return temp.Stretch();
                        }
                    default:
                        {
                            throw new Exception("错误的池化方式");
                        }
                }
            }
            else
            {
                switch (mode)
                {
                    case (PoolingMode.Max):
                        {
                            var temp = preOutput.MaxPool(poolSize);
                            weight = temp.Item2;
                            return temp.Item1;
                        }
                    case (PoolingMode.Average):
                        {
                            var temp = preOutput.AveragePool(poolSize);
                            weight = temp.Map(r => 1.0);
                            return temp;
                        }
                    default:
                        {
                            throw new Exception("错误的池化方式");
                        }
                }
            }
        }

        public override Tensor PushWithoutActivation(Tensor preOutput)
        {
            return Push(preOutput);
        }

        public override Tensor ComputeLoss(Tensor nextLoss, Tensor preOutput, Tensor nextWeight, LayerSign nextType)
        {
            if (nextType == LayerSign.PoolingLayer)//如果下一层是池化层。nextWeight为掩码矩阵
            {
                Tensor pureInput = this.PushWithoutActivation(preOutput);
                if (nextWeight[0, 0].ForAll(r => r == 1.0))
                {
                    return pureInput.Map(ActivationFunc.Derivate).PointMutiply(nextLoss.AverageUpSample((int)Math.Ceiling((double)nextWeight[0, 0].RowCount / (double)nextLoss[0, 0].RowCount), nextWeight[0, 0].RowCount, nextWeight[0, 0].ColumnCount));
                }
                else
                {
                    return pureInput.Map(ActivationFunc.Derivate).PointMutiply(nextLoss.MaxUpSample(nextWeight, (int)Math.Ceiling((double)nextWeight[0, 0].RowCount / (double)nextLoss[0, 0].RowCount)));
                }
            }
            else if (nextType == LayerSign.ConvolutionalLayer)
            {
                Tensor pureInput = this.PushWithoutActivation(preOutput);
                return pureInput.Map(ActivationFunc.Derivate).PointMutiply(nextLoss.InnerRot180().Convolve(nextWeight.OuterTranspose(), ConvolutionMode.Wide));
            }
            else if (nextType == LayerSign.FullConnectLayer)
            {
                return nextLoss.Fold(1, inputFatureCount, (int)Math.Ceiling((double)inputSize.Item1 / poolSize), (int)Math.Ceiling((double)inputSize.Item2 / poolSize));
            }
            else
            {
                throw new Exception("神经网络连接方式错误");
            }
        }
        /// <summary>
        /// 获取池化层梯度和损失，这里梯度直接设为weight，后续外部计算理应不涉及池化层梯度
        /// </summary>
        /// <param name="nextLoss"></param>
        /// <param name="preOutput"></param>
        /// <param name="nextWeight"></param>
        /// <param name="nextType"></param>
        /// <returns></returns>
        public override Tuple<Tensor, Tensor> GetGradient(Tensor nextLoss, Tensor preOutput, Tensor nextWeight, LayerSign nextType)
        {
            Tensor loss = ComputeLoss(nextLoss, preOutput, nextWeight, nextType);
            //Tensor gradient = preOutput.OuterTranspose().Convolve(loss, mode);
            return new Tuple<Tensor, Tensor>(weight, loss);
        }

        public override void BPRefresh(Tensor gradient, Tensor loss, double theta)
        {
            
        }
    }
}
