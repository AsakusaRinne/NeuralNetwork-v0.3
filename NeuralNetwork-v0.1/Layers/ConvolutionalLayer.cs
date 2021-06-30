using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Functions.Activation;
using NeuralNetwork.Struct;
using static System.Diagnostics.Debug;

namespace NeuralNetwork.Layers
{
    [Serializable]
    class ConvolutionalLayer:LayerBase
    {
        public ActivationFunction ActivationFunc;
        Tensor core;
        Tensor bias;
        ConvolutionMode mode;
        private bool locked = false;
        bool final;
        Tuple<int, int> inputSize;
        int inputFatureCount;
        int outputFatureCount;
        /// <summary>
        /// 输出的张量中矩阵的维度
        /// </summary>
        int mx;
        /// <summary>
        /// 输出的张量中矩阵的维度
        /// </summary>
        int my;

        public override bool Locked { get => locked; set => locked = value; }
        public override Tensor Weight => this.core;
        public override Tensor Bias => TensorBuilder.FromMatrix(1, outputFatureCount, Matrix<double>.Build.Dense(mx,my,0));
        public override LayerSign Sign => LayerSign.ConvolutionalLayer;

        public ConvolutionalLayer(ActivationFunction activationFunc,bool final,Tuple<int,int> inputSize,int coreSize,int inputFatureCount,int outputFatureCount,ConvolutionMode mode,IContinuousDistribution distribution)
        {
            this.ActivationFunc = activationFunc;
            this.mode = mode;
            this.core = TensorBuilder.Random(inputFatureCount, outputFatureCount, coreSize, coreSize, distribution);
            this.bias = TensorBuilder.Random(1, outputFatureCount, 1, 1, distribution);
            this.final = final;
            this.inputSize = inputSize;
            this.inputFatureCount = inputFatureCount;
            this.outputFatureCount = outputFatureCount;
            switch (mode)
            {
                case (ConvolutionMode.Narrow):
                    {
                        this.mx = inputSize.Item1 - coreSize + 1;
                        this.my = inputSize.Item2 - coreSize + 1;
                        break;
                    }
                case (ConvolutionMode.Same):
                {
                    this.mx= inputSize.Item1;
                    this.my = inputSize.Item2;
                    break;
                }
                case (ConvolutionMode.Wide):
                    {
                        this.mx = inputSize.Item1 + coreSize - 1;
                        this.my = inputSize.Item2 + coreSize - 1;
                        break;
                    }
            }
        }

        public ConvolutionalLayer(ActivationFunction activationFunc,bool final,Tuple<int,int> inputSize, Tensor core,Tensor bias, ConvolutionMode mode)
        {
            Assert(core.IsSameSize());
            Assert(core.DimensionX == bias.DimensionX && core.DimensionY == bias.DimensionY);
            this.ActivationFunc = activationFunc;
            this.mode = mode;
            this.core = core;
            this.bias = bias;
            this.final = final;
            this.inputSize = inputSize;
            this.inputFatureCount = core.DimensionX;
            this.outputFatureCount = core.DimensionY;
        }

        public override Tensor Push(Tensor preOutput)
        {
            if (final)
            {
                return (preOutput.Convolve(core, mode).OuterMapIndexed((x, y, r) => r + bias[x, y, 0, 0])).Map(ActivationFunc.Activate).Stretch();
            }
            else
            {
                return (preOutput.Convolve(core, mode).OuterMapIndexed((x, y, r) => r + bias[x, y, 0, 0])).Map(ActivationFunc.Activate);
            }
        }

        public override Tensor PushWithoutActivation(Tensor preOutput)
        {
            if (final)
            {
                return preOutput.Convolve(core, mode).OuterMapIndexed((x, y, r) => r + bias[x, y, 0, 0]).Stretch();
            }
            else
            {
                return preOutput.Convolve(core, mode).OuterMapIndexed((x, y, r) => r + bias[x, y, 0, 0]);
            }
        }

        public override Tensor ComputeLoss(Tensor nextLoss, Tensor preOutput, Tensor nextWeight,LayerSign nextType)
        {
            if (nextType==LayerSign.PoolingLayer)//如果下一层是池化层。nextWeight为掩码矩阵
            {
                Tensor pureInput = this.PushWithoutActivation(preOutput);
                if (nextWeight[0, 0].ForAll(r => r == 1.0))
                {
                    return pureInput.Map(ActivationFunc.Derivate).PointMutiply(nextLoss.AverageUpSample((int)Math.Ceiling((double)pureInput[0, 0].RowCount / (double)nextWeight[0, 0].RowCount), pureInput[0, 0].RowCount,pureInput[0, 0].ColumnCount)) ;
                }
                else
                {
                    return pureInput.Map(ActivationFunc.Derivate).PointMutiply(nextLoss.MaxUpSample(nextWeight, (int)Math.Ceiling((double)nextWeight[0, 0].RowCount / (double)nextLoss[0, 0].RowCount)));
                }
            }
            else if (nextType == LayerSign.ConvolutionalLayer)
            {
                Tensor pureInput = this.PushWithoutActivation(preOutput);
                return pureInput.Map(ActivationFunc.Derivate).PointMutiply(nextLoss.InnerRot180().Convolve(nextWeight, ConvolutionMode.Wide));
            }
            else if (nextType == LayerSign.FullConnectLayer)
            {
                return nextLoss.Fold(1, outputFatureCount, mx, my);
            }
            else
            {
                throw new Exception("神经网络连接方式错误");
            }
        }

        public override Tuple<Tensor, Tensor> GetGradient(Tensor nextLoss, Tensor preOutput, Tensor nextWeight, LayerSign nextType)
        {
            Tensor loss = ComputeLoss(nextLoss, preOutput, nextWeight, nextType);
            Tensor gradient = preOutput.OuterTranspose().Convolve(loss, mode);
            return new Tuple<Tensor, Tensor>(gradient, loss);
        }

        public override void BPRefresh(Tensor gradient, Tensor loss, double theta)
        {
            if (!locked)
            {
                this.core = this.core - theta * gradient;
                this.bias = this.bias - theta * loss.OuterMap(r => Matrix<double>.Build.Dense(1, 1, r.ColumnSums().Sum()));
            }
        }
    }
}
