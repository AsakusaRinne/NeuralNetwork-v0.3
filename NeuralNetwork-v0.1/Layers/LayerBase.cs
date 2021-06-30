using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Functions.Activation;
using NeuralNetwork.Struct;

namespace NeuralNetwork.Layers
{
    [Serializable]
    abstract class LayerBase
    {
        ActivationFunction ActivationFunc;
        public virtual LayerSign Sign
        {
            get;
        }
        /// <summary>
        /// 该项在全连接和softmax中代表的是权重矩阵（张量），在卷积层中则是指卷积核张量
        /// </summary>
        public abstract Tensor Weight
        {
            get;
        }
        public abstract Tensor Bias
        {
            get;
        }
        public abstract bool Locked
        {
            get;set;
        }
        public abstract Tensor Push(Tensor preOutput);
        public abstract Tensor PushWithoutActivation(Tensor preOutput);
        public abstract Tensor ComputeLoss(Tensor nextLoss, Tensor preOutput, Tensor nextWeight, LayerSign nextType);
        /// <summary>
        /// 计算梯度（同时返回损失，减少后续计算开销）
        /// </summary>
        /// <param name="nextLoss"></param>
        /// <param name="preOutput"></param>
        /// <param name="nextWeight"></param>
        /// <returns>第一个为梯度，第二个为损失</returns>
        public abstract Tuple<Tensor, Tensor> GetGradient(Tensor nextLoss, Tensor preOutput, Tensor nextWeight, LayerSign nextType);
        public abstract void BPRefresh(Tensor gradient, Tensor loss, double theta);
        public void SetActivationFunc(ActivationFunction f)
        {
            this.ActivationFunc = f;
        }
    }
}

namespace NeuralNetwork
{
    enum LayerSign
    {
        FullConnectLayer=0,
        SoftMaxLayer=1,
        ConvolutionalLayer=2,
        PoolingLayer=3,
        NormalizationLayer=4,
        Nothing=5
    }
}
