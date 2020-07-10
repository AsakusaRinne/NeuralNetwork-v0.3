using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Functions.Activation;
using NeuralNetwork.Struct;

namespace NeuralNetwork.Layers
{
    abstract class LayerBase
    {
        ActivationFunction ActivationFunc;
        /// <summary>
        /// 该项在全连接和softmax中代表的是权重矩阵（张量），在卷积层中则是指卷积核张量
        /// </summary>
        public virtual Tensor Weight
        {
            get;
        }
        public virtual Tensor Bias
        {
            get;
        }
        public virtual Tensor Push(Tensor preOutput)
        {
            return null;
        }
        public virtual Tensor PushWithoutActivation(Tensor preOutput)
        {
            return null;
        }
        public virtual Tensor ComputeLoss(Tensor nextLoss,Tensor preOutput,Tensor nextWeight)
        {
            return null;
        }
        /// <summary>
        /// 计算梯度（同时返回损失，减少后续计算开销）
        /// </summary>
        /// <param name="nextLoss"></param>
        /// <param name="preOutput"></param>
        /// <param name="nextWeight"></param>
        /// <returns>第一个为梯度，第二个为损失</returns>
        public virtual Tuple< Tensor,Tensor> GetGradient(Tensor nextLoss, Tensor preOutput, Tensor nextWeight)
        {
            return null;
        }
        public virtual void BPRefresh(Tensor gradient,Tensor loss,double theta)
        {

        }
    }

    
}
