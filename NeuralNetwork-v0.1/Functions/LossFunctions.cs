using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Functions
{
    interface LossFunction
    {
        Tensor Loss(Tensor realValue, Tensor predictedValue);
    }
    class CrossEntropyLoss:LossFunction
    {
        public Tensor Loss(Tensor realValue, Tensor predictedValue)
        {
            return -realValue.InnerTranspose() * predictedValue.Map(r => Math.Log(r));
        }
    }
}
