using NeuralNetwork.Data;
using NeuralNetwork.Models;
using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Evaluate
{
    interface IEvaluater
    {
        double Evaluate(IEnumerable<ProcessData> testCollection,IModel model,IClassifier classifier);
    }
    interface IClassifier
    {
        /// <summary>
        /// 将模型的输出转成单热编码
        /// </summary>
        /// <param name="predictedData"></param>
        /// <returns></returns>
        Tensor Classify(Tensor predictedData);
    }

    class SoftMaxClassifier:IClassifier
    {
        public Tensor Classify(Tensor predictedData)
        {
            return predictedData.Map(r => r < predictedData[0, 0].ToRowMajorArray().Max() ? 0 : 1);
        }
    }
}
