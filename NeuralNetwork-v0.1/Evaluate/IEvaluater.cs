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
        Tensor Classify(Tensor predictedData);
    }
}
