using NeuralNetwork.Models;
using NeuralNetwork.Optimize.LearnRate;
using NeuralNetwork.Train;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Context
{
    class NNContext
    {
        public IModel model;
        public LearnRateOptimizer learnRateOptimizer;
        public Trainer trainer;
    }
}
