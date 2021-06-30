using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Data;
using NeuralNetwork.Struct;
using MathNet.Numerics.LinearAlgebra;
using System.Text.RegularExpressions;
using System.IO;
using MathNet.Numerics;
using NeuralNetwork.Evaluate;
using NeuralNetwork.Functions.Activation;
using NeuralNetwork.Functions;
using NeuralNetwork.Models;
using NeuralNetwork.Optimize.LearnRate;
using NeuralNetwork.Train;
using MathNet.Numerics.Distributions;
using NeuralNetwork.Layers;

namespace NeuralNetwork.Tests.avila
{
    class AvilaDemo
    {
        public static void train(string dataPath)
        {
            //define dataprocesser
            AvilaDataProcessing dataProcessing = new AvilaDataProcessing(new AvilaDataConverter(new AvilaClassifier()));
            //define evaluater
            AvilaEvaluater evaluater = new AvilaEvaluater(new AvilaDataConverter(new AvilaClassifier()));
            //create dataset
            var inputDataSet = dataProcessing.GetData(dataPath);
            var processDataSet = dataProcessing.GetProcessDataCollection(10000, inputDataSet);
            ProcessData[] trainDataSet;
            ProcessData[] testDataSet;
            (trainDataSet, testDataSet) = dataProcessing.DivideCollection(0.7, processDataSet);
            trainDataSet = trainDataSet.Take(6000).ToArray();

            //define the model
            Model avilaFNN = new Model(new CrossEntropyLoss(), "AvilaFNN",
                new FullConnectLayer(new Direct(1), 10, 10, true, new Normal()),
                new FullConnectLayer(new Sigmoid(), 10, 16, false, new Normal()),
                new SoftMaxLayer(16, 12, new Normal())
                );
            //define optimizer
            Trainer trainer = new Trainer(100, false, 0, new ExponentialDelayOptimizer(2, 0.95, 0.1));

            //train
            for (int i = 0; i <= 19; i++)
            {
                trainDataSet = trainDataSet.SelectPermutation().ToArray();
                trainer = new Trainer(100, false, 0, new ExponentialDelayOptimizer(0.5, 0.9, 0.08));
                avilaFNN = (Model)trainer.Train(avilaFNN, trainDataSet);
                Console.WriteLine($"epoch {i}:" + evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            }

            for (int i = 0; i <= 19; i++)
            {
                trainDataSet = trainDataSet.SelectPermutation().ToArray();
                trainer = new Trainer(60, false, 0, new ExponentialDelayOptimizer(0.5, 0.9, 0.05));
                avilaFNN = (Model)trainer.Train(avilaFNN, trainDataSet);
                Console.WriteLine($"epoch {i}:" + evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            }

            for (int i = 0; i <= 19; i++)
            {
                trainDataSet = trainDataSet.SelectPermutation().ToArray();
                trainer = new Trainer(30, false, 0, new ExponentialDelayOptimizer(0.2, 0.9, 0.05));
                avilaFNN = (Model)trainer.Train(avilaFNN, trainDataSet);
                Console.WriteLine($"epoch {i}:" + evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            }

            //for (int i = 0; i <= 49; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToArray();
            //    trainer = new Trainer(5, false, 0, new ExponentialDelayOptimizer(0.05, 0.9, 0.1));
            //    avilaFNN = (Model)trainer.Train(avilaFNN, trainDataSet);
            //    Console.WriteLine($"epoch {i}:" + evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            //}
            //for (int i = 0; i <= 99; i++)
            //{
            //    trainer = new Trainer(64, false, 0, new ExponentialDelayOptimizer(0.2, 0.95, 0.1));
            //    avilaFNN = (Model)trainer.Train(avilaFNN, trainDataSet);
            //    Console.WriteLine($"epoch {i}:" + evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            //}

            //evaluate
            evaluater.EvaluateInDetails(trainDataSet, avilaFNN, new AvilaClassifier());
            evaluater.EvaluateInDetails(testDataSet, avilaFNN, new AvilaClassifier());
        }
    }
}
