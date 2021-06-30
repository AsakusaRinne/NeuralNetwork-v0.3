using NeuralNetwork.Models;
using NeuralNetwork.Optimize.LearnRate;
using NeuralNetwork.Train;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Evaluate;
using NeuralNetwork.Data;
using System.Data;
using System.IO;

namespace NeuralNetwork.Context
{
    class Context
    {
        public IModel model;
        public List<KeyValuePair<Trainer, int>> process = new List<KeyValuePair<Trainer, int>>();
        IEvaluater evaluater;
        IClassifier classifier;

        public Context(IModel model,IEvaluater evaluater,IClassifier classifier)
        {
            this.model = model;
            this.evaluater = evaluater;
            this.classifier = classifier;
        }

        public void AddProcess(Trainer trainer,int times)
        {
            this.process.Add(new KeyValuePair<Trainer, int>(trainer, times));
        }

        public void Train(IEnumerable<ProcessData> trainSet,IEnumerable<ProcessData> testDataSet,string saveDirectory,ModelSaveMode mode)
        {
            if (!Directory.Exists(saveDirectory))
            {
                try
                {
                    Directory.CreateDirectory(saveDirectory);
                }
                catch
                {
                    throw new Exception("非法路径");
                }
            }
            switch (mode)
            {
                case (ModelSaveMode.Final):
                    {
                        foreach (var item in process)
                        {
                            for (int i = 0; i <= item.Value; i++)
                            {
                                model = item.Key.Train(model, trainSet.ToArray());
                            }
                        }
                        double acc = evaluater.Evaluate(testDataSet, model, classifier);
                        model.SaveModel(Path.Combine(saveDirectory, model.Name +"-"+acc.ToString("f3")+ ".Model"));
                        break;
                    }
                case (ModelSaveMode.All):
                    {
                        foreach (var item in process)
                        {
                            for (int i = 0; i <= item.Value; i++)
                            {
                                model = item.Key.Train(model, trainSet.ToArray());
                                double acc = evaluater.Evaluate(testDataSet, model, classifier);
                                model.SaveModel(Path.Combine(saveDirectory, model.Name + "-" + acc.ToString("f3") + ".Model"));
                            }
                        }
                        break;
                    }

            }
        }

        public DataSet TrainAndShowLoss()
        {
            return null;
        }
    }
}
