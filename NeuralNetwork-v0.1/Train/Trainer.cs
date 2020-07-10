using MathNet.Numerics.Integration;
using NeuralNetwork.Data;
using NeuralNetwork.Extensions;
using NeuralNetwork.Models;
using NeuralNetwork.Optimize.LearnRate;
using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Train
{
    class Trainer
    {
        int batchSize;
        bool dropOut;
        double dropRate;
        LearnRateOptimizer learnRateOptimizer;

        public Trainer(int batchSize,bool dropOut,double dropRate,LearnRateOptimizer learnRateOptimizer)
        {
            this.batchSize = batchSize;
            this.dropOut = dropOut;
            this.dropRate = dropRate;
            this.learnRateOptimizer = learnRateOptimizer;
        }

        public virtual IModel Train(IModel model,ProcessData[] dataCollection)
        {
            if (dataCollection.Count() % batchSize != 0)
            {
                throw new Exception("当前版本暂时只支持整数倍小批量下降");
            }
            if (dropOut == false)
            {
                double[] theta = learnRateOptimizer.Init(model.Count, model.Layers.GetTypes());
                Tensor[] gradientList;
                Tensor[] lossList;
                for (int i = 0; i <= dataCollection.Count() - 1; i=i+batchSize)
                {
                    Stopwatch sw = new Stopwatch();
                    //sw.Start();
                    ProcessData[] tempGroup = new ProcessData[batchSize];
                    for (int j = i; j <= i + batchSize - 1; j++)
                    {
                        tempGroup[j-i] = dataCollection[j];
                    }
                    //sw.Stop();
                    //Console.WriteLine("PreWork:" + sw.Elapsed);
                    sw.Start();
                    (gradientList,lossList) = GetAverageGradient(model, tempGroup);
                    sw.Stop();
                    Console.WriteLine("GetGradientAndLoss:"+sw.Elapsed);
                    //sw.Restart();
                    //lossList = GetAverageLoss(model, tempGroup);
                    //sw.Stop();
                    //Console.WriteLine("GetLoss:"+sw.Elapsed);
                    //sw.Restart();
                    model = BPRefresh(model, gradientList, lossList, theta);
                    //sw.Stop();
                    //Console.WriteLine("BPRefresh:"+sw.Elapsed);
                    //sw.Restart();
                    theta = learnRateOptimizer.Optimize(model.Count, i, dataCollection.Count(), model.GetWeightedGradient(gradientList));
                    //sw.Stop();
                    //Console.WriteLine("LearnRateOptimize:"+sw.Elapsed);
                }
            }
            return model;
        }

        public IModel BPRefresh(IModel model,Tensor[] gradientList,Tensor[] lossList,double[] theta)
        {
            for(int i = 0; i <= model.Count - 1; i++)
            {
                model.Layers[i].BPRefresh(gradientList[i], lossList[i], theta[i]);
            }
            return model;
        }

        /// <summary>
        /// 获取一个批量内的平均梯度列表，为了减少运算开销同时给出损失列表
        /// </summary>
        /// <param name="model"></param>
        /// <param name="dataCollection"></param>
        /// <returns>第一个为梯度列表，第二个为损失列表</returns>
        public static Tuple<Tensor[],Tensor[]> GetAverageGradient(IModel model,ProcessData[] dataCollection)
        {
            Tensor[] gradientList = new Tensor[model.Count];
            Tensor[] lossList = new Tensor[model.Count];
            for (int i = 0; i <= model.Count - 1; i++)
            {
                gradientList[i]=TensorBuilder.AllZeros(model.Layers[i].Weight);
                lossList[i]= TensorBuilder.AllZeros(model.Layers[i].Bias); 
            }
            //foreach(var data in dataCollection)
            for(int j=0;j<=dataCollection.Length-1;j++)
            //Parallel.For(0, dataCollection.Length - 1, j =>
                {
                //var tempList = model.GetGradientList(data);
                var tempList = model.GetGradientList(dataCollection[j]);
                    for (int i = 0; i <= tempList.Item1.Length - 1; i++)
                    {
                        gradientList[i] = gradientList[i] + tempList.Item1[i];
                        lossList[i] = lossList[i] + tempList.Item2[i];
                    }
                }
            gradientList = gradientList.Select(r => r / dataCollection.Count()).ToArray();
            lossList = lossList.Select(r => r / dataCollection.Count()).ToArray();
            return new Tuple<Tensor[], Tensor[]>(gradientList, lossList);
        }

        public static List<Tensor> GetAverageLoss(IModel model, IEnumerable<ProcessData> dataCollection)
        {
            List<Tensor> lossList = new List<Tensor>();
            for (int i = 0; i <= model.Count - 1; i++)
            {
                lossList.Add(TensorBuilder.AllZeros(model.GetOutputAt(i,dataCollection.ElementAt(0).Data)));
            }
            foreach (var data in dataCollection)
            {
                var tempList = model.GetLossList(data);
                for (int i = 0; i <= tempList.Count - 1; i++)
                {
                    lossList[i] = lossList[i] + tempList[i] / dataCollection.Count();
                }
            }
            return lossList;
        }
    }
}
