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
using System.Data;

namespace NeuralNetwork.Train
{
    class Trainer
    {
        int batchSize;
        bool dropOut;
        double dropRate;
        LearnRateOptimizer learnRateOptimizer;
        public DataTable lossTable;

        public Trainer(int batchSize,bool dropOut,double dropRate,LearnRateOptimizer learnRateOptimizer)
        {
            this.batchSize = batchSize;
            this.dropOut = dropOut;
            this.dropRate = dropRate;
            this.learnRateOptimizer = learnRateOptimizer;
        }

        public virtual IModel Train(IModel model,ProcessData[] dataCollection)
        {
            lossTable = new DataTable();
            lossTable.Columns.Add("Times");
            lossTable.Columns.Add("loss");
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
                    ProcessData[] tempGroup = new ProcessData[batchSize];
                    for (int j = i; j <= i + batchSize - 1; j++)
                    {
                        tempGroup[j-i] = dataCollection[j];
                    }
                    (gradientList,lossList) = GetAverageGradient(model, tempGroup);
                    model = BPRefresh(model, gradientList, lossList, theta);
                    lossTable.Rows.Add(i, lossList[lossList.Length - 1].AbsoluteSumAll());
                    theta = learnRateOptimizer.Optimize(model.Count, i, dataCollection.Count(), model.GetWeightedGradient(gradientList));
                }
            }
            return model;
        }

        public virtual IModel Train(IModel model, ProcessData[] dataCollection,ProcessData[] verifyCollection)
        {
            lossTable = new DataTable();
            lossTable.Columns.Add("Times");
            lossTable.Columns.Add("loss");
            if (dataCollection.Count() % batchSize != 0)
            {
                throw new Exception("当前版本暂时只支持整数倍小批量下降");
            }
            if (dropOut == false)
            {
                double[] theta = learnRateOptimizer.Init(model.Count, model.Layers.GetTypes());
                Tensor[] gradientList;
                Tensor[] lossList;
                double prevLoss = 0;
                double vLoss = 0;
                for (int i = 0; i <= dataCollection.Count() - 1; i = i + batchSize)
                {
                    ProcessData[] tempGroup = new ProcessData[batchSize];
                    for (int j = i; j <= i + batchSize - 1; j++)
                    {
                        tempGroup[j - i] = dataCollection[j];
                    }
                    (gradientList, lossList) = GetAverageGradient(model, tempGroup);
                    model = BPRefresh(model, gradientList, lossList, theta);
                    
                    var tempv = GetFinalAbsoluteAverageLoss(model, verifyCollection).SumAll();
                    lossTable.Rows.Add(i, tempv);
                    if (tempv > vLoss && vLoss > prevLoss&&prevLoss!=0)
                    {
                        Console.WriteLine($"第{i}次时推出循环");
                        break;
                    }
                    else
                    {
                        prevLoss = vLoss;
                        vLoss = tempv;
                    }
                    theta = learnRateOptimizer.Optimize(model.Count, i, dataCollection.Count(), model.GetWeightedGradient(gradientList));
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
            var temp = model.GetGradientList(dataCollection[0]);
            for (int i = 0; i <= model.Count - 1; i++)
            {
                gradientList[i]=TensorBuilder.AllZeros(temp.Item1[i]);
                lossList[i]= TensorBuilder.AllZeros(temp.Item2[i]); 
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
            return lossList.Select(r => r / dataCollection.Count()).ToList();
        }

        public static Tensor GetFinalAbsoluteAverageLoss(IModel model, IEnumerable<ProcessData> dataCollection)
        {
            Tensor loss = TensorBuilder.AllZeros(model.GetFinalLoss(dataCollection.ElementAt(0)));
            foreach (var data in dataCollection)
            {
                var tempList = model.GetFinalLoss(data);
                loss = loss + model.GetFinalLoss(data).OuterMap(r => r.PointwiseAbs());
            }
            return loss / dataCollection.Count();
        }
    }
}
