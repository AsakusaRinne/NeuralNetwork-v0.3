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
    class AvilaDataConverter: IDataConvert<double[], string>
    {
        Dictionary<double, string> labelDecodeDic; 
        Dictionary<string, double[]> labelEncodeDic;
        IClassifier classifier;

        public AvilaDataConverter(IClassifier classifier)
        {
            this.classifier = classifier;

            labelDecodeDic = new Dictionary<double, string>();
            labelEncodeDic = new Dictionary<string, double[]>(new StringCompareByValue());

            labelEncodeDic.Add("A", new double[] { 1, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0});
            labelEncodeDic.Add("B", new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
            labelEncodeDic.Add("C", new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
            labelEncodeDic.Add("D", new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 });
            labelEncodeDic.Add("E", new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 });
            labelEncodeDic.Add("F", new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 });
            labelEncodeDic.Add("G", new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 });
            labelEncodeDic.Add("H", new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 });
            labelEncodeDic.Add("I", new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 });
            labelEncodeDic.Add("W", new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 });
            labelEncodeDic.Add("X", new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 });
            labelEncodeDic.Add("Y", new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
            labelDecodeDic = labelEncodeDic.ToDictionary(r => GetFromOnehot(r.Value), r => r.Key);
        }
        public ProcessData ConvertInputToProcess(InputData<double[], string> inputData)
        {
            return new ProcessData()
            {
                Data = TensorBuilder.FromMatrix(Matrix<double>.Build.DenseOfRowMajor(10, 1, inputData.OriginalData)),
                Label = TensorBuilder.FromMatrix(Matrix<double>.Build.Dense(12, 1, labelEncodeDic[inputData.OriginalLabel]))
            };
        }
        public OutputData<double[], string> ConvertProcessToOutput(ProcessData processData,IModel model)
        {
            return new OutputData<double[], string>()
            {
                OriginalData = processData.Data[0, 0].ToRowMajorArray(),
                RealResult = labelDecodeDic[GetFromOnehot(processData.Label[0, 0].ToRowMajorArray())],
                PredictedResult=labelDecodeDic[GetFromOnehot(classifier.Classify(model.Predict(processData.Data))[0, 0].ToRowMajorArray())]
            };
        }

        public double[] ConvertToOnehot(double originalData, int maxValue)
        {
            double[] result = new double[maxValue + 1];
            for (int i = 0; i <= result.Length - 1; i++)
            {
                if (i == originalData)
                {
                    result[i] = 1;
                }
                else
                {
                    result[i] = 0;
                }
            }
            return result;
        }

        public double GetFromOnehot(double[] oneHot)
        {
            double result = -1;
            for(int i = 0; i <= oneHot.Length - 1; i++)
            {
                if (oneHot[i] == 1)
                {
                    if (result == -1)
                    {
                        result = i;
                    }
                    else
                    {
                        throw new Exception("错误的单热编码");
                    }
                }
            }
            return result;
        }
    }
    class AvilaDataProcessing
    {
        IDataConvert<double[],string> dataConverter;
        public AvilaDataProcessing(IDataConvert<double[],string> dataConverter)
        {
            this.dataConverter = dataConverter;
        }
        public List<InputData<double[],string>> GetData(string fullPath)
        {
            List<InputData<double[], string>> dataList = new List<InputData<double[], string>>();
            FileStream f = new FileStream(fullPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            StreamReader f_reader = new StreamReader(f);
            string dataText= f_reader.ReadToEnd();

            string pattern = @"\b(,)([A-Za-z]+)\b";

            Dictionary<object[], object[]> data_dic = new Dictionary<object[], object[]>();
            int start_index = 0;
            dataText = dataText.Replace("\r\n", "\n");
            MatchCollection variety_name = Regex.Matches(dataText, pattern, RegexOptions.Multiline);

            foreach (Match x in variety_name)
            {
                //节省开销
                var xvalue = x.Value;
                dataList.Add(new InputData<double[], string>()
                {
                    OriginalData = Array.ConvertAll<string, double>
                    (dataText.Substring(start_index, x.Index - start_index).Split(','),
                    new Converter<string, double>(double.Parse)),
                    OriginalLabel = xvalue.Substring(1, xvalue.Length - 1).Trim()
                });
                start_index = x.Index + xvalue.Length;
            }
            return dataList;
        }

        public Tuple<ProcessData[], ProcessData[]> DivideCollection(double rate, List<ProcessData> dataCollection)
        {
            if (rate < 0 || rate > 1)
            {
                throw new Exception("分割比例要在0-1之间");
            }
            List<ProcessData> retainedCollection = new List<ProcessData>();
            List<ProcessData> dividedCollection = new List<ProcessData>();
            ILookup<Tensor, ProcessData> DataLookup = dataCollection.ToLookup<ProcessData,Tensor,ProcessData>(r => r.Label, r => r, comparer:new TensorCompareByValue());
            //对于Lookup中的每一个Group
            foreach (var species in DataLookup)
            {
                Random rnd = new Random();
                int retain_count = Convert.ToInt32(DataLookup[species.Key].Count() * rate);
                //切割得到一个rate%的随机组合
                var subset = species.ToList();
                var rndretain = subset.SelectCombination(retain_count).ToList();
                var rndsub = subset.Except(rndretain).ToList();


                for (int i = 0; i <= rndretain.Count() - 1; i++)
                {
                    retainedCollection.Add(rndretain[i]);
                }
                for (int i = 0; i <= rndsub.Count() - 1; i++)
                {
                    dividedCollection.Add(rndsub[i]);
                }
            }
            return new Tuple<ProcessData[], ProcessData[]>(retainedCollection.ToArray(), dividedCollection.ToArray());
        }
    
        public List<ProcessData> GetProcessDataCollection(int count, List<InputData<double[], string>> inputDataCollection)
        {
            if (count > inputDataCollection.Count)
            {
                throw new Exception("超出数据集数量上限");
            }
            List<ProcessData> processDataCollection = new List<ProcessData>();
            //随机抽取，避免分配不均
            var tempList = inputDataCollection.SelectCombination(count);
            foreach(var item in tempList)
            {
                processDataCollection.Add(dataConverter.ConvertInputToProcess(item));
            }
            return processDataCollection;
        }
    }

    class AvilaClassifier : IClassifier
    {
        public Tensor Classify(Tensor predictedData)
        {
            return predictedData.Map(r => r < predictedData[0, 0].ToRowMajorArray().Max() ? 0 : 1);
        }
    }

    class AvilaEvaluater : IEvaluater
    {
        IDataConvert<double[], string> converter;
        public AvilaEvaluater(IDataConvert<double[],string> converter)
        {
            this.converter = converter;
        }
        public double Evaluate(IEnumerable<ProcessData> testCollection,IModel model,IClassifier classifier)
        {
            int correct = 0;
            int wrong = 0;
            foreach(var item in testCollection)
            {
                if (classifier.Classify(model.Predict(item.Data)).Equals(item.Label))
                {
                    correct++;
                }
                else
                {
                    wrong++;
                }
            }
            return ((double)correct) / (double)(correct + wrong);
        }
        public void EvaluateInDetails(IEnumerable<ProcessData> testCollection, IModel model, IClassifier classifier)
        {
            int correct = 0;
            int wrong = 0;
            int i = 0;
            foreach (var item in testCollection)
            {
                i++;
                var prediction = converter.ConvertProcessToOutput(item, model);
                if (prediction.PredictedResult==prediction.RealResult)
                {
                    correct++;
                }
                else
                {
                    wrong++;
                }
                //Console.WriteLine($"{i}：Prediction is{prediction.PredictedResult}，Truth is {prediction.RealResult}，correct:{correct},wrong:{wrong} ");
            }
            Console.WriteLine($"acc: {((double)correct) / (double)(correct + wrong)}");
        }
    }

    public class doubleArrayCompareByValue : IEqualityComparer<double[]>
    {
        public bool Equals(double[] x, double[] y)
        {
            if (x == null || y == null || (x.Count() != y.Count()))
            {
                return false;
            }
            else
            {
                for (int i = 0; i <= x.Count() - 1; i++)
                {
                    if (!x[i].Equals(y[i]))
                    {
                        return false;
                    }
                }
                return true;
            }

        }

        public int GetHashCode(double[] obj)
        {
            if (obj == null)
                return 0;
            else
                //return obj.ToString().GetHashCode();
                return 1;
        }
    }
    public class StringCompareByValue : IEqualityComparer<string>
    {
        public bool Equals(string x, string y)
        {
            if (x == null || y == null)
                return false;
            if (x == y)
                return true;
            else
                return false;
        }

        public int GetHashCode(string obj)
        {
            if (obj == null)
                return 0;
            else
                return obj.GetHashCode();
        }
    }

    public class AvilaTest
    {
        public static void test()
        {
            AvilaDataProcessing dataProcessing = new AvilaDataProcessing(new AvilaDataConverter(new AvilaClassifier()));
            AvilaEvaluater evaluater = new AvilaEvaluater(new AvilaDataConverter(new AvilaClassifier()));
            var inputDataSet = dataProcessing.GetData(@"C:\Work\DataSets\avila\avila-tr.txt");
            var processDataSet = dataProcessing.GetProcessDataCollection(10000, inputDataSet);
            ProcessData[] trainDataSet;
            ProcessData[] testDataSet;
            (trainDataSet, testDataSet) = dataProcessing.DivideCollection(0.7, processDataSet);
            trainDataSet = trainDataSet.Take(6000).ToArray();
            //Model avilaFNN = new Model(new CrossEntropyLoss(),
            //    new FullConnectLayer(new Direct(), 10, 10, true, new Normal()),
            //    new FullConnectLayer(new Sigmoid(), 10, 12, false, new Normal()),
            //    new FullConnectLayer(new LeakyReLU(0.1), 12, 16, false, new Normal()),
            //    new SoftMaxLayer(16, 12, new Normal())
            //    );
            Model avilaFNN = new Model(new CrossEntropyLoss(),"AvilaFNN",
                new FullConnectLayer(new Direct(1), 10, 10, true, new Normal()),
                new FullConnectLayer(new Sigmoid(), 10, 16, false, new Normal(0, 2)),
                new SoftMaxLayer(16, 12, new Normal(0, 5))
                );
            Trainer trainer1 = new Trainer(100, false, 0, new ExponentialDelayOptimizer(1, 0.9, 0.1));
            //Trainer trainer2 = new Trainer(200, false, 0, new ExponentialDelayOptimizer(2, 0.95, 0.08));
            avilaFNN = (Model)trainer1.Train(avilaFNN, trainDataSet);
            trainDataSet = trainDataSet.SelectPermutation().ToArray();
            //Console.WriteLine(avilaFNN.layers[2].Weight[0,0]);
            Console.WriteLine(evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            avilaFNN = (Model)trainer1.Train(avilaFNN, trainDataSet);
            //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            Console.WriteLine(evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            //for (int i = 0; i <= 49; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToList();
            //    Trainer trainer2 = new Trainer(60, false, 0, new ExponentialDelayOptimizer(2, 0.95, 0.05));
            //    avilaFNN = (Model)trainer2.Train(avilaFNN, trainDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    Console.WriteLine($"第1轮第{i}次循环:"+evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            //}
            //for (int i = 0; i <= 49; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToList();
            //    Trainer trainer2 = new Trainer(200, false, 0, new ExponentialDelayOptimizer(5, 0.9, 0.05));
            //    avilaFNN = (Model)trainer2.Train(avilaFNN, trainDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    Console.WriteLine($"第2轮第{i}次循环:"+ evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            //}
            //for (int i = 0; i <= 1; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToList();
            //    Trainer trainer2 = new Trainer(600, false, 0, new ExponentialDelayOptimizer(10, 0.9, 0.02));
            //    avilaFNN = (Model)trainer2.Train(avilaFNN, trainDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    Console.WriteLine(evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            //}
            //for (int i = 0; i <= 99; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToList();
            //    Trainer trainer2 = new Trainer(200, false, 0, new ExponentialDelayOptimizer(5, 0.95, 0.02));
            //    avilaFNN = (Model)trainer2.Train(avilaFNN, trainDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    Console.WriteLine($"第3轮第{i}次循环:"+evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            //}
            //for (int i = 0; i <= 1; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToList();
            //    Trainer trainer2 = new Trainer(600, false, 0, new ExponentialDelayOptimizer(10, 0.9, 0.02));
            //    avilaFNN = (Model)trainer2.Train(avilaFNN, trainDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    Console.WriteLine(evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            //}
            //for (int i = 0; i <= 149; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToList();
            //    Trainer trainer2 = new Trainer(100, false, 0, new ExponentialDelayOptimizer(1, 0.98, 0.01));
            //    avilaFNN = (Model)trainer2.Train(avilaFNN, trainDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    Console.WriteLine($"第4轮第{i}次循环:"+evaluater.Evaluate(testDataSet, avilaFNN, new AvilaClassifier()));
            //}
            evaluater.EvaluateInDetails(testDataSet, avilaFNN, new AvilaClassifier());
        }
    }
}
