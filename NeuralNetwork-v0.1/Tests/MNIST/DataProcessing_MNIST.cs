using NeuralNetwork.Data;
using NeuralNetwork.Evaluate;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Struct;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Models;
using System.Text.RegularExpressions;
using System.IO;
using System.Windows.Forms;
using MathNet.Numerics;

namespace NeuralNetwork.Tests.MNIST
{
    class MNISTDataConverter : IDataConvert<double[], double>
    {
        IClassifier classifier;
        public MNISTDataConverter(IClassifier classifier)
        {
            this.classifier = classifier;

        }
        public ProcessData ConvertInputToProcess(InputData<double[], double> inputData)
        {
            return new ProcessData()
            {
                Data = TensorBuilder.FromMatrix(Matrix<double>.Build.DenseOfRowMajor(784, 1, inputData.OriginalData)),
                Label = TensorBuilder.FromMatrix(Matrix<double>.Build.Dense(10, 1, ConvertToOnehot(inputData.OriginalLabel, 9)))
            };
        }

        public OutputData<double[], double> ConvertProcessToOutput(ProcessData processData, IModel model)
        {
            return new OutputData<double[], double>()
            {
                OriginalData = processData.Data[0, 0].ToRowMajorArray(),
                RealResult = GetFromOnehot(processData.Label[0, 0].ToRowMajorArray()),
                PredictedResult = GetFromOnehot(classifier.Classify(model.Predict(processData.Data))[0, 0].ToRowMajorArray())
            };
        }

        public static double[] ConvertToOnehot(double originalData, int maxValue)
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

        public static double GetFromOnehot(double[] oneHot)
        {
            double result = -1;
            for (int i = 0; i <= oneHot.Length - 1; i++)
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

    class MNISTDataProcessing
    {
        const int width = 28;
        const int height = 28;
        IDataConvert<double[], double> dataConverter;
        public MNISTDataProcessing(IDataConvert<double[], double> dataConverter)
        {
            this.dataConverter = dataConverter;
        }
        /// <summary>
        /// 加载图像数据信息
        /// </summary>
        /// <param name="path"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        public List<double[]> LoadData(string path, int count)
        {
            List<double[]> datalist = new List<double[]>();
            byte[] data = new byte[width * height];
            FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            BinaryReader reader = new BinaryReader(fs);
            for (int i = 0; i <= 3; i++)
            {
                Console.WriteLine(reader.ReadInt32());
            }
            for (int i = 0; i <= count - 1; i++)
            {
                data = reader.ReadBytes(width * height);
                var temp = ConvertByteArrayToDouble(data);
                for(int j = 0; j <= temp.Length - 1; j++)
                {
                    temp[j] = temp[j] / 255;
                }
                datalist.Add(temp);
            }
            return datalist;
        }
        /// <summary>
        /// 加载图像标签信息
        /// </summary>
        /// <param name="path"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        public List<double> LoadLabel(string path, int count)
        {
            List<double> datalist = new List<double>();
            byte data;
            FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            BinaryReader reader = new BinaryReader(fs);
            for (int i = 0; i <= 1; i++)
            {
                Console.WriteLine(reader.ReadInt32());
            }
            for (int i = 0; i <= count - 1; i++)
            {
                data = reader.ReadByte();
                datalist.Add(data);
            }
            return datalist;
        }
        /// <summary>
        /// 将byte数组转为double数组
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static double[] ConvertByteArrayToDouble(byte[] input)
        {
            double[] result = new double[input.Length];
            for (int i = 0; i <= input.Length - 1; i++)
            {
                result[i] = (double)input[i];
            }
            return result;
        }
        public List<InputData<double[],double>> GetData(string dataPath,string labelPath,int count)
        {
            List<InputData<double[], double>> collection = new List<InputData<double[], double>>();
            List<double[]> data = LoadData(dataPath, count);
            double[] label = LoadLabel(labelPath, count).ToArray();
            if (data.Count != label.Count())
            {
                throw new Exception("数据和标签数量不一致");
            }
            else
            {
                //label = MinMaxNormalization(label, 0, 10);
                for (int i = 0; i <= data.Count - 1; i++)
                {
                    collection.Add(new InputData<double[], double>()
                    {
                        OriginalData = data[i],
                        OriginalLabel = label[i]
                    });
                }
                return collection;
            }
        }

        public Tuple<List<ProcessData>, List<ProcessData>> DivideCollection(double rate, List<ProcessData> dataCollection)
        {
            if (rate < 0 || rate > 1)
            {
                throw new Exception("分割比例要在0-1之间");
            }
            List<ProcessData> retainedCollection = new List<ProcessData>();
            List<ProcessData> dividedCollection = new List<ProcessData>();
            ILookup<Tensor, ProcessData> DataLookup = dataCollection.ToLookup<ProcessData, Tensor, ProcessData>(r => r.Label, r => r, comparer: new TensorCompareByValue());
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
            return new Tuple<List<ProcessData>, List<ProcessData>>(retainedCollection, dividedCollection);
        }

        public List<ProcessData> GetProcessDataCollection(int count, List<InputData<double[], double>> inputDataCollection)
        {
            if (count > inputDataCollection.Count)
            {
                throw new Exception("超出数据集数量上限");
            }
            List<ProcessData> processDataCollection = new List<ProcessData>();
            //随机抽取，避免分配不均
            var tempList = inputDataCollection.SelectCombination(count);
            foreach (var item in tempList)
            {
                processDataCollection.Add(dataConverter.ConvertInputToProcess(item));
            }
            return processDataCollection;
        }
    }

    class MNISTClassifier : IClassifier
    {
        public Tensor Classify(Tensor predictedData)
        {
            return predictedData.Map(r => r < predictedData[0, 0].ToRowMajorArray().Max() ? 0 : 1);
        }
    }

    class MNISTEvaluater : IEvaluater
    {
        IDataConvert<double[], double> converter;
        public MNISTEvaluater(IDataConvert<double[], double> converter)
        {
            this.converter = converter;
        }
        public double Evaluate(IEnumerable<ProcessData> testCollection, IModel model, IClassifier classifier)
        {
            int correct = 0;
            int wrong = 0;
            foreach (var item in testCollection)
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
        public void EvaluateInDetails(IEnumerable<ProcessData> testCollection, IModel model)
        {
            int correct = 0;
            int wrong = 0;
            int i = 0;
            foreach (var item in testCollection)
            {
                i++;
                var prediction = converter.ConvertProcessToOutput(item, model);
                if (prediction.PredictedResult == prediction.RealResult)
                {
                    correct++;
                }
                else
                {
                    wrong++;
                }
                Console.WriteLine($"{i}：Prediction is{prediction.PredictedResult}，Truth is {prediction.RealResult}，correct:{correct},wrong:{wrong} ");
            }
            Console.WriteLine(((double)correct) / (double)(correct + wrong));
        }
    }
}
