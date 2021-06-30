using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Context;
using NeuralNetwork.Models;
using NeuralNetwork.Layers;
using NeuralNetwork.Functions;
using NeuralNetwork.Functions.Activation;
using MathNet.Numerics.Distributions;
using NeuralNetwork.Evaluate;
using System.Windows.Forms;
using NeuralNetwork.Data;
using System.IO;
using System.Data;
using MathNet.Numerics;
using NeuralNetwork.Struct;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Train;
using NeuralNetwork.Optimize.LearnRate;

namespace NeuralNetwork.Tests.MNIST
{
    class MNISTDemo1
    {
        public static void TrainModel()
        {
            //模型建立部分
            Model MNIST_CNN = new Model(new CrossEntropyLoss(), "MNIST_CNN_v1",
                new ConvolutionalLayer(new Direct(1), false, new Tuple<int, int>(28, 28), 5, 1, 6, ConvolutionMode.Narrow, new Normal(0, 1)),
                new PoolingLayer(3, false, 6, new Tuple<int, int>(24, 24), PoolingMode.Max),
                new ConvolutionalLayer(new Direct(1), false, new Tuple<int, int>(8, 8), 5, 6, 16, ConvolutionMode.Narrow, new Normal(0, 1)),
                new PoolingLayer(2, true, 16, new Tuple<int, int>(4, 4), PoolingMode.Average),
                new FullConnectLayer(new Direct(1), 64, 64, true, new Normal(0, 1)),
                new FullConnectLayer(new Sigmoid(), 64, 32, false, new Normal(0, 2)),
                new SoftMaxLayer(32, 10, new Normal(0, 2))
                );
            //声明评估器
            MNISTEvaluater evaluater = new MNISTEvaluater(new MNISTDataConverter(new MNISTClassifier()));
            //声明分类器
            IClassifier classifier = new MNISTClassifier();
            //建立环境
            Context.Context context = new Context.Context(MNIST_CNN, evaluater, classifier);
            //准备数据集
            //string trainDataPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "train-images.idx3-ubyte");
            //string trainLabelPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "train-labels.idx1-ubyte");
            //string testDataPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "t10k-images.idx3-ubyte");
            //string testLabelPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "t10k-labels.idx1-ubyte");

            var dataConverter = new MNISTDataConverter(classifier);

            var dataView = new MNISTDataView(dataConverter);
            //dataView.GetInputData()

            context.AddProcess(new Trainer(1000, false, 0, new AdaGradOptimizer(0.5)),10);
            context.AddProcess(new Trainer(600, false, 0, new ExponentialDelayOptimizer(0.2,0.9,0.05)), 10);

            //context.Train(dataView.trainDataSet,dataView.verifyDataSet,)

        }
    }
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
                //Data = TensorBuilder.FromMatrix(Matrix<double>.Build.DenseOfRowMajor(784, 1, inputData.OriginalData)),
                Data = TensorBuilder.FromMatrix(Matrix<double>.Build.DenseOfRowMajor(28, 28, inputData.OriginalData)),
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
    class MNISTDataView : IDataView<double[], double>
    {
        const int width = 28;
        const int height = 28;
        public MNISTDataView(IDataConvert<double[], double> dataConverter)
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
                for (int j = 0; j <= temp.Length - 1; j++)
                {
                    temp[j] = temp[j] / 255;
                }
                datalist.Add(temp);
            }
            reader.Dispose();
            fs.Dispose();
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
            reader.Dispose();
            fs.Dispose();
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
        public override IEnumerable<InputData<double[], double>> GetInputData(string dataPath, string labelPath, int count)
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

        public override IEnumerable<ProcessData> GetProcessData(string dataPath, string labelPath, int count)
        {
            if (count > inputData.Count())
            {
                throw new Exception("超出数据集数量上限");
            }
            List<ProcessData> processDataCollection = new List<ProcessData>();
            //随机抽取，避免分配不均
            var tempList = inputData.SelectCombination(count);
            foreach (var item in tempList)
            {
                processDataCollection.Add(dataConverter.ConvertInputToProcess(item));
            }
            return processDataCollection;
        }
    }
}
