using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Linq;
using System.Windows.Forms;
using NeuralNetwork.Tests.avila;
using NeuralNetwork.Data;
using NeuralNetwork.Models;
using NeuralNetwork.Train;
using NeuralNetwork.Optimize.LearnRate;
using MathNet.Numerics;
using NeuralNetwork.Tests.MNIST;
using System.Threading;
using System.Windows.Forms.DataVisualization.Charting;
using MathNet.Numerics.Distributions;

namespace NeuralNetwork_v0._1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            //var m = Matrix<double>.Build.Random(4, 4);
            //Tensor t = tb.FromRowMajorMatrices(2, 3, new Matrix<double>[] { m, m.Map(r => r * 2), m.Map(r => r + 1), m.Map(r => r - 2), m.MapIndexed((x, y, r) => r * x * y), m.Map(r => 0.0) });
            //Console.WriteLine(m);
            //var result = t.MaxPool(2);
            //var n = Matrix<double>.Build.Dense(2, 2, new double[] { 1, 0, -1, 1 });
            //var p = result.Item1.MaxUpSample(result.Item2, 2);
            //Console.WriteLine(m.MaxPooling(2));
            //Tensor tr = t.Stretch();
            //Tensor tr2 = tr.Fold(2, 3, 4, 4);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            //以下为数据准备部分
            string trainDataPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "train-images.idx3-ubyte");
            string trainLabelPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "train-labels.idx1-ubyte");
            string testDataPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "t10k-images.idx3-ubyte");
            string testLabelPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "t10k-labels.idx1-ubyte");
            MNISTDataProcessing dataProcessing = new MNISTDataProcessing(new MNISTDataConverter(new MNISTClassifier()));
            MNISTEvaluater evaluater = new MNISTEvaluater(new MNISTDataConverter(new MNISTClassifier()));
            var inputDataSet = dataProcessing.GetData(trainDataPath, trainLabelPath, 60000);
            var processDataSet = dataProcessing.GetProcessDataCollection(60000, inputDataSet);
            ProcessData[] trainDataSet;
            processDataSet = processDataSet.Concat(processDataSet.Select(r => dataProcessing.HorizontalReverseImage(r))).ToList();
            List<ProcessData> testDataSet1;
            List<ProcessData> testDataSet2;
            //trainDataSet = processDataSet.ToArray();//训练集
            ProcessData[] verifyDataSet;
            (trainDataSet, verifyDataSet) = dataProcessing.DivideCollection(0.9, processDataSet);
            //ILookup<double, ProcessData> dataDistribution = trainDataSet.ToLookup(r => MNISTDataConverter.GetFromOnehot(r.Label[0, 0].ToRowMajorArray()));
            testDataSet1 = dataProcessing.GetProcessDataCollection(5000, dataProcessing.GetData(testDataPath, testLabelPath, 5000));//测试集1
            testDataSet2 = dataProcessing.GetProcessDataCollection(10000, dataProcessing.GetData(testDataPath, testLabelPath, 10000));//测试集2
            listBox1.Items.Add(DateTime.Now + " : DataPreparation Completed.");

            ////模型建立部分
            //Model avilaFNN = new Model(new CrossEntropyLoss(), "MNIST_LeNet_v1",
            //    //new NormaliztionLayer(trainDataSet, NormalizationMode.Standardization),
            //    new ConvolutionalLayer(new ReLU(), false, new Tuple<int, int>(28, 28), 5, 1, 6, ConvolutionMode.Narrow, new Normal(0, 1)),
            //    new PoolingLayer(3, false, 6, new Tuple<int, int>(24, 24), PoolingMode.Max),
            //    new ConvolutionalLayer(new Direct(0.01), false, new Tuple<int, int>(8, 8), 5, 6, 16, ConvolutionMode.Narrow, new Normal(0, 1)),
            //    new PoolingLayer(2, false, 16, new Tuple<int, int>(4, 4), PoolingMode.Average),
            //    new ConvolutionalLayer(new Direct(0.01), false, new Tuple<int, int>(2, 2), 2, 16, 120, ConvolutionMode.Narrow, new Normal(0, 1)),
            //    new PoolingLayer(1, true, 120, new Tuple<int, int>(1, 1), PoolingMode.Average),
            //    new FullConnectLayer(new Direct(1), 120, 120, true, new Normal(0, 1)),
            //    new FullConnectLayer(new Sigmoid(), 120, 156, false, new Normal(0, 2)),
            //    new SoftMaxLayer(156, 10, new Normal(0, 2))
            //    );


            Model avilaFNN = new Model(null, "1").LoadModel<Model>(@"C:\Users\liu_y\source\repos\CSahrp\NeuralNetwork-v0.3\NeuralNetwork-v0.1\bin\Debug\Models\MNIST_CNN_v6-0.868.model");
            listBox1.Items.Add(DateTime.Now + " : ModelBuild Completed.");
            avilaFNN.layers[5].Locked = false;
            avilaFNN.layers[6].Locked = false;

            double maxAcc = 0.868;
            Trainer trainer2 = new Trainer(1000, false, 0, new ExponentialDelayOptimizer(1, 0.9, 0.1));
            ////Trainer trainer2 = new Trainer(50, false, 0, new TriangularCyclicOptimizer(1500,0.2,0.005,0.98));
            //for (int i = 0; i <= 1; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToArray();

            //    var temp = (Model)trainer2.Train(avilaFNN, trainDataSet, verifyDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    //Console.WriteLine($"第1轮第{i}次循环:" + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            //    Console.WriteLine(DateTime.Now + $" : {i + 1}th Training Completed , Accuracy is:{evaluater.Evaluate(testDataSet2, temp, new AvilaClassifier())}");
            //    var acc = evaluater.EvaluateInDetails(testDataSet2, temp);
            //    if (acc >= maxAcc)
            //    {
            //        avilaFNN = temp;
            //        avilaFNN.SaveModel(accuracy: acc);
            //        maxAcc = acc;
            //    }
            //}
            //chart1.DataSource = trainer2.lossTable;
            //chart1.Series.Add("111");
            //chart1.Series[0].XValueMember = trainer2.lossTable.Columns[0].ColumnName;
            //chart1.Series[1].YValueMembers = trainer2.lossTable.Columns[1].ColumnName;
            //chart1.Series[0].ChartType = SeriesChartType.Line;


            //trainer2 = new Trainer(1000, false, 0, new ExponentialDelayOptimizer(1, 0.9, 0.05));
            //for (int i = 0; i <= 4; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToArray();
            //    var temp = (Model)trainer2.Train(avilaFNN, trainDataSet,verifyDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    //Console.WriteLine($"第1轮第{i}次循环:" + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            //    Console.WriteLine(DateTime.Now + $" : {i + 1}th Training Completed , Accuracy is:{evaluater.Evaluate(testDataSet2, temp, new AvilaClassifier())}");
            //    var acc = evaluater.EvaluateInDetails(testDataSet2, temp);
            //    if (acc >= maxAcc)
            //    {
            //        avilaFNN = temp;
            //        avilaFNN.SaveModel(accuracy: acc);
            //        maxAcc = acc;
            //    }
            //}
            //trainer2 = new Trainer(500, false, 0, new ExponentialDelayOptimizer(0.5, 0.95, 0.05));
            //for (int i = 0; i <= 4; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToArray();

            //    var temp = (Model)trainer2.Train(avilaFNN, trainDataSet,verifyDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    //Console.WriteLine($"第1轮第{i}次循环:" + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            //    Console.WriteLine(DateTime.Now + $" : {i + 1}th Training Completed , Accuracy is:{evaluater.Evaluate(testDataSet2, temp, new AvilaClassifier())}");
            //    var acc = evaluater.EvaluateInDetails(testDataSet2, temp);
            //    if (acc >= maxAcc)
            //    {
            //        avilaFNN = temp;
            //        avilaFNN.SaveModel(accuracy: acc);
            //        maxAcc = acc;
            //    }
            //}
            //trainer2 = new Trainer(1000, false, 0, new AdaGradOptimizer(1));
            //for (int i = 0; i <= 4; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToArray();
            //    var temp = (Model)trainer2.Train(avilaFNN, trainDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    //Console.WriteLine($"第1轮第{i}次循环:" + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            //    Console.WriteLine(DateTime.Now + $" : {i + 1}th Training Completed , Accuracy is:{evaluater.Evaluate(testDataSet2, temp, new AvilaClassifier())}");
            //    var acc = evaluater.EvaluateInDetails(testDataSet2, temp);
            //    avilaFNN = temp;
            //    if (acc >= maxAcc)
            //    {
            //        avilaFNN.SaveModel(accuracy: acc);
            //        maxAcc = acc;
            //    }
            //}
            //trainer2 = new Trainer(1000, false, 0, new ExponentialDelayOptimizer(0.3, 0.9, 0.05));
            //for (int i = 0; i <= 9; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToArray();
            //    var temp = (Model)trainer2.Train(avilaFNN, trainDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    //Console.WriteLine($"第1轮第{i}次循环:" + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            //    Console.WriteLine(DateTime.Now + $" : {i + 1}th Training Completed , Accuracy is:{evaluater.Evaluate(testDataSet2, temp, new AvilaClassifier())}");
            //    var acc = evaluater.EvaluateInDetails(testDataSet2, temp);
            //    avilaFNN = temp; 
            //    {

            //        avilaFNN.SaveModel(accuracy: acc);
            //        maxAcc = acc;
            //    }
            //}
            trainer2 = new Trainer(400, false, 0, new ExponentialDelayOptimizer(0.01, 0.9, 0.05));
            for (int i = 0; i <= 9; i++)
            {
                trainDataSet = trainDataSet.SelectPermutation().ToArray();
                var temp = (Model)trainer2.Train(avilaFNN, trainDataSet);
                //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
                //Console.WriteLine($"第1轮第{i}次循环:" + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
                Console.WriteLine(DateTime.Now + $" : {i + 1}th Training Completed , Accuracy is:{evaluater.Evaluate(testDataSet2, temp, new AvilaClassifier())}");
                var acc = evaluater.EvaluateInDetails(testDataSet2, temp);
                avilaFNN = temp;
                if (acc >= maxAcc)
                {
                    
                    avilaFNN.SaveModel(accuracy: acc);
                    maxAcc = acc;
                }
            }
            trainer2 = new Trainer(300, false, 0, new ExponentialDelayOptimizer(0.005, 0.95, 0.05));
            for (int i = 0; i <= 19; i++)
            {
                trainDataSet = trainDataSet.SelectPermutation().ToArray();
                var temp = (Model)trainer2.Train(avilaFNN, trainDataSet, verifyDataSet);
                //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
                //Console.WriteLine($"第1轮第{i}次循环:" + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
                Console.WriteLine(DateTime.Now + $" : {i + 1}th Training Completed , Accuracy is:{evaluater.Evaluate(testDataSet2, temp, new AvilaClassifier())}");
                var acc = evaluater.EvaluateInDetails(testDataSet2, temp);
                avilaFNN = temp;
                if (acc >= maxAcc)
                {

                    avilaFNN.SaveModel(accuracy: acc);
                    maxAcc = acc;
                }
            }
            trainDataSet = trainDataSet.SelectPermutation().ToArray();
            Trainer trainer4 = new Trainer(100, false, 0, new ExponentialDelayOptimizer(0.1, 0.9, 0.04));
            avilaFNN = (Model)trainer4.Train(avilaFNN, trainDataSet);
            Console.WriteLine(DateTime.Now + $" : 21th Training Completed, Accuracy is:{evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier())}");
            var accuracy = evaluater.EvaluateInDetails(testDataSet2, avilaFNN);
            avilaFNN.SaveModel(accuracy: accuracy);

            ////结果评估部分
            //evaluater.EvaluateInDetails(testDataSet1, avilaFNN);
            //Console.WriteLine("======================");
            //Console.WriteLine("-----Difficult Test-----");
            //var accuracy=evaluater.EvaluateInDetails(testDataSet2, avilaFNN);
            //avilaFNN.SaveModel(accuracy: accuracy);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            //以下为数据准备部分
            string trainDataPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "train-images.idx3-ubyte");
            string trainLabelPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "train-labels.idx1-ubyte");
            string testDataPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "t10k-images.idx3-ubyte");
            string testLabelPath = System.IO.Path.Combine(Application.StartupPath, "MNIST", "t10k-labels.idx1-ubyte");
            MNISTDataProcessing dataProcessing = new MNISTDataProcessing(new MNISTDataConverter(new MNISTClassifier()));
            MNISTEvaluater evaluater = new MNISTEvaluater(new MNISTDataConverter(new MNISTClassifier()));
            var inputDataSet = dataProcessing.GetData(trainDataPath, trainLabelPath, 60000);
            var processDataSet = dataProcessing.GetProcessDataCollection(60000, inputDataSet);
            ProcessData[] trainDataSet;
            List<ProcessData> testDataSet1;
            List<ProcessData> testDataSet2;
            trainDataSet = processDataSet.ToArray();//训练集
            ILookup<double, ProcessData> dataDistribution = trainDataSet.ToLookup(r => MNISTDataConverter.GetFromOnehot(r.Label[0, 0].ToRowMajorArray()));
            testDataSet1 = dataProcessing.GetProcessDataCollection(5000, dataProcessing.GetData(testDataPath, testLabelPath, 5000));//测试集1
            testDataSet2 = dataProcessing.GetProcessDataCollection(10000, dataProcessing.GetData(testDataPath, testLabelPath, 10000));//测试集2
            listBox1.Items.Add(DateTime.Now + " : DataPreparation Completed.");

            Model avilaFNN = new Model(null, "1").LoadModel<Model>(@"C:\Users\liu_y\source\repos\CSahrp\NeuralNetwork-v0.3\NeuralNetwork-v0.1\bin\Debug\Models\MNIST_CNN_v2-0.867.model");
            listBox1.Items.Add(DateTime.Now + " : ModelLoad Completed.");

            ////模型建立部分
            //Model avilaFNN = new Model(new CrossEntropyLoss(), "MNIST_CNN_v1",
            //    new ConvolutionalLayer(new Direct(), false, new Tuple<int, int>(28, 28), 5, 1, 3, ConvolutionMode.Narrow, new Normal(0, 2)),
            //    new PoolingLayer(2, false, 3, new Tuple<int, int>(24, 24), PoolingMode.Max),
            //    new ConvolutionalLayer(new Direct(), false, new Tuple<int, int>(12, 12), 5, 3, 5, ConvolutionMode.Narrow, new Normal(0, 2)),
            //    new PoolingLayer(2, true, 5, new Tuple<int, int>(8, 8), PoolingMode.Max),
            //    new FullConnectLayer(new Direct(), 80, 80, true, new Normal(0, 1)),
            //    new FullConnectLayer(new Sigmoid(), 80, 32, false, new Normal(0, 2)),
            //    new SoftMaxLayer(32, 10, new Normal(0.5, 1))
            //    );

            //Trainer trainer1 = new Trainer(400, false, 0, new ExponentialDelayOptimizer(2, 0.9, 0.08));
            //avilaFNN = (Model)trainer1.Train(avilaFNN, trainDataSet);
            //trainDataSet = trainDataSet.SelectPermutation().ToArray();
            ////Console.WriteLine(evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            //listBox1.Items.Add(DateTime.Now + $" : 1st Training Completed, Accuracy is:{evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier())}");
            //Trainer trainer3 = new Trainer(300, false, 0, new ExponentialDelayOptimizer(1.5, 0.9, 0.05));
            //avilaFNN = (Model)trainer3.Train(avilaFNN, trainDataSet);
            ////Console.WriteLine(evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            //listBox1.Items.Add(DateTime.Now + $" : 2nd Training Completed, Accuracy is:{evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier())}");

            
            //for (int i = 0; i <= 4; i++)
            //{
            //    trainDataSet = trainDataSet.SelectPermutation().ToArray();
            //    //Trainer trainer2 = new Trainer(300, false, 0, new TriangularCyclicOptimizer(1000,1,0.05,0.95));
            //    Trainer trainer2 = new Trainer(200, false, 0, new TriangularCyclicOptimizer(1000,1,0.1,0.95));
            //    avilaFNN = (Model)trainer2.Train(avilaFNN, trainDataSet);
            //    //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
            //    //Console.WriteLine($"第1轮第{i}次循环:" + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            //    listBox1.Items.Add(DateTime.Now + $" : {i + 1}th Training Completed , Accuracy is:{evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier())}");
            //    var acc = evaluater.EvaluateInDetails(testDataSet2, avilaFNN);
            //    avilaFNN.SaveModel(accuracy: acc);
            //}
            for (int i = 0; i <= 14; i++)
            {
                trainDataSet = trainDataSet.SelectPermutation().ToArray();
                //Trainer trainer2 = new Trainer(300, false, 0, new TriangularCyclicOptimizer(1000,1,0.05,0.95));
                Trainer trainer2 = new Trainer(200, false, 0, new ExponentialDelayOptimizer(0.035, 0.9, 0.1));
                avilaFNN = (Model)trainer2.Train(avilaFNN, trainDataSet);
                //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
                //Console.WriteLine($"第1轮第{i}次循环:" + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
                listBox1.Items.Add(DateTime.Now + $" : {i + 1}th Training Completed , Accuracy is:{evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier())}");
                var acc = evaluater.EvaluateInDetails(testDataSet2, avilaFNN);
                avilaFNN.SaveModel(accuracy: acc);
            }
            trainDataSet = trainDataSet.SelectPermutation().ToArray();
            Trainer trainer4 = new Trainer(100, false, 0, new ExponentialDelayOptimizer(0.02, 0.9, 0.1));
            avilaFNN = (Model)trainer4.Train(avilaFNN, trainDataSet);
            listBox1.Items.Add(DateTime.Now + $" : 22th Training Completed, Accuracy is:{evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier())}");

            var accuracy = evaluater.EvaluateInDetails(testDataSet2, avilaFNN);
            avilaFNN.SaveModel(accuracy: accuracy);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            AvilaDemo.train(@"C:\Work\DataSets\avila\avila-tr.txt");
        }
    }

}
