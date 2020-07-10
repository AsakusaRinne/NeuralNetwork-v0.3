using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NeuralNetwork.Functions.Activation;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Struct;
using tb= NeuralNetwork.Struct.TensorBuilder;
using NeuralNetwork.Extensions;
using System.Diagnostics;
using NeuralNetwork;
using NeuralNetwork.Layers;
using NeuralNetwork.Tests.avila;
using NeuralNetwork.Data;
using NeuralNetwork.Models;
using NeuralNetwork.Functions;
using MathNet.Numerics.Distributions;
using NeuralNetwork.Train;
using NeuralNetwork.Optimize.LearnRate;
using MathNet.Numerics;
using NeuralNetwork.Tests.MNIST;

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
            List<ProcessData> testDataSet1;
            List<ProcessData> testDataSet2;
            trainDataSet = processDataSet.ToArray();//训练集
            ILookup<double, ProcessData> dataDistribution = trainDataSet.ToLookup(r => MNISTDataConverter.GetFromOnehot(r.Label[0, 0].ToRowMajorArray()));
            testDataSet1 = dataProcessing.GetProcessDataCollection(5000, dataProcessing.GetData(testDataPath, testLabelPath, 5000));//测试集1
            testDataSet2 = dataProcessing.GetProcessDataCollection(10000, dataProcessing.GetData(testDataPath, testLabelPath, 10000));//测试集2
            listBox1.Items.Add(DateTime.Now + " : DataPreparation Completed.");

            //模型建立部分
            Model avilaFNN = new Model(new CrossEntropyLoss(),
                new FullConnectLayer(new Direct(), 784, 784, true, new Normal()),
                //new FullConnectLayer(new Sigmoid(), 784, 256, false, new Normal(0, 3)),
                new FullConnectLayer(new Sigmoid(), 784, 256, false, new Normal(0, 2)),
                new SoftMaxLayer(256, 10, new Normal(0, 5))
                );
            listBox1.Items.Add(DateTime.Now + " : ModelBuild Completed.");

            //模型训练部分
            Trainer trainer1 = new Trainer(100, false, 0, new ExponentialDelayOptimizer(1, 0.9, 0.1));
            avilaFNN = (Model)trainer1.Train(avilaFNN, trainDataSet);
            trainDataSet = trainDataSet.SelectPermutation().ToArray();
            //Console.WriteLine(evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            listBox1.Items.Add(DateTime.Now + " : 1st Training Completed, Accuracy: " + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            avilaFNN = (Model)trainer1.Train(avilaFNN, trainDataSet);
            //Console.WriteLine(evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            listBox1.Items.Add(DateTime.Now + " : 2nd Training Completed, Accuracy: " + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            for (int i = 0; i <= 1; i++)
            {
                trainDataSet = trainDataSet.SelectPermutation().ToArray();
                Trainer trainer2 = new Trainer(200, false, 0, new ExponentialDelayOptimizer(2, 0.95, 0.05));
                avilaFNN = (Model)trainer2.Train(avilaFNN, trainDataSet);
                //Console.WriteLine(avilaFNN.layers[2].Weight[0, 0]);
                //Console.WriteLine($"第1轮第{i}次循环:" + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
                listBox1.Items.Add(DateTime.Now + $" : {i + 3}th Training Completed, Accuracy: " + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));
            }
            trainDataSet = trainDataSet.SelectPermutation().ToArray();
            Trainer trainer3 = new Trainer(500, false, 0, new ExponentialDelayOptimizer(1, 0.9, 0.05));
            avilaFNN = (Model)trainer3.Train(avilaFNN, trainDataSet);
            listBox1.Items.Add(DateTime.Now + " : 5th Training Completed, Accuracy: " + evaluater.Evaluate(testDataSet1, avilaFNN, new AvilaClassifier()));

            //结果评估部分
            evaluater.EvaluateInDetails(testDataSet1, avilaFNN);
            Console.WriteLine("======================");
            Console.WriteLine("-----Difficult Test-----");
            evaluater.EvaluateInDetails(testDataSet2, avilaFNN);
        }
    }

   
}
