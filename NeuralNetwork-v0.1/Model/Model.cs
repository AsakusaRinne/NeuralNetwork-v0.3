using NeuralNetwork.Data;
using NeuralNetwork.Functions;
using NeuralNetwork.Layers;
using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System.Runtime;
using System.Diagnostics;

namespace NeuralNetwork.Models
{
    class Model:IModel,IModelSave
    {
        public LayerBase[] layers;
        public int layerCount;
        public LossFunction LossFunc;
        string name;

        public Model(LossFunction lossFunction,params LayerBase[] layers)
        {
            this.layers = layers;
            this.layerCount = layers.Length;
            this.LossFunc = lossFunction;
        }

        public LayerBase this[int index]
        {
            get
            {
                return this.layers[index];
            }
            set
            {
                this.layers[index] = value;
            }
        }

        public LayerBase[] GetLayers()
        {
            return this.layers;
        }

        public int Count
        {
            get
            {
                return this.layerCount;
            }
        }

        public LayerBase[] Layers
        {
            get
            {
                return layers;
            }
            set
            {
                this.layers = value;
            }
        }

        public Tensor Predict(Tensor inputData)
        {
            Tensor result = inputData.Clone();
            for(int i = 0; i <= layers.Length - 1; i++)
            {
                result = layers[i].Push(result);
            }
            return result;
        }
        /// <summary>
        /// 获取对某一对数据的最终的误差
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        //public double GetFinalLoss(ProcessData data)
        //{
        //    return this.LossFunc.Loss(data.Label, this.Predict(data.Data)).SumAll();
        //}
        /// <summary>
        /// 对于某一对数据，获取其在某个下标对应的层的输出
        /// </summary>
        /// <param name="index"></param>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public Tensor GetOutputAt(int index,Tensor inputData)
        {
            Tensor result = inputData.Clone();
            for (int i = 0; i <= index; i++)
            {
                result = layers[i].Push(result);
            }
            return result;
        }
        /// <summary>
        /// 获取一个列表，包含对于一组数据每一层的输出
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public List<Tensor> GetAllOutputs(Tensor inputData)
        {
            List<Tensor> outputList = new List<Tensor>();
            Tensor data = inputData.Clone();
            for (int i = 0; i <= layers.Length - 1; i++)
            {
                data = layers[i].Push(data);
                outputList.Add(data);
            }
            return outputList;
        }
        /// <summary>
        /// 获取梯度列表，同时为了简化运算同时输出损失列表
        /// </summary>
        /// <param name="data"></param>
        /// <returns>第一个为梯度列表，第二个为损失列表</returns>
        public Tuple<Tensor[],Tensor[]> GetGradientList(ProcessData data)
        {
            Tensor[] gradientList = new Tensor[this.Count];
            Tensor[] lossList = new Tensor[this.Count];
            List<Tensor> pushes = this.GetAllOutputs(data.Data);
            Tensor loss;
            //这里对softmax层进行了特别对待处理，应该如何统一化？
            if (layers[layerCount - 1].GetType() == typeof(SoftMaxLayer))//反射是否开销太大了？
            {
                loss = data.Label-pushes[this.layerCount - 1];
            }
            else
            {
                loss = LossFunc.Loss(data.Label, pushes[this.layerCount - 1]);
            }
            Tensor nextWeight = TensorBuilder.FromMatrix(Matrix<double>.Build.DenseIdentity(data.Label[0, 0].RowCount));
            for (int i= this.layerCount - 1; i>=2; i--)
            {
                (gradientList[i],loss)=layers[i].GetGradient(loss, pushes[i - 1], nextWeight);
                lossList[i] = loss;
                //loss = layers[i].ComputeLoss(loss, pushes[i - 1], nextWeight);
                nextWeight = layers[i].Weight;
            }
            (gradientList[1],loss)=layers[1].GetGradient(loss, pushes[0], nextWeight);
            lossList[1] = loss;
            //loss = layers[1].ComputeLoss(loss, pushes[0], nextWeight);
            nextWeight = layers[1].Weight;
            (gradientList[0],lossList[0])=layers[0].GetGradient(loss, data.Data, nextWeight);
            return new Tuple<Tensor[], Tensor[]>(gradientList, lossList);
        }

        public List<Tensor> GetLossList(ProcessData data)
        {
            Tensor[] lossList = new Tensor[this.Count];
            List<Tensor> pushes = this.GetAllOutputs(data.Data);
            Tensor loss;
            //这里对softmax层进行了特别对待处理，应该如何统一化？
            if (layers[layerCount - 1].GetType() == typeof(SoftMaxLayer))
            {
                loss = data.Label - pushes[this.layerCount - 1];
            }
            else
            {
                loss = LossFunc.Loss(data.Label, pushes[this.layerCount - 1]); ;
            }
            Tensor nextWeight = TensorBuilder.FromMatrix(Matrix<double>.Build.DenseIdentity(data.Label[0, 0].ColumnCount));
            for (int i = this.layerCount - 1; i >= 2; i--)
            {
                loss = layers[i].ComputeLoss(loss, pushes[i - 1], nextWeight);
                lossList[i]=loss;
                nextWeight = layers[i].Weight;
            }
            loss = layers[1].ComputeLoss(loss, pushes[0], nextWeight);
            lossList[1]=loss;
            nextWeight = layers[1].Weight;
            loss = layers[0].ComputeLoss(loss, data.Data, nextWeight);
            lossList[0]=loss;
            return lossList.ToList();
        }

        public double[] GetWeightedGradient(Tensor[] gradientList)
        {
            double[] result = new double[this.layerCount];
            for (int i= 0; i <= layerCount - 1; i++)
            {
                //if (!layers[i].Weight.IsSameSize())
                //{
                //    throw new Exception("目前版本暂时不接受内部维度不同的张量");
                //}
                //result[i] = gradientList[i].AbsoluteSumAll() / (layers[i].Weight.DimensionX * layers[i].Weight.DimensionY * layers[i].Weight[0, 0].ColumnCount * layers[i].Weight[0, 0].RowCount);
                result[i] = gradientList[i].AbsoluteMean();
            }
            return result;
        }

        public virtual void SaveModel(string path = null, double accuracy = -1)
        {
            if (path == null)
            {
                path = System.IO.Path.Combine(Application.StartupPath, "Models");
            }
            if (Directory.Exists(path) == false)
            {
                Directory.CreateDirectory(path);
            }
            if (accuracy == -1)
            {
                path = System.IO.Path.Combine(Application.StartupPath, "A" + accuracy.ToString("f3"));
            }
            path = System.IO.Path.Combine(path, this.GetType().ToString() + DateTime.Now.ToString("yyyyMMdd-HH"));
            Filehelper.SaveObject(path, this);
            Console.WriteLine("The Model has been save to" + path);
        }

        public virtual T LoadModel<T>(string path)
        {
            return Filehelper.GetObjectFromFile<T>(path);
        }
    }
}
