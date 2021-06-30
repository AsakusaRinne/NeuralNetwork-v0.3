using NeuralNetwork.Data;
using NeuralNetwork.Layers;
using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Models
{
    interface IModel:IModelSave
    {
        LayerBase this[int index]
        {
            get;
            set;
        }
        int Count
        {
            get;
        }
        LayerBase[] Layers
        {
            get;
            set;
        }
        string Name
        {
            get;
        }
        Tensor Predict(Tensor inputData);
        //double GetFinalLoss(ProcessData data);
        Tensor GetOutputAt(int index,Tensor inputData);
        List<Tensor> GetAllOutputs(Tensor inputData);
        List<Tensor> GetLossList(ProcessData data);
        /// <summary>
        /// 获取梯度列表，同时为了简化运算同时输出损失列表
        /// </summary>
        /// <param name="data"></param>
        /// <returns>第一个为梯度列表，第二个为损失列表</returns>
        Tuple<Tensor[],Tensor[]> GetGradientList(ProcessData data);
        double[] GetWeightedGradient(Tensor[] gradientList);
        Tensor GetFinalLoss(ProcessData data);
    }

    interface IModelSave
    {
        void SaveModel(string path = null, double accuracy = -1);

        T LoadModel<T>(string path);
    }

    enum ModelSaveMode
    {
        Final=0,
        MaxAcc=1,
        All=2,
        MinLoss=3
    }
}
