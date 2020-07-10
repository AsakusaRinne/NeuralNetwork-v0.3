using NeuralNetwork.Models;
using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Data
{
    /// <summary>
    /// 初始得到的数据，可以是任意类型
    /// </summary>
    /// <typeparam name="TData"></typeparam>
    /// <typeparam name="TLabel"></typeparam>
    class InputData<TData,TLabel>
    {
        public TData OriginalData;
        public TLabel OriginalLabel;
    }
    /// <summary>
    /// 进行模型训练等中间过程处理的数据，数据和标签都要规范到张量的形式，需要注意label张量中内容应该是行向量（1*x矩阵形式）
    /// </summary>
    class ProcessData
    {
        public Tensor Data;
        public Tensor Label;
    }
    /// <summary>
    /// 进行输出的数据，在一个完整的流程中应当和输入数据的类型一致
    /// </summary>
    /// <typeparam name="TData"></typeparam>
    /// <typeparam name="TLabel"></typeparam>
    class OutputData<TData,TLabel>
    {
        public TData OriginalData;
        public TLabel RealResult;
        public TLabel PredictedResult;
    }
    /// <summary>
    /// 对三种类型数据进行转换的接口
    /// </summary>
    /// <typeparam name="TData"></typeparam>
    /// <typeparam name="TLabel"></typeparam>
    interface IDataConvert<TData, TLabel>
    {
        ProcessData ConvertInputToProcess(InputData<TData, TLabel> inputData);
        OutputData<TData, TLabel> ConvertProcessToOutput(ProcessData processData,IModel model);
    }
}
