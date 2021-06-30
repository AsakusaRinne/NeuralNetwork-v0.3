using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Data;
using System.Data;
using NeuralNetwork.Struct;
using MathNet.Numerics;

namespace NeuralNetwork.Data
{

    abstract class IDataView<TData,TLabel>
    {
        public IDataConvert<TData, TLabel> dataConverter;

        public IEnumerable<InputData<TData, TLabel>> inputData;
        public IEnumerable<ProcessData> trainDataSet;
        public IEnumerable<ProcessData> verifyDataSet;
        public IEnumerable<ProcessData> testDataSet;
        //public IDataView(IDataConvert<TData, TLabel> dataConverter)
        //{
        //    this.dataConverter = dataConverter;
        //}

        public abstract IEnumerable<InputData<TData, TLabel>> GetInputData(string dataPath, string labelPath, int count);

        public abstract IEnumerable<ProcessData> GetProcessData(string dataPath, string labelPath, int count);

        public void DivideDataSet(IEnumerable<ProcessData> dataSet,double trainPart)
        {
            var r1 = DataProcessing.DivideCollection(trainPart, dataSet);
            this.trainDataSet = r1.Item1;
            this.testDataSet = r1.Item2;
        }

        public void DivideDataSet(IEnumerable<ProcessData> dataSet, double trainPart,double verifyPart)
        {
            var r1 = DataProcessing.DivideCollection(trainPart, dataSet);
            var r2 = DataProcessing.DivideCollection(verifyPart, r1.Item1);
            this.trainDataSet = r2.Item1;
            this.verifyDataSet = r2.Item2;
            this.testDataSet = r1.Item2;
        }


    }

    class DataProcessing
    {
        public static Tuple<List<ProcessData>, List<ProcessData>> DivideCollection(double rate, IEnumerable<ProcessData> dataCollection)
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

        public static DataTable ToDataTable(IEnumerable<ProcessData> data, string tableName)
        {
            DataTable table = new DataTable(tableName);
            for (int i = 0; i <= data.ElementAt(0).Data.Storage.Length - 1; i++)
            {
                table.Columns.Add("Feature" + (i + 1).ToString());
            }
            table.Columns.Add("Label");


            foreach (var item in data)
            {
                table.Rows.Add(item.Data, item.Label);
            }

            return table;
        }
        /// <summary>
        /// 对图像进行水平翻转
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        public static ProcessData HorizontalReverseImage(ProcessData image)
        {
            return new ProcessData()
            {
                Data = image.Data.InnerHorizontalReverse(),
                Label = image.Label
            };
        }
    }
}
