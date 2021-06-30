using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Extensions;
using MathNet.Numerics.Distributions;

namespace NeuralNetwork.Struct
{
    class TensorBuilder
    {
        /// <summary>
        /// 由一个矩阵生成张量，张量的XY维度的长度均为1
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static Tensor FromMatrix(Matrix<double> source)
        {
            var newStorage = new Matrix<double>[1];
            newStorage[0] = source;
            return new Tensor(1,1,newStorage);
        }
        /// <summary>
        /// 生成外层大小为x，y的张量，所有元素都是相同的矩阵
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static Tensor FromMatrix(int x, int y, Matrix<double> source)
        {
            var newStorage = new Matrix<double>[x*y];
            for (int i = 0; i <= newStorage.Length - 1; i++)
            {
                newStorage[i] = source;
            }
            return new Tensor(x,y,newStorage);
        }
        /// <summary>
        /// 生成外层大小为x，y的张量，根据输入的矩阵数组，按照行优先原则
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="matrices"></param>
        /// <returns></returns>
        public static Tensor FromRowMajorMatrices(int x, int y, Matrix<double>[] matrices)
        {
            if (x * y != matrices.Length)
            {
                throw new Exception("维度设定错误");
            }
            var newStorage = new Matrix<double>[x*y];
            matrices.CopyTo(newStorage, 0);
            return new Tensor(x,y,newStorage);
        }
        /// <summary>
        /// 生成外层大小为x，y的张量，根据输入的矩阵数组，按照列优先原则（开销大，尽量不使用）
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="matrices"></param>
        /// <returns></returns>
        public static Tensor FromColumnMajorMatrices(int x, int y, Matrix<double>[] matrices)
        {
            if (x * y != matrices.Length)
            {
                throw new Exception("维度设定错误");
            }
            var newStorage = new Matrix<double>[x*y];
            for (int i = 0; i <= x - 1; i++)
            {
                for (int j = 0; j <= y - 1; j++)
                {
                    newStorage[i*y+ j] = matrices[j * x + i];
                }
            }
            return new Tensor(x,y,newStorage);
        }
        /// <summary>
        /// 根据二维的矩阵数组生成张量（开销大，尽量不用）
        /// </summary>
        /// <param name="source"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor FromMatrixArray(Matrix<double>[,] source)
        {
            return new Tensor(source.GetLength(0),source.GetLength(1),source.ToRowArray());
        }
        /// <summary>
        /// 由一个特定的随机分布初始化每个矩阵从而生成张量，这个方法中强制保持每个矩阵维度相同
        /// </summary>
        /// <param name="x">外层行数</param>
        /// <param name="y">外层列数</param>
        /// <param name="p">内层行数</param>
        /// <param name="q">内层列数</param>
        /// <param name="distribution">随机分布</param>
        /// <param name="name">张量名称</param>
        /// <returns></returns>
        public static Tensor FromDistribution(int x, int y, int p, int q, IContinuousDistribution distribution)
        {
            Matrix<double>[] storge = new Matrix<double>[x*y];
            for (int i = 0; i <= storge.Length - 1; i++)
            {
                storge[i] = Matrix<double>.Build.Random(p, q, distribution);
            }
            return new Tensor(x,y,storge);
        }
        /// <summary>
        /// 由IEnumerable生成张量，行优先原则
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="source"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor FromRowMajorIEnumerable(int x, int y, IEnumerable<Matrix<double>> source)
        {
            if (x * y != source.Count())
            {
                throw new Exception("维度设定错误");
            }
            return new Tensor(x,y,source.ToArray());
        }
        /// <summary>
        /// 由IEnumerable生成张量，列优先原则，开销大，尽量不要用
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="source"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor FromColumnMajorIEnumerable(int x, int y, IEnumerable<Matrix<double>> source)
        {
            if (x * y != source.Count())
            {
                throw new Exception("维度设定错误");
            }
            var newStorage = new Matrix<double>[x*y];
            for (int i = 0; i <= x - 1; i++)
            {
                for (int j = 0; j <= y - 1; j++)
                {
                    newStorage[i*y+ j] = source.ElementAt(j * x + i);
                }
            }
            return new Tensor(x,y,newStorage);
        }
        /// <summary>
        /// 获取指定大小的全0张量
        /// </summary>
        /// <param name="x">外层行数</param>
        /// <param name="y">外层列数</param>
        /// <param name="p">内层行数</param>
        /// <param name="q">内层列数</param>
        /// <param name="name">新张量的名称</param>
        /// <returns></returns>
        public static Tensor AllZeros(int x, int y, int p, int q)
        {
            Matrix<double>[] newStorage = new Matrix<double>[x* y];
            for (int i = 0; i <= newStorage.Length-1; i++)
            {
                newStorage[i] = Matrix<double>.Build.Dense(p, q, 0);
            }
            return new Tensor(x,y,newStorage);
        }
        /// <summary>
        /// 产生一个跟已有张量形状相同的全0张量
        /// </summary>
        /// <param name="refer">参考张量</param>
        /// <param name="name">新张量的名称</param>
        /// <returns></returns>
        public static Tensor AllZeros(Tensor refer)
        {
            return refer.Map(r => 0.0);
        }
        /// <summary>
        /// 建立一个指定大小的空张量
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor Empty(int x, int y)
        {
            Matrix<double>[] storage = new Matrix<double>[x*y];
            return new Tensor(x,y,storage);
        }
        /// <summary>
        /// 生成数值随机的张量
        /// </summary>
        /// <param name="x">外层行数</param>
        /// <param name="y">外层列数</param>
        /// <param name="p">内层行数</param>
        /// <param name="q">内层列数</param>
        /// <param name="distribution">随机分布</param>
        /// <returns></returns>
        public static Tensor Random(int x,int y,int p,int q,IContinuousDistribution distribution)
        {
            Matrix<double>[] storage = new Matrix<double>[x * y];
            for(int i = 0; i <= storage.Length - 1; i++)
            {
                storage[i] = Matrix<double>.Build.Random(p, q, distribution);
            }
            return new Tensor(x, y, storage);
        }
    }
}
