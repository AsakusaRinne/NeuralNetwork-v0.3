using MathNet.Numerics.LinearAlgebra.Complex;
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
    static class TensorBuilder
    {
        /// <summary>
        /// 从已有的张量再次生成一个新的张量，只保留了数值部分到新张量中
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static Tensor FromTensor(Tensor source,string name=null)
        {
            Matrix<double>[,] newStorge = new Matrix<double>[source.DimensionX, source.DimensionY];
            for (int i = 0; i <= source.DimensionX - 1; i++)
            {
                for (int j = 0; j <= source.DimensionY - 1; j++)
                {
                    newStorge[i, j] = source[i, j];
                }
            }
            //Parallel.For(0, source.DimensionX - 1, r =>
            //{
            //    Parallel.For(0, source.DimensionY - 1, q =>
            //     {
            //         newStorge[r, q] = source[r, q];
            //     });
            //});
            return new Tensor(newStorge, name);
        }
        /// <summary>
        /// 从已有张量生成一个新的张量，新张量将拥有指定的形状。如果大于原本张量则补null，小于则截短。原则是行优先。
        /// </summary>
        /// <param name="source"></param>
        /// <param name="dimensionX"></param>
        /// <param name="dimensionY"></param>
        /// <returns></returns>
        public static Tensor FromTensor(Tensor source, int dimensionX, int dimensionY,string name=null)
        {
            Matrix<double>[,] newStorage = new Matrix<double>[dimensionX, dimensionY];
            Matrix<double>[] elements = source.Storage.ToRowArray();
            if (dimensionX * dimensionY > elements.Length)
            {
                for (int i = 0; i <= dimensionX - 1; i++)
                {
                    for (int j = 0; j <= dimensionY - 1; j++)
                    {
                        if (i * dimensionY + j + 1 <= elements.Length)
                        {
                            newStorage[i, j] = elements[i * dimensionY + j];
                        }
                        else
                        {
                            newStorage[i, j] = null;
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i <= dimensionX - 1; i++)
                {
                    for (int j = 0; j <= dimensionY - 1; j++)
                    {
                        newStorage[i,j]=elements[i * dimensionY + j];
                    }
                }
            }
            return new Tensor(newStorage,name);
        }
        /// <summary>
        /// 由一个矩阵生成张量，张量的XY维度的长度均为1
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static Tensor FromMatrix(Matrix<double> matrix,string name=null)
        {
            var newStorage = new Matrix<double>[1,1];
            newStorage[0, 0] = matrix;
            return new Tensor(newStorage,name);
        }
        /// <summary>
        /// 生成外层大小为x，y的张量，所有元素都是相同的矩阵
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static Tensor FromMatrix(int x, int y, Matrix<double> matrix,string name=null)
        {
            var newStorage = new Matrix<double>[x,y];
            for (int i = 0; i <= x - 1; i++)
            {
                for (int j = 0; j <= y - 1; j++)
                {
                    newStorage[i, j] = matrix;
                }
            }
            return new Tensor(newStorage,null);
        }
        /// <summary>
        /// 生成外层大小为x，y的张量，根据输入的矩阵数组，按照行优先原则
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="matrices"></param>
        /// <returns></returns>
        public static Tensor FromRowMajorMatrices(int x, int y, Matrix<double>[] matrices,string name=null)
        {
            if (x * y != matrices.Length)
            {
                throw new Exception("维度设定错误");
            }
            var newStorage = new Matrix<double>[x, y];
            for (int i = 0; i <= x - 1; i++)
            {
                for (int j = 0; j <= y - 1; j++)
                {
                    newStorage[i,j]=matrices[i * y + j];
                }
            }
            return new Tensor(newStorage,name);
        }
        /// <summary>
        /// 生成外层大小为x，y的张量，根据输入的矩阵数组，按照列优先原则
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="matrices"></param>
        /// <returns></returns>
        public static Tensor FromColumnMajorMatrices(int x, int y, Matrix<double>[] matrices, string name = null)
        {
            if (x * y != matrices.Length)
            {
                throw new Exception("维度设定错误");
            }
            var newStorage = new Matrix<double>[x, y];
            for (int i = 0; i <= x - 1; i++)
            {
                for (int j = 0; j <= y - 1; j++)
                {
                    newStorage[i, j] = matrices[j * x + i];
                }
            }
            return new Tensor(newStorage, name);
        }
        /// <summary>
        /// 根据二维的矩阵数组生成张量
        /// </summary>
        /// <param name="source"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor FromMatrixArray(Matrix<double>[,] source,string name=null)
        {
            return new Tensor(source, name);
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
        public static Tensor FromDistribution(int x,int y,int p,int q,IContinuousDistribution distribution,string name=null)
        {
            Matrix<double>[,] storge = new Matrix<double>[x, y];
            for(int i = 0; i <= x - 1; i++)
            {
                for(int j = 0; j <= y - 1; j++)
                {
                    storge[i, j] = Matrix<double>.Build.Random(p, q, distribution);
                }
            }
            return new Tensor(storge,name);
        }
        /// <summary>
        /// 由IEnumerable生成张量，行优先原则
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="source"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor FromRowMajorIEnumerable(int x,int y,IEnumerable<Matrix<double>> source,string name = null)
        {
            if (x * y != source.Count())
            {
                throw new Exception("维度设定错误");
            }
            var newStorage = new Matrix<double>[x, y];
            for (int i = 0; i <= x - 1; i++)
            {
                for (int j = 0; j <= y - 1; j++)
                {
                    newStorage[i, j] =source.ElementAt(i * y + j);
                }
            }
            return new Tensor(newStorage, name);
        }
        /// <summary>
        /// 由IEnumerable生成张量，列优先原则
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="source"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor FromColumnMajorIEnumerable(int x, int y, IEnumerable<Matrix<double>> source, string name = null)
        {
            if (x * y != source.Count())
            {
                throw new Exception("维度设定错误");
            }
            var newStorage = new Matrix<double>[x, y];
            for (int i = 0; i <= x - 1; i++)
            {
                for (int j = 0; j <= y - 1; j++)
                {
                    newStorage[i, j] = source.ElementAt(j * x + i);
                }
            }
            return new Tensor(newStorage, name);
        }
        /// <summary>
        /// 建立一个指定大小的空张量
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor Empty(int x,int y,string name=null)
        {
            Matrix<double>[,] storage = new Matrix<double>[x, y];
            return new Tensor(storage,name);
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
        public static Tensor AllZeros(int x,int y,int p,int q,string name = null)
        {
            Matrix<double>[,] newStorage = new Matrix<double>[x, y];
            for(int i = 0; i <= x - 1; i++)
            {
                for(int j = 0; j <= x - 1; j++)
                {
                    newStorage[i, j] = Matrix<double>.Build.Dense(p, q, 0);
                }
            }
            return new Tensor(newStorage, name);
        }
        /// <summary>
        /// 产生一个跟已有张量形状相同的全0张量
        /// </summary>
        /// <param name="refer">参考张量</param>
        /// <param name="name">新张量的名称</param>
        /// <returns></returns>
        public static Tensor AllZeros(Tensor refer,string name = null)
        {
            return refer.Map(r => 0.0, true, name);
        }
    }
}
