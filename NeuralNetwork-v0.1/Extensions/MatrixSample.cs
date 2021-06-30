using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Extensions
{
    public static class MatrixSampleExtension
    {
        /// <summary>
        /// 最大池化
        /// </summary>
        /// <param name="pm">被池化的矩阵</param>
        /// <returns>第一个为池化后矩阵，第二个为最大池化的掩码表</returns>
        public static Tuple<Matrix<double>, Matrix<double>> MaxPooling(this Matrix<double> pooledMatrix, int poolSize)
        {
            var pm = pooledMatrix.Clone();
            //创建一个掩码矩阵，该矩阵中被最大池化的点将会被赋值为1，其余为0
            Matrix<double> mask = Matrix<double>.Build.Dense(pooledMatrix.RowCount, pooledMatrix.ColumnCount, 0);
            //如果大小不能被池化大小所整除，就先补无穷小
            if (pm.RowCount % poolSize != 0)
            {
                for (int i = 0; i <= poolSize - pooledMatrix.RowCount % poolSize - 1; i++)
                {
                    pm = pm.InsertRow(pm.RowCount, Vector<double>.Build.Dense(pm.ColumnCount, double.MinValue));
                }
            }
            if (pm.ColumnCount % poolSize != 0)
            {
                for (int i = 0; i <= poolSize - pooledMatrix.ColumnCount % poolSize - 1; i++)
                {
                    pm = pm.InsertColumn(pm.ColumnCount, Vector<double>.Build.Dense(pm.RowCount, double.MinValue));
                }
            }

            //进行池化
            Matrix<double> result = Matrix<double>.Build.Dense(pm.RowCount / poolSize, pm.ColumnCount / poolSize);
            for (int i = 0; i <= pm.RowCount - 1; i = i + poolSize)
            {
                for (int j = 0; j <= pm.ColumnCount - 1; j = j + poolSize)
                {
                    //切割矩阵，得到一个poolsize*poolsize大小的矩阵进行最大值提取
                    var temp = pm.SubMatrix(i, poolSize, j, poolSize);
                    var maxValue = temp.ToColumnMajorArray().Max();
                    //寻找temp矩阵中的最大值及其下标
                    var info = temp.Find(r => r == maxValue);
                    //生成池化后的矩阵
                    result[i / poolSize, j / poolSize] = info.Item3;
                    //建立池化掩码表
                    mask[i + info.Item1, j + info.Item2] = 1;
                }
            }
            return new Tuple<Matrix<double>, Matrix<double>>(result, mask);
            //return new Tuple<int, int, Matrix<double>, Dictionary<int[], int[]>>(originalrowcount, originalcolumncount, result, poolmap);
        }
        /// <summary>
        /// 平均池化
        /// </summary>
        /// <param name="pooledMatrix">被池化的矩阵</param>
        /// <returns></returns>
        public static Matrix<double> AveragePooling(this Matrix<double> pooledMatrix, int poolSize)
        {
            var pm = pooledMatrix.Clone();

            //如果大小不能被池化大小所整除，就先补无穷小
            if (pm.RowCount % poolSize != 0)
            {
                for (int i = 0; i <= poolSize - pooledMatrix.RowCount % poolSize - 1; i++)
                {
                    pm = pm.InsertRow(pm.RowCount, Vector<double>.Build.Dense(pm.ColumnCount, 0));
                }
            }
            if (pm.ColumnCount % poolSize != 0)
            {
                for (int i = 0; i <= poolSize - pooledMatrix.ColumnCount % poolSize - 1; i++)
                {
                    pm = pm.InsertColumn(pm.ColumnCount, Vector<double>.Build.Dense(pm.RowCount, 0));
                }
            }

            //进行池化
            Matrix<double> result = Matrix<double>.Build.Dense(pm.RowCount / poolSize, pm.ColumnCount / poolSize);
            for (int i = 0; i <= pm.RowCount - 1; i = i + poolSize)
            {
                for (int j = 0; j <= pm.ColumnCount - 1; j = j + poolSize)
                {
                    //切割矩阵，得到一个poolsize*poolsize大小的矩阵进行平均值提取
                    var temp = pm.SubMatrix(i, poolSize, j, poolSize).Storage.AsColumnMajorArray().Average();
                    //生成池化后的矩阵
                    result[i / poolSize, j / poolSize] = temp;
                }
            }
            return result;
        }
        /// <summary>
        /// 最大池化的上采样
        /// </summary>
        /// <param name="sampledMatrix"></param>
        /// <param name="mask">掩码矩阵</param>
        /// <param name="poolSize">池化大小</param>
        /// <returns></returns>
        public static Matrix<double> MaxUpSample(this Matrix<double> sampledMatrix, Matrix<double> mask, int poolSize)
        {
            if (sampledMatrix.RowCount * poolSize == mask.RowCount && sampledMatrix.ColumnCount * poolSize == mask.ColumnCount)
            {
                return mask.PointwiseMultiply(sampledMatrix.Expand(poolSize));
            }
            else
            {
                return mask.PointwiseMultiply(sampledMatrix.Expand(poolSize).SubMatrix(0, mask.RowCount, 0, mask.ColumnCount));
            }
        }
        /// <summary>
        /// 平均池化的上采样
        /// </summary>
        /// <param name="sampledMatrix"></param>
        /// <param name="poolSize">池化大小</param>
        /// <param name="destinyRowCount">目标矩阵（池化前矩阵）的行数</param>
        /// <param name="destinyColumnCount">目标矩阵（池化前矩阵）的列数</param>
        /// <returns></returns>
        public static Matrix<double> AverageUpSample(this Matrix<double> sampledMatrix, int poolSize, int destinyRowCount, int destinyColumnCount)
        {
            if (sampledMatrix.RowCount * poolSize == destinyRowCount && sampledMatrix.ColumnCount * poolSize == destinyColumnCount)
            {
                return sampledMatrix.Expand(poolSize);
            }
            else
            {
                return sampledMatrix.Expand(poolSize).SubMatrix(0, destinyRowCount, 0, destinyColumnCount);
            }
        }

        /// <summary>
        /// 扩展一个矩阵到原来的某一个倍数大小
        /// </summary>
        /// <param name="source"></param>
        /// <param name="multiple">倍数</param>
        /// <returns></returns>
        public static Matrix<double> Expand(this Matrix<double> source, int multiple)
        {
            var result = Matrix<double>.Build.Dense(source.RowCount * multiple, source.ColumnCount * multiple);
            for (int i = 0; i <= source.RowCount - 1; i++)
            {
                for (int j = 0; j <= source.ColumnCount - 1; j++)
                {
                    for (int p = i * multiple; p <= i * multiple + multiple - 1; p++)
                    {
                        for (int q = j * multiple; q <= j * multiple + multiple - 1; q++)
                        {
                            result[p, q] = source[i, j];
                        }
                    }
                }
            }
            return result;
        }
    }
}
namespace NeuralNetwork
{
    public enum PoolingMode
    {
        Max = 0,
        Average = 1
    }
}
