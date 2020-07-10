using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Extensions
{
    public static class MatrixConvolutionExtension
    {
        public static Matrix<double> Convolve(this Matrix<double> a, Matrix<double> b, int step)
        {
            if (a.RowCount < b.RowCount || a.ColumnCount < b.ColumnCount)
            {
                throw new Exception("卷积核大小必须小于原矩阵大小");
            }
            a = a.Pad(0, (a.RowCount - b.RowCount) % step, 0, (a.ColumnCount - b.ColumnCount) % step);
            Matrix<double> result = Matrix<double>.Build.Dense((a.RowCount - b.RowCount) / step + 1, (a.ColumnCount - b.ColumnCount) / step + 1, 0);
            for (int i = 0; i <= (a.RowCount - b.RowCount) / step; i++)
            {
                for (int j = 0; j <= (a.ColumnCount - b.ColumnCount) / step; j++)
                {
                    //先求hadamard积，然后求和
                    result[i, j] = a.SubMatrix(i * step, b.RowCount, j * step, b.ColumnCount).PointwiseMultiply(b).RowSums().Sum();
                    //for (int k = i; k <= i + b.RowCount-1; k++)
                    //{
                    //    var tempRow = a.Row(k, j, b.ColumnCount);
                    //    result[i, j] = result[i, j] + tempRow * b.Row(k - i);
                    //}
                }
            }
            return result;
        }

        public static Matrix<double> Convolve(this Matrix<double> a, Matrix<double> b, ConvolutionMode mode)
        {
            switch (mode)
            {
                case ConvolutionMode.Narrow:
                    {
                        return a.Convolve(b, 1);
                    }

                case ConvolutionMode.Wide:
                    {
                        Matrix<double> result = Matrix<double>.Build.Dense(a.RowCount + b.RowCount - 1, a.ColumnCount + b.ColumnCount - 1);
                        var aCopy = a.Clone();
                        aCopy = aCopy.Pad(b.RowCount - 1, b.RowCount - 1, b.ColumnCount - 1, b.ColumnCount - 1);
                        for (int i = 0; i <= aCopy.RowCount - b.RowCount; i++)
                        {
                            for (int j = 0; j <= aCopy.ColumnCount - b.ColumnCount; j++)
                            {
                                //先求hadamard积，然后求和
                                result[i, j] = aCopy.SubMatrix(i, b.RowCount, j, b.ColumnCount).PointwiseMultiply(b).RowSums().Sum();
                            }
                        }
                        return result;
                    }

                case ConvolutionMode.Same:
                    {
                        if (b.RowCount % 2 == 0 || b.ColumnCount % 2 == 0)
                        {
                            throw new Exception("等宽卷积的卷积核大小必须为奇数");
                        }
                        Matrix<double> result = Matrix<double>.Build.Dense(a.RowCount , a.ColumnCount );
                        var aCopy = a.Clone();
                        aCopy = aCopy.Pad((b.RowCount - 1)/2, (b.RowCount - 1)/2,( b.ColumnCount - 1)/2, (b.ColumnCount - 1)/2);
                        for (int i = 0; i <= aCopy.RowCount - b.RowCount; i++)
                        {
                            for (int j = 0; j <= aCopy.ColumnCount - b.ColumnCount; j++)
                            {
                                //先求hadamard积，然后求和
                                result[i, j] = aCopy.SubMatrix(i, b.RowCount, j, b.ColumnCount).PointwiseMultiply(b).RowSums().Sum();
                            }
                        }
                        return result;
                    }
                default:
                    {
                        throw new Exception("需要从给定类型中选择");
                    }
            }
        }
    }
}
namespace NeuralNetwork
{
    public enum ConvolutionMode
    {
        Narrow = 0,
        Wide = 1,
        Same = 2
    }
}
