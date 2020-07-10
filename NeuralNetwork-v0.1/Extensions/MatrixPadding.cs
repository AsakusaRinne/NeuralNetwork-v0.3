using MathNet.Numerics.LinearAlgebra.Complex;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Extensions
{
    public static class MatrixPaddingExtension
    {
        /// <summary>
        /// 对矩阵进行边缘填充
        /// </summary>
        /// <param name="source"></param>
        /// <param name="up"></param>
        /// <param name="down"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static Matrix<double> Pad(this Matrix<double> source,int up,int down,int left,int right,int value = 0)
        {
            Matrix<double> newMatrix = source.Clone();
            if (up != 0)
            {
                for(int i = 0; i <= up - 1; i++)
                {
                    newMatrix = newMatrix.InsertRow(0, Vector<double>.Build.Dense(newMatrix.ColumnCount, value));
                }
            }
            if (down != 0)
            {
                for (int i = 0; i <= down - 1; i++)
                {
                    newMatrix = newMatrix.InsertRow(newMatrix.RowCount, Vector<double>.Build.Dense(newMatrix.ColumnCount, value));
                }
            }
            if (left != 0)
            {
                for (int i = 0; i <= left - 1; i++)
                {
                    newMatrix = newMatrix.InsertColumn(0, Vector<double>.Build.Dense(newMatrix.RowCount, value));
                }
            }
            if (right != 0)
            {
                for (int i = 0; i <= right - 1; i++)
                {
                    newMatrix = newMatrix.InsertColumn(newMatrix.ColumnCount , Vector<double>.Build.Dense(newMatrix.RowCount, value));
                }
            }
            return newMatrix;
        }
        /// <summary>
        /// 对一个矩阵四周均匀填充
        /// </summary>
        /// <param name="source"></param>
        /// <param name="count"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static Matrix<double> PadAround(this Matrix<double> source, int count, int value = 0)
        {
            Matrix<double> newMatrix = source.Clone();
            if (count != 0)
            {
                for (int i = 0; i <= count - 1; i++)
                {
                    newMatrix = newMatrix.InsertRow(0, Vector<double>.Build.Dense(newMatrix.ColumnCount, value));
                }
                for (int i = 0; i <= count - 1; i++)
                {
                    newMatrix = newMatrix.InsertRow(newMatrix.RowCount , Vector<double>.Build.Dense(newMatrix.ColumnCount, value));
                }
                for (int i = 0; i <= count - 1; i++)
                {
                    newMatrix = newMatrix.InsertColumn(0, Vector<double>.Build.Dense(newMatrix.RowCount, value));
                }
                for (int i = 0; i <= count - 1; i++)
                {
                    newMatrix = newMatrix.InsertColumn(newMatrix.ColumnCount , Vector<double>.Build.Dense(newMatrix.RowCount, value));
                }
            }
            return newMatrix;
        }

    }
}
