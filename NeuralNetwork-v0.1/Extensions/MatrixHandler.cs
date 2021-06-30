using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Extensions
{
    public static class MatrixHandlerExtension
    {
        public static Matrix<double> Rot180(this Matrix<double> source)
        {
            IEnumerable<double> temp = source.ToColumnMajorArray().Reverse();
            return Matrix<double>.Build.DenseOfColumnMajor(source.RowCount, source.ColumnCount, temp);
        }

        public static Matrix<double> HorizontalReverse(this Matrix<double> source)
        {
            Matrix<double> r = Matrix<double>.Build.DenseOfMatrix(source);
            double temp;
            for(int i = 0; i <= r.RowCount - 1; i++)
            {
                for(int j = 0; j <= (r.ColumnCount - 1) / 2; j++)
                {
                    temp = r[i, j];
                    r[i, j] = r[i, r.ColumnCount - j - 1];
                    r[i, r.ColumnCount - j - 1] = temp;
                }
            }
            return r;
        }

    }
}
