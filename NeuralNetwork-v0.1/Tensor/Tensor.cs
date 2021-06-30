using MathNet.Numerics.LinearAlgebra.Complex;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Runtime.CompilerServices;
using NeuralNetwork.Extensions;
using System.Collections;

namespace NeuralNetwork.Struct
{
    [Serializable]
    public class Tensor:IEquatable<Tensor>,IEnumerable<Matrix<double>>
    {
        protected internal Matrix<double>[] Storage;
        public readonly int DimensionX;
        public readonly int DimensionY;

        private bool Locked = false;

        protected internal Tensor(int x,int y,Matrix<double>[] storage)
        {
            this.Storage = storage;
            DimensionX = x;
            DimensionY = y;
        }

        public Matrix<double> this[int x, int y]
        {
            get
            {
                return Storage[x * DimensionY + y];
            }
            set
            {
                if (!Locked)
                {
                    Storage[x * DimensionY + y] = value;
                }
                else
                {
                    throw new Exception("张量已被锁定");
                }
            }
        }
        public double this[int x,int y,int p,int q]
        {
            get
            {
                return Storage[x*DimensionY+y][p, q];
            }
        }

        public Tensor Clone()
        {
            Matrix<double>[] newStorage = new Matrix<double>[Storage.Length];
            Storage.CopyTo(newStorage, 0);
            return new Tensor(DimensionX, DimensionY, newStorage)
            {
                Locked = this.Locked
            };
        }

        public bool Equals(Tensor other)
        {
            return Enumerable.SequenceEqual(this.Storage, other.Storage);
        }

        public IEnumerator<Matrix<double>> GetEnumerator()
        {
            return ((IEnumerable<Matrix<double>>)Storage).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable<Matrix<double>>)Storage).GetEnumerator();
        }

        /// <summary>
        /// 检测当前张量中每个矩阵是否是相同维度
        /// </summary>
        /// <returns></returns>
        public bool IsSameSize()
        {
            if (Storage[0] == null)
            {
                throw new Exception("张量中有未实例化的矩阵");
            }
            else
            {
                int rc = Storage[0].RowCount;
                int cc = Storage[0].ColumnCount;
                for(int i = 0; i <= Storage.Length - 1; i++)
                {
                    if(Storage[i] == null)
                    {
                        throw new Exception("张量中有未实例化的矩阵");
                    }
                    else if (Storage[i].RowCount != rc || Storage[i].ColumnCount != cc)
                    {
                        return false;
                    }
                }
                return true;
            }
        }

        public void Lock()
        {
            this.Locked = true;
        }
        public void UnLock()
        {
            this.Locked = false;
        }
        public bool IsLocked()
        {
            return Locked;
        }
        /// <summary>
        /// 对张量中的每个矩阵的每个元素执行一个操作
        /// </summary>
        /// <param name="f"></param>
        /// <returns>新的张量</returns>
        public Tensor Map(Func<double,double> f)
        {
            Tensor newTensor = TensorBuilder.Empty(this.DimensionX, this.DimensionY);
            for(int i = 0; i <= Storage.Length - 1; i++)
            {
                newTensor.Storage[i] = this.Storage[i].Map(f);
            }
            return newTensor;
        }
        /// <summary>
        /// 对张量中的每个矩阵按照下标执行操作
        /// </summary>
        /// <param name="f"></param>
        /// <param name="keepName"></param>
        /// <param name="newName"></param>
        /// <returns></returns>
        public Tensor MapIndexed(Func<int, int, double, double> f)
        {
            Tensor newTensor = TensorBuilder.Empty(this.DimensionX, this.DimensionY);
            for (int i = 0; i <= Storage.Length - 1; i++)
            {
                newTensor.Storage[i] = this.Storage[i].MapIndexed(f);
            }
            return newTensor;
        }
        /// <summary>
        /// 对张量中的每个矩阵进行整体操作
        /// </summary>
        /// <param name="f"></param>
        /// <param name="keepName"></param>
        /// <param name="newName"></param>
        /// <returns></returns>
        public Tensor OuterMap(Func<Matrix<double>, Matrix<double>> f)
        {
            Tensor newTensor = TensorBuilder.Empty(this.DimensionX, this.DimensionY);
            for (int i = 0; i <= Storage.Length - 1; i++)
            {
                newTensor.Storage[i] = f(this.Storage[i]);
            }
            newTensor.OuterAct(r => r.CoerceZero(1e-15));
            return newTensor;
        }

        public void OuterAct(Action<Matrix<double>> g)
        {
            for(int i = 0; i <= this.Storage.Length - 1; i++)
            {
                g(this.Storage[i]);
            }
        }
        /// <summary>
        /// 根据下标对张量中的每个矩阵进行整体操作
        /// </summary>
        /// <param name="f"></param>
        /// <param name="keepName"></param>
        /// <param name="newName"></param>
        /// <returns></returns>
        public Tensor OuterMapIndexed(Func<int, int, Matrix<double>, Matrix<double>> f, bool keepName = true, string newName = null)
        {
            Tensor newTensor = TensorBuilder.Empty(this.DimensionX, this.DimensionY);
            for (int i = 0; i <= Storage.Length - 1; i++)
            {
                newTensor.Storage[i] = f(i/DimensionY,i%DimensionY,this.Storage[i]);
            }
            newTensor.OuterAct(r => r.CoerceZero(1e-15));
            return newTensor;
        }

        /// <summary>
        /// 将向量中的每个矩阵进行转置
        /// </summary>
        /// <param name="keepName"></param>
        /// <param name="newName"></param>
        /// <returns></returns>
        public Tensor InnerTranspose()
        {
            return this.OuterMap(r => r.Transpose());
        }

        /// <summary>
        /// 将张量中的每个矩阵转为对角矩阵（必须所有矩阵有一维为1否则报错）
        /// </summary>
        /// <param name="keepName"></param>
        /// <param name="newName"></param>
        /// <returns></returns>
        public Tensor InnerDiagonal()
        {
            Func<Matrix<double>, Matrix<double>> ToDiagonal = r =>
            {
                if (r.RowCount == 1)
                {
                    return Matrix<double>.Build.DenseDiagonal(r.ColumnCount, r.ColumnCount, t => r[0, t]);
                }
                else if (r.ColumnCount == 1)
                {
                    return Matrix<double>.Build.DenseDiagonal(r.RowCount, r.RowCount, t => r[t, 0]);
                }
                else
                {
                    throw new Exception("转对角要求必须是有一维度为1的矩阵");
                }
            };
            return this.OuterMap(ToDiagonal);
        }
        /// <summary>
        /// 对张量中每个矩阵的每个元素相加起来
        /// </summary>
        /// <returns></returns>
        public double SumAll()
        {
            double result = 0;
            for (int i = 0; i <= Storage.Length-1; i++)
            {
                result = result + Storage[i].ColumnSums().Sum();
            }
            return result;
        }
        /// <summary>
        /// 对张量中每个矩阵的每个元素的绝对值相加起来
        /// </summary>
        /// <returns></returns>
        public double AbsoluteSumAll()
        {
            double result = 0;
            for (int i = 0; i <= Storage.Length - 1; i++)
            {
                result = result + Storage[i].ColumnAbsoluteSums().Sum();
            }
            return result;
        }
        /// <summary>
        /// 求平均值
        /// </summary>
        /// <returns></returns>
        public double MeanAll()
        {
            double result = 0;
            for (int i = 0; i <= Storage.Length - 1; i++)
            {
                result = result + Storage[i].Storage.AsColumnMajorArray().Average();
                //result = result + Storage[i].ToColumnMajorArray().Average();
            }
            return result / Storage.Length;
        }
        /// <summary>
        /// 求绝对值平均
        /// </summary>
        /// <returns></returns>
        public double AbsoluteMeanAll()
        {
            double result = 0;
            for (int i = 0; i <= Storage.Length - 1; i++)
            {
                result = result + Storage[i].ColumnAbsoluteSums().Average();
            }
            return result / Storage.Length;
        }

        public Tensor Normalize(NormalizationMode mode)
        {
            switch (mode)
            {
                case (NormalizationMode.Standardization):
                    {
                        var avg = this.MeanAll();
                        double variance = this.Map(r => Math.Pow(r - avg, 2)).MeanAll();
                        return this.Map(r => (r - avg) / Math.Sqrt(variance + 0.001));
                    }
            }
            return null;
        }

        /// <summary>
        /// 卷积，外层是乘积法则，内层是卷积
        /// </summary>
        /// <param name="core">卷积核平面</param>
        /// <param name="mode">卷积模式</param>
        /// <returns></returns>
        public Tensor Convolve(Tensor core, ConvolutionMode mode)
        {
            if (this.DimensionY != core.DimensionX)
            {
                throw new Exception("进行卷积的两个向量外层维度必须符合矩阵乘法规律");
            }
            Tensor newTensor = TensorBuilder.Empty(this.DimensionX, core.DimensionY);
            //创建一个与后续卷积结果大小相等的0矩阵
            Matrix<double> temp = this[0, 0].Convolve(core[0, 0], mode).Map(r => 0.0);
            for (int i = 0; i <= newTensor.DimensionX - 1; i++)
            {
                for(int j = 0; j <= newTensor.DimensionY - 1; j++)
                {
                    for(int k = 0; k <= this.DimensionY - 1; k++)
                    {
                        temp = temp + this[i, k].Convolve(core[k, j], mode);
                    }
                    newTensor[i, j] = temp;
                    temp = temp.Map(r => 0.0);
                }
            }
            newTensor.OuterAct(r => r.CoerceZero(1e-15));
            return newTensor;
        }
        /// <summary>
        /// 最大池化
        /// </summary>
        /// <param name="poolSize"></param>
        /// <returns>第一个为池化后张量，第二个为掩码表</returns>
        public Tuple<Tensor,Tensor> MaxPool(int poolSize)
        {
            Tensor pooledTensor = TensorBuilder.Empty(DimensionX, DimensionY);
            Tensor maskTensor = TensorBuilder.Empty(DimensionX, DimensionY);
            for(int i = 0; i <= this.Storage.Length - 1; i++)
            {
                (pooledTensor.Storage[i], maskTensor.Storage[i]) = this.Storage[i].MaxPooling(poolSize);
            }
            return new Tuple<Tensor, Tensor>(pooledTensor, maskTensor);
        }
        /// <summary>
        /// 平均池化
        /// </summary>
        /// <param name="poolSize"></param>
        /// <returns></returns>
        public Tensor AveragePool(int poolSize)
        {
            return this.OuterMap(r => r.AveragePooling(poolSize));
        }
        /// <summary>
        /// 最大池化上采样
        /// </summary>
        /// <param name="mask"></param>
        /// <param name="poolSize"></param>
        /// <returns></returns>
        public Tensor MaxUpSample(Tensor mask,int poolSize)
        {
            return this.OuterMapIndexed((x, y, r) => r.MaxUpSample(mask[x, y], poolSize));
        }
        /// <summary>
        /// 平均池化上采样
        /// </summary>
        /// <param name="poolSize">池化大小</param>
        /// <param name="destinyRowCount">目标矩阵行数</param>
        /// <param name="destinyColumnCount">目标矩阵列数</param>
        /// <returns></returns>
        public Tensor AverageUpSample(int poolSize,int destinyRowCount,int destinyColumnCount)
        {
            return this.OuterMap(r => r.AverageUpSample(poolSize, destinyRowCount, destinyColumnCount));
        }

        /// <summary>
        /// 内部每个矩阵水平翻转
        /// </summary>
        /// <returns></returns>
        public Tensor InnerHorizontalReverse()
        {
            return this.OuterMap(r => r.HorizontalReverse());
        }

        /// <summary>
        /// 内外部同时点乘
        /// </summary>
        /// <param name="rightSide"></param>
        /// <returns></returns>
        public Tensor PointMutiply(Tensor rightSide)
        {
            if (this.DimensionX == rightSide.DimensionX && this.DimensionY == rightSide.DimensionY)
            {
                Tensor newTensor = TensorBuilder.Empty(this.DimensionX, this.DimensionY);
                for(int i = 0; i <= this.Storage.Length - 1; i++)
                {
                    newTensor.Storage[i] = this.Storage[i].PointwiseMultiply(rightSide.Storage[i]);
                }
                newTensor.OuterAct(r => r.CoerceZero(1e-15));
                return newTensor;
            }
            else
            {
                throw new Exception("张量维度不匹配");
            }
        }
        /// <summary>
        /// 内外部同时点除
        /// </summary>
        /// <param name="rightSide"></param>
        /// <returns></returns>
        public Tensor PointDivide(Tensor rightSide)
        {
            if (this.DimensionX == rightSide.DimensionX && this.DimensionY == rightSide.DimensionY)
            {
                Tensor newTensor = TensorBuilder.Empty(this.DimensionX, this.DimensionY);
                for (int i = 0; i <= this.Storage.Length - 1; i++)
                {
                    newTensor.Storage[i] = this.Storage[i].PointwiseDivide(rightSide.Storage[i]);
                }
                newTensor.OuterAct(r => r.CoerceZero(1e-15));
                return newTensor;
            }
            else
            {
                throw new Exception("张量维度不匹配");
            }
        }

        /// <summary>
        /// 将每个矩阵转置并乘以一个矩阵
        /// </summary>
        /// <param name="rightSide"></param>
        /// <returns></returns>
        public Tensor InnerTransposeAndMultiply(Tensor rightSide)
        {
            Tensor newTensor = TensorBuilder.Empty(this.DimensionX, this.DimensionY);
            for(int i = 0; i <= this.Storage.Length - 1; i++)
            {
                newTensor.Storage[i] = this.Storage[i].TransposeThisAndMultiply(rightSide.Storage[i]);
            }
            newTensor.OuterAct(r => r.CoerceZero(1e-15));
            return newTensor;
        }
        /// <summary>
        /// 内层矩阵全部旋转180度
        /// </summary>
        /// <returns></returns>
        public Tensor InnerRot180()
        {
            return this.OuterMap(r => r.Rot180());
        }
        /// <summary>
        /// 外层转置
        /// </summary>
        /// <returns></returns>
        public Tensor OuterTranspose()
        {
            var newTensor = TensorBuilder.Empty(this.DimensionY, this.DimensionX);
            for(int i = 0; i <= DimensionX-1; i++)
            {
                for(int j = 0; j <= DimensionY-1; j++)
                {
                    newTensor[j, i] = this[i, j];
                }
            }
            return newTensor;
        }

        public Tensor Stretch()
        {
            Tensor newTensor = TensorBuilder.Empty(1, 1);
            var temp = Storage[0].ToRowMajorArray().AsEnumerable();
            for (int i = 1; i <= this.Storage.Length - 1; i++)
            {
                temp = temp.Concat(Storage[i].ToRowMajorArray());
            }
            return TensorBuilder.FromMatrix(Matrix<double>.Build.DenseOfRowMajor(temp.Count(), 1, temp));
        }

        public Tensor Fold(int x,int y,int p,int q)
        {
            Tensor newTensor = TensorBuilder.Empty(x,y);
            var temp = this.Storage[0].ToRowMajorArray();
            if (this.Storage.Length == 1&& Storage[0].RowCount * Storage[0].ColumnCount==x*y*p*q)
            {
                for(int i = 0; i <= x*y-1; i++)
                {
                    newTensor.Storage[i] = Matrix<double>.Build.DenseOfRowMajor(p, q, temp.AsSpan().Slice(i * p * q, p * q).ToArray());
                }
            }
            else
            {
                throw new Exception("要折叠的张量必须是1*1");
            }
            return newTensor;
        }

        public static Tensor operator +(Tensor leftSide, Tensor rightSide)
        {
            if ((leftSide.DimensionX != 1 || leftSide.DimensionY != 1)
                && (rightSide.DimensionX != 1 || rightSide.DimensionY != 1)
                && (leftSide.DimensionX != rightSide.DimensionX || leftSide.DimensionY != rightSide.DimensionY))
            {
                throw new Exception($"张量外层维度错误，左侧为{leftSide.DimensionX}*{leftSide.DimensionY}，右侧为{rightSide.DimensionX}*{rightSide.DimensionY}");
            }
            if (leftSide.DimensionX == 1 && leftSide.DimensionY == 1)
            {
                Tensor newTensor = TensorBuilder.Empty(rightSide.DimensionX, rightSide.DimensionY);
                for (int i = 0; i <= newTensor.Storage.Length - 1; i++)
                {
                    newTensor.Storage[i] = leftSide.Storage[0] + rightSide.Storage[i];
                }
                newTensor.OuterAct(r => r.CoerceZero(1e-15));
                return newTensor;
            }
            else if (rightSide.DimensionX == 1 && rightSide.DimensionY == 1)
            {
                Tensor newTensor = TensorBuilder.Empty(leftSide.DimensionX, leftSide.DimensionY);
                for (int i = 0; i <= newTensor.Storage.Length - 1; i++)
                {
                    newTensor.Storage[i] = leftSide.Storage[i] + rightSide.Storage[0];
                }
                newTensor.OuterAct(r => r.CoerceZero(1e-15));
                return newTensor;
            }
            else
            {
                Tensor newTensor = TensorBuilder.Empty(rightSide.DimensionX, rightSide.DimensionY);
                for (int i = 0; i <= newTensor.Storage.Length - 1; i++)
                {
                    newTensor.Storage[i] = leftSide.Storage[i] + rightSide.Storage[i];
                }
                newTensor.OuterAct(r => r.CoerceZero(1e-15));
                return newTensor;
            }
        }
        public static Tensor operator -(Tensor leftSide, Tensor rightSide)
        {
            if ((rightSide.DimensionX != 1 || rightSide.DimensionY != 1)
                && (leftSide.DimensionX != rightSide.DimensionX || leftSide.DimensionY != rightSide.DimensionY))
            {
                throw new Exception($"张量外层维度错误，左侧为{leftSide.DimensionX}*{leftSide.DimensionY}，右侧为{rightSide.DimensionX}*{rightSide.DimensionY}");
            }
            if (rightSide.DimensionX == 1 && rightSide.DimensionY == 1)
            {
                Tensor newTensor = TensorBuilder.Empty(leftSide.DimensionX, leftSide.DimensionY);
                for (int i = 0; i <= newTensor.Storage.Length - 1; i++)
                {
                    newTensor.Storage[i] =leftSide.Storage[i] - rightSide.Storage[0];
                }
                newTensor.OuterAct(r => r.CoerceZero(1e-15));
                return newTensor;
            }
            else
            {
                Tensor newTensor = TensorBuilder.Empty(leftSide.DimensionX, leftSide.DimensionY);
                for (int i = 0; i <= newTensor.Storage.Length - 1; i++)
                {
                    newTensor.Storage[i] = leftSide.Storage[i] - rightSide.Storage[i];
                }
                newTensor.OuterAct(r => r.CoerceZero(1e-15));
                return newTensor;
            }
        }
        public static Tensor operator -(Tensor rightSide)
        {
            Tensor newTensor = TensorBuilder.Empty(rightSide.DimensionX, rightSide.DimensionY);
            for (int i = 0; i <= newTensor.Storage.Length - 1; i++)
            {
                newTensor.Storage[i] = -rightSide.Storage[i];
            }
            newTensor.OuterAct(r => r.CoerceZero(1e-15));
            return newTensor;
        }
        /// <summary>
        /// 外层点积，内层矩阵乘法
        /// </summary>
        /// <param name="leftSide"></param>
        /// <param name="rightSide"></param>
        /// <returns></returns>
        public static Tensor operator *(Tensor leftSide, Tensor rightSide)
        {
            if (leftSide.DimensionX == rightSide.DimensionX && leftSide.DimensionY == rightSide.DimensionY)
            {
                Tensor newTensor = TensorBuilder.Empty(leftSide.DimensionX, leftSide.DimensionY);
                for (int i = 0; i <= newTensor.Storage.Length - 1; i++)
                {
                    newTensor.Storage[i] = leftSide.Storage[i] * rightSide.Storage[i];
                }
                newTensor.OuterAct(r => r.CoerceZero(1e-15));
                return newTensor;
            }
            else
            {
                throw new Exception("维度不符合");
            }
        }
        public static Tensor operator *(double leftSide, Tensor rightSide)
        {
            return rightSide.Map(r => leftSide * r);
        }
        public static Tensor operator *(Tensor leftSide, double rightSide)
        {
            return leftSide.Map(r => rightSide * r);
        }
        public static Tensor operator /(Tensor leftSide, double rightSide)
        {
            return leftSide.Map(r => r / rightSide);
        }
        public static Tensor operator /(Tensor leftSide, Tensor rightSide)
        {
            if (rightSide.DimensionX == 1 && rightSide.DimensionY == 1 && rightSide[0, 0].RowCount == 1 && rightSide[0, 0].ColumnCount == 1)
            {
                return leftSide.Map(r => r / rightSide.Storage[0][0, 0]);
            }
            else
            {
                throw new Exception("目前张量除法只支持除以数值，如果张量之间相除，必须右侧为1*1*1*1大小");
            }
        }

    }

    public class TensorCompareByValue : IEqualityComparer<Tensor>
    {
        public bool Equals(Tensor x, Tensor y)
        {
            if (x == null || y == null || x.DimensionX != y.DimensionX || x.DimensionY != y.DimensionY || !x.IsSameSize() || !y.IsSameSize())
            {
                return false;
            }
            else
            {
                return x.Equals(y);
            }

        }

        public int GetHashCode(Tensor obj)
        {
            if (obj == null)
                return 0;
            else if (obj.Storage[0] == null)
                return 0;
            else
                return obj.Storage[0][0, 0].GetHashCode();
            //return 1;
        }
    }

    class TensorEnumerator : IEnumerator<Matrix<double>>
    {
        private int _index;
        private Matrix<double>[] _collection;
        private Matrix<double> value;
        public TensorEnumerator(Matrix<double>[] colletion)
        {
            _collection = colletion;
            _index = -1;
        }
        Matrix<double> IEnumerator<Matrix<double>>.Current
        {
            get { return value; }
        }
        public object Current
        {
            get { return value; }
        }
        public bool MoveNext()
        {
            _index++;
            if (_index >= _collection.Length) { return false; }
            else { value = _collection[_index]; }
            return true;
        }
        public void Reset()
        {
            _index = -1;
        }

        public void Dispose()
        {
            this.Dispose();
        }
    }
}
