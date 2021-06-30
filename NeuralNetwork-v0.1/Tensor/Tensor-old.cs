using MathNet.Numerics.LinearAlgebra.Complex;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Reflection.Emit;
using NeuralNetwork.Extensions;
using System.Runtime.InteropServices;
using MathNet.Numerics.Statistics;

namespace NeuralNetwork.Struct
{
    [Serializable]
    public class Tensor //:IEnumerable<Matrix<double>>
    {
        protected internal Matrix<double>[,] Storage;
        public readonly int DimensionX;
        public readonly int DimensionY;
        public readonly string Name;

        /// <summary>
        /// 是否可以重塑形状
        /// </summary>
        public bool CanReshape = true;
        /// <summary>
        /// 是否锁定该张量。锁定后不可以更改任何数据。
        /// </summary>
        private bool Locked = false;

        protected internal Tensor(Matrix<double>[,] storage,string name)
        {
            Storage = storage;
            DimensionX = storage.GetLength(0);
            DimensionY = storage.GetLength(1);
            Name = name;
        }

        public Matrix<double> this[int x,int y]
        {
            get
            {
                return Storage[x, y];
            }
            set
            {
                if (Locked == true)
                {
                    throw new Exception("该张量已锁定，不可更改");
                }
                Storage[x, y] = value;
            }
        }
        public double this[int x,int y,int p,int q]
        {
            get
            {
                return Storage[x, y][p, q];
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
        /// 检测张量中所有矩阵是否是相同大小
        /// </summary>
        /// <returns></returns>
        public bool IsSameSize()
        {
            if (Storage[0,0] == null)
            {
                for (int i = 0; i <= DimensionX - 1; i++)
                {
                    for (int j = 0; j <= DimensionY - 1; j++)
                    {
                        if (Storage[i, j]!=null)
                        {
                            return false;
                        }
                    }
                }
                return true;
            }
            else
            {
                int rc = Storage[0, 0].RowCount;
                int cc = Storage[0, 0].ColumnCount;
                for(int i = 0; i <= DimensionX - 1; i++)
                {
                    for(int j = 0; j <= DimensionY - 1; j++)
                    {
                        if(Storage[i,j].RowCount!=rc || Storage[i, j].ColumnCount != cc)
                        {
                            return false;
                        }
                    }
                }
                return true;
            }
        }

        public bool Equals(Tensor other)
        {
            if (this.DimensionX != other.DimensionX || this.DimensionY != other.DimensionY)
            {
                return false;
            }
            else
            {
                for(int i = 0; i <= DimensionX - 1; i++)
                {
                    for( int j = 0;j <= DimensionY - 1; j++)
                    {
                        if (!this.Storage[i, j].Equals(other.Storage[i, j]))
                        {
                            return false;
                        }
                    }
                }
                return true;
            }
        }
        /// <summary>
        /// 对张量中的每个矩阵的每个元素执行一个操作
        /// </summary>
        /// <param name="f"></param>
        /// <returns>新的张量</returns>
        public Tensor Map(Func<double, double> f, bool keepName = true,string newName=null)
        {
            Tensor newTensor = TensorBuilder.Empty(this.DimensionX, this.DimensionY, keepName ? this.Name : newName);
            for(int i = 0; i <= DimensionX - 1; i++)
            {
                for(int j = 0; j <= DimensionY - 1; j++)
                {
                    newTensor[i, j] = Storage[i, j].Map(f);
                }
            }
            return newTensor;
        }
        /// <summary>
        /// 对张量中特定位置的矩阵中的每个元素执行相同操作
        /// </summary>
        /// <param name="x">外层行下标</param>
        /// <param name="y">外层列下标</param>
        /// <param name="f">函数</param>
        /// <param name="keepName">是否保持原实例名称</param>
        /// <param name="newName">新名称</param>
        /// <returns></returns>
        public Tensor MapAt(int x,int y,Func<double,double> f)
        {
            Tensor newTensor = this.Clone();
            newTensor[x, y] = Storage[x, y].Map(f);
            ///this.Storage[x, y] = this.Storage[x, y].Map(f);
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
            for (int i = 0; i <= DimensionX - 1; i++)
            {
                for (int j = 0; j <= DimensionY - 1; j++)
                {
                    newTensor[i, j] = Storage[i, j].MapIndexed(f);
                }
            }
            return newTensor;
        }
        /// <summary>
        /// 对张量中特定位置中的矩阵中的元素按照下标执行操作
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="f"></param>
        /// <param name="keepName"></param>
        /// <param name="newName"></param>
        /// <returns></returns>
        public Tensor MapAtIndexed(int x,int y,Func<int,int,double,double>f)
        {
            Tensor newTensor = this.Clone();
            newTensor[x, y] = Storage[x, y].MapIndexed(f);
            return newTensor;
        }
        /// <summary>
        /// 对张量中的每个矩阵进行整体操作
        /// </summary>
        /// <param name="f"></param>
        /// <param name="keepName"></param>
        /// <param name="newName"></param>
        /// <returns></returns>
        public Tensor OuterMap(Func<Matrix<double>,Matrix<double>> f)
        {
            Matrix<double>[,] newStorage = new Matrix<double>[this.DimensionX, this.DimensionY];
            for(int i = 0; i <= DimensionX - 1; i++)
            {
                for(int j = 0; j <= DimensionY - 1; j++)
                {
                    newStorage[i, j] = f(this.Storage[i, j]);
                }
            }
            return new Tensor(newStorage,null);
        }
        /// <summary>
        /// 根据下标对张量中的每个矩阵进行整体操作
        /// </summary>
        /// <param name="f"></param>
        /// <param name="keepName"></param>
        /// <param name="newName"></param>
        /// <returns></returns>
        public Tensor OuterMapIndexed(Func<int,int,Matrix<double>, Matrix<double>> f)
        {
            Matrix<double>[,] newStorage = this.Storage;
            for (int i = 0; i <= DimensionX - 1; i++)
            {
                for (int j = 0; j <= DimensionY - 1; j++)
                {
                    newStorage[i, j] = f(i,j,Storage[i, j]);
                }
            }
            return new Tensor(newStorage, null);
        }
        /// <summary>
        /// 外层行切片，包括下标stop处的数据
        /// </summary>
        /// <param name="start"></param>
        /// <param name="stop"></param>
        /// <param name="step"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor RowSlice(int start,int stop,int step,string name=null)
        {
            List<Matrix<double>> list = new List<Matrix<double>>();
            int x = 0;//记录新张量的外层行数
            for(int i = start; i <= stop; i = i + step)
            {
                for(int j = 0; j <= DimensionY - 1; j++)
                {
                    list.Add(Storage[i, j]);
                }
                x++;
            }
            return TensorBuilder.FromRowMajorIEnumerable(x, DimensionY, list, name);
        }
        /// <summary>
        /// 外层列切片，包括下标stop处的数据
        /// </summary>
        /// <param name="start"></param>
        /// <param name="stop"></param>
        /// <param name="step"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor ColumnSlice(int start, int stop, int step, string name = null)
        {
            List<Matrix<double>> list = new List<Matrix<double>>();
            int y = 0;//记录新张量的外层列数
            for (int j = start; j <= stop; j = j + step)
            {
                for (int i = 0; i <= DimensionX - 1; i++)
                {
                    list.Add(Storage[i, j]);
                }
                y++;
            }
            return TensorBuilder.FromColumnMajorIEnumerable(y, DimensionX, list, name);
        }
        /// <summary>
        /// 垂直方向连接张量，加在下方，外层列数必须一致
        /// </summary>
        /// <param name="down"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor VerticalConcat(Tensor down, string name = null)
        {
            if (this.DimensionY != down.DimensionY)
            {
                throw new Exception("要连接的张量外层维度不一致");
            }
            //Matrix<double>[,] newStorage = new Matrix<double>[this.DimensionX + down.DimensionX, this.DimensionY];
            //for (int i = 0; i <= this.DimensionX - 1; i++)
            //{
            //    for (int j = 0; j <= this.DimensionY - 1; j++)
            //    {
            //        newStorage[i, j] = this.Storage[i, j];
            //    }
            //}
            //for (int i = 0; i <= down.DimensionX - 1; i++)
            //{
            //    for (int j = 0; j <= this.DimensionY - 1; j++)
            //    {
            //        newStorage[i + this.DimensionX, j] = down.Storage[i, j];
            //    }
            //}
            return TensorBuilder.FromRowMajorIEnumerable(DimensionX + down.DimensionX, DimensionY, Storage.ToRowArray().Concat(down.Storage.ToRowArray()));
            //return new Tensor(newStorage, name);
            //var newStorage = this.Storage.ToRowArray().Concat(down.Storage.ToRowArray());
            //return TensorBuilder.FromRowMajorIEnumerable(this.DimensionX + down.DimensionX, this.DimensionY, newStorage);
        }
        /// <summary>
        /// 水平方向连接张量，加在右方，外层行数必须一致
        /// </summary>
        /// <param name="right"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor HorizontalConcat(Tensor right,string name=null)
        {
            if (this.DimensionX != right.DimensionX)
            {
                throw new Exception("要连接的张量外层维度不一致");
            }
            //Matrix<double>[,] newStorage = new Matrix<double>[this.DimensionX , this.DimensionY+right.DimensionY];
            //for (int i = 0; i <= this.DimensionX - 1; i++)
            //{
            //    for (int j = 0; j <= this.DimensionY - 1; j++)
            //    {
            //        newStorage[i, j] = this.Storage[i, j];
            //    }
            //}
            //for (int i = 0; i <= this.DimensionX - 1; i++)
            //{
            //    for (int j = 0; j <= right.DimensionY- 1; j++)
            //    {
            //        newStorage[i , j+this.DimensionY] = right.Storage[i, j];
            //    }
            //}
            //return new Tensor(newStorage, name);
            return TensorBuilder.FromRowMajorIEnumerable(DimensionX,DimensionY + right.DimensionY, Storage.ToRowArray().Concat(right.Storage.ToColumnArray()));
        }
        /// <summary>
        /// 删除下标为index的行
        /// </summary>
        /// <param name="index"></param>
        /// <param name="keepName"></param>
        /// <param name="newName"></param>
        /// <returns></returns>
        public Tensor RemoveRow(int index,bool keepName=true,string newName=null)
        {
            Matrix<double>[,] newStorage = new Matrix<double>[this.DimensionX - 1, this.DimensionY];
            for(int i = 0; i < index; i++)
            {
                for(int j = 0; j <= this.DimensionY - 1; j++)
                {
                    newStorage[i, j] = this.Storage[i, j];
                }
            }
            if (index < this.DimensionX-1)
            {
                for (int i = index + 1; i <= this.DimensionX - 1; i++)
                {
                    for (int j = 0; j <= this.DimensionY - 1; j++)
                    {
                        newStorage[i-1, j] = this.Storage[i, j];
                    }
                }
            }
            return new Tensor(newStorage, keepName ? this.Name : newName);
        }
        /// <summary>
        /// 删除下标为index的列
        /// </summary>
        /// <param name="index"></param>
        /// <param name="keepName"></param>
        /// <param name="newName"></param>
        /// <returns></returns>
        public Tensor RemoveColumn(int index, bool keepName = true, string newName = null)
        {
            Matrix<double>[,] newStorage = new Matrix<double>[this.DimensionX , this.DimensionY-1];
            for (int i = 0; i <= this.DimensionX-1; i++)
            {
                for (int j = 0; j < index; j++)
                {
                    newStorage[i, j] = this.Storage[i, j];
                }
            }
            if (index < this.DimensionY - 1)
            {
                for (int i = 0; i <= this.DimensionX - 1; i++)
                {
                    for (int j = index + 1; j <= this.DimensionY - 1; j++)
                    {
                        newStorage[i , j-1] = this.Storage[i, j];
                    }
                }
            }
            return new Tensor(newStorage, keepName ? this.Name : newName);
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
                    return  Matrix<double>.Build.DenseDiagonal(r.ColumnCount, r.ColumnCount, t => r[0, t]);
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
            for(int i = 0; i <= DimensionX-1; i++)
            {
                for(int j = 0; j <= DimensionY - 1; j++)
                {
                    result = result + Storage[i, j].ColumnSums().Sum();
                }
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
            for (int i = 0; i <= DimensionX - 1; i++)
            {
                for (int j = 0; j <= DimensionY - 1; j++)
                {
                    result = result + Storage[i, j].ColumnSums().PointwiseAbs().Sum();
                }
            }
            return result;
        }

        public double AbsoluteMean()
        {
            double result = 0;
            for(int i = 0; i <= DimensionX - 1; i++)
            {
                for(int j = 0; j <= DimensionY - 1; j++)
                {
                    result = result + Storage[i, j].PointwiseAbs().ToRowMajorArray().Average();
                }
            }
            return result;
        }

        public void  Add( Tensor rightSide)
        {
            if ( (rightSide.DimensionX != 1 || rightSide.DimensionY != 1)
                && (DimensionX != rightSide.DimensionX || DimensionY != rightSide.DimensionY))
            {
                throw new Exception($"张量外层维度错误，左侧为{DimensionX}*{DimensionY}，右侧为{rightSide.DimensionX}*{rightSide.DimensionY}");
            }
            if (rightSide.DimensionX == 1 && rightSide.DimensionY == 1)
            {
                for (int i = 0; i <= DimensionX - 1; i++)
                {
                    for (int j = 0; j <= DimensionY - 1; j++)
                    {
                        Storage[i, j] = Storage[i, j] + rightSide[0, 0];
                    }
                }
            }
            else
            {
                for (int i = 0; i <= DimensionX - 1; i++)
                {
                    for (int j = 0; j <= DimensionY - 1; j++)
                    {
                        Storage[i, j] = Storage[i, j] + rightSide[i, j];
                    }
                }
            }

        }

        public Tensor Convolve(Tensor core,ConvolutionMode mode)
        {
            return null;
        }




        public Tensor Clone(string name=null)
        {
            Matrix<double>[,] newStorage = new Matrix<double>[DimensionX, DimensionY];
            for(int i = 0; i <= DimensionX - 1; i++)
            {
                for(int j = 0; j <= DimensionY - 1; j++)
                {
                    newStorage[i, j] = Storage[i, j];
                }
            }
            return new Tensor(newStorage, name)
            {
                CanReshape = this.CanReshape,
                Locked = this.Locked
            };
        }
        public override string ToString()
        {
            var result = new StringBuilder();
            for(int i = 0; i <= DimensionX - 1; i++)
            {
                for(int j = 0; j <= DimensionY - 1; j++)
                {
                    result .Append(Storage[i, j].ToString());
                }
            }
            return result.ToString();
        }

        public static Tensor operator +(Tensor leftSide,Tensor rightSide)
        {
            if((leftSide.DimensionX!=1||leftSide.DimensionY!=1)
                &&(rightSide.DimensionX!=1||rightSide.DimensionY!=1)
                && (leftSide.DimensionX != rightSide.DimensionX || leftSide.DimensionY != rightSide.DimensionY))
            {
                throw new Exception($"张量外层维度错误，左侧为{leftSide.DimensionX}*{leftSide.DimensionY}，右侧为{rightSide.DimensionX}*{rightSide.DimensionY}");
            }
            if (leftSide.DimensionX == 1 && leftSide.DimensionY == 1)
            {
                Tensor newTensor = rightSide.Clone();
                for (int i = 0; i <= newTensor.DimensionX - 1; i++)
                {
                    for (int j = 0; j <= newTensor.DimensionY - 1; j++)
                    {
                        newTensor[i, j] = newTensor[i, j] + leftSide[i, j];
                    }
                }
                return newTensor;
            }
            else if (rightSide.DimensionX == 1 && rightSide.DimensionY == 1)
            {
                Tensor newTensor = leftSide.Clone();
                for (int i = 0; i <= newTensor.DimensionX - 1; i++)
                {
                    for (int j = 0; j <= newTensor.DimensionY - 1; j++)
                    {
                        newTensor[i, j] = newTensor[i, j] + rightSide[0,0];
                    }
                }
                return newTensor;
            }
            //else if (leftSide == null)
            //{
            //    return rightSide.Clone();
            //}
            //else if (rightSide == null)
            //{
            //    return leftSide.Clone();
            //}
            else
            {
                Tensor newTensor = leftSide.Clone();
                for (int i = 0; i <= newTensor.DimensionX - 1; i++)
                {
                    for (int j = 0; j <= newTensor.DimensionY - 1; j++)
                    {
                        newTensor[i, j] = newTensor[i, j] + rightSide[i, j];
                    }
                }
                return newTensor;
            }
            
        }
        public static Tensor operator -(Tensor leftSide, Tensor rightSide)
        {
            if ( (rightSide.DimensionX != 1 || rightSide.DimensionY != 1)
                && (leftSide.DimensionX != rightSide.DimensionX || leftSide.DimensionY != rightSide.DimensionY))
            {
                throw new Exception($"张量外层维度错误，左侧为{leftSide.DimensionX}*{leftSide.DimensionY}，右侧为{rightSide.DimensionX}*{rightSide.DimensionY}");
            }
            if (rightSide.DimensionX == 1 && rightSide.DimensionY == 1)
            {
                Tensor newTensor = leftSide.Clone();
                for (int i = 0; i <= newTensor.DimensionX - 1; i++)
                {
                    for (int j = 0; j <= newTensor.DimensionY - 1; j++)
                    {
                        newTensor[i, j] = newTensor[i, j] - rightSide[0, 0];
                    }
                }
                return newTensor;
            }
            else
            {
                Tensor newTensor = leftSide.Clone();
                for (int i = 0; i <= newTensor.DimensionX - 1; i++)
                {
                    for (int j = 0; j <= newTensor.DimensionY - 1; j++)
                    {
                        newTensor[i, j] = newTensor[i, j] - rightSide[i, j];
                    }
                }
                return newTensor;
            }

        }
        public static Tensor operator -(Tensor rightSide)
        {
                Tensor newTensor = rightSide.Clone();
                for (int i = 0; i <= newTensor.DimensionX - 1; i++)
                {
                    for (int j = 0; j <= newTensor.DimensionY - 1; j++)
                    {
                        newTensor[i, j] =  - rightSide[i, j];
                    }
                }
                return newTensor;
            }
        /// <summary>
        /// 外层点积，内层矩阵乘法
        /// </summary>
        /// <param name="leftSide"></param>
        /// <param name="rightSide"></param>
        /// <returns></returns>
        public static Tensor operator *(Tensor leftSide,Tensor rightSide)
        {
            if (leftSide.DimensionX == rightSide.DimensionX&&leftSide.DimensionY==rightSide.DimensionY)
            {
                Matrix<double>[,] newStorage = new Matrix<double>[leftSide.DimensionX, leftSide.DimensionY];
                for(int i = 0; i <= leftSide.DimensionX-1; i++)
                {
                    for(int j = 0; j <= leftSide.DimensionY - 1; j++)
                    {
                        newStorage[i, j] = leftSide.Storage[i, j] * rightSide.Storage[i, j];
                    }
                }
                return new Tensor(newStorage, null);
            }
            else
            {
                throw new Exception("维度不符合");
            }
        }
        public static Tensor operator *(double leftSide,Tensor rightSide)
        {
            return rightSide.Map(r => leftSide * r);
        }
        public static Tensor operator *(Tensor leftSide,double rightSide)
        {
            return leftSide.Map(r => rightSide * r);
        }
        public static Tensor operator /(Tensor leftSide,double rightSide)
        {
            return leftSide.Map(r => r / rightSide);
        }
        public static Tensor operator /(Tensor leftSide,Tensor rightSide)
        {
            if (rightSide.DimensionX == 1 && rightSide.DimensionY == 1 && rightSide[0, 0].RowCount == 1 && rightSide[0, 0].ColumnCount == 1)
            {
                return leftSide.Map(r => r / rightSide[0, 0][0, 0]);
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
                for (int i = 0; i <= x.DimensionX - 1; i++)
                {
                    for (int j = 0; j <= x.DimensionY - 1; j++)
                    {
                        if (!x[i, j].Equals(y[i, j]))
                        {
                            return false;
                        }
                    }
                }
                return true;
            }

        }

        public int GetHashCode(Tensor obj)
        {
            if (obj == null)
                return 0;
            else
                return obj.ToString().GetHashCode();
            //return 1;
        }
    }
}
