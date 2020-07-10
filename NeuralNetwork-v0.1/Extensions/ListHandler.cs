using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Extensions
{
    public static class ListHandlerExtension
    {
        public static List<T> GetAllElements<T>(this List<List<T>> source)
        {
            if(source.Count!=0&&source.All(r=>r.Count != 0))
            {
                List<T> result = new List<T>();
                for(int i = 0; i <= source.Count - 1; i++)
                {
                    for(int j = 0; j <= source[i].Count-1; j++)
                    {
                        result.Add(source[i][j]);
                    }
                }
                return result;
            }
            else
            {
                throw new Exception("数据为空");
            }
        }
        /// <summary>
        /// 逐行取出元素归并到一个一维数组中
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public static T[] ToRowArray<T>(this T[,] source)
        {
            int rowcount = source.GetUpperBound(0) + 1;
            int columncount = source.GetUpperBound(1) + 1;
            T[] result = new T[rowcount* columncount];
            for (int i = 0; i <= rowcount-1; i++)
            {
                for (int j = 0; j <= columncount-1; j++)
                {
                    result[i*columncount+j]=source[i,j];
                }
            }
            return result;
        }
        /// <summary>
        /// 逐列取出元素归并到一个一维数组中
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public static T[] ToColumnArray<T>(this T[,] source)
        {
            int rowcount = source.GetUpperBound(0) + 1;
            int columncount = source.GetUpperBound(1) + 1;
            T[] result = new T[rowcount * columncount];
            for (int i = 0; i <= rowcount - 1; i++)
            {
                for (int j = 0; j <= columncount - 1; j++)
                {
                    result[j * rowcount + i] = source[i, j];
                }
            }
            return result;
        }
        /// <summary>
        /// 获取一个数组中每个元素对应类型的数组
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public static Type[] GetTypes<T>(this T[] source)
        {
            Type[] types = new Type[source.Length];
            for(int i = 0; i <= source.Length - 1; i++)
            {
                types[i] = source[i].GetType();
            }
            return types; 
        }

    }
}
