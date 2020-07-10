using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data;

namespace NeuralNetwork
{
    /// <summary>
    /// 文件存写与读取的静态类，包括二进制序列化等
    /// </summary>
     static class Filehelper
    {
        /// <summary>
        /// 序列化存储对象
        /// </summary>
        /// <param name="path">存储路径</param>
        /// <param name="obj">要存储的对象</param>
        public static void SaveObject(string path,Object obj)
        {
            System.IO.MemoryStream MStream = new System.IO.MemoryStream();
            System.IO.FileStream FStream = new System.IO.FileStream(path, FileMode.Create);
            System.Runtime.Serialization.Formatters.Binary.BinaryFormatter Formatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
            Formatter.Serialize(MStream, obj);
            FStream.Write(MStream.GetBuffer(), 0, MStream.GetBuffer().Length);
            FStream.Close();
        }
        /// <summary>
        /// 反序列化获得存储的对象
        /// </summary>
        /// <typeparam name="T">存储进去时候的类型</typeparam>
        /// <param name="path">存储路径</param>
        /// <returns></returns>
        public static T GetObjectFromFile<T>(string path)
        {
            T ReturnObject;
            System.IO.FileStream FStream = new System.IO.FileStream(path, FileMode.Open);
            System.Runtime.Serialization.Formatters.Binary.BinaryFormatter Formatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
            ReturnObject = (T)Formatter.Deserialize(FStream);
            FStream.Close();
            FStream.Dispose();
            return ReturnObject;
        }
    }
}
