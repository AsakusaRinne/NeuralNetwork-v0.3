using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Extensions
{
    /// <summary>
    /// 一些并非Tensor主要功能的作为扩展放在这里
    /// </summary>
    static class TensorExtensions
    {
        /// <summary>
        /// 不允许Tensor中出现0，强制修改为其他值
        /// </summary>
        /// <param name="source"></param>
        /// <param name="value">要修改的值</param>
        /// <returns></returns>
        public static Tensor ForceZerosToValue(this Tensor source,double value)
        {
            return source.Map(r => r == 0 ? r = value:r);
        }
    }
}
