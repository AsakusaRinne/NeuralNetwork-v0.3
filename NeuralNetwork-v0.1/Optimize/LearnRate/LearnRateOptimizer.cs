using MathNet.Numerics.Integration;
using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Optimize.LearnRate
{
    interface LearnRateOptimizer
    {
        /// <summary>
        /// 学习率初始化
        /// </summary>
        /// <param name="count">学习率超参数个数</param>
        /// <param name="layerTypes">神经网络层的类型列表</param>
        /// <returns></returns>
        double[] Init(int count, Type[] layerTypes);
        /// <summary>
        /// 优化学习率
        /// </summary>
        /// <param name="count">学习率向量长度</param>
        /// <param name="step">当前次数</param>
        /// <param name="totalStep">总次数</param>
        /// <param name="gradient">神经网络层的梯度绝对值除以对应参数个数</param>
        /// <returns></returns>
        double[] Optimize(int count,int step, int totalStep, double[] gradient);
    }
    /// <summary>
    /// 指数衰减学习率优化
    /// </summary>
    class ExponentialDelayOptimizer:LearnRateOptimizer
    {
        double original;
        double exponentialRate;
        double exponentialStep;

        /// <summary>
        /// 初始化
        /// </summary>
        /// <param name="originalLearnRate">初始学习率</param>
        /// <param name="exponentialRate">衰减率，即指数衰减公式的底数，要求在0-1之间</param>
        /// <param name="exponentialStep">衰减步长，即每次衰减时的步数/总步数，范围应为0-1</param>
        public ExponentialDelayOptimizer(double originalLearnRate,double exponentialRate, double exponentialStep)
        {
            if (exponentialRate > 1 || exponentialRate <= 0||exponentialStep>1||exponentialStep<0)
            {
                throw new Exception("衰减率必须在0-1之间");
            }
            this.original = originalLearnRate;
            this.exponentialRate = exponentialRate;
            this.exponentialStep = exponentialStep;
        }

        public double[] Init(int count,Type[] layerTypes=null)
        {
            double[] result = new double[count];
            for(int i = 0; i <= count - 1; i++)
            {
                result[i] = original;
            }
            return result;
        }

        public double[] Optimize(int count, int step, int totalStep, double[] gradient = null)
        {
            double[] result = new double[count];
            double rate = original * Math.Pow(exponentialRate, (int)(step / (exponentialStep * totalStep)));
            for (int i = 0; i <= count - 1; i++)
            {
                result[i] = rate;
            }
            return result;
        }
    }
}
