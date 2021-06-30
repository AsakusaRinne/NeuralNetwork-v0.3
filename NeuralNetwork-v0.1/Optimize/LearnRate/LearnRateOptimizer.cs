using MathNet.Numerics.Integration;
using NeuralNetwork.Struct;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
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

    class TriangularCyclicOptimizer:LearnRateOptimizer
    {
        int period;
        double upper;
        double lower;
        double declineRate;
        /// <summary>
        /// 三角循环学习率的构造函数
        /// </summary>
        /// <param name="period">循环周期，即多少step一个循环</param>
        /// <param name="upper">第一个周期中学习率的上界</param>
        /// <param name="lower">第一个周期中学习率的下界</param>
        /// <param name="declineRate">上下界随周期的衰减速率</param>
        public TriangularCyclicOptimizer(int period,double upper,double lower,double declineRate)
        {
            this.period = period;
            this.upper = upper;
            this.lower = lower;
            this.declineRate = declineRate;
        }

        public double[] Init(int count, Type[] layerTypes)
        {
            double[] result = new double[count];
            for (int i = 0; i <= count - 1; i++)
            {
                result[i] = lower;
            }
            return result;
        }

        public double[] Optimize(int count, int step, int totalStep, double[] gradient = null)
        {
            var m = Math.Floor((double)(1 + step / (2 * period)));
            var b = Math.Abs(step / period - 2 * m + 1);
            upper = upper * Math.Pow(declineRate, m-1);
            lower= lower* Math.Pow(declineRate, m - 1);
            var newRate = upper + (upper - lower) * Math.Max(0, 1 - b);
            double[] result = new double[count];
            for (int i = 0; i <= count - 1; i++)
            {
                result[i] = newRate;
            }
            return result;
        }

    }

    class AdaGradOptimizer : LearnRateOptimizer
    {
        double original;
        double[] accumulation;

        /// <summary>
        /// 初始化
        /// </summary>
        /// <param name="originalLearnRate">初始学习率</param>
        public AdaGradOptimizer(double originalLearnRate)
        {
            this.original = originalLearnRate;
        }

        public double[] Init(int count, Type[] layerTypes = null)
        {
            double[] result = new double[count];
            accumulation = new double[count];
            for (int i = 0; i <= count - 1; i++)
            {
                result[i] = original;
            }
            return result;
        }

        public double[] Optimize(int count, int step, int totalStep, double[] gradient = null)
        {
            System.Diagnostics.Debug.Assert(gradient.Length == accumulation.Length);
            for(int i = 0; i <= accumulation.Length-1; i++)
            {
                accumulation[i] = accumulation[i] + Math.Pow(gradient[i],2);
            }
            double[] result = new double[count];
            for (int i = 0; i <= count - 1; i++)
            {
                result[i] = result[i] - original / (Math.Sqrt(accumulation[i] + 1e-8)) * gradient[i];
            }
            return result;
        }
    }
}
