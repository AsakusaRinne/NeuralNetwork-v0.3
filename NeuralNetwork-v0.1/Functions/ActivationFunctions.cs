using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.Remoting;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Functions.Activation
{
    /// <summary>
    /// 激活函数类接口
    /// </summary>
    interface ActivationFunction
    {
        double Activate(double input);
        double Derivate(double input);
    }
    /// <summary>
    /// 带可学习参数的激活函数类接口
    /// </summary>
    interface LearnableActivationFunction :ActivationFunction
    {
        double Activate(double input,double learnedParameter);
        double Derivate(double input, double learnedParameter);
    }

    public class Direct : ActivationFunction
    {
        public double Activate(double input)
        {
            return input;
        }
        public double Derivate(double input)
        {
            return 1;
        }
    }

    public class ReLU : ActivationFunction
    {
        public double Activate(double input)
        {
            return input >= 0 ? input : 0;
        }
        public double Derivate(double input)
        {
            return input >= 0 ? 1 : 0;
        }
    }

    public class Sigmoid : ActivationFunction
    {
        public double Activate(double input)
        {
            return 1 / (1 + (double)Math.Exp(-input));
        }
        public double Derivate(double input)
        {
            var temp = Activate(input);
            return temp * (1 - temp);
        }
    }

    public class LeakyReLU:ActivationFunction
    {
        double coefficient;
        public LeakyReLU(double coefficient)
        {
            this.coefficient = coefficient;
        }

        public double Activate(double input)
        {
            return input >= 0 ? input : coefficient * input;
        }
        public double Derivate(double input)
        {
            return input >= 0 ? 1 : coefficient;
        }
    }

}
