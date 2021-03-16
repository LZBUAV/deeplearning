import numpy as np

class ActivateFun(object):
    def __init__(self):
        pass

    #阶跃函数
    def step_fun(self, x):
        return 1 if x > 0 else 0

    #线性函数y = x
    def liner_fun(self, x):
        return x

    #sigmod函数
    def sigmod(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmod_forward(self, x):#存在溢出风险，如何解决？
        if x.all() > 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

    def sigmod_backward(self, output):
        return output * (1.0 - output)

    def relu_forward(self, x):
        return x

    def relu_backward(self, x):
        return 1