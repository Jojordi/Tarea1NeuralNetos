import numpy as np


class Step:

    def apply(self, x):
        if x < 0:
            return 0
        else:
            return 1

    def derivative(self, x):
        if x == 0:
            raise Exception('Can\'t diferentiate at 0')
        else:
            return 0
    def print(self):
        return "Step"


class Sigmoid:

    def apply(self, x):
        return 1 / (1 + np.exp(-x))


    def derivative(self,x):
        return self.apply(x)*(1-self.apply(x))

    def print(self):
        return "Sigmoid"


class Tanh:

    def apply(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


    def derivative(self,x):
        return 1 - (self.apply(x)*self.apply(x))

    def print(self):
        return "Tanh"
