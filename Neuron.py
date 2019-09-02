import numbers
import numpy as np
import Functions
class Neuron:

    def __init__(self, inputs=None, weights=None, bias=None, f=Functions.Sigmoid()):
        self.inputs=inputs
        self.weights=weights
        self.bias=bias
        self.output=None
        self.lr=0.1
        self.f=f
        self.delta=None
        if weights is None and inputs is not None:
            self.weights=[0]*len(inputs)
            for i in range(len(inputs)):
                self.weights[i]=np.random.random()*np.random.randint(1,3)*np.power(-1,np.random.randint(0,500));
        if bias is None:
            self.bias = np.random.random() * np.random.randint(1, 3) * np.power(-1, np.random.randint(0, 500))

        self.check_correctness(self.inputs)
        self.check_correctness(self.weights)

    @staticmethod
    def check_correctness(array):
        if array is None:
            raise Exception('Empty Array!')
        for i in range (len(array)):
            if not isinstance(array[i], numbers.Number):
                raise Exception('One of the terms in the array is not a Number: '+str(i))

    def get_output(self):
        return self.output

    #Actualizamos la neurona al realizar la backpropagation de la red
    def update(self):
        for i in range(len(self.weights)):
            self.weights[i]=self.weights[i]+(self.lr*self.delta*self.inputs[i])
        self.bias = self.bias + (self.lr*self.delta)

    def set_inputs(self,inputs):
        self.check_correctness(inputs)
        self.inputs=inputs

    def set_threshold(self,threshold):
        if not isinstance(threshold, numbers.Number):
            raise Exception('Value given for threshold is not a number')
        self.bias=threshold

    def set_weights(self,weights):
        self.check_correctness(weights)
        self.weights=weights

    def get_weight(self,i):
        return self.weights[i]

    def set_function(self,function):
        self.f=function

    def get_function(self):
        return self.f

    def set_delta(self,delta):
        self.delta=delta

    def get_delta(self):
        return self.delta

    def feed(self,inputs):
        self.set_inputs(inputs)
        self.compute()

    def train(self, inputs, desired_output):
        diff = desired_output - self.feed(inputs)
        for i in range(self.weights):
            self.weights[i]=self.weights[i]+(self.lr*self.inputs*diff)
        self.bias = self.bias + (self.lr * diff)

    def print(self):
        if self.inputs is not None and self.weights is not None and len(self.inputs)==len(self.weights):
            for i in range(len(self.inputs)):
                print("Input "+str(i)+": "+str(self.inputs[i])+"    Weight "+str(i)+": "+str(self.weights[i]))
        print("Bias: "+str(self.bias))
        print("Delta: "+str(self.delta))
        print("Function: "+self.f.print())
        print("Last Output :"+str(self.output))
    #Función que computa el output de la neurona
    def compute(self):
        if len(self.inputs)==len(self.weights):
            result = 0
            for i in range (len(self.weights)):
                result=result+self.inputs[i]*self.weights[i]
            result=result+self.bias
            self.output=self.f.apply(result)
            return self.get_output()
        raise Exception('The number of inputs differs in size with the number of stored weights')

