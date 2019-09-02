import Neuron
import Functions

class Layer:

    def __init__(self,inputs,neurons,function=Functions.Sigmoid()):
        self.neurons=[0]*neurons
        self.function=function
        for i in range (neurons):
            self.neurons[i]=Neuron.Neuron(inputs=[0]*inputs,f=function)

    def get_outputs(self):
        outputs=[0]*len(self.neurons)
        for i in range(len(self.neurons)):
            outputs[i]=self.neurons[i].get_output()
        return outputs

    def update(self):
        for i in self.neurons:
            i.update()

    def get_function(self):
        return self.function

    def set_deltas(self,deltas):
        for i in range(len(deltas)):
            self.neurons[i].set_delta(deltas[i])

    def get_deltas(self):
        deltas = [0]*len(self.neurons)
        for i in range(len(self.neurons)):
            deltas[i]=self.neurons[i].get_delta()
        return deltas

    def get_neurons(self):
        return self.neurons

    def set_weights(self,weights):
        for i in range (len(self.neurons)):
            self.neurons[i].set_weights(weights[i])

    def set_function(self,function):
        for i in range (len(self.neurons)):
            self.neurons[i].set_function(function)

    def feed(self,inputs):
        output = [0]*len(self.neurons)
        for i in range (len(self.neurons)):
            output[i] = self.neurons[i].feed(inputs)
        return output

    def train(self,inputs,desired_outputs):
        for i in range (len(self.neurons)):
            self.neurons[i].train(inputs,desired_outputs[i])

    def print(self):
        for i in range (len(self.neurons)):
            print("Neuron " +str(i)+":\n")
            self.neurons[i].print()
            print("\n")





