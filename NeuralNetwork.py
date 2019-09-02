import Layer
import numbers


class NeuralNetwork:

    def __init__(self,layers,neurons_per_layer,inputs_on_first_layer,outputs):
        self.outputs=outputs
        self.number_of_layers=layers
        self.layers = [0]*layers
        self.layers[0]=Layer.Layer(inputs_on_first_layer,neurons_per_layer[0])
        for i in range(1,layers):
            self.layers[i]=Layer.Layer(neurons_per_layer[i-1],neurons_per_layer[i])

    def print(self):
        for i in range(self.number_of_layers):
            print("Layer "+str(i)+ ": \n")
            self.layers[i].print()
            print("\n")

    def feed(self,inputs):
        current_output=inputs
        for i in range(self.number_of_layers):
            self.layers[i].feed(current_output)
            current_output = self.layers[i].get_outputs()
        return current_output

    def backpropagate(self):
        for i in range (1,self.number_of_layers):
            current_layer_neurons = self.layers[self.number_of_layers-i-1].get_neurons()
            current_next_layer_neurons = self.layers[self.number_of_layers-i].get_neurons()

            for j in range (len(current_layer_neurons)):
                current_neuron=current_layer_neurons[j]
                error = 0

                for k in range (len(current_next_layer_neurons)):
                    current_next_neuron = current_next_layer_neurons[k]
                    error=error+current_next_neuron.get_weight(j)*current_next_neuron.get_delta()

                delta = error*current_neuron.get_function().derivative(current_neuron.get_output())
                current_neuron.set_delta(delta)

    def update(self):
        for i in self.layers:
            i.update()

    def train(self, inputs,goal_output):
        desired_output = goal_output

        if isinstance(desired_output, numbers.Number):
            desired_output=[desired_output]

        result = self.feed(inputs)

        deltas=[0]*len(desired_output)
        outputs=self.layers[self.number_of_layers-1].get_outputs()
        f = self.layers[self.number_of_layers-1].get_function()

        for i in range(len(desired_output)):
            error=desired_output[i]-outputs[i]
            deltas[i]=error*f.derivative(outputs[i])

        self.layers[self.number_of_layers-1].set_deltas(deltas)
        self.backpropagate()
        self.update()
        return result



