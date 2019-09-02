import NeuralNetwork
import DataManager
import Functions
import random as rn


NN = NeuralNetwork.NeuralNetwork(5,[7,6,5,5,3],7,3)
Dataset = DataManager.DataManager().get_n_matrix("seeds_dataset.txt")
slicer = slice(7)
rn.shuffle(Dataset)
epochs = 100

for j in range(epochs):
    for i in range(len(Dataset)-54):
        current_input =Dataset[i][slicer]
        current_expected_output=Dataset[i][7]
        NN.train(current_input,current_expected_output)
    if j%100==0:
        print(str(j))

for i in range(len(Dataset)-54,len(Dataset)):
    current_input =Dataset[i][slicer]
    current_expected_output=Dataset[i][7]
    print("Output obtained : ")
    print(NN.feed(current_input))
    print("Output expected :")
    print(current_expected_output)
