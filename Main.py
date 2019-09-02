import NeuralNetwork
import DataManager
import Functions
import random as rn
import numpy as np
import matplotlib.pyplot as plt

#Creación de la red
NN = NeuralNetwork.NeuralNetwork(5,[7,6,5,5,3],7,3)
#Creación de una matriz que contenga los datos del irisdataset de forma amigable para la red, es decir en forma matricial
Dataset = DataManager.DataManager().get_n_matrix("seeds_dataset.txt")
#Se corta el input y el output deseado
slicer = slice(7)
#Revolvemos el dataset
rn.shuffle(Dataset)
#Definimos las epochas
epochs = 100
successes=[0]*epochs
epoch_array=[0]*epochs
error=[0]*epochs
#Entrenamos la red epocas-veces con el 75% de los datos
for j in range(epochs):
    error[j] = 0
    for i in range(len(Dataset)-54):
        current_input =Dataset[i][slicer]
        current_expected_output=Dataset[i][7]
        result = NN.train(current_input,current_expected_output)
        if np.argmax(result)==np.argmax(current_expected_output):
            successes[j]=successes[j]+1
        epoch_array[j]=epoch_array[j]+1
        error[j]=error[j]+np.power(np.argmax(result)-np.argmax(current_expected_output),2)
    error[j]=error[j]/54

results=[0]*54
expected=[0]*54
j=0
#Testeamos la red con el otro 25%
for i in range(len(Dataset)-54,len(Dataset)):
    current_input =Dataset[i][slicer]
    current_expected_output=Dataset[i][7]
    results[j]=np.argmax(NN.feed(current_input))
    expected[j]=np.argmax(current_expected_output)
    j=j+1

#Creamos la matriz de confusión
confusion_m=[0]*3
confusion_m[0]=[0]*3
confusion_m[1]=[0]*3
confusion_m[2]=[0]*3
for i in range(len(results)):
    confusion_m[results[i]-1][expected[i]-1]=confusion_m[results[i]-1][expected[i]-1]+1
print("              |GoldLabel0|GoldLabel1|GoldLabel2")
for i in range(3):
    print("Predicted "+ str(i)+""+"    |    "+str(confusion_m[i][0])+"    |    "+str(confusion_m[i][1])+"    |    "+str(confusion_m[i][2])+"|")
#Ploteamos los aciertos y el error
plt.plot(successes,epoch_array,error,epoch_array)
plt.show()
