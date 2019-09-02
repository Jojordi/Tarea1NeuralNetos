**Como correr la Tarea:**

Para inicializar y entrenar una red neuronal de 5 capas escondidas, que recibe un input
de 7 características y entrega un output de 3 correr el archivo Main.py

Lo que hará dicho archivo es:

    -Primero crear la red neuronal.
    -Entrenarla por 100 épocas con el 80% de los datos del dataset Iris.
    -Computar el error como se describe en el enunciado de la tarea
    -Computar la cantidad de éxitos por época
    -Crear e imprimir en consola la matriz de confusión
    -Crear un gráfico con los aciertos por época, (en azul), y el error,(en rojo).

**Implementación**


Respecto a la implementación de la tarea, se crea primero la clase Neuron encargada del funcionamiento de las neuronas.

Luego se crea la clas Layer que se encarga del correcto funcionmiento de un grupo de neuronas todas en la misma capa.

Finalmente se creó la clase NeuralNetwork la cual hace uso de la clase de capas y la implementación de neuronas.

Se debe destacar que al implementar el método de backpropagation para NeuralNetwork se trabajó directamente con neuronas 
y como aquellas en forma de capa.

Se creó una clase encargada de leer los datos desde un archivo de texto, parsearlos, normalizarlos, hacer el 1-hot
encoding y finalmente pasarlo a una matriz que puediese ser aceptada por la red

**Dificultades**

La principal dificultad encontrada durante el desarrollo de esta tarea fue la implementación de backpropagation.
Debido a la falta de claridad por como funciona este método se cree que existen bastantes errores potenciales que pueden
explicar el desempeño mediocre que obtuvo la red.




    
    
