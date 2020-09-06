# Basic Neural Network
This is an implementation of a basic neural network. The goal is to understand and implement the fundamental concepts. 

## Structure and Compenents
A neural network consists of layers of neurons, where $n^l$ is the number of neurons in the $l$-th layer.  
Each neuron is connected to all neurons in the adjacent layers. These connections are weighted and generally just called **weights**. The weights from the ($l-1$)-th layer to the $l$-th layer are represented as a $n^{l-1} \times n^l$ Matrix $W^{l}$, where $W^{l}_{jk}$ is the weight from the $k$-th neuron in the ($l-1$)-th layer to the $j$-th neuron in the $l$-th layer.  
Additionally a nonlinear **activation function** $g$ is needed and each neuron also has a **bias** value. The bias values for the $l$-th layer are represented as a $n^l \times 1$ vector $b^l$.

## Feedforward
To generate an ouput for a given input, the values for each neuron have to be calculated from the first to the last layer, this is called **feedforward**.  
The values in the $l$-th layer $y^l_1, y^l_2, ...$ are represented as a single $n^l \times 1$ vector $y^l$. The input specifies these for the first layer. In the following layers the bias and all values from the previous layer multiplied with their weight are summed up. This can be done efficiently with matrix multiplication $z^l = W^ly^{l-1}$. Then the activation function $g$ is applied and the result $y^l = g(z^l)$ is the value of the neuron. The values in the last layer represent the output.