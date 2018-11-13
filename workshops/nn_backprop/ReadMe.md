# Neural networks and backpropagation workshop

This folder contains workshop that complements Neural networks and back-propagation lecture. The purpose of the workshop is to test understanding of back-propagation algorithm through Python implementation.

Whole workshop is implemented in one Python file *nn_bacprop.py*. In the file there is Python implementation of the whole neural network including:
* Sigmoid and identity activation functions
* Softmax with cross entropy layer
* Fully connected layer
* Neural network that combines previous functionality

Each layer implements two methods *forward* and *backward*. Implementations are dummy with comments marked with *TODO* explaining what needs to be implemented.

Network task is to discriminate handwritten digits from *MNIST* dataset (more details about this dataset can be found [here](http://yann.lecun.com/exdb/mnist/))

## Tasks

1) **Implement forward functionality**

For each layer find its *forward* method with dummy implementation. In the method there is a TODO comment explaining how to properly implement forward logic.
After implementing forward logic one can test correctness by running:

```
python nn_backprop.py test
```

Before implementing forward logic output should be:
```
Test neural net .\model\nn.pkl.gz on test set in .\data\mnist.pkl.gz
Test error: 90.420000
```

After implementing forward logic output should be:
```
Test neural net .\model\nn.pkl.gz on test set in .\data\mnist.pkl.gz
Test error: 2.590000
```

As can be seen, after loading trained weights network is able to discriminate handwritten digits with ~97.5% accuracy. With random weights accuracy is ~9% (equal to random guessing).

2) **Implement backward functionality**

NOTE: Implementing backward functionality requires implemented forward from the previous task.

For each layer find its *backward* method with dummy implementation. In the method there is a TODO comment explaining how to properly implement backward logic.
After implementing backward logic one can test correctness by running:

```
python nn_backprop.py train
```
If everything is implemented correctly expected output would be similar to:

```
Epoch   TrainingError%% ValidationError%%       TestError%%
0       4.542000        4.390000        4.740000
1       3.060000        3.400000        3.570000
2       2.310000        2.960000        3.120000
3       1.748000        2.800000        2.860000
4       1.362000        2.720000        2.670000
5       1.134000        2.560000        2.620000
6       0.870000        2.600000        2.600000
7       0.672000        2.460000        2.470000
8       0.528000        2.430000        2.390000
9       0.384000        2.430000        2.370000
```

As can be seen training produces network whose accuracy is similar to network from 1).

## Further work

To further test understanding of the back-propagation one can:

1) Implement sigmoid activation function in *nn_backprop.py*. Replace existing tanh activation with sigmoid. Observe effects.
2) Implement new layer type from scratch and integrate it properly with code in *nn_backprop.py*. New layer can implement y = x^2 functionality for example.