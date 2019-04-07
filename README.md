# naive-neuralnet
A naive implementation of dynamic dense neural network training on MNIST-digits

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to run the software and how to install them

```
Python 3.*
Numpy
```

### Installing

A step by step series of examples that tell you how to get a development env running

```
git clone https://github.com/AndreFagereng/naive-neauralnet.git
```
Installing requirements
```
pip3 install -r requirements
```
Running the solution to download and train the neural network.

Options: 
- Set the layer dimensions for the neural network
- Set -v for verbose training information
- Set -g for accuracy/loss graphs after training
- Set -b for batch-size
- Set -r for learning rate
- Set -e for epochs
```
Default values

Dimensions:    [784,128,10]
Learning rate: 0.01
Batch-size:    128
Epochs:        100

Example:

- python3 main.py -v -g 

- python3 main.py -v -g -l 784,128,64,32,10 -b 64 -r 0.1 -e 1000 

```


