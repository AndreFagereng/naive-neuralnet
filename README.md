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

```
Example:

Verbose training information and default layer-dimensions [784, 128, 10]
- python3 main.py -v 

Verbose training information and choosen layer-dimensions [784, 128, 64, 32, 10]
- python3 main.py -v -l 784,128,64,32,10
```


