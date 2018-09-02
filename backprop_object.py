'''code adapted from https://dev.to/shamdasani/
build-a-flexible-neural-network-with-backpropagation-in-python'''

import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X/np.amax(X, axis=0)  # maximum of X array (each column)
y = y/100  # max test score is 100


class NeuralNetwork(object):
    def __init__(self):
        # constructor -> creates an instance
        # parameters
        self.input_size = 2
        self.output_size = 1
        self.hidden_size = 3
        # (3x2) weight matrix from input to hidden layer:
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        # (3x1) weight matrix from hidden to output layer:
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        '''forward propagation through our network'''
        # dot product of X (input) and first set of 3x2 weights
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)  # activation function
        # dot product of hidden layer (z2) and second set of 3x1 weights
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def sigmoid(self, s):
        '''activation function'''
        return 1/(1+np.exp(-s))

    def sigmoid_prime(self, s):
        '''derivative of sigmoid'''
        return s * (1 - s)

    def backward(self, X, y, o):
        '''backward propgate through the network'''
        self.o_error = y - o  # error in output
        # applying derivative of sigmoid to error
        self.o_delta = self.o_error*self.sigmoid_prime(o)

        # z2 error: how much hidden layer weights contributed to output error
        self.z2_error = self.o_delta.dot(self.W2.T)
        # applying derivative of sigmoid to z2 error
        self.z2_delta = self.z2_error*self.sigmoid_prime(self.z2)

        # adjusting first set (input --> hidden) weights
        self.W1 += X.T.dot(self.z2_delta)
        # adjusting second set (hidden --> output) weights
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


NN = NeuralNetwork()  # this is an 'instance'
for i in range(1000):  # trains the NN 1,000 times
    NN.train(X, y)

print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" + str(NN.forward(X)))
# mean sum squared loss
print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))))
print("\n")
