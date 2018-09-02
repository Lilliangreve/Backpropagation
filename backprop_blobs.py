
import numpy as np
from sklearn.metrics import log_loss
from sklearn.datasets import make_blobs, make_classification
import matplotlib.pyplot as plt

X = np.array([1, 0])
y = np.array([1])

X, y = make_blobs(centers=2, random_state=42)
# X, y = make_classification(n_features=2, n_redundant=0, n_informative=1,
#                             n_clusters_per_class=1)
print(X.shape, y.shape)


# plt.plot(X[:,0], X[:,1], 'bo')
# plt.show()


def sigmoid(x):
    """Calculates the sigmoid function"""
    return 1/(1+np.exp(-x))


def initialize_weights(n):
    """Returns random weights"""
    # could try normal biases
    np.random.seed(42)
    weights = np.random.rand(n)*0.01
    weights = weights.reshape((3, 3))
    return weights


def loss(ytrue, ypred):
    """returns a scalar that tells how far we are off"""
    return (ytrue-ypred)**2


def sigmoid_derivative(x):
    """returns the values of sig'(x)"""
    sig = sigmoid(x)
    return sig*(1-sig)


def feed_forward(x, w, activation):
    """Runs data through the neural network and returns a prediction"""
    # hidden layer
    in_data = np.array([x[:, 0], x[:, 1], np.ones(x.shape[0])])
    neuron1 = np.dot(in_data.T, w[0])
    n1 = activation(neuron1)
    neuron2 = np.dot(in_data.T, w[1])
    n2 = activation(neuron2)

    # output layer
    in2 = np.array([n1, n2, 1])
    output = np.dot(in2, w[2])

    ypred = activation(output)
    return (neuron1, neuron2, n1, n2, output, ypred)


def backprop(x, y, w, learning_rate, n_iter):
    """Return optimized weights for the network"""
    for i in range(n_iter):
        # Calculate the output of both layers
        neuron1, neuron2, n1, n2, out, ypred = feed_forward(X, w, sigmoid)

        # Calculate the loss (difference of predicted and correct output)
        error = loss(y, ypred)

        # Modify each weight of the output layer by:
        # sig'(output) * loss * hidden_output
        oldw = w.copy()
        w[2, 0] -= sum(sigmoid_derivative(out)*n1*(ypred-y))*learning_rate
        w[2, 1] -= sum(sigmoid_derivative(out)*n2*(ypred-y))*learning_rate
        w[2, 2] -= sum(sigmoid_derivative(out) * 1*(ypred-y))*learning_rate

        # Calculate hidden_loss as sig'(output) * output_weight for each
        # hidden neuron
        # Modify each weight in the hidden layer by:
        # -sig'(hidden_output) * hidden_loss * input features
        # PROBLEM: doesn't backpropagate to hidden layer
        w[0, 0] -= sum(sigmoid_derivative(out)*(ypred-y)*oldw[2, 0]
                       * sigmoid_derivative(neuron1)*X[:, 0])*learning_rate
        w[0, 1] -= sum(sigmoid_derivative(out)*(ypred-y)*oldw[2, 0]
                       * sigmoid_derivative(neuron1)*X[:, 1])*learning_rate
        w[0, 2] -= sum(sigmoid_derivative(out)*(ypred-y)*oldw[2, 0]
                       * sigmoid_derivative(neuron1))*learning_rate
        w[1, 0] -= sum(sigmoid_derivative(out)*(ypred-y)*oldw[2, 1]
                       * sigmoid_derivative(neuron2)*X[:, 0])*learning_rate
        w[1, 1] -= sum(sigmoid_derivative(out)*(ypred-y)*oldw[2, 1]
                       * sigmoid_derivative(neuron2)*X[:, 1])*learning_rate
        w[1, 2] -= sum(sigmoid_derivative(out)*(ypred-y)*oldw[2, 1]
                       * sigmoid_derivative(neuron2))*learning_rate

        _, _, _, _, _, ypred = feed_forward(X, w, sigmoid)
        print("loss: ", sum(np.abs(loss(y, ypred))))

    return w


if __name__ == '__main__':

    w = initialize_weights(9)
    w = backprop(X, y, w, 0.01, 1000)

    print("-" * 40)
    print("FINAL RESULT:")
    #print("X: ", X)
    #print("w: ", w)

    _, _, _, _, _, ypred = feed_forward(X, w, sigmoid)
    yp = np.round(ypred)
    # print("ypred: ", yp)

    acc = 1 - sum(np.abs(yp - y)) / 100
    print(acc)

    error1 = loss(y, ypred)
    print("loss: ", sum(np.abs(error1)))
