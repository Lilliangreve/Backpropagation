import numpy as np

'''Backprop is a method for calculating the gradient of the model’s error with
respect to every weight in the model. We do it so that we can slightly update
each weight via gradient descent in order to reduce the model’s error.'''


def main():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.float64)
    w = initialize_weights(9)
    w = backpropagation(x, y, w, 0.1, 1000)
    print("-" * 40)
    print("FINAL RESULT:")
    print("X: ", x)
    print("w: ", w)
    _, _, ypred = feed_forward(x, w, sigmoid)
    print("ypred: ", ypred)
    print("total loss: ", total_loss(y, ypred))


def sigmoid(x):
    '''activation function'''
    return 1/(1 + np.exp(-x))


def initialize_weights(n):
    '''Returns random weights'''
    w = np.random.rand(n) * 0.1
    w = w.reshape((3, 3))
    return w


def total_loss(ytrue, ypred):
    '''returns a scalar that tell how far we are off'''
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(ypred, ytrue))
    #loss = log_loss(ytrue, ypred)
    total_loss = sum(ytrue-ypred)**2
    return total_loss


def sigmoid_derivative(x):
    '''returns the values of sig'(x)'''
    sig = sigmoid(x)
    return sig*(1-sig)


def feed_forward(x, w, activation):
    '''runs data through the neural network and returns a prediction'''
    neuron1 = (x[:, 0]*w[0, 0]) + (x[:, 1]*w[0, 1]) + w[0, 2]
    neuron1 = activation(neuron1)
    neuron2 = (x[:, 0]*w[1, 0])+(x[:, 1]*w[1, 1])+w[1, 2]
    neuron2 = activation(neuron2)
    output = (neuron1*w[2, 0])+(neuron2*w[2, 1])+w[2, 2]
    return (neuron1, neuron2, activation(output))


def backpropagation(x, y, w, learning_rate, epochs):
    '''run the feed forward network, calculate the loss, adjust the weights of
    the output layer, propagate to the hidden layer, adjust weights, iterate'''
    for i in range(epochs):
        # 2
        n1, n2, ypred = feed_forward(x, w, sigmoid)
        # 3. calculate the loss for each data point
        loss = (y - ypred) ** 3
        print("loss: ", sum(loss))
        # 4. modify each weight
        oldw = w.copy()
        w[2, 0] += sum(sigmoid_derivative(ypred)*loss*n1)
        w[2, 1] += sum(sigmoid_derivative(ypred)*loss*n2)
        w[2, 2] += sum(sigmoid_derivative(ypred)*loss*1)
        # 5
        hidden_loss1 = sigmoid_derivative(ypred) * oldw[2, 0]
        hidden_loss2 = sigmoid_derivative(ypred) * oldw[2, 1]
        # 6
        w[0, 0] -= sum(sigmoid_derivative(n1) *
                       hidden_loss1 * x[:, 0]) * learning_rate
        w[0, 1] -= sum(sigmoid_derivative(n1) *
                       hidden_loss1 * x[:, 1]) * learning_rate
        w[0, 2] -= sum(sigmoid_derivative(n1) * loss * 1.0) * learning_rate
        w[1, 0] -= sum(sigmoid_derivative(n1) *
                       hidden_loss1 * x[:, 0]) * learning_rate
        w[1, 1] -= sum(sigmoid_derivative(n1) *
                       hidden_loss1 * x[:, 1]) * learning_rate
        w[1, 2] -= sum(sigmoid_derivative(n1) * loss * 1.0) * learning_rate

    return w


if __name__ == '__main__':
    main()
