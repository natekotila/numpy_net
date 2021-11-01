"""
Implementing a Logistic Deep Neural Network in Numpy
"""

import numpy as np

"""
Activation Functions
"""


def sigmoid(Z):
    """
    Implements the sigmoid function s(Z) = 1 / (1 + exp(-Z))

    Parameters:
        Z : array_like
            Real valued matrix for activation layer

    Returns:
        S : sigmoid function value for Z
    """

    S = 1 / (1 + np.exp(-Z))

    return S


def sigmoid_grad(Z):
    """
    Calculates the derivative of the sigmoid function, evaluated at Z

    Parameters:
        Z : array_like

    Returns:
        sg : array_like
        (1 - sigmoid(Z))*(sigmoid(Z))
    """
    s = sigmoid(Z)
    sg = s * (1 - s)

    return sg

def softmax(Z):
    """
    Calculates the softmax vector S = e^z_i / sum(e^z_i)
    for i = 1, 2, ..., K possible outcomes

    Parameters:
        Z: array_like
    
    Returns:
        S: softmax function for Z
    """

    e_Z = np.exp(Z)
    denom = sum(e_Z)

    S = e_Z / denom

    assert(S.shape == Z.shape)

    return S

def tanh(Z):
    """
    Calculates the hyperbolic tangent of Z:
    tanh(Z) = (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))

    Parameters:
        Z : array_like

    Returns:
        tan_h : array_like
    """

    numerator = np.exp(Z) - np.exp(-Z)
    denominator = np.exp(Z) + np.exp(-Z)

    tan_h = numerator / denominator

    return tan_h


def tanh_grad(Z):
    """
    Calculates the derivative of the hyperbolic tangent of Z

    Parameters:
        Z : array_like

    Returns:
        tg : array_like
        tanh'(Z) = 1 - (tanh(Z)) ** 2
    """

    t = tanh(Z)
    tg = 1 - t ** 2

    return tg


def relu(Z):
    """
    Calculates the Rectified Linear Unit of a real value Z

    Parameters:
        Z : array_like

    Returns:
        r : array_like
            Elementwise max(Z, 0)
    """
    # Absolute value of Z is used to avoid getting -0 values
    r = (Z > 0) * abs(Z)

    return r


def relu_grad(Z):
    """
    Calculates the derivative of the ReLU function, with a minor modification:
    relu'(Z < 0) = -.01 instead of 0 to help with gradient descent

    Parameters:
        Z : array_like

    Returns:
        rg: array_like
            1 if Z > 0, -.01 otherwise
    """

    rg = 1. * (Z > 0) - .01 * (Z < 0)

    return rg


def linear(A, W, b):
    """
    Calculates the linear step preceding each activation:
    Z = WA + b

    Parameters:
        A : array_like
            Activation matrix for previous layer of the Network
            X if this is the first linear transformation

        W : array_like
            Weight Matrix

        b : array_like
            Biases for current layer of the network
    """

    Z = np.dot(W, A) + b

    return Z


def logistic_cost(Y_hat, Y, weights):
    """
    Computes the cost function associated with logistic regression

    Parameters:
        Y_hat : array_like
                Predicted values from the neural network for current step

        Y : array_like
            True values for input files (binary classification)

        weights : dictionary

    Returns:
        cost : float
    """

    m = Y.shape[1]

    arg = -weights['positive'] * Y * np.log(Y_hat) \
        - weights['negative'] * (1 - Y) * np.log(1 - Y_hat)

    cost = 1 / m * np.sum(arg)
    assert cost.shape == ()

    return cost


def initialize_weights(layer_dims):
    """
    Initialize the weights and biases.

    Each set of weights will be a matrix with dimensions:
    layer_dims[l] by layer_dims[l-1], and will randomly be initialized near 0.

    Biases will all be column vectors of dimension
    layer_dims[l] initialized to 0.

    Parameters:
        layer_dims: list
                    The number of nodes in each layer of the Network

    Returns:
        params: dictionary
                A Dictionary of all weights and biases for the network
    """

    L = len(layer_dims)
    params = {}

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l],
                                               layer_dims[l-1]) * .1
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return params


def forward_propagation(A_prev, W, b, activation='relu'):
    """
    Single Step of Forward Propagation for m samples

    Parameters:
        A_prev : array_like
            Dimensions: (layer_dims, m)
            Real valued input for linear transformation
            Activation function output for previous step,
            or X if this is step 1

        W : array_like

            Dimensions: (A.shape[0], A_prev.shape[0])

        b : array_like
            Bias Vector
            Dimensions: (A.shape[0], 1)

        activation : String
            Determines which function is performed after linear step.
            Chlices are: 'relu' Regularized Linear Unit
                         'sigmoid' Sigmoid Function
                         'tanh' Hyperbolic Tangent Function

    Returns:
        A : array_like
            Matrix for Activation Layer

        Z : array_like
            Linear transformation of Previous Layer with Weights and Biases
            Stored for back propagation step
    """

    Z = linear(A_prev, W, b)

    if (activation == 'relu'):
        A = relu(Z)
    elif (activation == 'sigmoid'):
        A = sigmoid(Z)
    elif (activation == 'tanh'):
        A = tanh(Z)
    elif (activation == 'softmax'):
        A = softmax(Z)
    return A, Z


def forward_propagation_L_layers(X, params, activations=None):
    """
    Implementation of Forward Propagation through all layers of a Deep Network

    Parameters:
        X : array_like
            Input values for neural network

        params : array_like
                 Weights and Biases calculated at this point

        activations : list of String values
                      Optional argument for using different activation
                      functions throughout the network. Default is ReLu
                      for Hidden Layers and Sigmoid for Output Layer

    Returns:
        Y_hat : array_like
                Predictions for the current loop through the network

        caches : dictionary
                 Computed Z and A values at each layer of the network.
                 Stored for future use in Backward Propagation
    """
    A_prev = X
    L = len(params) // 2

    caches = {}
    caches['A0'] = X

    # Loop through the Hidden Layers of the Neetwork layers 1 -> L-1
    for l in range(1, L):
        A_prev, caches['Z' + str(l)] = forward_propagation(
                                                          A_prev,
                                                          params['W' + str(l)],
                                                          params['b' + str(l)],
                                                          activation='relu')
        caches['A' + str(l)] = A_prev

    # Output layer of Network
    Y_hat, caches['Z' + str(L)] = forward_propagation(
                                                     A_prev,
                                                     params['W' + str(L)],
                                                     params['b' + str(L)],
                                                     activation='sigmoid')
    caches['A' + str(L)] = Y_hat

    return Y_hat, caches


def back_propagation(Y, cache, params, weights):
    """
    Compute the gradient values for each step of the Neural Network
    """
    grads = {}
    L = len(params) // 2
    Y_hat = cache['A' + str(L)]
    m = Y_hat.shape[1]

    grads['dA' + str(L)] = -weights['positive'] * np.divide(Y, Y_hat) \
        + weights['negative'] * np.divide(1 - Y, 1 - Y_hat)
    grads['dZ' + str(L)] = grads['dA' + str(L)] * sigmoid_grad(
                                                    cache['Z' + str(L)])
    assert grads['dZ' + str(L)].shape == cache['Z' + str(L)].shape

    grads['dW' + str(L)] = 1 / m * np.dot(
                                          grads['dZ' + str(L)],
                                          cache['A' + str(L - 1)].T)
    grads['db' + str(L)] = 1 / m * np.sum(
                                          grads['dZ' + str(L)],
                                          axis=1,
                                          keepdims=True)

    for l in reversed(range(1, L)):
        grads['dA' + str(l)] = np.dot(
                                      params['W' + str(l + 1)].T,
                                      grads['dZ' + str(l + 1)])

        grads['dZ' + str(l)] = np.multiply(
                                           grads['dA' + str(l)],
                                           relu_grad(cache['Z' + str(l)]))
        assert grads['dZ' + str(l)].shape == cache['Z' + str(l)].shape

        grads['dW' + str(l)] = 1 / m * np.dot(
                                              grads['dZ' + str(l)],
                                              cache['A' + str(l - 1)].T)
        assert grads['dW' + str(l)].shape == params['W' + str(l)].shape

        grads['db' + str(l)] = 1 / m * np.sum(
                                              grads['dZ' + str(l)],
                                              axis=1,
                                              keepdims=True)

    return grads


def update_params(params, grads, learning_rate):
    """

    """
    L = len(params) // 2

    for l in range(1, L + 1):
        params['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        params['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return params
