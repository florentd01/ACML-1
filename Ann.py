from random import random
import numpy as np
import math



class Ann:

    # Neural network class. Each robot will have its own neural network instance.

    def __init__(self, layers, alpha):

        """
        In this example I use a list with 5 input values since I used 5 neurons in the first layer,
        but we can also use multiple input values for each neuron
        :param layers: The architecture of the network
        """
        self.layers = layers
        self.weights = self.initialize_weights(layers[0], layers[1], layers[2])
        self.alpha = alpha

        # Initialize a network
    def initialize_weights(self, n_inputs, n_hidden, n_outputs):
        layers = list()
        hidden_layer = [[random() for i in range(n_inputs + 1)] for i in range(n_hidden)]
        layers.append(hidden_layer)
        output_layer = [[random() for i in range(n_hidden + 1)] for i in range(n_outputs)]
        layers.append(output_layer)
        return layers

    def split_list(self, weights_list, n):

        """
        Given the lists of weights for one layer, this method splits it into different lists,
        each of which refers to a particular neuron that has to be computed in the next layer
        :param weights_list: One of the previously calculated lists of weights
        :param n: The number that indicates how many parts the list should be divided into
        :return: n different lists of equal length
        """

        weights_array = np.array(weights_list)
        weights_array = np.split(weights_array, n)
        matrix_weights = np.array(weights_array)
        return matrix_weights

    def relu(self, Z):
        """
        Just the implementation of the relu activation function
        :param Z: Tne value of the neuron
        :return: 0 if the neuron is turned off, Z otherwise
        """
        return np.maximum(0, Z)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def feedforward(self, input):
        activation = list()
        input.append(1)
        #weights = self.split_list(self.weights[0], self.layers[0])
        weights = self.weights[0]
        activation = self.sigmoid(np.dot(weights, input))
        for i in range(1, len(self.layers)-1):
            #weights = self.split_list(self.weights[i], self.layers[i])
            weights = self.weights[i]
            activation = np.append(activation, 1)
            activation = self.sigmoid(np.dot(weights, activation))
        
        return activation


    def feedforward2(self, sensor_input, weights_lists):

        real_input = sensor_input + self.prev_output
        """
        Feedforward routine implementation
        :param weights_lists: All the lists of weights previously computed
        :return: The output of the network, which are the two wheel velocities
        """
        i = 1
        weights = self.split_list(weights_lists[0], self.layers[0])
        layer = self.sigmoid(np.dot(weights, real_input))
        while i < len(self.layers)-1:
            weights = self.split_list(weights_lists[i], self.layers[i])
            layer = self.sigmoid(np.dot(weights, layer))
            i += 1
        weights = self.split_list(weights_lists[i], self.layers[i])
        layer = np.dot(weights, layer)
        

        return layer
    
    def backpropagation(self, input, target):
        """
        Backpropagation algorithm for training neural network
        :param input: Input data for training
        :param target: Target output for training
        :return: Error for monitoring training process
        """
        alpha = self.alpha
        weights = list()
        for i in range(len(self.weights)):
            weights.append(np.array(self.weights[i]))


        output = self.feedforward(input)
        output_error = target - output

        delta = output_error * (output * (1 - output))
        weight_deltas = []

        for i in range(len(self.layers) - 1, 0, -1):
            prev_layer_output = output if i == len(self.layers) - 1 else self.feedforward(input)

            weight_delta = alpha * np.outer(delta, prev_layer_output)
            weight_deltas.insert(0, weight_delta)

            weights_transpose = weights[i - 1].T
            delta = np.dot(weights_transpose[:-1, :], delta) * (prev_layer_output * (1 - prev_layer_output))

        for i in range(len(self.weights)):
            self.weights[i] += weight_deltas[i]

        error = np.sum(output_error ** 2)
        return error

