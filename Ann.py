import random
import numpy as np
import math
output = [0, 0]


class Ann:

    # Neural network class. Each robot will have its own neural network instance.

    def __init__(self, layers, alpha):

        """
        In this example I use a list with 5 input values since I used 5 neurons in the first layer,
        but we can also use multiple input values for each neuron
        :param layers: The architecture of the network
        :param weights: The weights, which are basically the genotype
        """
        self.layers = layers

        self.prev_output = [0, 0]
        self.weights = self.initialize_weights(layers[0], layers[1], layers[2])
        self.alpha = alpha

        # Initialize a network
    def initialize_weights(n_inputs, n_hidden, n_outputs):
        network = list()
        hidden_layer = [[random() for i in range(n_inputs + 1)] for i in range(n_hidden)]
        network.append(hidden_layer)
        output_layer = [[random() for i in range(n_hidden + 1)] for i in range(n_outputs)]
        network.append(output_layer)
        return network

    def create_weights_lists(self):

        """
        This method creates different weights lists given the whole genotype.
        Each list represents the weights for a specific layer.
        :return: Different weights lists, the amount of lists is equal to the number of layers - 1
        """
        weights_lists = []
        i = 0
        weights_lists.append(self.genotype[:self.layers[0] * self.layers[0] + self.layers[0] * len(self.prev_output)])
        weights = self.genotype[self.layers[0] * self.layers[0] + self.layers[0] * len(self.prev_output):]
        while i < len(self.layers) - 1:
            weights_lists.append(weights[:self.layers[i] * self.layers[i + 1]])
            weights = weights[self.layers[i] * self.layers[i + 1]:]
            i += 1
        return weights_lists

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

    def feedforward(self, sensor_input, weights_lists):

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
        new_l = layer[0]
        new_r = layer[1]
        if new_l > 0:
            new_l = min(new_l, self.max_vel)
        else:
            new_l = max(new_l, -self.max_vel)
        if new_r > 0:
            new_r = (min(new_r, self.max_vel))
        else:
            new_r = (max(new_r, -self.max_vel))

        self.prev_output = [new_l, new_r]

        return self.prev_output

