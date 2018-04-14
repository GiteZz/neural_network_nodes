import numpy as np
import random
from minst_loader import get_mnist_data
import _pickle
from multiprocessing import Pool as ThreadPool
import itertools

class neural_network:
    def __init__(self, layers, batch_amount):
        # layer is array with layer size
        self.layers = []
        self.amount_layers = len(layers)
        for layer_index in range(len(layers)):
            layer_size = layers[layer_index]
            self.layers.append([])
            for _ in range(layer_size):
                if layer_index == 0:
                    self.layers[layer_index].append(neural_node(random.gauss(0.0,1.0), [], [],batch_amount))
                else:
                    self.layers[layer_index].append(neural_node(random.gauss(0.0, 1.0)
                                                                , rand_float_array(layers[layer_index - 1])
                                                                , self.layers[layer_index - 1]
                                                                , batch_amount))

    def calc_output(self, input_vector, index=0):
        for input_index in range(len(input_vector)):
            self.layers[0][input_index].outputs[index] = input_vector[input_index]

        for layer in self.layers[1:]:
            for node in layer:
                node.update(index=index)

    def get_output(self, input_vector):
        self.calc_output(input_vector)
        last_node_activations = []

        for node in self.layers[self.amount_layers - 1]:
            last_node_activations.append(node.get_activation())

        max_index = last_node_activations.index(max(last_node_activations))

        output = [0] * len(self.layers[self.amount_layers - 1])

        output[max_index] = 1

        return output

    # calculates error of output layer, this is a different formula then the other layers
    def calc_error_last(self, desired, index=0):
        # iterate over last layer
        last_layer = self.layers[self.amount_layers - 1]
        for i in range(len(last_layer)):
            # BP formula 1
            node = last_layer[i]
            node.add_error((node.outputs[index] - desired[i])*sigmoid_prime(node.weighted_input[index]))

    def calc_error(self, layer_index, index=0):
        this_layer = self.layers[layer_index]
        next_layer = self.layers[layer_index + 1]

        for this_node_index in range(len(this_layer)):
            this_node = this_layer[this_node_index]
            sum = 0
            for next_node_index in range(len(next_layer)):
                next_node = next_layer[next_node_index]
                sum += next_node.weights[this_node_index] * next_node.errors[index]
            this_node.errors[index] = sum * sigmoid_prime(this_node.weighted_input[index])

    def back_propagate_batch(self, batch):
        # runs through batch and errors/activations get saved in the node itself
        batch_amount = len(batch)

        pool = ThreadPool(batch_amount)
        pool.starmap(self.back_propagate,
                     zip(range(batch_amount), batch))

        pool.close()
        pool.join()

    def back_propagate(self, index, tup):
        self.calc_output(tup[0], index=index)
        # this calculates the error of the last layer, diff from other layers
        self.calc_error_last(tup[1], index=index)
        # loop starts at second to last layer and runs to first layer
        for layer_index in range(self.amount_layers - 2, 0, -1):
            self.calc_error(layer_index, index=index)

    def adjust_weights_bias(self, eta):
        for i in range(self.amount_layers - 1, 0, -1):
            this_layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            for node_index in range(len(this_layer)):
                this_node = this_layer[node_index]

                # adjusts weights
                for weight_index in range(len(this_node.weights)):
                    adj_sum = 0
                    for batch_index in range(this_node.batch_amount):
                        adj_sum += prev_layer[weight_index].outputs[batch_index] * this_node.errors[batch_index]
                    this_node.weights[weight_index] -= (eta/this_node.batch_amount) * adj_sum

                # adjusts bias
                this_node.bias -= (eta/this_node.batch_amount) * sum(this_node.errors)
                this_node.reset_batch()

    def test_batch(self, batch, eta):
        # batch[0] is the input for the NN, batch[1] contains the desired output
        self.back_propagate_batch(batch)
        self.adjust_weights_bias(eta)

    def learn(self, eta, batch_size, training, epoch):
        training_size = len(training)
        for _ in range(epoch):
            random.shuffle(training)
            for i in range(0, training_size, batch_size):
                print("currently on batch: " + str(i))
                batch = training[i: i + batch_size]
                self.test_batch(batch, eta)

    def test(self, testing):
        test_amount = len(testing)
        hits = 0
        for i in range(len(testing)):
            hits += self.get_output(testing[i][0]) == testing[i][1]

        return hits/test_amount


class neural_node:
    def __init__(self, bias, weigths, input_nodes, batch_amount):
        self.outputs = [0.0] * batch_amount
        self.weighted_input = [0.0] * batch_amount
        self.bias = bias
        self.weights = weigths
        self.input_nodes = input_nodes
        self.errors = [0.0] * batch_amount
        self.batch_amount = batch_amount

    def update(self, index=0):
        total_sum = self.bias

        for node_index in range(len(self.input_nodes)):
            total_sum = self.weights[node_index] * self.input_nodes[node_index].outputs[index]
        self.weighted_input[index] = total_sum
        self.outputs[index] = sigmoid(self.weighted_input[index])


    def set_activation(self, value, index=0):
        self.outputs[index] = value

    def get_activation(self, index=0):
        return self.outputs[0]

    def add_error(self, error, index=0):
        self.errors[index] = error

    def get_error(self, index=0):
        return self.errors[index]

    def reset_batch(self):
        self.errors = [0.0] * self.batch_amount
        self.outputs = [0.0] * self.batch_amount
        self.weighted_input = [0.0] * self.batch_amount


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def rand_float_array(size):
    ret_ar = []

    for _ in range(size):
        ret_ar.append(random.gauss(0.0,1.0))

    return ret_ar


def cost_function(desired, output):
    sum = 0
    for i in range(len(desired)):
        sum += (desired[i] - output[i])*(desired[i] - output[i])
    sum /= (2*len(desired))
    return sum


if __name__ == "__main__":
    nn = neural_network([784,30,10], 10)
    training, testing = get_mnist_data()
    nn.learn(3, 10, training, 29)
    """
    with open('outfile', 'rb') as fp:
        tuple_list = _pickle.load(fp)
    """
    #nn.learn(0.001, 10, tuple_list, 3)
    print("hitrate: " + str(nn.test(testing)))
    #print(nn.get_output([0,0,0]))
    a = 5