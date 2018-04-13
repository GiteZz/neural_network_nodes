import numpy as np
import random
from minst_loader import get_mnist_data


class neural_network:
    def __init__(self, layers):
        # layer is array with layer size
        self.layers = []
        self.amount_layers = len(layers)
        for layer_index in range(len(layers)):
            layer_size = layers[layer_index]
            self.layers.append([])
            for _ in range(layer_size):
                if layer_index == 0:
                    self.layers[layer_index].append(neural_node(random.gauss(0.0,1.0), [], []))
                else:
                    self.layers[layer_index].append(neural_node(random.gauss(0.0, 1.0)
                                                                , rand_float_array(layers[layer_index - 1])
                                                                , self.layers[layer_index - 1]))

    def get_output(self, input_vector):
        for input_index in range(len(input_vector)):
            self.layers[0][input_index].set_activation(input_vector[input_index])

        for layer in self.layers[1:]:
            for node in layer:
                node.update()

        last_node_activations = []

        for node in self.layers[self.amount_layers - 1]:
            last_node_activations.append(node.get_activation())

        max_index = last_node_activations.index(max(last_node_activations))

        output = [0]*len(self.layers[self.amount_layers - 1])

        output[max_index] = 1

        return output

    # calculates error of output layer, this is a different formula then the other layers
    def calc_error_last(self, desired):
        # iterate over last layer
        last_layer = self.layers[self.amount_layers - 1]
        for i in range(len(last_layer)):
            # BP formula 1
            node = last_layer[i]
            node.add_error((node.get_activation() - desired[i])*sigmoid_prime(node.weighted_input))

    def calc_error(self, index):
        this_layer = self.layers[index]
        next_layer = self.layers[index + 1]

        for this_index in range(len(this_layer)):
            this_node = this_layer[this_index]
            sum = 0
            for next_index in range(len(next_layer)):
                next_node = next_layer[next_index]
                sum += next_node.weights[this_index] * next_node.get_error()
            this_node.add_error(sum * sigmoid_prime(this_node.weighted_input))

    def back_propagate(self,x,y):
        # runs through batch and errors/activations get saved in the node itself
        for input, output in zip(x, y):
            self.get_output(input)
            # this calculates the error of the last layer, diff from other layers
            self.calc_error_last(output)
            # loop starts at second to last layer and runs to first layer
            for i in range(self.amount_layers -2, -1, -1):
                self.calc_error(i)

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
                        sum += prev_layer[weight_index].outputs[batch_index] * this_node.errors[batch_index]
                    this_node.weights[weight_index] -= (eta/this_node.batch_amount) * adj_sum

                # adjusts bias
                this_node.bias -= (eta/this_node.batch_amount) * sum(this_node.errors)
                this_node.reset_batch()

    def test_batch(self, batch, eta):
        # batch[0] is the input for the NN, batch[1] contains the desired output
        self.back_propagate(batch[0], batch[1])
        self.adjust_weights_bias(eta)

    def learn(self, eta, batch_size, training, epoch):
        training_size = len(training)
        for _ in range(epoch):
            random.shuffle(training)
            for i in range(0,training_size, batch_size):
                batch = training[i: i + batch_size]
                self.test_batch(batch, eta)

    def test(self, testing):
        test_amount = len(testing)
        hits = 0
        for i in range(len(testing)):
            hits += self.get_output(testing[i][0]) == testing[i][1]

        return hits/test_amount


class neural_node:
    def __init__(self, bias, weigths, input_nodes):
        self.outputs = []
        self.last_output = None
        self.weighted_input = None
        self.bias = bias
        self.weights = weigths
        self.input_nodes = input_nodes
        self.errors = []
        self.last_error = None
        self.batch_amount = 0

    def update(self):
        total_sum = self.bias

        for node_index in range(len(self.input_nodes)):
            total_sum = self.weights[node_index] * self.input_nodes[node_index].get_activation()
        self.weighted_input = total_sum
        new_value = sigmoid(self.weighted_input)
        self.outputs.append(new_value)
        self.last_output = new_value
        self.batch_amount += 1

    def set_activation(self, value):
        self.outputs.append(value)
        self.last_output = value

    def get_activation(self):
        return self.last_output

    def add_error(self, error):
        self.last_error = error
        self.errors.append(error)

    def get_error(self):
        return self.last_error

    def reset_batch(self):
        self.batch_amount = 0
        self.last_output = None
        self.last_error = None
        self.errors = []
        self.outputs = []


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
    nn = neural_network([784,784,10])

    training, testing = get_mnist_data()
    nn.learn(0.001, 20, training, 1)
    print("hitrate: " + str(nn.test(testing)))
    print(nn.get_output([0.5,0.3,0.9]))
    a = 5