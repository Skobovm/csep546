import numpy as np
import random


RELU_LEAK = 0.01

def relu(x):
    retval = max(x * RELU_LEAK, x)
    #retval = max(0.0, x)
    return retval

def drelu(x):
    #retval = 1.0 if x > 0.0 else RELU_LEAK
    retval = 1.0 if x > 0.0 else 0.0
    return retval

def act_func(x, sigmoid=True):
    if sigmoid:
        return 1.0/(1.0+np.exp(-x))
    else:
        zero = np.zeros(x.shape)
        #f = np.vectorize(relu)
        #return f(x)
        return np.maximum(x, zero)


def dact_func(x, sigmoid=True):
    if sigmoid:
        return act_func(x)*(1-act_func(x))
    else:
        #f = np.vectorize(drelu)
        #return f(x)
        return np.greater(x, 0).astype(float)


class NeuralNetwork(object):
    def __init__(self, num_input, num_hidden, num_output, use_sigmoid=True):
        self.num_layers = 3
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.use_sigmoid = use_sigmoid

        if self.use_sigmoid:
            input_range = 1.0 / self.num_input ** (1 / 2)
            output_range = 1.0 / self.num_hidden ** (1 / 2)

            # Note: need to be "backwards" for dot product
            self.weights_h_i = np.random.normal(loc=0, scale=input_range, size=(self.num_hidden, self.num_input))
            self.weights_o_h = np.random.normal(loc=0, scale=output_range, size=(self.num_output, self.num_hidden))

        else:
            input_range = (2 / (self.num_input + self.num_hidden)) ** (1 / 2)
            output_range = (2 / (self.num_output + self.num_hidden)) ** (1 / 2)

            # Note: need to be "backwards" for dot product
            self.weights_h_i = np.random.normal(loc=0, scale=input_range, size=(self.num_hidden, self.num_input))
            self.weights_o_h = np.random.normal(loc=0, scale=output_range, size=(self.num_output, self.num_hidden))


        self.errors = []

    def train(self, training_data, epochs, batch_size, eta, momentum=0.0, weight_decay=0.0, test_data=None):
        self.data_len = len(training_data)
        self.momentum = momentum
        self.batch_change_h_i = np.zeros(self.weights_h_i.shape)
        self.batch_change_o_h = np.zeros(self.weights_o_h.shape)
        self.weight_decay = weight_decay


        for epoch in range(epochs):
            random.shuffle(training_data)

            batch_index = 0
            mini_batches = [training_data[k:k + batch_size] for k in range(0, self.data_len, batch_size)]
            for mini_batch in mini_batches:
                self.train_batch(mini_batch, eta)

                # Plot errors every half epoch
                if batch_index == len(mini_batches) / 2 or batch_index == len(mini_batches) - 1:
                    # Use test_data as the switch for evaluation
                    if test_data:
                        tr_s_e = self.test_squared_loss(training_data) / self.data_len
                        te_s_e = self.test_squared_loss(test_data) / len(test_data)

                        tr_a_e = self.test_accuracy(training_data)
                        te_a_e = self.test_accuracy(test_data)

                        error_tuple = (tr_s_e, te_s_e, tr_a_e, te_a_e)
                        print(error_tuple)
                        self.errors.append(error_tuple)
                batch_index += 1

            print('Epoch %s complete' % epoch)

    def train_batch(self, mini_batch, learning_rate):
        batch_delta_w_h_i = np.zeros(self.weights_h_i.shape)
        batch_delta_w_o_h = np.zeros(self.weights_o_h.shape)

        for input, target in mini_batch:
            self.feed_forward(input)
            self.back_propogate(target)
            batch_delta_w_h_i += self.delta_w_h_i
            batch_delta_w_o_h += self.delta_w_o_h

        self.batch_change_h_i = (learning_rate / len(mini_batch)) * batch_delta_w_h_i * (1 - self.momentum) + self.batch_change_h_i * self.momentum
        self.batch_change_o_h = (learning_rate / len(mini_batch)) * batch_delta_w_o_h * (1 - self.momentum) + self.batch_change_o_h * self.momentum

        if self.use_sigmoid:
            self.weights_h_i = self.weights_h_i - self.batch_change_h_i
            self.weights_o_h = self.weights_o_h - self.batch_change_o_h
        else:
            self.weights_h_i = self.weights_h_i - self.batch_change_h_i
            self.weights_o_h = self.weights_o_h - self.batch_change_o_h

        # Decay
        self.weights_h_i *= (1.0 - self.weight_decay)
        self.weights_o_h *= (1.0 - self.weight_decay)

    def feed_forward(self, input):
        self.input_nodes = input
        self.input_output = np.dot(self.weights_h_i, self.input_nodes)
        self.hidden_nodes = act_func(self.input_output, True)

        self.hidden_output = np.dot(self.weights_o_h, self.hidden_nodes)
        self.output_nodes = act_func(self.hidden_output, self.use_sigmoid)


    def back_propogate(self, target):
        grad_o_h = dact_func(self.hidden_output, self.use_sigmoid)
        delta_o_h = (self.output_nodes - target) * grad_o_h
        self.delta_w_o_h = np.dot(delta_o_h, self.hidden_nodes.transpose())

        grad_h_i = dact_func(self.input_output, True)
        layer = self.weights_o_h
        delta = np.dot(layer.transpose(), delta_o_h) * grad_h_i
        self.delta_w_h_i = np.dot(delta, self.input_nodes.transpose())

    # This is the same as feed_forward, but doesn't modify any class properties
    def evaluate(self, input):
        # Input to hidden
        hidden_vals = act_func(np.dot(self.weights_h_i, input), True)

        # Hidden to output
        output = act_func(np.dot(self.weights_o_h, hidden_vals), self.use_sigmoid)
        return output

    def test_squared_loss(self, test_data):
        error = 0.0
        for input, target in test_data:
            prediction = self.evaluate(input)
            error += np.sum(.5 * (target - prediction) ** 2)

        return error

    def test_accuracy(self, test_data):
        predictions = [0] * 10
        correct = 0
        for input, target in test_data:
            prediction = self.evaluate(input)
            max_index = np.argmax(prediction)
            predictions[max_index] += 1
            target_index = np.argmax(target)
            if max_index == target_index:
                correct += 1

        print(predictions)
        return 1 - correct / len(test_data)
