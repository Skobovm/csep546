import numpy as np
from idx_reader import IdxReader
from PIL import Image
import copy
from network import NeuralNetwork
import matplotlib.pyplot as plt


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def plot_errors(errors):
    tr_l = []
    te_l = []
    tr_a = []
    te_a = []
    x_axis = []
    count = 0
    for error in errors:
        count += 1
        tr_l.append(error[0])
        te_l.append(error[1])
        tr_a.append(error[2])
        te_a.append(error[3])
        x_axis.append(count)

    return plt.plot(x_axis, tr_l, 'r--', x_axis, te_l, 'bs')

if __name__ == '__main__':
    data_file = '/Users/skobovm/repos/csep546/assignment3/MNIST_PCA/train-images-pca.idx2-double'
    label_file = '/Users/skobovm/repos/csep546/assignment3/MNIST_PCA/train-labels.idx1-ubyte'
    test_data_file = '/Users/skobovm/repos/csep546/assignment3/MNIST_PCA/t10k-images-pca.idx2-double'
    test_label_file = '/Users/skobovm/repos/csep546/assignment3/MNIST_PCA/t10k-labels.idx1-ubyte'

    data_reader = IdxReader(data_file, 'double')
    label_reader = IdxReader(label_file, 'ubyte')
    test_data_reader = IdxReader(test_data_file, 'double')
    test_label_reader = IdxReader(test_label_file, 'ubyte')

    training_inputs = [np.reshape(x, (50, 1)) for x in data_reader]
    training_results = [vectorized_result(y) for y in label_reader.data]
    training_data = list(zip(training_inputs, training_results))

    test_inputs = [np.reshape(x, (50, 1)) for x in test_data_reader]
    test_results = [vectorized_result(y) for y in test_label_reader.data]
    test_data = list(zip(test_inputs, test_results))

    # Good
    net = NeuralNetwork(num_input=50, num_hidden=10, num_output=10)
    net.train(training_data, 50, 50, .05, momentum=.7, test_data=test_data)
    errors1 = net.errors
    with open('h10.txt', 'w') as fp:
        fp.write('\n'.join(str(s) for s in errors1))

    # OK
    net = NeuralNetwork(num_input=50, num_hidden=50, num_output=10)
    net.train(training_data, 40, 50, .06, momentum=.7, test_data=test_data)
    errors2 = net.errors
    with open('h50.txt', 'w') as fp:
        fp.write('\n'.join(str(s) for s in errors2))

    # OK
    net = NeuralNetwork(num_input=50, num_hidden=100, num_output=10)
    net.train(training_data, 40, 50, .1, momentum=.9, test_data=test_data)
    errors3 = net.errors
    with open('h100.txt', 'w') as fp:
        fp.write('\n'.join(str(s) for s in errors3))

    #Good
    net = NeuralNetwork(num_input=50, num_hidden=500, num_output=10)
    net.train(training_data, 50, 100, .1, momentum=.9, test_data=test_data)
    errors4 = net.errors
    with open('h500.txt', 'w') as fp:
        fp.write('\n'.join(str(s) for s in errors4))

    # Good
    net = NeuralNetwork(num_input=50, num_hidden=1000, num_output=10)
    net.train(training_data, 40, 100, .08, momentum=.9, test_data=test_data)
    errors5 = net.errors
    with open('h1000.txt', 'w') as fp:
        fp.write('\n'.join(str(s) for s in errors5))

    # ReLU
    learning_rates = [.0001]

    for rate in learning_rates:
        try:
            print('Rate: ', rate)
            net = NeuralNetwork(num_input=50, num_hidden=500, num_output=10, use_sigmoid=False)
            net.train(training_data, 50, 1, rate, momentum=0.0, weight_decay=0.0, test_data=test_data)
            errors6 = net.errors
        except Exception as e:
            print(e)
            pass

        with open('rh500.txt', 'w') as fp:
            fp.write('\n'.join(str(s) for s in errors6))
