import configparser
import csv

from datetime import datetime
from os import path
from random import random, shuffle, uniform

from matplotlib import pyplot as plt

from settings import PROJECT_ROOT

class LogicData:
    and_data_scheme_unipolar = [([0, 0], 0), ([1, 0], 0), ([0, 1], 0), ([1, 1], 1)]
    or_data_scheme_unipolar =  [([0, 0], 0), ([1, 0], 1), ([0, 1], 1), ([1, 1], 1)]
    xor_data_scheme_unipolar = [([0, 0], 0), ([1, 0], 1), ([0, 1], 1), ([1, 1], 0)]
    and_data_scheme_bipolar =  [([-1, -1], -1), ([1, -1], -1), ([-1, 1], -1), ([1, 1], 1)]
    or_data_scheme_bipolar =   [([-1, -1], -1), ([1, -1], 1), ([-1, 1], 1), ([1, 1], 1)]
    xor_data_scheme_bipolar =  [([-1, -1], -1), ([1, -1], 1), ([-1, 1], 1), ([1, 1], -1)]

    def __init__(self):
        self.data = []

    def read_file_data(self, **params):
        with open(params['data_path'], 'r') as csv_file:
            data = list(csv_file.read())
        self.data = [(elem[:-1], elem[-1]) for elem in data]

    def generate(self, n_samples, activation='unipolar', logical_fun='or', offset=0.1):
        logical_fun = logical_fun.lower()
        data_generations = {
            'unipolar': {
                'and': self._generate_scheme_samples(n_samples, offset, self.and_data_scheme_unipolar),
                'or': self._generate_scheme_samples(n_samples, offset, self.or_data_scheme_unipolar),
                'xor': self._generate_scheme_samples(n_samples, offset, self.xor_data_scheme_unipolar)
            },
            'bipolar': {
                'and': self._generate_scheme_samples(n_samples, offset, self.and_data_scheme_bipolar),
                'or': self._generate_scheme_samples(n_samples, offset, self.or_data_scheme_bipolar),
                'xor': self._generate_scheme_samples(n_samples, offset, self.xor_data_scheme_bipolar)
            }
        }
        self.data = data_generations[activation][logical_fun]
        print("Generated samples: \n {}".format(self.data))

    def _generate_scheme_samples(self, n_samples, offset, data_schemes):
        return [[self._sample(inputs=scheme[0], offset=offset), scheme[1]]
                for scheme in data_schemes
                for _ in range(n_samples//len(data_schemes))]

    def _sample(self, inputs, offset, rounded=4):
        return [round(uniform(value, value - offset) if value > offset else
                      uniform(value, value + offset), rounded) for value in inputs]

    def data_to_csv(self):
        with open(path.join(PROJECT_ROOT, 'input', 'data_{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))), 'w') as csv_file:
            wr = csv.writer(csv_file)
            for row in self.data:
                wr.writerow(row)


class Perceptron:
    def __init__(self, data=None, activation='unipolar', loss='discreet', weights_range=(-1, 1)):
        self.inputs, self.labels = self.initialize_data(data)
        self.weights, self.bias = self.initialize_weights(weights_range=weights_range,
                                                          input_len=len(self.inputs[0]))
        print("Initial weights: \n {}".format(self.weights))
        print("Initial bias: {}".format(self.bias))
        self.activation = self.set_activation(activation)
        print("Used {} activation function.".format(activation))
        self.loss = self.set_loss(loss)

    def initialize_data(self, data):
        return list(zip(*data))

    def initialize_weights(self, weights_range=(-1, 1), input_len=0):
        return [uniform(*weights_range) for _ in range(input_len)], random()

    def set_activation(self, activation):
        functions = {
            'unipolar': self.unipolar,
            'bipolar': self.bipolar
        }
        return functions[activation]

    def set_loss(self, loss):
        functions = {
            'discreet': self.discreet_loss,
            'adaline': self.adaline_squared_loss,
            'adaline_squared': self.adaline_squared_loss
        }
        return functions[loss]

    def train(self, epochs, learning_rate, learning_loss_threshold=5, shuffle_data=True, inputs=None, labels=None):
        if not inputs or not labels:
            print("Train inputs was not given fully. Used perceptron initial inputs.")
            inputs, labels = self.inputs, self.labels
        epoch = 0
        epoch_loss = learning_loss_threshold + 1
        while epoch < epochs and abs(epoch_loss) > learning_loss_threshold:
            labeled_data = list(zip(inputs, labels))
            if shuffle_data:
                shuffle(labeled_data)
            epoch_loss = 0
            print("Training set size: {}".format(len(labeled_data)))
            for sample, label in labeled_data:
                epoch_loss += self._optimize_weights(sample, label, learning_rate)
            print(epoch_loss)
            print("Epoch {} loss: {} \n\t Bias: {} | Weights: {}".format(epoch, epoch_loss, self.bias,
                                                                         self.weights))
            epoch += 1

    def _optimize_weights(self, sample, label, learning_rate):
        error = self.error(sample, label)
        delta = learning_rate * error
        for i in range(len(self.weights)):
            self.weights[i] += delta * sample[i]
        self.bias += delta
        return error

    def error(self, sample, label):
        predicted = self.predict(sample)
        error = self.loss(predicted if self.loss == self.discreet_loss else sample, label)
        print("Predicted: {} | Label: {} | Error: {} | Sample: {}".format(predicted, label, error, sample))
        return error

    def predict(self, sample):
        return self.activation(sample)

    def unipolar(self, sample):
        return 1 if self._calculate_activation(sample) > 0 else 0

    def bipolar(self, sample):
        return 1 if self._calculate_activation(sample) > 0 else -1

    def _calculate_activation(self, sample):
        return self.propagate_sample(sample) + self.bias

    def propagate_sample(self, sample):
        return sum(sample[i] * self.weights[i] for i in range(len(sample)))

    def discreet_loss(self, predicted, label):
        print((predicted, label))
        return label - predicted

    def adaline_squared_loss(self, sample, label):
        return self.discreet_loss(self.propagate_sample(sample), label) ** 2

    def validate(self, data):
        error = 0
        mispredicted = []
        print("Validation result on each sample:")
        for x, y in data:
            sample_error = self.error(x, y)
            error += sample_error
            if sample_error != 0:
                mispredicted.append([x, sample_error, y])
        data_size = len(data)
        accuracy = 100 - abs(len(mispredicted)) / data_size
        print("Accuracy: {} on {} samples".format(accuracy, data_size))
        print("Mispredicted examples: \n {}".format(mispredicted))
        return accuracy

    def visualize_separating_function_on_samples(self):
        self._plot_samples()
        self._plot_saparating_function()
        plt.show()

    def _plot_samples(self):
        xs = [elem[0] for elem in self.inputs]
        ys = [elem[1] for elem in self.inputs]
        colors = ['green' if y == 1 else 'red' for y in self.labels]
        for x, y, c in zip(xs, ys, colors):
            plt.scatter(x, y, color=c)

    def _plot_saparating_function(self):
        x0_plot = 0
        y0_plot = 0
        point_0 = (x0_plot, (-self.weights[0] * x0_plot - self.bias) / self.weights[1])
        point_1 = ((-self.weights[1] * y0_plot - self.bias) / self.weights[0], y0_plot)
        print((point_0, point_1, self.bias))
        plt.plot(point_0, point_1, marker='o')


def main():
    CONFIG = configparser.ConfigParser()
    CONFIG.read(path.join(PROJECT_ROOT, 'configs', 'perceptron_config_OR.ini'))
    data = LogicData()
    data.generate(500, offset=0.3, logical_fun='and', activation='bipolar')
    perceptron = Perceptron(data.data, activation='bipolar')
    perceptron.visualize_separating_function_on_samples()
    perceptron.train(500, 0.01)
    data.generate(500, offset=0.1, logical_fun='and', activation='bipolar')
    validating_data = data.data
    test_sample = [-1, -1]
    print(test_sample, (perceptron.predict(test_sample)))
    test_sample = [-1, 1]
    print(test_sample, (perceptron.predict(test_sample)))
    test_sample = [1, -1]
    print(test_sample, (perceptron.predict(test_sample)))
    test_sample = [1, 1]
    print(test_sample, (perceptron.predict(test_sample)))
    perceptron.validate(validating_data)
    perceptron.visualize_separating_function_on_samples()

if __name__ == "__main__":
    main()
