import numpy as np
import random
import math

# maybe detect it later from data
labels = {"Iris-setosa": [1, 0, 0],
          "Iris-versicolor": [0, 1, 0],
          "Iris-virginica": [0, 0, 1]}
label_count = 3

data = []
data_labels = dict()
training_data = []
test_data = []
validation_data = []
layers = []
number_of_hidden_layers = 2
hidden_layers_neuron_count = (4, 4)
logistic_function = lambda x: 1 / (1 + pow(math.e, -x))
activation_function = logistic_function


def normalize_and_scale_data():
    means = np.mean(data, axis=0, dtype=np.float64)

    std_deviations = np.subtract(np.amax(data, axis=0), np.amin(data, axis=0))
    for column in range(data.shape[1]):
        data[:, column] -= means[column]
        data[:, column] /= std_deviations[column]


def generate_data_sets():
    global training_data, test_data, validation_data
    data_count = data.shape[0]
    indices = [i for i in range(data_count)]
    random.shuffle(indices)
    training_data = indices[0:int(0.6 * data_count)]
    test_data = indices[int(0.6 * data_count):int(0.6 * data_count) + int(0.2 * data_count)]
    validation_data = indices[-int(0.2 * data_count):]


def initialize_network():
    layers.append(np.matrix(np.random.random_sample((data.shape[1] + 1, data.shape[1])) - 0.5))

    previous_layer_neuron_count = data.shape[1]
    for hidden_layer in range(number_of_hidden_layers):
        layers.append(np.matrix(
            np.random.random_sample((previous_layer_neuron_count + 1, hidden_layers_neuron_count[hidden_layer])) - 0.5))
        previous_layer_neuron_count = hidden_layers_neuron_count[hidden_layer]

    layers.append(np.matrix(np.random.random_sample((previous_layer_neuron_count + 1, label_count)) - 0.5))


def feed_forward(data_row):
    internal_result = np.copy(data_row)
    for index, layer in enumerate(layers):
        data_row_with_bias = np.matrix(np.insert(internal_result, 0, 1, axis=0))
        internal_result = np.vectorize(activation_function)(data_row_with_bias * layer).A1

    return internal_result


def learn_network():
    for training_row in training_data[1:2]:
        answer = feed_forward(data[training_row])
        expected_answer = data_labels[training_row]
        print(training_row, expected_answer, answer)


def run():
    read_data("iris.data")
    normalize_and_scale_data()
    generate_data_sets()
    initialize_network()
    learn_network()


def read_data(data_file):
    global data
    f = open(data_file, "r")
    for index, line in enumerate(f):
        split_line = line.replace("\n", "").split(",")
        data.append(split_line[:-1])
        data_labels[index] = labels[split_line[-1:][0]]
    data = np.array(data, dtype=np.float64)


if __name__ == "__main__":
    run()
