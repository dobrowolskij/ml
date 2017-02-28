import numpy as np
import random
import math
import sys

# maybe detect it later from data
labels = {"Iris-setosa": np.array([1, 0, 0]),
          "Iris-versicolor": np.array([0, 1, 0]),
          "Iris-virginica": np.array([0, 0, 1])}
label_count = 3

data = []
data_labels = dict()
training_data = []
test_data = []
validation_data = []
layers = []
W = []
learning_rate = 0.03
hidden_layers_neuron_count = (5, 5)
number_of_hidden_layers = len(hidden_layers_neuron_count)
logistic_function = lambda x: 1 / (1 + pow(math.e, -x))
logistic_function_derivative = lambda x: logistic_function(x) * logistic_function(-x)
activation_function = logistic_function
derivative_function = logistic_function_derivative


def initialize_network():
    bias = 1
    W.append(create_weights_for_input_layer(data.shape[1] + bias, hidden_layers_neuron_count[0]))

    for hidden_layer in range(number_of_hidden_layers - 1):
        W.append(create_weights_for_layer(hidden_layers_neuron_count[hidden_layer] + bias,
                                          hidden_layers_neuron_count[hidden_layer + 1]))

    W.append(create_weights_for_output_layer(hidden_layers_neuron_count[-1] + bias, label_count))


def add_biases(matrix_to_enhance):
    biases = np.matrix([1])
    return np.concatenate((biases.T, matrix_to_enhance), axis=1)


def feed_forward(input_data):
    Z = []
    S = []
    F = []
    for weights in W:
        s = add_biases(input_data if not Z else Z[-1]) * weights
        S.append(s)
        Z.append(np.vectorize(activation_function)(s))
        F.append(np.vectorize(derivative_function)(-s))
    return S, Z, F


def get_deltas(D, index):
    return D[index] if index + 1 == len(D) else D[index][1:]


def back_propagation(Z, F, expected_answer):
    D = get_initialized_deltas()
    D[-1] = (Z[-1] - expected_answer).transpose()
    for index, s in reversed(list(enumerate(zip(F[:-1], D[:-1])))):
        D[index] = np.matrix(add_biases(F[index]).A1 * (W[index + 1] * get_deltas(D, index + 1)).A1).transpose()
    return D


def update_weights(D, input_data, Z):
    for index in range(len(W)):
        weight_delta = (D[index][1:] if index + 1 < len(W) else D[index]) * add_biases(
            Z[index - 1] if index > 0 else input_data)
        W[index] += -learning_rate * weight_delta.transpose()


def learn_network():
    random.shuffle(training_data)
    for training_row in training_data:
        training_data_matrix = np.matrix(data[training_row])
        S, Z, F = feed_forward(training_data_matrix)
        expected_answer = data_labels[training_row]
        D = back_propagation(Z, F, expected_answer)
        update_weights(D, training_data_matrix, Z)


def evaluate_training_data():
    return evaluate_data(training_data)


def evaluate_test_data():
    return evaluate_data(test_data)


def evaluate_validation_data():
    return evaluate_data(validation_data)


def evaluate_data(data_set):
    error = 0.0
    correct_answers = 0
    for data_row in data_set:
        data_matrix = np.matrix(data[data_row])
        _, Z, _ = feed_forward(data_matrix)
        expected_answer = data_labels[data_row]
        difference = (Z[-1] - expected_answer).A1
        row_error = sum(list(map(lambda x: pow(x, 2), difference)))
        error += row_error
        if is_correct_answer(difference):
            correct_answers += 1
    return error, correct_answers/len(data_set)


def run():
    read_data("iris.data")
    normalize_and_scale_data()
    generate_data_sets()
    initialize_network()
    for iteration in range(5000):
        learn_network()
        if iteration % 100 == 0:
            training_error, training_correct_answers = evaluate_training_data()
            test_error, test_correct_answers = evaluate_test_data()
            validation_error, validation_correct_answers = evaluate_validation_data()
            print("Iteration {0} error: {1:0.5f} ({2:0.2f}%) {3:0.5f} ({4:0.2f}%) {5:0.5f} ({6:0.2f}%)".format(iteration, training_error,
                                                                       training_correct_answers,
                                                                       test_error, test_correct_answers,
                                                                       validation_error, validation_correct_answers
                                                                       ))
            sys.stdout.flush()


def get_initialized_deltas():
    D = []
    for weights in W[:-1]:
        D.append(np.matrix(np.zeros((weights.shape[1] + 1, 1))))
    D.append(np.matrix(np.zeros((W[-1].shape[1], 1))))
    return D


def read_data(data_file):
    global data
    f = open(data_file, "r")
    for index, line in enumerate(f):
        split_line = line.replace("\n", "").split(",")
        data.append(split_line[:-1])
        data_labels[index] = labels[split_line[-1:][0]]
    data = np.array(data, dtype=np.float64)


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


def create_weights_for_input_layer(feature_count, first_hidden_layer_neuron_count):
    return create_weights_for_layer(feature_count, first_hidden_layer_neuron_count)


def create_weights_for_layer(rows_count, columns_count):
    return np.matrix(np.ones((rows_count, columns_count)))


def create_weights_for_output_layer(last_hidden_layer_neuron_count, label_count):
    return create_weights_for_layer(last_hidden_layer_neuron_count, label_count)


def pm(matrix):
    print(*matrix, sep="\n")


def is_correct_answer(difference):
    return all(list(map(lambda x: x < 0.05, difference)))


if __name__ == "__main__":
    run()
