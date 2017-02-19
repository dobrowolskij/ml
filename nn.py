import numpy as np
import random

labels = {"Iris-setosa": [1, 0, 0],
          "Iris-versicolor": [0, 1, 0],
          "Iris-virginica": [0, 0, 1]}

data = None
data_labels = None
training_data = None
test_data = None
validation_data = None


def normalize_and_scale_data(data):
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


def run():
    read_data("iris.data")
    normalize_and_scale_data(data)
    generate_data_sets()


def read_data(data_file):
    global data, data_labels
    f = open(data_file, "r")
    data = []
    data_labels = dict()
    for index, line in enumerate(f):
        split_line = line.replace("\n", "").split(",")
        data.append(split_line[:-1])
        data_labels[index] = labels[split_line[-1:][0]]
    data = np.array(data, dtype=np.float64)


if __name__ == "__main__":
    run()
