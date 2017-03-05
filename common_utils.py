import numpy as np
import random

labels = {"Iris-setosa": np.array([1, 0, 0]),
          "Iris-versicolor": np.array([0, 1, 0]),
          "Iris-virginica": np.array([0, 0, 1])}
label_count = 3

data = []
data_labels = dict()
training_data = []
validation_data = []


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
    global training_data, validation_data
    data_count = data.shape[0]
    indices = [i for i in range(data_count)]
    random.shuffle(indices)
    training_data = indices[0:int(0.7 * data_count)]
    validation_data = indices[-int(0.3 * data_count):]