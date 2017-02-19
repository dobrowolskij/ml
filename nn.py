import numpy as np

labels = {"Iris-setosa": [1, 0, 0],
          "Iris-versicolor": [0, 1, 0],
          "Iris-virginica": [0, 0, 1]}


def normalize_and_scale_data(data):
    means = np.mean(data, axis=0, dtype=np.float64)

    std_deviations = np.subtract(np.amax(data, axis=0), np.amin(data, axis=0))
    for column in range(data.shape[1]):
        data[:, column] -= means[column]
        data[:, column] /= std_deviations[column]


def run():
    data, data_labels = read_data("iris.data")
    normalize_and_scale_data(data)


def read_data(data_file):
    f = open(data_file, "r")
    data = []
    data_labels = dict()
    for index, line in enumerate(f):
        split_line = line.replace("\n", "").split(",")
        data.append(split_line[:-1])
        data_labels[index] = labels[split_line[-1:][0]]
    return np.array(data, dtype=np.float64), data_labels


if __name__ == "__main__":
    run()
