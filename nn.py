
labels = {"Iris-setosa": [1, 0, 0],
          "Iris-versicolor": [0, 1, 0],
          "Iris-virginica": [0, 0, 1]}


def run():
    data = read_data("iris.data")

    for data_row in data:
        print(data_row)


def read_data(data_file):
    f = open(data_file, "r")
    data = []
    for line in f:
        splitted_line = line.replace("\n", "").split(",")
        data.append((splitted_line[:-1], labels[splitted_line[-1:][0]]))
    return data


if __name__ == "__main__":
    run()
