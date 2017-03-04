import tensorflow as tf
import numpy as np
import random
import sys

labels = {"Iris-setosa": np.array([1, 0, 0]),
          "Iris-versicolor": np.array([0, 1, 0]),
          "Iris-virginica": np.array([0, 0, 1])}
label_count = 3

data = []
data_labels = dict()
training_data = []
training_labels = []
validation_data = []
validation_labels = []
hidden_layers_neuron_count = (15, 15, 15)
learning_rate = 0.001


def read_data(data_file):
    global data
    f = open(data_file, "r")
    for index, line in enumerate(f):
        split_line = line.replace("\n", "").split(",")
        data.append(split_line[:-1])
        data_labels[index] = labels[split_line[-1:][0]]
    data = np.array(data, dtype=np.float64)


def generate_data_sets():
    global training_data, training_labels, validation_data, validation_labels
    data_count = data.shape[0]
    indices = [i for i in range(data_count)]
    random.shuffle(indices)
    training_data = np.array([data[i] for i in indices[0:int(0.7 * data_count)]])
    training_labels = np.array([data_labels[i] for i in indices[0:int(0.7 * data_count)]])
    validation_data = np.array([data[i] for i in indices[-int(0.3 * data_count):]])
    validation_labels = np.array([data_labels[i] for i in indices[-int(0.3 * data_count):]])


def run():
    read_data("iris.data")
    generate_data_sets()

    input_data = tf.placeholder(tf.float32, [None, training_data.shape[1]])
    input_labels = tf.placeholder(tf.float32, [None, training_labels.shape[1]])

    weights = {
        'hidden_1': tf.Variable(tf.random_normal([training_data.shape[1], hidden_layers_neuron_count[0]])),
        'hidden_2': tf.Variable(tf.random_normal([hidden_layers_neuron_count[0], hidden_layers_neuron_count[1]])),
        'hidden_3': tf.Variable(tf.random_normal([hidden_layers_neuron_count[1], hidden_layers_neuron_count[2]])),
        'output': tf.Variable(tf.random_normal([hidden_layers_neuron_count[2], label_count]))
    }

    biases = {
        'hidden_1': tf.Variable(tf.random_normal([hidden_layers_neuron_count[0]])),
        'hidden_2': tf.Variable(tf.random_normal([hidden_layers_neuron_count[1]])),
        'hidden_3': tf.Variable(tf.random_normal([hidden_layers_neuron_count[2]])),
        'output': tf.Variable(tf.random_normal([label_count]))
    }

    hidden_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_data, weights['hidden_1']), biases['hidden_1']))
    hidden_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_1, weights['hidden_2']), biases['hidden_2']))
    hidden_layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_2, weights['hidden_3']), biases['hidden_3']))
    output_layer = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_3, weights['output']), biases['output']))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, input_labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(500000):
            _, error = sess.run([optimizer, cost], feed_dict={input_data: training_data, input_labels: training_labels})
            if i % 100 == 0:
                result = sess.run([output_layer], feed_dict={input_data: validation_data, input_labels: validation_labels})
                correct_answers = 0
                for index, (row, label) in enumerate(zip(result[0], validation_labels)):
                    difference = row - label
                    if is_correct_answer(difference):
                        correct_answers += 1
                print("{} {}".format(error, (correct_answers / len(result[0]) * 100)))
                sys.stdout.flush()


def is_correct_answer(difference):
    return all(list(map(lambda x: x < 0.05, difference)))


if __name__ == "__main__":
    run()
