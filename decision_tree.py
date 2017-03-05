import common_utils as cu
# from sklearn import tree

def run():
    cu.read_data("iris.data")
    cu.generate_data_sets()
    cu.normalize_and_scale_data()
    # clf = tree.DecisionTreeClassifier()
    X = [cu.data[i].tolist() for i in cu.training_data]
    Y = [cu.data_labels[i].tolist() for i in cu.training_data]
    print(Y)
    # clf = clf.fit(X, Y)

if __name__ == "__main__":
    run()