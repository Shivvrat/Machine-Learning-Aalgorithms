import itertools
import import_mnist
import MLPClassifier


hidden_layer_sizes = [(50, 50,), (100, 100,), (100, 50, 10,), (200, 100, 50,)]
X_train, y_train, X_test, y_test = import_mnist.import_mnist_data()
activation = ["tanh", "relu", "logistic"]
for each in list(itertools.product(hidden_layer_sizes, activation)):
    predicted_y = MLPClassifier.MLP(each[0], each[1], X_train, X_test, y_train)
    error = MLPClassifier.get_error(predicted_y, y_test)
    print each[0], each[1], error
