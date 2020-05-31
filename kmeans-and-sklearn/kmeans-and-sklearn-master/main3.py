import itertools
import import_mnist
import GradientBoostingClassifier

max_features = ['sqrt']
criterion = ['friedman_mse']
max_depth = [7]

X_train, y_train, X_test, y_test = import_mnist.import_mnist_data()
for each in list(itertools.product(max_features , criterion, max_depth)):
    predicted_y = GradientBoostingClassifier.GradientBoostingModel(each[0], each[1], each[2], X_train, X_test, y_train)
    error = GradientBoostingClassifier.get_error(predicted_y, y_test)
    print each[0], each[1], each[2], error
