from numpy import genfromtxt


def import_mnist_data():
    """
    This function is used to import the mnist data from the csv files
    @return: 4 arrays containing data
    """
    y_test = genfromtxt('data/y_test.csv', delimiter=',')
    y_train = genfromtxt('data/y_train.csv', delimiter=',')
    X_test = genfromtxt('data/X_test.csv', delimiter=',')
    X_train = genfromtxt('data/X_train.csv', delimiter=',')
    return X_train, y_train, X_test, y_test
