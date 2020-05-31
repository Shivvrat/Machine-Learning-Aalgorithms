from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def MLP(hidden_layer_sizes, activation_function, X_train, X_test, y_train):
    """
    This is the function used to train the classifer
    @param hidden_layer_sizes: Tuple which gives the hidden layer orientation
    @param activation_function: The activation function for nn
    @param X_train: The train data
    @param X_test: The test data
    @param y_train: The train labels
    @return: The predicted values for X-test
    """
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation_function)
    model = model.fit(X_train, y_train)
    predicted_y = model.predict(X_test)
    return predicted_y


def get_error(predicted_y, y_test):
    return 1 - accuracy_score(y_test, predicted_y)
