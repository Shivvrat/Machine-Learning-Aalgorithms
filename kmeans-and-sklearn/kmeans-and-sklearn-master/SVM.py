from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def SVM(taken_kernel, penalty_parameter, X_train, X_test, y_train):
    model = SVC(kernel=taken_kernel, C=penalty_parameter, cache_size=700)
    model = model.fit(X_train, y_train)
    predicted_y = model.predict(X_test)
    return predicted_y


def get_error(predicted_y, y_test):
    return 1 - accuracy_score(y_test, predicted_y)
