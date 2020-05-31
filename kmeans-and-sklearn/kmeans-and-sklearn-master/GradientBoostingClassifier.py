from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def GradientBoostingModel(max_features , criterion, max_depth, X_train, X_test, y_train):
    """
    This function is used to run the gradient boosting model from sklearn
    @param max_features: These are the maximum number of features for the model
    @param criterion: This is the criterion as per sklearn
    @param max_depth: This is the max depth a model can take
    @param X_train: The train data
    @param X_test: The test output
    @param y_train: The train output
    @return:
    """
    model = GradientBoostingClassifier(max_features = max_features, criterion =criterion, max_depth=max_depth)
    # We fit the model here
    model = model.fit(X_train, y_train)
    predicted_y = model.predict(X_test)
    return predicted_y


def get_error(predicted_y, y_test):
    """
    This function is used to get the error for the function
    @param predicted_y: These are the predicted values
    @param y_test:  This is the actual outpyt
    @return:  Error of the given model
    """
    return 1 - accuracy_score(y_test, predicted_y)
