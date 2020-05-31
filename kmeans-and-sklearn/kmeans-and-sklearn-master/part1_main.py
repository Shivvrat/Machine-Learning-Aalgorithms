import sys
import warnings
import import_mnist
import MLPClassifier
import SVM
import GradientBoostingClassifier

warnings.filterwarnings("ignore")

# Here we take the inputs given in the command line
arguments = list(sys.argv)
try:
    algorithm_name = str(arguments[1])
    parameter_1 = str(arguments[2])
    parameter_2 = str(arguments[3])
except:
    print "You have not provided enough arguments, please read the readme"
    print "Usage for SVM: python part1_main svm <kernel_name> <penalty_parameter_value>"
    print "Usage for Gradient boosting: python part1_main gb <max_depth> <max_feature> <criterion>"
    print "Usage for Neural Networks: python part1_main nn <hidden_layer_nodes> <Activation_function>"
    exit(-1)

try:
    parameter_3 = str(arguments[4])
    print "You are trying to run the Grading Boosting algorithm else you have given wrong(more) number of parameters"
    print "Usage for Gradient boosting: python part1_main gb <max_depth> <max_feature> <criterion>"
except:
    print "You are trying to run the SVM or MLC algorithm else you have given wrong(more) number of parameters"


def main():
    """
    This is the main function which is used to run the algorithm
    :return: Nothing just print the error value
    """
    X_train, y_train, X_test, y_test = import_mnist.import_mnist_data()
    if algorithm_name == "svm":
            predicted_y = SVM.SVM(parameter_1, int(parameter_2), X_train, X_test, y_train)
            error = SVM.get_error(predicted_y, y_test)
    elif algorithm_name == "gb":
            predicted_y = GradientBoostingClassifier.GradientBoostingModel(parameter_2, parameter_3, int(parameter_1),
                                                                           X_train, X_test, y_train)
            error = GradientBoostingClassifier.get_error(predicted_y, y_test)
    elif algorithm_name == "nn":
            hidden_layers = tuple(map(int, parameter_1[1:-1].split(',')))
            predicted_y = MLPClassifier.MLP(hidden_layers, parameter_2, X_train, X_test, y_train)
            error = MLPClassifier.get_error(predicted_y, y_test)
    print "The error for the classifier is ", error


if __name__ == "__main__":
    main()
