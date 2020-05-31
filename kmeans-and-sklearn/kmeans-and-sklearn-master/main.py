import itertools
import import_mnist
import SVM

kernels = ['poly']
X_train, y_train, X_test, y_test = import_mnist.import_mnist_data()
penalty_parameter = [2**3,2**-3]
#penalty_parameter = [2**-5, 1, 2**5, 2**3, 2**-10]
for each in list(itertools.product(kernels, penalty_parameter)):
    predicted_y = SVM.SVM(each[0],each[1], X_train, X_test, y_train)
    error = SVM.get_error(predicted_y, y_test)
    print each[0], each[1], error
