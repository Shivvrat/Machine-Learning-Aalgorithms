"""

__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""
import numpy
import pandas
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train(examples, class_of_examples):
	"""
	This function just creates the training data
	:param examples: The examples from given dataset
	:param class_of_examples: The class of the given examples
	:return: A matrix containing both the examples and its respective classes
	"""
	return numpy.hstack((examples, class_of_examples))


def test_classification(train_data, test_point, index_of_prediction_value = -1, k_value = 3):
	"""
	This function is used to predict the class for the given test point
	:param train_data: The training data set
	:param test_point: The point for which we want to find the class
	:param index_of_prediction_value: The index of the class variable in the dataset
	:param k_value: The value of k
	:return: The class of the test example
	"""
	distance = []
	for each_point in train_data:
		distance.append(numpy.linalg.norm(each_point[:-1] - test_point))
	train_data = numpy.array(train_data).tolist()
	zipped_pairs = zip(distance, train_data)
	train_points = [x for _, x in sorted(zipped_pairs)]
	train_points = numpy.array(train_points)
	output_val = train_points[:k_value, index_of_prediction_value]
	return numpy.mode(output_val)


def test_regression(train_data, test_point, index_of_prediction_value = -1, k_value = 3):
	"""
	This function is used to predict the value of output for the given test point
	:param train_data: The training data set
	:param test_point: The point for which we want to find the value of output
	:param index_of_prediction_value: The index of the output variable in the dataset
	:param k_value: The value of k
	:return: The value of output for the test example
	"""
	distance = []
	for each_point in train_data:
		distance.append(numpy.linalg.norm(each_point[:-1] - test_point))
	train_data = numpy.array(train_data).tolist()
	zipped_pairs = zip(distance, train_data)
	train_points = [x for _, x in sorted(zipped_pairs)]
	train_points = numpy.array(train_points)
	output_val = train_points[:k_value, index_of_prediction_value]
	return numpy.mean(output_val)


def evaluation_classification(true_class, predicted_class):
	"""
	This function is used to evaluate the given model
	:param true_class: The true class vector
	:param predicted_class: The classes predicted for the same testing examples
	:return: The accuracy
	"""
	return accuracy_score(true_class, predicted_class)


def evaluation_regression(true_val, predicted_val):
	"""
	This function is used to evaluate the given model
	:param true_class: The true class vector
	:param predicted_class: The classes predicted for the same testing examples
	:return: The mean_squared_error
	"""
	return mean_squared_error(true_val, predicted_val)


def knn_classification(train_dataset, test_dataset, k_value):
	"""
	This is the main function to do classification by knn
	:param train_dataset: This contains the examples on which we will train our model
	:param test_dataset: THis contains the examples on which we will test our model
	:param k_value: The value of k in knn
	:return: The accuracy score
	"""
	predicted_class = []
	for each_test_example in test_dataset:
		class_of_test_example = test_classification(train_dataset, each_test_example[:-1], k_value)
		predicted_class.append(class_of_test_example)
	accuracy = evaluation_classification(test_dataset[:, :-1], predicted_class)
	return accuracy


def knn_regression(train_dataset, test_dataset, k_value):
	"""
	This is the main function to do regression by knn
	:param train_dataset: This contains the examples on which we will train our model
	:param test_dataset: THis contains the examples on which we will test our model
	:param k_value: The value of k in knn
	:return: The mean squared error
	"""
	predicted_val = []
	for each_test_example in test_dataset:
		val_of_test_example = test_regression(train_dataset, each_test_example[:-1], -1, k_value)
		predicted_val.append(val_of_test_example)
	accuracy = evaluation_regression(test_dataset[:, -1], predicted_val)
	return accuracy


def read_data(directory):
	"""
	This function is used to read data from the given directory
	:param directory: The directory in which the data is stored
	:return: The data as a pandas dataframe
	"""
	from numpy import genfromtxt
	my_data = pandas.read_table(directory, header = "infer")
	return my_data


def dataset_split(data):
	"""
	Here we split the dataset into train and test parts
	:param data: The given whole dataset
	:return: Train data and test data as numpy arrays
	"""
	train_data, test_data = train_test_split(data, test_size = 0.2)
	return train_data, test_data


if __name__ == "__main__":
	data = read_data(r"./dataset/datasets_9590_13660_fruit_data_with_colors.txt")
	del data['fruit_name']
	del data['fruit_subtype']
	data = data.apply(pandas.to_numeric)
	data = pandas.DataFrame.to_numpy(data)
	train_data, test_data = dataset_split(data)
	accuracy = knn_regression(train_data, test_data, 3)
	print(accuracy)
