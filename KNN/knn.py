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
	return numpy.hstack((examples, class_of_examples))


def test_classification(train_data, test_point, index_of_prediction_value = -1, k_value = 3):
	distance = []
	for each_point in train_data:
		distance.append(numpy.linalg.norm(each_point[:-1] - test_point))
	train_points, distance = zip(*sorted(zip(train_data, distance)))
	classes = train_points[:k_value, index_of_prediction_value]
	return numpy.median(classes)


def test_regression(train_data, test_point, index_of_prediction_value = -1, k_value = 3):
	distance = []
	for each_point in train_data:
		distance.append(numpy.linalg.norm(each_point[:-1] - test_point))
	train_data = numpy.array(train_data).tolist()
	zipped_pairs = zip(distance, train_data)
	train_points = [x for _, x in sorted(zipped_pairs)]
	train_points = numpy.array(train_points)
	color_score = train_points[:k_value, index_of_prediction_value ]
	return numpy.mean(color_score)


def evaluation_classification(true_class, predicted_class):
	return accuracy_score(true_class, predicted_class)


def evaluation_regression(true_val, predicted_val):
	return mean_squared_error(true_val, predicted_val)


def knn_classification(train_dataset, test_dataset, k_value):
	predicted_class = []
	for each_test_example in test_dataset:
		class_of_test_example = test_classification(train_dataset, each_test_example[:-1], k_value)
		predicted_class.append(class_of_test_example)
	accuracy = evaluation_classification(test_dataset[:, :-1], predicted_class)
	return accuracy


def knn_regression(train_dataset, test_dataset, k_value):
	predicted_val = []
	for each_test_example in test_dataset:
		val_of_test_example = test_regression(train_dataset, each_test_example[:-1], -1, k_value)
		predicted_val.append(val_of_test_example)
	accuracy = evaluation_regression(test_dataset[:, -1], predicted_val)
	return accuracy


def read_data(directory):
	from numpy import genfromtxt
	my_data = pandas.read_table(directory, header = "infer")
	return my_data


def dataset_split(data):
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
