"""

__author__      = "Shivvrat Arya"
__version__     = "Python3.7"
"""
from random import randrange

import numpy


def find_distance(input1, input2):
	return numpy.linalg.norm(input1 - input2)


def find_best_matching_unit(codebooks, test_example):
	distance = []
	for each_cookbook in codebooks:
		distance_for_this_example = find_distance(each_cookbook, test_example)
		distance.append((each_cookbook, distance_for_this_example))
	distance.sort(key = lambda tup: tup[1])
	return distance[0][0]

def generate_random_cookbook(train_data):
	number_of_records, number_of_features = numpy.shape(train_data)
	cookbook =  [train_data[randrange(number_of_records)][i] for i in range(number_of_features)]
	return cookbook


def train(train_data, num_of_codeblocks, alpha, number_of_epochs):
	codebooks = [generate_random_cookbook(train) for i in range(num_of_codeblocks)]
	for each_epoch in range(number_of_epochs):
		error = 0
		for each_example in train_data:
			best_macthing_unit = find_best_matching_unit(codebooks, each_example)
			for i in range(len(each_example)-1):
				error += each_example[i] - best_macthing_unit[i]**2
				if best_macthing_unit[-1] == each_example[-1]:
					best_macthing_unit[i] += alpha * error
				else:
					best_macthing_unit[i] -= alpha * error
		print('number of epochs = %d, learning rate = %.3f, error = %.3f' % (each_epoch, alpha, error))

