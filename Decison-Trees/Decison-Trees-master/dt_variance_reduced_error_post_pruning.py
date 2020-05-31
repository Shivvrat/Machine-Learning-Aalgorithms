import os

import numpy as np
import pandas as pd
import copy as copy

"""
This function is used to import the data. Put the data in a folder named all_data in the directory of the code
"""


def import_data(dt_name):
    """

    :param dt_name: Name of the Dataset
    :return: Three pandas frames which correspond to training, testing and validation data
    """
    # First we get the directory of our project and then take all the three files and import them to the respective
    # names.
    d = os.getcwd()
    test_data = pd.read_csv(os.path.join(os.path.join(d, "all_data"), "test_{0}.csv".format(dt_name)), header=None)
    train_data = pd.read_csv(os.path.join(os.path.join(d, "all_data"), "train_{0}.csv".format(dt_name)), header=None)
    validation_data = pd.read_csv(os.path.join(os.path.join(d, "all_data"), "valid_{0}.csv".format(dt_name)),
                                  header=None)
    # Now we will return the data frames
    return [test_data, train_data, validation_data]


"""
This function is defined to get the labels/classes and  attribute values in different variables
"""


def get_attributes_and_labels(data):
    """

    :param data: The dataset to be divided
    :return: Two panda frames which are in order of classes and attributes
    """
    # Here we divide our attributes and classes features for a given dataset
    return [data.iloc[:, -1], data.iloc[:, :-1]]


"""
This function is used to find the variance which is our impurity heuristic for this algorithm(for one row/column)
"""


def get_variance(data):
    """

    :param data: THese are the values for which we want to find the variance of. We pass a whole vector of values which
    correspond to the attribute of importance and find variance for that vector.
    :return: variance for the given vector
    """
    variance_value = 0
    temp, unique_count = np.unique(data, return_counts=True)
    # We will use the formula mentioned in the slides to calculate the value of variance for both the options (i.e,
    # 1 and 0)
    variance_value = (unique_count[0] * unique_count[1]) / (np.size(data)) * (np.size(data))
    return variance_value


"""
This function is used to find the  gain by using the variance for the given sub-tree/tree. The  gain is used to find the
attribute we will use to do further branching
"""


def Variance_Heuristic(examples, attributes, target_attribute):
    """

    :param examples: The data for whihc we want to find the information gain
    :param attributes: the values of the attributes available (the column number)
    :param target_attribute: the target attribute we are trying to find
    :return: Information Gain of the given sub-tree.
    """
    # Here we find the variance for the root node
    previous_variance = get_variance(target_attribute)
    Gain = []
    for each_attribute in attributes:
        unique_value_of_attribute, counts_of_attribute = np.unique(examples[each_attribute], return_counts=True)
        # Since I have hardcoded the array_after_division arrays we will try to the first values for 0.
        if unique_value_of_attribute[0] == 1:
            counts_of_attribute = np.flip(counts_of_attribute)
            unique_value_of_attribute = np.flip(unique_value_of_attribute)
        array_after_division_1 = []
        array_after_division_0 = []
        # This loop is for 0 and 1
        # I need to find the number of 1's and 0's in target value when the given attribute value is something
        # particular
        total_data = pd.concat([examples, target_attribute], axis=1, sort=False)
        # Here I concatenated the data frames so that only one df is used to both lookup the value and find the value
        # to append
        row_names = total_data.index.values
        list_of_row_names = list(row_names)
        for each in list_of_row_names:
            value_to_append = int(total_data.iloc[:, -1][each])
            if examples[each_attribute][each] == 1:
                array_after_division_1.append(value_to_append)
            else:
                array_after_division_0.append(value_to_append)
        # Here I will use try catch since if the target_attribute have only one unique value then it will give an
        # error if we try to use the second index (i.e. 2). and if we have only one unique value then our imputrity
        # is 0 and thus variance is 0
        try:
            value_of_new_inpurity = (counts_of_attribute[0] / np.size(examples[each_attribute])) * get_variance(
                array_after_division_0) + (counts_of_attribute[1] / np.size(examples[each_attribute])) * get_variance(
                array_after_division_1)
        except IndexError:
            value_of_new_inpurity = 0
        temp = previous_variance - value_of_new_inpurity
        Gain.append(temp)
    return Gain


"""
This function is the main function for our algorithm. The decision_tree function is used recursively to create new nodes
and make the tree while doing the training.
"""

def decision_tree_construction(examples, target_attribute, attributes):
    """

    :param examples: The data we will use to train the tree(x)
    :param target_attribute: The label we want to classify(y)
    :param attributes: The number(index) of the labels/attributes of the data-set
    :return: The tree corresponding to the given data
    """
    # This is the first base condition of the algorithm. It is used if the attributes variable is empty, then we return
    # the single-node tree Root, with label = most common value of target_attribute in examples
    # The base condition for the recursion when we check if all the variables are same or not in the node and if they
    # are same then we return that value as the node
    if len(attributes) == 0 or len(np.unique(target_attribute)) == 1:
        unique_value_of_attribute, counts_of_attribute = np.unique(target_attribute, return_counts=True)
        try:
            if counts_of_attribute[0] < counts_of_attribute[1]:
                unique_value_of_attribute = np.flipud(unique_value_of_attribute)
        except IndexError:
            i = 0
        if unique_value_of_attribute[0] == 1:
            # More positive values
            return 1
        elif unique_value_of_attribute[0] == 0:
            # More negative values
            return 0
    # This is the recursion part of the algorithm in which we try to find the sub-tree's by using recursion and
    # information gain
    else:
        Gain = Variance_Heuristic(examples, attributes, target_attribute)
        best_attribute_number = attributes[np.argmax(Gain)]
        # Since we now have the best_attribute(A in algorithm) we will create the root node of the tree/sub-tree with
        # that and name the root as the best attribute among all Here we make the tree as a dictionary for testing
        # purposes
        tree = dict([(best_attribute_number, dict())])
        if isinstance(tree, int):
            tree[best_attribute_number]["type_of_node"] = "leaf"
            tree[best_attribute_number]["majority_target_attribute"] = best_attribute_number
            tree[best_attribute_number]["best_attribute_number"] = best_attribute_number
        else:
            tree[best_attribute_number]["type_of_node"] = "node"
            unique_value_of_attribute, counts_of_attribute = np.unique(target_attribute, return_counts=True)
            try:
                if counts_of_attribute[0] < counts_of_attribute[1]:
                    counts_of_attribute = np.flipud(counts_of_attribute)
                    unique_value_of_attribute = np.flipud(unique_value_of_attribute)
            except IndexError:
                i = 0
            tree[best_attribute_number]["majority_target_attribute"] = unique_value_of_attribute[0]
            tree[best_attribute_number]["best_attribute_number"] = best_attribute_number
        attributes.remove(best_attribute_number)
        # Now we do the recursive algorithm which will be used to create the tree after the root node.
        for each_unique_value in np.unique(examples[best_attribute_number]):
            # We use those values for which the examples[best_attribute_number] == each_unique_value
            class1 = each_unique_value
            new_target_attribute = pd.DataFrame(target_attribute)
            total_data = pd.concat([examples, new_target_attribute], axis=1, sort=False)
            # WE do this step so that we can pick the values which belong to the best_attribute = [0,1], i.e. We now
            # want to divide our data so that the values for the best_attribute is divided among the branches. And
            # thus we will have 4 arrays now, two for the data and two for target attribute.
            new_data_after_partition = total_data.loc[total_data[best_attribute_number] == class1]
            new_target_attribute, new_examples_after_partition = get_attributes_and_labels(new_data_after_partition)
            # This is also a condition for our algorithm in which we check if the number of examples after the
            # partition are positive or not. If the values are less than 1 then we return the most frequent value in
            # the node
            if len(new_examples_after_partition) == 0:
                unique_value_of_attribute, counts_of_attribute = np.unique(target_attribute, return_counts=True)
                try:
                    if counts_of_attribute[0] < counts_of_attribute[1]:
                        counts_of_attribute = np.flipud(counts_of_attribute)
                        unique_value_of_attribute = np.flipud(unique_value_of_attribute)
                except IndexError:
                    i = 0
                if unique_value_of_attribute[0] == 1:
                    # More positive values
                    return 1
                elif unique_value_of_attribute[0] == 0:
                    # More negative values
                    return 0
            # This is the recursion step, in which we make new deicison trees till the case when any of the base
            # cases are true
            new_sub_tree_after_partition = decision_tree_construction(new_examples_after_partition,
                                                                      new_target_attribute, attributes)
            tree[best_attribute_number][each_unique_value] = new_sub_tree_after_partition
            if isinstance(new_sub_tree_after_partition, int):
                tree[best_attribute_number]["type_of_node"] = "leaf"
                tree[best_attribute_number]["majority_target_attribute"] = unique_value_of_attribute[0]
                tree[best_attribute_number]["best_attribute_number"] = best_attribute_number
            else:
                tree[best_attribute_number]["type_of_node"] = "node"
                unique_value_of_attribute, counts_of_attribute = np.unique(target_attribute, return_counts=True)
                try:
                    if counts_of_attribute[0] < counts_of_attribute[1]:
                        counts_of_attribute = np.flipud(counts_of_attribute)
                        unique_value_of_attribute = np.flipud(unique_value_of_attribute)
                except IndexError:
                    i = 0
                tree[best_attribute_number]["majority_target_attribute"] = unique_value_of_attribute[0]
                tree[best_attribute_number]["best_attribute_number"] = best_attribute_number
    return tree


"""
This function is used to change the value for a particular key in a given tree
"""


def change_value_of_sub_tree(tree, new_value, key_value, class_type):
    """

    :param tree: The given tree on which we want to do pruning
    :param new_value: This sis the value we want to put in the tree
    :param key_value: This is the key on which we want to change the value
    :param class_type: This is the class of the node, i.e. 1 or 0
    :return: the tree with new value as the value for key key_value
    """
    if isinstance(tree, dict):
        for key_for_this_tree, next_tree in tree.items():
            if key_for_this_tree == key_value:
                try:
                    tree[key_for_this_tree][class_type] = new_value
                except TypeError:
                    i = 0
                return tree
            else:
                tree = change_value_of_sub_tree(next_tree, new_value, key_value, class_type)
    return tree


"""
This function is used to find the node which is best for pruning and then we return the new tree with pruning so 
that we can find the find the best pruned tree at last. 
"""


globals()["index"] = 0
def reduced_error_post_pruning(main_tree, given_tree, set_of_pruned_trees, validation_x, validation_y,
                               max_value_in_target_attribute, accuracy_for_main_tree):
    """

    :param main_tree: This is the main tree
    :param set_of_pruned_trees: this is the set we will return which will contain each main tree with pruning which leads to imporved accuracy
    :param max_value_in_target_attribute: This is the maximum occurring value in the train dataset
    :param validation_y: The data given as validation set labels
    :param validation_x: The data given as validation set attributes
    :param given_tree: The tree on which we want to apply the reduced-error post pruning
    :return: The pruned tree
    """
    if isinstance(given_tree, dict):
        current_index = list(given_tree.keys())[0]
    else:
        return given_tree
    try:
        if isinstance(given_tree[current_index][0], dict):
            pruned_0 = reduced_error_post_pruning(main_tree, given_tree[current_index][0], set_of_pruned_trees, validation_x,
                                                  validation_y, max_value_in_target_attribute, accuracy_for_main_tree)
        else:
            return given_tree
    except KeyError:
        i = 1
    try:
        if isinstance(given_tree[current_index][1], dict):
            pruned_1 = reduced_error_post_pruning(main_tree, given_tree[current_index][1], set_of_pruned_trees, validation_x,
                                                  validation_y, max_value_in_target_attribute, accuracy_for_main_tree)
        else:
            return given_tree
    except KeyError:
        i = 1
    copy_of_given_tree = copy.deepcopy(given_tree)
    if isinstance(given_tree, dict):
        given_tree[current_index][0] = given_tree[current_index]["majority_target_attribute"]
        given_tree[current_index][1] = given_tree[current_index]["majority_target_attribute"]
        given_tree[current_index]["type_of_node"] = "leaf"
    else:
        return given_tree
#    predicted_y = decision_tree_test(validation_x, main_tree, max_value_in_target_attribute)
#    accuracy_for_main_tree = decision_tree_accuracy(validation_y, predicted_y)
    copy_of_main_tree = copy.deepcopy(main_tree)
    # Now we change the value for the pruned tree to compare with the main tree validation accuracy
    try:
        change_value_of_sub_tree(copy_of_main_tree, pruned_0, current_index, 0)
    except UnboundLocalError:
        i = 1
    try:
        change_value_of_sub_tree(copy_of_main_tree, pruned_1, current_index, 1)
    except UnboundLocalError:
        i = 0
    predicted_y = decision_tree_test(validation_x, copy_of_main_tree, max_value_in_target_attribute)
    pruned_accuracy = decision_tree_accuracy(validation_y, predicted_y)
    if accuracy_for_main_tree < pruned_accuracy:
        set_of_pruned_trees.update({globals()["index"]: copy_of_main_tree})
        globals()["index"] = globals()["index"] + 1
    return copy_of_given_tree



def choose_best_pruned_tree(main_tree, validation_x, validation_y, max_value_in_target_attribute):
    predicted_y = decision_tree_test(validation_x, main_tree, max_value_in_target_attribute)
    current_best_score = decision_tree_accuracy(validation_y, predicted_y)
    next_best_score = 1
    i = 0
    while next_best_score > current_best_score and i < 75:
        i = i + 1
        set_of_pruned_trees = {}
        predicted_y = decision_tree_test(validation_x, main_tree, max_value_in_target_attribute)
        current_best_score = decision_tree_accuracy(validation_y, predicted_y)
        copy_of_main_tree = copy.deepcopy(main_tree)
        reduced_error_post_pruning(copy_of_main_tree, copy_of_main_tree, set_of_pruned_trees, validation_x, validation_y, max_value_in_target_attribute, current_best_score)
        for temp, main_tree_pruned in set_of_pruned_trees.items():
            predicted_y = decision_tree_test(validation_x, main_tree_pruned, max_value_in_target_attribute)
            accuracy_for_pruned_main_tree = decision_tree_accuracy(validation_y, predicted_y)
            if accuracy_for_pruned_main_tree > current_best_score:
                main_tree = main_tree_pruned
                next_best_score = accuracy_for_pruned_main_tree
    return main_tree


"""
This function is used to predict the new and unseen test data point by using the created tree and the given instance
"""


def decision_tree_predict(tree, testing_example, max_value_in_target_attribute):
    """

    :param max_value_in_target_attribute: If we are not able to classify due to less data, we return this value when testing
    :param tree: This is the trained tree which we will use for finding the class of the given instance
    :param testing_example: These are the instance on which we want to find the class
    :return: the predicted value for given datapoint
    """
    predicted_values = []
    # We take each attribute for the datapoint anc check if that attribute is the root node for the tree we are on
    try:
        max_value_in_target_attribute = tree[list(tree)[0]]["majority_target_attribute"]
    except KeyError:
        i = 0
    for each_attribute in list(testing_example.index):
        if each_attribute in tree:
            try:
                # I have used a try catch here since we  trained the algo on a part of the data and it's not
                # necessary that we will have a branch which can classify the data point at all.
                next_dict = tree[each_attribute][testing_example[each_attribute]]
                new_tree = next_dict
            except KeyError:
                # There are two things we can do here, first is to show an error and return no class and the second
                # thing we can do is return the max value in our training target array. error = "The algorithm cannot
                # classify the given instance right now" return error....Need more training
                return max_value_in_target_attribute
            if type(next_dict) == dict:
                # In this case we see if the value predicted is a tree then we again call the recursion if not we
                # return the value we got
                return decision_tree_predict(new_tree, testing_example, max_value_in_target_attribute)
            else:
                return new_tree


"""
This function is used to find the output for the given testing dataset by using the tree created during the training process
"""


def decision_tree_test(test_x, tree, max_value_in_target_attribute):
    """

    :param test_x: This is input attributes from the testing data
    :param tree: This is the tree created after the training process
    :param max_value_in_target_attribute:  This is the most occurring target_attribute in our training data
    :return: The output for the given testing data
    """
    output = []
    # In this function we just use the decision_tree_predict and predict the value for all the instances
    for index, row in test_x.iterrows():
        try:
            output.append(int(decision_tree_predict(tree, row, max_value_in_target_attribute)))
        except TypeError:
            i = 0
    return output


"""
In this function we try to find the accuracy for our predictions and return the accuracy
"""


def decision_tree_accuracy(test_y, predicted_y):
    """

    :param test_y: The classes given with the data
    :param predicted_y: The predicted classes
    :return: The accuracy for the given tree
    """
    test_y = test_y.tolist()
    right_predict = 0
    # Here we predict the accuracy by the formula accuracy = right_predictions/total_data_points_in_the_set
    for each in range(np.size(test_y)):
        if test_y[each] == predicted_y[each]:
            right_predict = right_predict + 1
    return right_predict / len(test_y)


"""
In this function we will try to iterate over the whole naive decision tree with variance as the impurity heuristic
"""


def main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, validation_x, validation_y):
    """

    :param validation_y: This is the validation data
    :param validation_x: This is the validation class
    :param train_x: This is the training data
    :param train_y: This is the training class
    :param test_x:  This is the testing data
    :param test_y: This is the testing class
    :return: The accuracy for created tree and the pruned tree
    """
    train_y = pd.DataFrame(train_y)
    max_value_in_target_attribute, temp = np.unique(train_y, return_counts=True)
    try:
        if temp[0] < temp[1]:
            counts_of_attribute = np.flipud(temp)
            unique_value_of_attribute = np.flipud(max_value_in_target_attribute)
    except IndexError:
        i = 0

    max_value_in_target_attribute = max_value_in_target_attribute[0]
    # First we do the training process
    tree = decision_tree_construction(train_x, train_y, list(train_x.columns))
    pruned_tree = choose_best_pruned_tree(tree, validation_x, validation_y, max_value_in_target_attribute)
    output_test = decision_tree_test(test_x, pruned_tree, max_value_in_target_attribute)
    accuracy2 = decision_tree_accuracy(test_y, output_test)
    return accuracy2