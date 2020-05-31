import sys
from collections import OrderedDict
from random import randint

import numpy as np

import helper
import variable_elimination


def w_cutset(num_of_var, var_in_clique, w_cutset_bound):
    """
    Find the variables which make the tree width of the given graph less than the w-cutset bound
    @param num_of_var: The number of variables(Nodes) in the PGM
    @param var_in_clique: The variables in all the cliques
    @param w_cutset_bound: The cutset bound for the given machine
    @return: The variables that help to make w-cutset bound
    """
    X = set()
    count_of_var_to_remove = 0
    count_for_var, cluster, num_of_var_in_cluster = get_clusters(num_of_var, var_in_clique)
    sorted_variable = sorted(count_for_var, key=lambda k: count_for_var[k], reverse=True)
    for var_to_remove in sorted_variable:
        max_cluster_size = num_of_var_in_cluster[max(num_of_var_in_cluster, key=num_of_var_in_cluster.get)]
        if max_cluster_size <= w_cutset_bound + 1:
            break
        for each_cluster in cluster:
            if cluster[each_cluster] is not None and var_to_remove in cluster[each_cluster]:
                cluster[each_cluster].remove(var_to_remove)
                num_of_var_in_cluster[each_cluster] -= 1
                X.add(var_to_remove)
    return list(X)


def get_clusters(num_of_var, var_in_clique):
    """
    Returns the clusters according to the tree decomposition
    @param num_of_var: the number of variables in PGM
    @param var_in_clique: the variables in clique
    @return: The clusters and count for each variable for how many times it comes in a cluster
    """
    min_degree_for_each_var, sorted_variable = helper.compute_ordering(num_of_var, var_in_clique, evidence=[])
    sorted_variable = list(sorted_variable)
    cluster = OrderedDict()
    var_in_bucket = OrderedDict()
    count_for_var = dict()
    num_of_var_in_cluster = dict()
    var_in_clique_new = var_in_clique.copy()
    for each_var in sorted_variable:
        cluster[each_var] = set()
        var_in_bucket[each_var] = set()
        count_for_var[each_var] = 0
        num_of_var_in_cluster[each_var] = 0
    for each_clique in var_in_clique_new:
        for each_var in each_clique:
            for each_var_2 in each_clique:
                if each_var == each_var_2:
                    continue
                else:
                    if list.index(sorted_variable, each_var) < list.index(sorted_variable, each_var_2):
                        var_in_bucket[each_var].add(each_var_2)
    each_clique = 0
    while each_clique < len(var_in_clique_new):
        for each_var in var_in_clique_new[each_clique]:
            for each_var_2 in var_in_clique_new[each_clique]:
                if each_var == each_var_2:
                    continue
                else:
                    if list.index(sorted_variable, each_var) < list.index(sorted_variable, each_var_2):
                        if each_var_2 not in cluster[each_var]:
                            num_of_var_in_cluster[each_var] += 1
                            count_for_var[each_var_2] += 1
                            cluster[each_var].add(each_var_2)
                        var_in_clique_new.append(var_in_bucket[each_var])
        each_clique += 1
    return count_for_var, cluster, num_of_var_in_cluster


def sampling_VE(num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array,
                w_cutset_bound, num_samples):
    """
    This is the main function used to do sampling VE
    @param num_of_var: The numbre of variables in the PGM
    @param cardinalities: The cardialities of the PGM
    @param num_of_cliques: The number of cliques in the PGM
    @param num_of_var_in_clique: The number of variables in each clique
    @param var_in_clique: The variables in each clique
    @param distribution_array: The distribution array for given PGM
    @param w_cutset_bound: The cutset bound for the PGM
    @param num_samples: The number of samples for the algorithm
    @return: The predicted value of Z or probability
    """
    z = 0
    X = w_cutset(num_of_var, var_in_clique, w_cutset_bound)
    cardinalities = np.array(cardinalities)
    num_of_var_in_x = len(X)
    cardinalities_of_x = cardinalities[X]
    num_of_cliques_in_X = 1
    var_in_clique_X = X
    if num_of_var_in_x != 0:
        sum_of_log_of_cardinalities = np.sum(np.log10(cardinalities_of_x))
        distribution_array_X = dict()
        distribution_array_X["FIRST"] = [1 / sum_of_log_of_cardinalities]
        uniform_dist = 1 / sum_of_log_of_cardinalities
    else:
        distribution_array_X = [1]
        uniform_dist = 1
    num = {}
    denom = 0
    weights = []
    for each_N in range(num_samples):
        distribution_array1 = distribution_array.copy()
        var_in_clique1 = var_in_clique.copy()
        num_of_var_in_clique1 = num_of_var_in_clique.copy()
        num_of_var1 = num_of_var
        evidence = []
        for each_evidence in range(num_of_var_in_x):
            evidence.append((var_in_clique_X[each_evidence], randint(0, cardinalities_of_x[each_evidence] - 1)))
        # evidence = [(each_var, each_val) for (each_var, each_val) in zip(var_in_clique_X, sample)]
        var_in_clique1, distribution_array1 = helper.instantiate(num_of_var1, evidence, cardinalities, var_in_clique1,
                                                                 distribution_array1)
        var_elem_sol = variable_elimination.variable_elimination(num_of_var1, cardinalities, num_of_cliques,
                                                                 num_of_var_in_clique1, var_in_clique1,
                                                                 distribution_array1, evidence)
        sample = [one_value for (var, one_value) in evidence]
        sample_string = sample.__str__()
        evidence_tuple = helper.get_index_given_truth_values(var_in_clique_X, sample, cardinalities_of_x)
        if distribution_array_X is list:
            Q = uniform_dist
        else:
            if sample_string in distribution_array_X:
                Q = distribution_array_X[sample_string]
            else:
                Q = uniform_dist
        weight = var_elem_sol / Q
        weights.append(weight)
        denom += weight
        if sample_string not in num:
            num[sample_string] = weight
        else:
            num[sample_string] += weight
        if each_N % 100 == 0 and each_N != 0 and distribution_array_X is not list:
            distribution_array_X = {}
            for each in num:
                distribution_array_X[each] = num[each] / denom
    weights = np.array(weights)
    if (weights[1:] == weights[:-1]).all:
        z = np.sum(weights)
    else:
        z = helper.threshold(helper.logsumexp(weights))
    """if weight != float('inf'):
            z += weight
        else:
            z += sys.float_info.max * np.random.uniform(0,2)"""
    return z / num_samples