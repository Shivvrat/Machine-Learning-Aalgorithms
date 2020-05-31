import glob
import itertools
import random
import time
from math import log10

import get_data_uai
import get_evidence_data
import get_pr
import helper
import sampling_VE
import sampling_VE_adaptive_proposal_distribution


def main():
    """
    This is the main function which is used to run all the algorithms
    :return:
    """

    uai_file = glob.glob("D:\Spring 20\Stats in AI and ML\HW\HW3\Code\1.uai")
    evid_file = glob.glob("D:\Spring 20\Stats in AI and ML\HW\HW3\Code\1.uai.evid")
    pr_file = glob.glob("D:\Spring 20\Stats in AI and ML\HW\HW3\Code\1.uai.PR")
    files = zip(uai_file, evid_file, pr_file)
    num_samples = [10]
    w_cutset = [1]
    c = tuple(itertools.product(w_cutset, num_samples))
    final_output = dict()
    timer = dict()
    for each_file in files:
        file_name = each_file[0].split("\\")
        file_name = file_name[1].split(".")[0]
        print("The file we are processing is", file_name)
        uai_file_name = each_file[0]
        evidence_file_name = each_file[1]
        pr_file_name = each_file[2]
        final_output[file_name] = {}
        final_output[file_name]["normal"] = {}
        final_output[file_name]["adaptive"] = {}
        timer[file_name] = {}
        timer[file_name]["normal"] = {}
        timer[file_name]["adaptive"] = {}
        for each_val in c:
            print("The value of c is", each_val)
            error_for_iter_normal = []
            error_for_iter_adaptive = []
            time_for_iter_normal = []
            time_for_iter_adaptive = []
            for each_iter in range(1):
                random.seed(each_iter)
                start_time_normal = time.time()
                num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array = get_data_uai.get_uai_data(
                    uai_file_name)
                evidence = get_evidence_data.get_evidence(evidence_file_name)
                var_in_clique, distribution_array = helper.instantiate(num_of_var_in_clique, evidence, cardinalities,
                                                                       var_in_clique, distribution_array)
                estimate_1 = sampling_VE.sampling_VE(num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique,
                                                     var_in_clique, distribution_array, each_val[0], each_val[1])
                time_normal = (time.time() - start_time_normal)
                time_for_iter_normal.append(time_normal)
                start_time_adaptive = time.time()
                num_of_var, cardinalities, num_of_cliques, num_of_var_in_clique, var_in_clique, distribution_array = get_data_uai.get_uai_data(
                    uai_file_name)
                evidence = get_evidence_data.get_evidence(evidence_file_name)
                var_in_clique, distribution_array = helper.instantiate(num_of_var_in_clique, evidence, cardinalities,
                                                                       var_in_clique, distribution_array)
                estimate_2 = sampling_VE_adaptive_proposal_distribution.sampling_VE(num_of_var, cardinalities,
                                                                                    num_of_cliques,
                                                                                    num_of_var_in_clique,
                                                                                    var_in_clique, distribution_array,
                                                                                    each_val[0], each_val[1])
                time_adaptive = (time.time() - start_time_adaptive)
                time_for_iter_adaptive.append(time_adaptive)
                print(estimate_1, estimate_2)
                log_actual_pr = log10(get_pr.get_pr(pr_file_name))
                log_predicted_pr = log10(estimate_1)
                log_predicted_pr_adaptive = log10(estimate_2)
                actual = (log_actual_pr - log_predicted_pr) / log_actual_pr
                adaptive = (log_actual_pr - log_predicted_pr_adaptive) / log_actual_pr
                error_for_iter_normal.append(actual)
                error_for_iter_adaptive.append(adaptive)
                print("Normal :-", actual, time_normal)
                print("adaptive :-", adaptive, time_adaptive)

            timer[file_name]["adaptive"][each_val] = time_adaptive
            timer[file_name]["normal"][each_val] = time_normal
            final_output[file_name]["adaptive"][each_val] = error_for_iter_adaptive
            final_output[file_name]["normal"][each_val] = error_for_iter_normal
        try:
            error_file_name = "error_" + file_name + ".txt"
            f = open(error_file_name, "w")
            f.write(str(final_output))
            f.close()
            time_file_name = "time_" + file_name + ".txt"
            f = open(time_file_name, "w")
            f.write(str(timer))
            f.close()
        except:
            print("IO error")
    print(final_output)
    print(timer)

    return final_output, timer


if __name__ == "__main__":
    print(main())
