import sys
import warnings
import kmeans

warnings.filterwarnings("ignore")

# Here we take the inputs given in the command line
arguments = list(sys.argv)
try:
    input_path = str(arguments[1])
    K = arguments[2]
    K = tuple(map(int, K.split(',')))
    output_path = str(arguments[3])
    number_of_iterations = int(arguments[4])
    number_of_initializations = int(arguments[5])
except:
    print "You have not provided enough arguments, please read the readme"
    print "Usage for K-means :- python part2_main.py <input-image-path> <k> <output-image-path> <number_of_iterations> <number_of_initializations>"
    exit(-1)


def main():
    compression_ratio_average, compression_ratio_variance = kmeans.save_after_kmeans(K, input_path,
                                                                                     number_of_iterations, output_path, number_of_initializations)
    print "The compression ratio average is ", compression_ratio_average
    print "The compression ratio variance is ", compression_ratio_variance


if __name__ == "__main__":
    main()
