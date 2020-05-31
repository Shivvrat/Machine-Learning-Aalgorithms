import numpy as np
from matplotlib import image
from matplotlib import pyplot
import os


def import_image(path):
    """
    This function is used to import the image
    @param path: the path on which the image is present
    @return: Return the image as an array of rgb values
    """
    img = image.imread(path)
    arr = np.array(img)
    return arr


def image_size(image):
    """
    This function is used to get the image size
    @param image: The image as an array
    @return: The height and width of the image
    """
    height, width, temp = map(int, np.shape(image))
    return height, width


def initialize_means(image, k):
    """
    Here we initialize the means for k-means
    @param image: The image as array
    @param k: The value of k
    @return:
    """
    indices = np.random.choice((image.shape)[0], k)
    return image[indices]


def update_assignments(image, means, k):
    """
    This function is used to update the assignments for the k-means algorithm
    @param image: The image as array
    @param means: The means
    @param k: The value of parameter k
    @return: The updated assignments
    """
    assignments = []
    for pixel in image:
        distance_val = []
        for mean_point in means:
            # Here we get the distance of the given pixel with each mean
            distance_val.append(np.linalg.norm(pixel - mean_point))
        # We get the closest mean to a pixel by taking the argmin of the array
        closest_mean = np.argmin(distance_val)
        assignments.append(closest_mean)
    return assignments


def update_means(image, means, assignments, k):
    """
    This function is used to update the means
    @param image: the image as an array
    @param means: the means
    @param assignments: the assignments values
    @param k: The value of parameter k
    @return: The updated means
    """
    updated_means = []
    for each_mean in range(len(means)):
        new_pixels_for_current_mean = []
        for each_closest_mean_for_given_pixel in assignments:
            # Check if the current pixel has this given mean
            if each_mean == each_closest_mean_for_given_pixel:
                current_pixel = assignments.index(each_closest_mean_for_given_pixel)
                # Add the current pixel to the pixels which have the given mean
                new_pixels_for_current_mean.append(image[current_pixel])
        if len(new_pixels_for_current_mean) != 0:
            # Update the means
            updated_means.append(calculate_mean(new_pixels_for_current_mean))
        else:
            continue
    return updated_means


def calculate_mean(pixels):
    """
    This function is used to find the means for given values of pixels
    @param pixels: The pixels selected for current means
    @return: The new mean
    """
    new_mean = []
    for count1 in range(np.shape(pixels)[1]):
        # For each channel
        sum = 0
        for count2 in range(np.shape(pixels)[0]):
            # For each pixel
            sum = sum + pixels[count2][count1]
        average = sum / float(np.shape(pixels)[0])
        # We append the value for each channel here
        new_mean.append(average)
    return new_mean


def kmeans(image, means, k, number_of_iterations):
    """
    This function is used to run the k means algorithm
    @param image: The image as array
    @param means:
    @param k:
    @param number_of_iterations:
    @return: The updated means and assignments
    """
    for each_iteration in range(number_of_iterations):
        assignments = update_assignments(image, means, k)
        new_means = update_means(image, means, assignments, k)
        # Here we check if the new and old means are same then we have converged
        if np.array_equal(new_means, means):
            # If equal then we just return
            break
        else:
            means = new_means
    return means, assignments


def compress_image(image, means, assignments):
    """
    This is the function used to compress the image
    @param image: The image in array form
    @param means: The final means for the image pixels
    @param assignments: The final assignments for the array
    @return: The final compressed image
    """
    final_image = []
    for each_closest_mean in assignments:
        # Each pixel value is replaced by its mean value
        final_image.append(means[each_closest_mean])
    return final_image


def reshape_array_to_image(array, length, width):
    """
    This function is used to reshape an array to an image
    @param array: The array to be reshaped
    @param length: The length of the image
    @param width: The width of the image
    @return: The image
    """
    return np.reshape(array, (length, width, 3))


def reshape_image_to_array(image):
    """
    This function is used to create an array from an image
    @param image: The image we want to convert
    @return:
    """
    height, width = image_size(image)
    return image.reshape(height * width, 3)


def run_k_means(image_name, k, number_of_iterations):
    """
    This is the function used to run k-means
    @param image_name: The name/path of the image
    @param k: The value of parameter K
    @param number_of_iterations: The number of iterations for which k-means runs
    @return: The final image after compression
    """
    image = import_image(image_name)
    height, width = image_size(image)
    image = reshape_image_to_array(image)
    means = initialize_means(image, k)
    means, assignments = kmeans(image, means, k, number_of_iterations)
    final_array = compress_image(image, means, assignments)
    final_image = reshape_array_to_image(final_array, height, width)
    return final_image


def save_after_kmeans(values_of_k, path, number_of_iterations, output_path, number_of_initializations):
    """
    This function is used to save the image
    @param path: The name/path of the image
    @param values_of_k: These value of parameter K
    @param number_of_iterations: The number of iterations for which k-means runs
    @param output_path:
    @param number_of_initializations:
    @return:
    """
    compression_ratio_average = []
    compression_ratio_variance = []
    for each_k in values_of_k:
        compression_ratio = []
        for count1 in range(number_of_initializations):
            final_image = run_k_means(path, each_k, number_of_iterations)
            image_name = output_path
            # I saved the image
            pyplot.imsave(image_name, final_image / float(255))
            original_image = os.stat(path)
            new_image = os.stat(image_name)
            # Here we find the compression ratio of the image
            compression_ratio.append(original_image.st_size / float(new_image.st_size))
        # here we find the average and variance
        compression_ratio_average.append(np.mean(compression_ratio))
        compression_ratio_variance.append(np.var(compression_ratio))
    return compression_ratio_average, compression_ratio_variance
