# k-means-and-sklearn



## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project
1. Neural Networks, Gradient Boosting and SVMs [30 points] • For this part, you will use scikit learn.
• Download the MNIST dataset available at http://yann.lecun.com/exdb/mnist/. The dataset has a training set of 60,000 examples, and a test set of 10,000 examples where the digits have been centered inside 28x28 pixel images. You can also use scikit-learn to download and rescale the dataset using the following code:
    ```
    from sklearn.datasets import fetch_openml 
    # Load data from https://www.openml.org/d/554 
    X, y = fetch_openml(’mnist_784’, version=1, return_X_y=True) 
    X = X / 255.
    # rescale the data, use the traditional train/test split 
    # (60K: Train) and (10K: Test) 
    X_train, X_test = X[:60000], X[60000:] 
    y_train, y_test = y[:60000], y[60000:]
    ```
    • Use the SVM classifier in scikit learn and try different kernels and values of penalty parameter. Important: Depending on your computer hardware, you may have to carefully select the parameters (see the documentation on scikit learn for details) in order to speed up the computation. Report the error rate for at least 10 parameter settings that you tried (see how it is reported on http://yann.lecun.com/exdb/mnist/). Make sure to precisely describe the parameters used so that your results are reproducible.
    • Use the MLPClassifier in scikit learn and try different architectures, gradient descent schemes, etc. Depending on your computer hardware, you may have to carefully select the parameters of MLPClassifier in order to speed up the computation. Report the error rate for at least 10 parameters that you tried. Make sure to precisely describe the parameters used so that your results are reproducible.
    • Use the GradientBoostingClassifier in scikit learn and try different parameters (see the documentation for details). Again depending on your computer hardware, you may have to carefully select the parameters in order to speed up the computation. Report the error rate for at least 10 parameters that you tried. Make sure to precisely describe the parameters used so that your results are reproducible.
    • What is the best error rate you were able to reach for each of the three classifiers?

2. K-means clustering on images [30 points]
In this problem, you will use K-means clustering for image compression. We have provided you with two images. 
• Display the images after data compression using K-means clustering for different values of K (2, 5, 10, 15, 20).
• What are the compression ratios for different values of K? Note that you have to repeat the experiment multiple times with different initializations and report the average as well as variance in the compression ratio.
• Is there a tradeoff between image quality and degree of compression. What would be a good value of K for each of the two images?
### Built With

* [Python 3.7](https://www.python.org/downloads/release/python-370/)


## Getting Started

Lets see how to run this program on a local machine.

### Prerequisites

You will need the following modules 
```
1 import sys
2 import warnings
3 from sklearn.neural_network import MLPClassifier 
4 from sklearn.svm import SVC
5 from sklearn.ensemble import GradientBoostingClassifier 
6 from sklearn.metrics import accuracy_score 7 from numpy import genfromtxt
7 import numpy as np
8 from matplotlib import image 
9 from matplotlib import pyplot 6 import os
```
### Installation

1. Clone the repo
```sh
https://github.com/Shivvrat/kmeans-and-sklearn.git
```
Use the main.py to run the algorithm.


<!-- USAGE EXAMPLES -->
## Usage
Please enter the following command line argument for part 1:-
```sh
python part1_main.py <algorithm_name > <parameter_1 > <parameter_2 > < parameter_3 >
```
Please use the following command line parameters for the main.py file :-

* SVM classifier The code should look like

```python part1_main.py svm <parameter_1 > <parameter_2 > ```

We only have 2 parameters for the SVM algorithm which are: 

    1 ```<parameter_1 >``` (kernel) The kernel can take values from :
    * ‘linear’, 
    * ‘poly’, 
    * ‘rbf’, 
    * ‘sigmoid’ and 
    * ‘precomputed’
    2  ```<parameter_2 >``` (penalty parameter) 
        The penalty parameter can take any float value
        
***The neural network classifier (MLP classifier)*** 
The code should look like :

```python part1_main.py nn <parameter_1 > <parameter_2 >```

Please provide parameter 1 as a string. 
We only have 2 parameters for the neural network algorithm which are: 

• ```<parameter_1 >``` (hidden_layer_nodes)

The value of the hidden layer nodes should take a tuple where the lenght of the tuple should be length = n layers - 2, for example
– ”(50, 50)”, 
– ”(100, 100)”, 
– ”(100, 50, 10)”, 
– ”(200, 100, 50)”

• ```<parameter_2 > ```(Activation_function) 
The value of Activation function can be
– ‘identity’, 
– ‘logistic’, 
– ‘tanh’, 
– ‘relu’

***Gradient Boosting Classifier***

```python part1_main.py gb <parameter_1 > <parameter_2 > <parameter_3 >```

We only have 3 parameters for the gradient boosting algorithm which are: 
1. ```<parameter_1 >``` (max_depth) 
    This parameter can take an integer input.
2. ```<parameter_2 >``` (max_feature)
This parameter can take values from the following :– ”sqrt” – ”log2”
3. ```<parameter_3 >``` (criterion)
This parameter can take values from the following :– ”friedman mse” – ”mse”

***Part 2.*** The code should look like :
    ```python part2_main.py <input_image_path > <k> <output_image_path > < number_of_iterations > <number_of_initializations >```
    We only have 5 parameter for the part 2 which is: 
    
    1. ```input image path``` 
    This the path at which the input image is stored
    2. ```K``` 
    This is the value of the K parameter for the algorithm
    3. ```output image path``` 
    This is path at which you want to store the output compressed image
    4. ```number of iterations``` 
    These are the number of iterations for the K-means algorithm.
    5. ```number of initializations``` 
    These are the number of initilizations we want for the kmeans to get the average and the variance for the different compression ratios.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - Shivvrat Arya[@ShivvratA](https://twitter.com/ShivvratA) - shivvratvarya@gmail.com

Project Link: [https://github.com/Shivvrat/kmeans-and-sklearn.git](https://github.com/Shivvrat/kmeans-and-sklearn.git)
