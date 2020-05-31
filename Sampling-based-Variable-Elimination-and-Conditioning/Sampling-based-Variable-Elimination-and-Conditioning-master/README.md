# Sampling-based-Variable-Elimination-and-Conditioning






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
In this project we will implement the Sampling-based Variable Elimination and Conditioning algorithm for inference on a given probabilistic graphical model.
The algorithm is as follows :-
![Sampling-based Variable Elimination and Conditioning algorithm](https://github.com/Shivvrat/Sampling-based-Variable-Elimination-and-Conditioning/blob/master/algo.jpg)
We will use the following algorithm to find the w-cutset for a given graphical model :-
![How to find w-cutset](https://github.com/Shivvrat/Sampling-based-Variable-Elimination-and-Conditioning/blob/master/w-cutset.jpg)


### Built With

* [Python 3.7](https://www.python.org/downloads/release/python-370/)


## Getting Started

Lets see how to run this program on a local machine.

### Prerequisites

You will need the following modules 
```
1 import sys
2 import numpy
3 from numpy import log10 
4 from numpy import logaddexp 
5 from numpy import log
6 from scipy.special import logsumexp as logsum1 
7 import sys
8 import warnings
9 from math import log10 
10 import random 
11 import time 
12 import numpy 
13 import random 
14 import sys
15 from collections import OrderedDict 
16 from random import randint 
17 import numpy as np
18 from math import log10
```
### Installation

1. Clone the repo
```sh
git clone https://github.com/Shivvrat/Sampling-based-Variable-Elimination-and-Conditioning.git
```
Use the main.py to run all three algorithm.


<!-- USAGE EXAMPLES -->
## Usage
Please enter the following command line argument to run the first part :-
```
python part_1.py <uai_file_directory_and_name > <evidence_file_directory_and_name > < pr_file_directory_and_name > <type_of_algorithm > <w_cutset_size > <number_of_samples >
```
Please enter the following command line argument to run the third part :-
```
python part_3.py <uai_file_directory_and_name > <evidence_file_directory_and_name > < pr_file_directory_and_name > <type_of_algorithm > <w_cutset_size > <number_of_samples >
```
Please use the following command line parameters for the main.py file :-

• ```<uai_file_directory_and_name>```
Please provide the file that will be used as the .uai file

• ```<evidence_file_directory_and_name>```
Please provide the file that will be used as the uai.evid file

• ```<pr_file_directory_and_name>```
Please provide the file that will be used as the uai.PR file

• ```type_of_algorithm``` 
Please provide the algorithm you want to use :
1. ***-vec*** Sampling-based Variable Elimination and Conditioning
2. ***-avec*** Adaptive Sampling-based Variable Elimination and Conditioning

• ```w-cutset-size```
Please provide the w-cutset allowed size

• ```number-of-samples```
Please provide the number of samples to be used to find the estimate


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - Shivvrat Arya[@ShivvratA](https://twitter.com/ShivvratA) - shivvratvarya@gmail.com

Project Link: [https://github.com/Shivvrat/Sampling-based-Variable-Elimination-and-Conditioning.git](https://github.com/Shivvrat/Sampling-based-Variable-Elimination-and-Conditioning.git)
