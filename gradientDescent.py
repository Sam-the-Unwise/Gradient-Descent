################################################################################
#
# AUTHOR(S): Josh Holguin
#            Samantha Muellner
#            Jacob Christiansen
# DESCRIPTION: program that contains the code for the gradient descent algorithm
# VERSION: 1.0.0v
#
################################################################################

import numpy as np
import csv
from math import sqrt
import matplotlib.pyplot as plt

# Function: gradientDescent
# INPUT ARGS:
#   X : a matrix of numeric inputs {Obervations x Feature}
#   y : a vector of binary outputs {0,1}
#   stepSize : learning rate - epsilon parameters
#   max_iterations : pos int that controls how many steps to take
# Return: weight_matrix
def gradientDescent(X, y, step_size, max_iterations):

    # VARIABLES

    # tuple of array dim (row, col)
    arr_dim = X.shape
    #print(arr_dim)

    # num of input features
    X_arr_col = arr_dim[1]

    wm_total_entries = X_arr_col * max_iterations

    # variable that initiates to the weight vector
    weight_vector = np.zeros(X_arr_col)

    # matrix for real numbers
    #   row of #s = num of inputs
    #   num of cols = maxIterations
    weight_matrix = np.array(np
                        .zeros(wm_total_entries)
                        .reshape(X_arr_col, max_iterations))

    # ALGORITHM
    weight_vector_transpose = np.transpose(weight_vector)

    for iteration in range(0, max_iterations):
        #calculate y_tid
        for index in range(0, X.shape[1]):

            grad_log_losss = 0
            verctor_mult = 0
            inner_exp = 0

            y_tild = -1

            if(y[index] == 1):
                y_tild = 1

            # variables for simplification
            verctor_mult = np.multiply(weight_vector_transpose, X[index,:])
            inner_exp = np.multiply(y_tild, verctor_mult)
            # calculate gradient

            gradient = (1/(1+np.exp(inner_exp))*
                        (np.exp(inner_exp))*
                        ((-y_tild)*X[index,:]))

            grad_log_losss += gradient

        mean_grad_log_loss = grad_log_losss/X.shape[1]

        # update weight_vector depending on positive or negative
        weight_vector = weight_vector - np.multiply(step_size, mean_grad_log_loss)

        # store the resulting weight_vector in the corresponding column weight_matrix
        weight_matrix[: ,index] = weight_vector


    # end of algorithm
    return weight_matrix