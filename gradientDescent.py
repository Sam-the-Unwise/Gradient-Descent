import numpy as np
import csv
from math import sqrt

# Function: gradientDescent
# INPUT ARGS:
#   X : a matrix of numeric inputs {Obervations x Feature}
#   y : a vector of binary outputs {0,1}
#   stepSize : learning rate - epsilon parameters
#   max_iterations : pos int that controls how many steps to take
# Return: weight_matrix
def gradientDescent(X, y, step_size, max_iterations):

    # VARIABLES

    # tuple of array dimensions: (row, col)
    arr_dim = X.shape
    #print(arr_dim)

    # num of input features
    X_arr_col = arr_dim[1]

    wm_total_entries = X_arr_col * max_iterations

    # variable that initiates to the weight vector -- should start as zero vector
    weight_vector = np.zeros(X_arr_col)

    # matrix for real numbers
    #   row of #s = num of inputs
    #   num of cols = maxIterations
    weight_matrix = np.array(np
                        .zeros(wm_total_entries,dtype=float)
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
        weight_matrix[: ,index] = weight_vector[:]

    # end of algorithm
    return weight_matrix[:,:]

# Function: scale
# INPUT ARGS:
#   matrix : the matrix that we need to scale
# Return: [none]
def scale(matrix):
    matrix_t = np.transpose(matrix)
    counter = 0

    for column in matrix_t:
        counter += 1
        col_sq_sum = 0

        sum = np.sum(column)
        shape = column.shape
        col_size = shape[0]
        mean = sum/col_size

        for item in column:
            col_sq_sum += ((item - mean)**2)

        std = sqrt(col_sq_sum/col_size)

        column -= mean
        column /= std

# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    with open(file_name, 'r') as data_file:
        file = list(csv.reader(data_file, delimiter = " "))

    data_matrix_full = np.array(spam_file[0:], dtype=np.float)

    return data_matrix_full

# Function: main
# INPUT ARGS:
#   [none]
# Return: [none]
def main():
    data_matrix_full = convert_data_to_matrix("spam.data")

    np.random.shuffle(data_matrix_full)

    col_length = data_matrix_full.shape[1]

    data_matrix_test = np.delete(data_matrix_full, col_length - 1, 1)

    binary_vector = data_matrix_full[:,57]

    scale(data_matrix_test)

    weight_matrix = gradientDescent(data_matrix_test,binary_vector,.05,500)

    print(np.sum(weight_matrix))


    # get 60-20-20 of data for train-validation-test

    # go through and count items in matrix belonging to 0 and 1 groups
    count_of_zeros = 0
    count_of_ones = 0

    for item in data_matrix_full:
        if int(item[col_length - 1]) == 1:
            count_of_ones += 1

        if int(item[col_length - 1]) == 0:
            count_of_zeros += 1

    # print(count_of_zeros)
    # print(count_of_ones)



    # np.random.shuffle(data_matrix_test)

    # apply gradient descent to matrix
    learned_weight_matrix = gradientDescent(data_matrix_test, binary_vector, .05, 100)

    np.savetxt("plzwork.csv", weight_matrix, delimiter = " ")

# call our main
main()
