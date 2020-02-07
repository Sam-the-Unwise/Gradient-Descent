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

    # num of input features
    X_arr_col = arr_dim[1]

    wm_total_entries = X_arr_col * max_iterations

    # variable that initiates to the weight vector -- should start as zero vector
    weight_vector = np.zeros(X_arr_col)

    # matrix for real numbers
    #   row of #s = num of inputs
    #   num of cols = maxIterations
    weight_matrix = np.array(np
                        .zeros(wm_total_entries)
                        .reshape(X_arr_col, max_iterations))


    # ALGORITHM
    for index in range(0, max_iterations):
        # calculate y tilda
        y_tild = -1

        if(y[index] == 1):
            y_tild = 1


        # variables for simplification
        weight_vector_transpose = np.transpose(weight_vector)
        vector_mult = np.multiply(weight_vector_transpose, X[index,:])
        inner_exp = np.multiply(y_tild, vector_mult)
        
        # calculate gradient
        gradient = (1/(1+np.exp(inner_exp))*
                    (np.exp(inner_exp))*
                    ((-y_tild)*X[index,:]))


        # update weight_vector depending on positive or negative
        # If negative, you add to the steps
        if y_tild == -1:
            weight_vector += gradient

        # If positive, you subtract it
        elif y_tild == 1:
            weight_vector -= gradient

        #print(weight_vector)
        # store the resulting weight_vector in the corresponding column weight_matrix
        weight_matrix[: ,index] = weight_vector


    # end of algorithm
    return weight_matrix

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

        #print(f"{counter} : mean {mean} std: {std}")
        column -= mean
        #print(f"col - {column}\n")
        column /= std
        #print(f"col / {column}")

# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    with open(file_name, 'r') as data_file:
        spam_file = list(csv.reader(data_file, delimiter = " "))

    data_matrix_full = np.array(spam_file[0:], dtype=np.float)

    return data_matrix_full

# Function: main
# INPUT ARGS:
#   [none]
# Return: [none]
def main():
    # get the data from our CSV file
    data_matrix_full = convert_data_to_matrix("spam.data")

    # get necessary variables
    # shape yields tuple : (row, col)
    col_length = data_matrix_full.shape[1]

    data_matrix_test = np.delete(data_matrix_full, col_length - 1, 1) # delete last column
    binary_vector = data_matrix_full[:,57]

    test_col_length = data_matrix_full.shape[1]

    # scale our matrix so our data is between 0 and 1
    scale(data_matrix_test)

    # get 60-20-20 of data for train-validation-test

    # go through and count items in matrix belonging to 0 and 1 groups
    count_of_zeros = 0
    count_of_ones = 0

    for item in data_matrix_full:
        if int(item[col_length - 1]) == 1:
            count_of_ones += 1

        if int(item[col_length - 1]) == 0:
            count_of_zeros += 1

    print(count_of_zeros)
    # print(count_of_ones)

    # divide num of zero items into 60-20-20
    zero_train_data_count = count_of_zeros * .6
    print(zero_train_data_count)
    zero_val_data_count = count_of_zeros * .2
    print(zero_val_data_count)
    zero_test_data_count = count_of_zeros * .2
    print(zero_test_data_count)

    # divide num of one items into 60-20-20
    one_train_data_count = int(count_of_ones * .6)
    one_val_data_count = int(count_of_ones * .2)
    one_test_data_count = int(count_of_ones * .2)

    # loop through matrix and assign certain items to different matrixes
    #       assign 60 to train, 20 to val, and 20 to test
    zero_train_data = np.array([])
    zero_val_data = np.array([])    
    zero_test_data = np.array([])

    # to do this, we must assume that the data_matrix_test is still in the same order as the binary_vector#

    # create index count that will keep track of the index we're at in the data_matrix_test and binary_vector
    index_count = 0
    # create count that will count the total number of zero-data we've added
    zero_count = 0

    for data_list in data_matrix_test:
        # check if this row is one of 0 data by checking binary_vector
        if int(binary_vector[index_count]) is 0:

            # append the first 60% of data to the zero_train_data
            if zero_count < zero_train_data_count:
                zero_train_data = np.append(zero_train_data, [data_list])

            # append the next 20% of data to the zero_val_data
            elif zero_count < (zero_train_data_count + zero_val_data_count):
                zero_val_data = np.append(zero_val_data, data_list)
            
            # append the last 20% of data to the zero_test_data
            elif zero_count < (zero_train_data_count + zero_val_data_count + zero_test_data_count):
                zero_test_data = np.append(zero_test_data, data_list)

            zero_count += 1

        index_count += 1

    for item in zero_train_data:
        print(item)




    one_train_data = np.array(np.zeros(one_train_data_count))
    one_val_data = np.array(np.zeros(one_val_data_count))
    one_test_data = np.array(np.zeros(one_test_data_count))


    # np.random.shuffle(data_matrix_test)

    # apply gradient descent to matrix
    learned_weight_matrix = gradientDescent(data_matrix_test, binary_vector, .05, 100)
    



# call our main
main()
