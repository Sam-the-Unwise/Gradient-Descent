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
        spam_file = list(csv.reader(data_file, delimiter = " "))

    data_matrix_full = np.array(spam_file[0:], dtype=np.float)
    return data_matrix_full



# Function: calculate_train_test_and_val_data
# INPUT ARGS:
#   data_matrix_test : modified data matrix (lacking last row of 1s and 0s)
#   data_matrix_full : original data matrix
#   binary_vector : vector containing the list of 1s and 0s in correspondance to
#       data_matrix_test (obtained from data_matrix_full)
# Return: [none]
def calculate_train_test_and_val_data(data_matrix_test, data_matrix_full, binary_vector):
    # get 60-20-20 of data for train-validation-test

    row_length = data_matrix_full.shape[0]
    col_length = data_matrix_full.shape[1]

    # divide num of zero items into 60-20-20
    train_data_count = int(row_length * .6)
    val_data_count = int(row_length * .2)
    test_data_count = int(row_length * .2)
    
    # create zero-arrays to initialize the numpy array for each set
    #   allows us to add the arrays from the data_matrix_full since they will now
    #   be the same size
    array_of_zeros = []

    for i in range(col_length - 1):
        array_of_zeros.append(0)

    train_data = np.array(array_of_zeros)
    val_data = np.array(array_of_zeros)
    test_data = np.array(array_of_zeros)
    
    # NOTE: to do this, we must assume that the data_matrix_test is still in the same order as the binary_vector#

    # create index count that will keep track of the index we're at in the data_matrix_test and binary_vector
    index_count = 0

    # create count that will count the total number of zero-data we've added
    zero_test_data_count = 0
    zero_train_data_count = 0
    zero_val_data_count = 0
    one_test_data_count = 0
    one_train_data_count = 0
    one_val_data_count = 0

    train_binary_vector = []
    test_binary_vector = []
    val_binary_vector = []

    # loop through matrix and assign certain items to different matrixes
    #       assign 60 to train, 20 to val, and 20 to test for both 0s and 1s
    for data_list in data_matrix_test:

        # append the first 60% of data to the zero_train_data
        if index_count < train_data_count:
            train_data = np.vstack((train_data, np.array(data_list)))

            if int(binary_vector[index_count]) is 0:
                zero_train_data_count += 1
            else:
                one_train_data_count += 1

            train_binary_vector.append(binary_vector[index_count])

        # append the next 20% of data to the zero_val_data
        elif index_count < (train_data_count + val_data_count):
            val_data = np.vstack((val_data, np.array(data_list)))

            if int(binary_vector[index_count]) is 0:
                zero_val_data_count += 1
            else:
                one_val_data_count += 1

            val_binary_vector.append(binary_vector[index_count])
            
        # append the last 20% of data to the zero_test_data
        elif index_count < (train_data_count + val_data_count + test_data_count):
            test_data = np.vstack((test_data, np.array(data_list)))

            if int(binary_vector[index_count]) is 0:
                zero_test_data_count += 1
            else:
                one_test_data_count += 1

            test_binary_vector.append(binary_vector[index_count])

        index_count += 1

    # get ride of added array of 0s we made when initializing array
    train_data = np.delete(train_data, 0, 0)
    val_data = np.delete(val_data, 0, 0)
    test_data = np.delete(test_data, 0, 0)

    # print out amount of 0s and 1s in each set
    print("                y")
    print("set              0     1")
    print("test           " + str(zero_test_data_count) + "  " + str(one_test_data_count))
    print("train          " + str(zero_train_data_count) + "  " + str(one_train_data_count))
    print("validation     " + str(zero_val_data_count) + "  " + str(one_val_data_count))


    # apply gradient descent to train data
    weight_matrix = gradientDescent(train_data, train_binary_vector, .5, 500)

    # save our weight matrix
    np.savetxt("plzwork.csv", weight_matrix, delimiter = " ")
    data_matrix_full = convert_data_to_matrix("spam.data")


    # multiply weight_matrix with train_matrix and val_matrix
    multi_train_data = np.array(np
                                .zeros(weight_matrix.shape[1]*train_data.shape[0])
                                .reshape(train_data.shape[0], weight_matrix.shape[1]))
    multi_val_data = np.array(np
                                .zeros(weight_matrix.shape[1]*val_data.shape[0])
                                .reshape(val_data.shape[0], weight_matrix.shape[1]))

    # multiply matrices -- save in out = [matrix_name]
    np.matmul(train_data, weight_matrix, out = multi_train_data)
    np.matmul(val_data, weight_matrix, out = multi_val_data)

    plt.plot(multi_train_data, color='black', label='train')
    plt.plot(multi_val_data, color='red', label='validation')
    plt.xlabel('interations')
    plt.show()


# Function: main
# INPUT ARGS:
#   [none]
# Return: [none]
def main():
    # get the data from our CSV file
    data_matrix_full = convert_data_to_matrix("spam.data")

    np.random.shuffle(data_matrix_full)

    # get necessary variables
    # shape yields tuple : (row, col)
    col_length = data_matrix_full.shape[1]

    data_matrix_test = np.delete(data_matrix_full, col_length - 1, 1)

    binary_vector = data_matrix_full[:,57]

    scale(data_matrix_test)

    # calculate train, test, and validation data
    weight_matrix = calculate_train_test_and_val_data(data_matrix_test, 
                                                    data_matrix_full, 
                                                    binary_vector)



# call our main
main()
