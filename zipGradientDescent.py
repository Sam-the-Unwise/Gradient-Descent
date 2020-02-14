###############################################################################
#
# AUTHOR(S): Josh Holguin
#            Samantha Muellner
#            Jacob Christiansen
# DESCRIPTION: program that will find and graph gradientDescent on the
#       provided data set -- in this case SAheart.data
# VERSION: 2.3.2v
#
###############################################################################

import numpy as np
import csv
from math import sqrt
import sklearn.metrics



# Function: calculate_gradient
# INPUT ARGS:
#   matrix : input matrix row with obs and features
#   y_tild : modified y val to calc gradient
#   step_size : step fir gradient
#
# Return: [none]
def calculate_gradient(x_row, y_tild, step_size, weight_vector_transpose):
    # calculate elements of the denominator
    verctor_mult = np.multiply(weight_vector_transpose, x_row)
    inner_exp = np.multiply(y_tild, verctor_mult)
    denom = 1 + np.exp(inner_exp)

    numerator = np.multiply(x_row, y_tild)
    
    # calculate gradient
    gradient = numerator/denom

    return gradient



# Function: gradientDescent
# INPUT ARGS:
#   X : a matrix of numeric inputs {Obervations x Feature}
#   y : a vector of binary outputs {0,1}
#   stepSize : learning rate - epsilon parameters
#   max_iterations : pos int that controls how many steps to take
# Return: weight_matrix
def gradientDescent(X, y, step_size, max_iterations):
    # tuple of array dim (row, col)
    arr_dim = X.shape

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
        
        for index in range(0, X.shape[1]):
            #calculate y_tid
            y_tild = -1

            if(y[index] == 1):
                y_tild = 1


            grad_log_losss = 0
            verctor_mult = 0
            inner_exp = 0

            # variables for simplification
            gradient = calculate_gradient(X[index,:], y_tild, step_size, weight_vector_transpose)

            grad_log_losss += gradient

        
        mean_grad_log_loss = grad_log_losss/X.shape[1]

        # update weight_vector depending on positive or negative
        weight_vector -= np.multiply(step_size, mean_grad_log_loss)

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
    data_matrix_full = np.genfromtxt( file_name, delimiter = " " )
    return data_matrix_full


# Function: split matrix
# INPUT ARGS:
#   [none]
# Return: [none]
def split_matrix(X):
    train, validation, test = np.split( X, [int(.6 * len(X)), int(.8 * len(X))])

    return (train, validation, test)



def calculate_sigmoid(y):
    y_tilde_i = 1/(1 + np.exp(-y))

    return y_tilde_i



def get_t_and_v_data(test_data, validation_data, pred):
    test_output = np.matmul(test_data, pred)
    validation_output = np.matmul(validation_data, pred)

    return (test_output, validation_output)


# Function: main
# INPUT ARGS:
#   [none]
# Return: [none]
def main():
    # get the data from our CSV file
    data_matrix_full = convert_data_to_matrix("zip.train")

    np.random.shuffle(data_matrix_full)

    # get necessary variables
    # shape yields tuple : (row, col)
    col_length = data_matrix_full.shape[1]

    data_matrix_test = np.delete(data_matrix_full, 0, 1)

    binary_vector = data_matrix_full[:,57]
    # calculate train, test, and validation data
    #weight_matrix = calculate_train_test_and_val_data(data_matrix_test,
    #                                                data_matrix_full,
    #
    count = 0
    for item in data_matrix_full[:,col_length - 1]:

        if item != 0 or item != 1:
            np.delete(data_matrix_full, count, 0)
        count += 1

    train, validation, test = split_matrix(data_matrix_full)

    X_train_data = np.delete(train, 0, 1)
    X_validation_data = np.delete(validation, 0, 1)
    X_test_data = np.delete(test, 0, 1)

    scale(X_train_data)
    scale(X_validation_data)
    scale(X_test_data)

    # print out amount of 0s and 1s in each set
    # print("                y")
    # print("set              0     1")
    # print("test           " + str(X_train_data.count(0)) + "  " + str(X_train_data.count(1)))
    # print("train          " + str(validation_data.count(0)) + "  " + str(validation_data.count(1)))
    # print("validation     " + str(test_data.count(0)) + "  " + str(test_data.count(1)))

    y_train_vector = train[0]
    y_validation_vector = validation[0]
    y_test_vector = test[0]

    max_iterations = 1500
    step_size = .5

    train_pred_matrix = gradientDescent(X_train_data, y_train_vector, step_size, max_iterations)
    val_pred_matrix = gradientDescent(X_validation_data, y_validation_vector, step_size, max_iterations)
    test_pred_matrix = gradientDescent(X_test_data, y_test_vector, step_size, max_iterations)

    #test_data_output, validation_data_output = get_t_and_v_data(test_data, validation_data, pred_matrix)



    ######################## CALCULATE LOGISTIC REGRESSION ########################

    # get dot product of matrixes
    training_prediction = np.dot(X_train_data, train_pred_matrix)
    validation_prediction = np.dot(X_validation_data, val_pred_matrix)
    test_prediction = np.dot(X_test_data, -test_pred_matrix)

    sigmoid_vector = np.vectorize(calculate_sigmoid)

    # used to set numbers above 0 to 1 and below 0 to -1
    training_prediction = sigmoid_vector(training_prediction)
    validation_prediction = sigmoid_vector(validation_prediction)
    test_prediction = sigmoid_vector(test_prediction)

    # used to round numbers -- unsure if we need this but Jacob said fractions were a problem when graphing *shrugs*
    training_prediction = np.around(training_prediction)
    validation_prediction = np.around(validation_prediction)
    test_prediction = np.around(test_prediction)


    # calculate minumum
    train_sum_matrix = []
    validation_sum_matrix = []

    for count in range(1, max_iterations):
        mean = np.mean(y_train_vector != training_prediction[:, count-1])

        train_sum_matrix.append(mean)

    # must use enumerate otherwise get the error ""'numpy.float64' object is not iterable"
    train_min_index, train_min_value = min(enumerate(train_sum_matrix))
    validation_min_index, validation_min_value = min(enumerate(train_sum_matrix))


    # calculate loss

    training_loss_result_matrix = []
    validation_loss_result_matrix = []

    print(y_train_vector.shape)
    print(training_prediction.shape)

    # create loss validation matrices
    for number in range(max_iterations):
        training_loss_result_matrix.append(sklearn.metrics.log_loss(y_train_vector, training_prediction[:, number]))
        validation_loss_result_matrix.append(sklearn.metrics.log_loss(y_validation_vector, validation_prediction[:, number]))


    # print(training_loss_result_matrix)
    # print(validation_loss_result_matrix)

    with open("zipLogLoss.csv", mode = 'w') as roc_file:

        fieldnames = ['train loss', 'validation loss']
        writer = csv.DictWriter(roc_file, fieldnames = fieldnames)

        writer.writeheader()

        for index in range(max_iterations):
            writer.writerow({'train loss': training_loss_result_matrix[index],
                            "validation loss": validation_loss_result_matrix[index]})


    ######################## CALCULATE ROC CURVE ########################

    # print(train_min_index)
    # print(train_min_value)


    #fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test_vector, test_prediction)[:, validation_min_index]

    # calculate roc curves for logistic regression and baseline
    #fpr, tpr, thresho= roc_curve(y_test, sig_v(np.dot(X_test, weightMatrix))[:, val_min_index])
    # log_roc.append((fpr_log, tpr_log))



    # with open("ROC.csv", mode = 'w') as roc_file:

    #     fieldnames = ['FPR', 'TPR']
    #     writer = csv.DictWriter(roc_file, fieldnames = fieldnames)

    #     writer.writeheader()

    #     writer.writerow({'FPR': fpr, "TPR": tpr})




# call our main
main()