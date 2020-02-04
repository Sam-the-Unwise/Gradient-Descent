import numpy as np
import csv

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
    for index in range(0, max_iterations):
        #calculate y_tid
        y_tild = -1

        if(y[index] == 1):
            y_tild = 1

        # variables for simplification
        weight_vector_transpose = np.transpose(weight_vector)
        verctor_mult = np.multiply(weight_vector_transpose, X[index,:])
        inner_exp = np.multiply(y_tild, verctor_mult)
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

# Function: gradientDescent
# INPUT ARGS:
#   X : a matrix of numeric inputs {Obervations x Feature}
# Return: weight_matrix
#def calc_step_size(X):


with open("spam.data", 'r') as data_file:
    spam_file = list(csv.reader(data_file, delimiter = " "))

data_matrix_full = np.array(spam_file[0:], dtype=np.float)

data_matrix_test = np.delete(data_matrix_full, -1, 1)
print(data_matrix_test)

binary_vector = data_matrix_full[:,57]

gradientDescent(data_matrix_test,binary_vector,.5,4601)
