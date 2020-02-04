import numpy as np

# function that does gradient descent
# INPUT ARGS:
#   X : a matrix of numeric inputs {Obervations x Feature}
#   y : a vector of binary outputs {0,1}
#   stepSize : learning rate - epsilon parameters
#   maxIterations : pos int that controls how many steps to take
def gradientDescent(X, y, stepSize, max_iterations):

    # VARIABLES

    # tuple of array dim (row, col)
    arr_dim = X.shape

    # num of input features
    X_arr_col = arr_dim[1]

    wm_total_entries = X_arr_col * max_iterations
    # variable that initiates to the zero vector
    weight_vector = np.zeros(X_arr_col)

    # matrix for real numbers
    #   row of #s = num of inputs
    #   num of cols = maxIterations
    weight_matrix = np.array(np
                        .zeros(wm_total_entries)
                        .reshape(X_arr_col, max_iterations))


    # ALGORITHM
    for(index in range(0, maxIterations)):

        #calculate y_tid
        y_tild = -1

        if(y == 1):
            y_tild = 1

        # calculate gradiant
        gradient = (1/(1+numpy.exp((-y_tild)*weight_vector*X[index]))*
                    (numpy.exp((-y_tild)*weight_vector*X[index]))*
                    ((-y_tild)*X[index]))

        # update weight_vector depending on positive or negative
        # If negative, you add to the steps
        np.multiply(weight_vector, 1)

        # If positive, you subtract it
        if y_tild == 1:
            np.multiply(weight_vector, -1)

        # store the resulting weight_vector in the corresponding column weight_matrix
        weight_matrix.item((index, 1)) = weight_vector


    # end of algorithm
    return weight_matrix


matrix = np.array([ [0,0], [0,0], [0,0] ])

gradientDescent(matrix,0,0,5)