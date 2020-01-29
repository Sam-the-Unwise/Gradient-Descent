import numpy

# function that does gradient descent
# INPUT ARGS: 
#   X : a mtrix of numeric inputs
#   y : a vector of binary outputs
#   stepSize : learning rate - epsilon parameters
#   maxIterations : pos int that controls how many steps to take
def gradientDescent(X, y, stepSize, maxIterations):

    # VARIABLES

    # variable that initiates to the zero vector
    weightVector = array([0, 0])

    # matrix for real numbers
    #   row of #s = num of inputs
    #   num of cols = maxIterations
    weightMatrix = [len(X)][maxIterations]

    # ALGORITHM

    # include a for loop over iterations (from 1 to maxIterations)
    # during each iteration
    #   first comput the gradient give the current weightVector
    #   then update weightVector by taking a step in the negative gradient direction
    #   then store the resulting weightVector in the corr. column of weightMatrix

    # end of algorithm
    return weightMatrix

