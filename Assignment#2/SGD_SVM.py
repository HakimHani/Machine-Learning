'''
this script will calculate the hinge loss of three linear classifiers w1,w2,w3 as a multiclass classifier
by taking the maximum dot product of the weight vectors w1,w2,w3 and the point to be clssified
'''

import parser
import numpy as np
import sys
import copy

'''
Note: if a parameter is denoted as an array, it is an np.array
'''

''' Returns 1 or -1 as our class guess
    
    Params: weights: weight array with bias as weights[0]
            point:   data point array with true class as x_n[0]
    '''

def get_sign(weights, point):
    # remove class from the line, and bias from weights when calculating
    # vector multiplication
    sign = weights[0] + sum(np.array(point[1:])*weights[1:])
    return 1 if sign >= 0.0 else -1

''' Returns: gradients for weights and bias
                 wgradient: weight gradient array
                 bgradient: bias gradient float
    Params: weights: weight array with bias as weights[0]
            x_n:     data point array with true class as x_n[0]
            N:       number of training points in training data
            C:       arbitrary capacity
    '''

def get_gradients(weights, x_n, N, C=.05):
    bias = weights[0]
    y_n = x_n[0]
    wgradient = bgradient = 0
    if (y_n * (sum(weights[1:]*x_n[1:])+bias)) < 1:
        # incorrect guess
        wgradient = (1/N)*(weights[1:]) - (C*y_n*x_n[1:])
        bgradient = (-C)*(y_n)
    else:
        # correct guess
        wgradient = (1/N)*(weights[1:])
    return wgradient, bgradient

''' Returns: trained weights in np.array
    Params: train_data:    training data
            learning_rate: learning rate
            epochs:        number of times to loop through training data
            C:             arbitrary capacity
    '''

def train_weights(training_data, learning_rate, epochs, C=.05):
    weight_length = len(training_data[0])
    weights = np.array([0] * weight_length).astype(float)
    N = len(training_data)
    for iteration in range(epochs):  # loop epochs
        for point in training_data:  # [0.2, 0.4, ..., 1]
            wgradient, bgradient = get_gradients(
                weights, point, N, C)   # gradients
            # update 
            weights[0] -= learning_rate*bgradient  
            weights[1:] -= learning_rate*wgradient
    return weights

def main(): 
    # load datasets
    a_array = []
    file = open("fg_inputs.txt", "r")
    for line in file:
        for num in line.strip().split(','):
            a_array.append(float(num))
    b_array = np.array(a_array)
    training_data = b_array.reshape(200, 3)
    print(training_data)

    epochs = 4
    learning_rate = 0.45
    C = 0.0127

    print("Training...")
    weights = train_weights(training_data, learning_rate, epochs, C)

    print("the weights are:", weights)

    print("Weights are trained " + str(epochs) + " times, with a learning rate of "
          + str(learning_rate) + " and a capacity of " + str(C))
    # print "Testing..."
    # print "Tested on " + str(file_test) + ", accuracy is " + \
    # str(svm(test_data, weights)*100) + "%"

if __name__ == '__main__':
    main()
