'''
this script is an implementation of the Stochastic Gradiant Descent of SoftSVM that plots the binary and hinge-loss
for different regulariztion parameter lamda.
'''

import numpy.matlib
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
import scipy
import random
from random import randint


def soft_svm(td, y_label, T, lamda):
    theta = np.zeros(4)
    w = theta
    hinge_loss = []
    binary_loss = []
    weights = []
    indices = []
    N = len(td) - 1

    for i in range(T):
        w = (1/lamda)*theta
        rand_i = random.randint(0, N)
        weights.append(w)
        indices.append(i+1)
        b = td[rand_i]
        # hinge_loss.append(hinge_loss(w, y_label, lamda))
        hinge_loss.append(max(0, 1-(y_label[rand_i] * np.dot(b, w))))
        # print("loss:", 1-y_label[rand_i] * np.dot(b,w))
        # binary_loss.append(zero_one_loss((y_label * (np.dot(training_data, w))) > 0, y))

        if ((y_label[rand_i] * np.dot(b, w)) < 1):
            theta = theta + y_label[rand_i] * td[rand_i]
            print("dot is:", y_label[rand_i] * np.dot(b, w))
            print("theta is:", theta)
            print("label is", y_label[rand_i])

    return (indices, hinge_loss)


def main():  # learning_rate, epochs, C):ei
    T = 500
    lamda = 1

    # load datasets
    a_array = []
    y_label = []
    file = open("fg_inputs.txt", "r")
    for line in file:
        for num in line.strip().split(','):
            a_array.append(float(num))

    b_array = np.array(a_array)
    training_data = b_array.reshape(200, 3)
    ylabel_p = np.ones(100)
    ylabel_n = -1*np.ones(100)
    y_label = np.r_[ylabel_p, ylabel_n]
    bias = np.ones(200)
    training_data = np.c_[bias, training_data]
    # print(training_data)

    print("Training...")
    (indices, hinge_loss) = soft_svm(training_data, y_label, T, lamda)
    # print("hinge_loss:", hinge_loss)
    # print("indices:", indices)

    plt.plot(indices, hinge_loss)
    plt.show()

    # print ("the weights are:", weights)

    # print ("Weights are trained " + str(epochs) + " times, with a learning rate of "
    #      + str(learning_rate) + " and a capacity of " + str(C))
    # print "Testing..."
    # print "Tested on " + str(file_test) + ", accuracy is " + \
    # str(svm(test_data, weights)*100) + "%"

if __name__ == '__main__':
    main()
