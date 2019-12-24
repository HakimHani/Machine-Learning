'''
this script is created to plot the cost vs number of clusters of k-means algorithm
'''

import numpy.matlib
import numpy as np
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
import scipy 
import random 
import logging
from random import randint
from scipy.spatial.distance import pdist, squareform


def dist(X,Y):
    distance = ((X[0] - Y[0])**2 + (X[1] - Y[1])**2)**0.5
    return distance

def k_means_cost(X, mean):
    cost = 0
    # print("the clusters are:", clusters)
    for j in range(1,len(X)):
        cost += (X[j][0] - mean[0] )**2 + (X[j][1] - mean[1])**2
    return cost    

def k_means(X, k):
    n = X.shape[0]
    d = X.shape[1]
    # max_dist = 0
    # max_dist_1 = 0
    # max_dist_2 = 0
    distances = np.zeros((n,k))
    print(distances.shape)

    # manually picking the mean values
    # new_centers = np.array([[-1.28, -1.9], [3.5, 1.7], [3.34, 6]])
    # randomly picking the mean values
    # new_centers = np.random.uniform(np.min(X), np.max(X), size=(k, d))
    new_centers = X[np.random.choice(n, k, replace= False)]

    # picking the furthest distanced centers
    '''A = np.random.randint(np.min(X), np.max(X), size=(1, d))
    for i in range(n):
        temp = dist(X[i], A[0])
        if temp > max_dist:
            max_dist = temp
            B = X[i]
    for i in range(n):
        temp1 = dist(X[i], A[0])
        temp2 = dist(X[i], B)
        if temp1 + temp2 > max_dist_1 + max_dist_2:
             max_dist_1 = temp1
             max_dist_2 = temp2
             C = X[i]'''

    # new_centers = np.array([A[0], B, C])
        
    # print("centers is:",new_centers)
    old_centers = np.zeros(new_centers.shape)

    clusters = np.zeros(n)

    error = np.linalg.norm(new_centers - old_centers)
    # print (error)

    while error !=0:
        for i in range(k):
            # distances = dist(X[i], new_centers)
            distances[:,i] = np.linalg.norm(X - new_centers[i], axis=1)
            clusters = np.argmin(distances, axis=1)
            # print("clusters are:", cluster)
            # clusters[i] = cluster
        
        old_centers = deepcopy(new_centers)

        for i in range(k):
            # points = [X[j] for j in range(len(X)) if clusters[j] == i]
            # new_centers[i] = np.mean(points, axis=0)
            if len(X[clusters == i]) == 0:
                continue
            new_centers[i] = np.mean(X[clusters == i], axis=0)
    
        error = np.linalg.norm(new_centers - old_centers)
    # print("clusters are:", X[clusters == 1])
    return clusters, new_centers

def main():	
    # load datasets
    a_array = [] 
    WCSS_array=np.array([])
    file = open("twodpoints.txt", "r")
    for line in file:
    	for num in line.strip().split(','):
    	  a_array.append(float(num))
    b_array = np.array(a_array)
    input_data = b_array.reshape(200,2)
    
    # compute the clustering and centeriods
    for i in range(1,11):
        (clusters, centeriods) = k_means(input_data, i)
        print("centriods are:", centeriods)
        wcss=0
        for j in range(i):
            if len(input_data[clusters == j]) == 0:
                continue
            else:
                wcss+=np.sum((input_data[clusters == j]- centeriods[j,:])**2)
        #         wcss+=k_means_cost(input_data[clusters == j], centeriods[j,:])
        #         avg_loss = sum(wcss)/len(wcss)
        #         print("cluster number:", j, input_data[clusters == j])
        # c = k_means_cost(input_data, clusters, centeriods)  
        # cost.append(c)
        WCSS_array = np.append(WCSS_array,wcss) 
    x = np.linspace(1,10,10)
    plt.plot(x, WCSS_array)
    plt.ylabel('cost')
    plt.xlabel('number of clusters')
    plt.show()
    
if __name__ == '__main__':
    main()
