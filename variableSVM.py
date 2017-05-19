#******************************************************************************
#Title: Variable RBF Kernel SVM
#Author: Brandon Campbell
#Date: December 17, 2016
#******************************************************************************

import numpy as np
import scipy.spatial as sps
from sklearn.svm import SVC
from sklearn import preprocessing

import matplotlib.pyplot as plt


##################################################################################
# Defintions
##################################################################################
def sigmaMatrix(x):
    pairDist = sps.distance.pdist(x, 'euclidean')
    squarePDist = sps.distance.squareform(pairDist)
    nearestNeighbors = np.sort(squarePDist, axis = 0)
    sigma = np.mean(nearestNeighbors[1:11, :], axis = 0)
    sigmaSquared = np.power(sigma, 2)
    sigmaSquared = np.transpose(np.expand_dims(sigmaSquared, axis =1))

    return sigmaSquared

def gramMatrix(x, sigmaSquared):
    pairDist = sps.distance.pdist(x, 'euclidean')
    squarePDist = sps.distance.squareform(pairDist)

    distanceDivSigmaSquared = (squarePDist/ sigmaSquared[:, None])
    argument = 0.5 * distanceDivSigmaSquared * 100

    rbfKernel = np.exp(-argument)

    kval = rbfKernel[0, :, :]

    return kval

def trainingGram(x, y, sigmavalues):
    x_i = np.linalg.norm(x, axis=1)
    x_j = np.linalg.norm(y, axis=0)
    y = []

    for i in x_i:
        for j in x_j:
            value = i - j
            y.append(value)

    numerator = np.asarray(y)

    row_dim = np.shape(x)
    col_dim = np.shape(y)

    num = np.reshape(numerator, (row_dim[0], col_dim[1]))

    distanceDivSigmaSquared = (num / sigmavalues[:, None])
    argument = 0.5 * distanceDivSigmaSquared * 100
    testingKernel = np.transpose(np.exp(-argument))

    return testingKernel
####################################################################################
####################################################################################




#Load peak data from CSV files (Comma delimited)
#Columns are Water, 1st Fat, 2nd Fat, 3rd Fat, R2*
WAT = np.loadtxt('/home/path/',delimiter=',')
BAT = np.loadtxt('/home/path/,delimiter=',')

WAT = np.transpose(WAT)
BAT = np.transpose(BAT)


#Create xData matrix by stacking WAT and BAT
xData = np.vstack((WAT,BAT))

#Create group data
#This creates a group using the number of columns from data
group = np.ones((xData.shape[0]), dtype=np.int)
group[WAT.shape[0]:] = [-1] # Changes the latter half of group to -1


###### Compute gram matrix and train SVM
sigma = sigmaMatrix(xData)
gram = gramMatrix(xData, sigma)

svmModel = SVC(kernel='precomputed', probability=True, C=1.0)

svmModel.fit(gram, group)

###########
#########
#######

testData = np.loadtxt('/home/path/',delimiter=',')

testingKernel = trainingGram(xData, testData, sigma)



