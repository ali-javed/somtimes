#python libraries needed in code
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from collections import defaultdict
import time
import multiprocessing
from joblib import Parallel, delayed
from dtaidistance import dtw
from tslearn import metrics
from scipy.spatial.distance import cdist
import copy
#from tqdm import tqdm


class SelfOrganizingMap:
    def __init__(self, inputSize, hiddenSize):
        ##############################
        # author: Ali Javed
        # October 14 2020
        # email: ajaved@uvm.edu
        #############################
        # Description: Class initilizer. This function creates an instance of neural network saving the parameters and setting random weights
        # inputsSize: number of input nodes needed i.e. 5.
        # hiddenSize: number of hidden layer nodes [2,3] will create a 2x3 node grid
        ########################################
        #set random see for reproducibility
        #np.random.seed(0)
        # initilize variables
        self.hiddenSize = np.asarray(hiddenSize)
        self.inputSize = inputSize
        # always start learning rate at 0.9
        self.learningRateInitial = 0.9
        self.learningRate = 0.9
        self.neighborhoodSizeInitial = int(self.hiddenSize[0] / 2)
        self.neighborhoodSize = int(self.hiddenSize[0] / 2)
        self.Umatrix = np.zeros((self.hiddenSize[0],self.hiddenSize[1]))

        
        # initilize weights between 0 and 1 for a 3d weights matrix
        self.weights_Kohonen = np.zeros((self.hiddenSize[0]*self.hiddenSize[1], self.inputSize))

    
    def getWeights(self):
        return self.weights_Kohonen


    def getUmatrix(self):
        return self.Umatrix

    def saveWeights(self,path):
        np.save(path,self.weights_Kohonen,)

    def loadWeights(self,path):
        self.weights_Kohonen = np.load(path)
