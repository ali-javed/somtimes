
#import libraries
from SelfOrganizingMap import SelfOrganizingMap
import numpy as np


##############################
# author: Ali Javed
# October 14 2020
# email: ajaved@uvm.edu
#############################
        
        
        

time_series = [[1,2,3,4,5],[0,1,2,3,5],[6,7,8,9,5], [0,1,2,3,3]]

time_series = np.asarray(time_series)

#optionall have labels (i.e., ground truth if they are available).
labels = [0,1,0,1]


print('Creating Multivariate SOM...')
#############################
# Description: Class initilizer. This function creates an instance of neural network saving the parameters and setting random weights
# inputsSize: number of input nodes needed i.e. 5.
# hiddenSize: number of hidden layer nodes [2,3] will create a 2x3 node grid
########################################
hiddenSize = [10,10]
SOM = SelfOrganizingMap(inputSize = np.shape(time_series)[1], hiddenSize = hiddenSize)


##################################
# Description: Function iterates to organize the Kohonen layer

# inputs: all inputs
# epochs: epochs to iterate for
# k: optional to generate hard clusters
##################################
fname = 'Demo_'
stats = SOM.iterate(time_series,epochs = 40)

