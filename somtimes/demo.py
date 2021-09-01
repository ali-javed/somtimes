
#import libraries
from SelfOrganizingMap import SelfOrganizingMap
import numpy as np
import warnings

##############################
# author: Ali Javed
# October 14 2020
# email: ajaved@uvm.edu
#############################
        
warnings.filterwarnings("ignore")
np.random.seed(0)

#set window size for DTW --- using zero in this example and this will replicate Euclidean distance
wSize = 0.1
#set epochs
epochs =20

#generate random RGB values, not time series but they allow a good illustation.
inputs = np.random.rand(500,3)
inputs = inputs

#set mesh size
n = len(inputs)
slength = len(inputs[0])
totalNodes = 5*np.sqrt(len(inputs))
ss = int(np.sqrt(totalNodes))

#no need to create giangatic maps
if ss>20:
    ss =20

print('Creating SOM... ')

hiddenSize = [ss,ss]
print('Hidden Size is: '+str(hiddenSize))
SOM = SelfOrganizingMap(inputSize = len(inputs[0]), hiddenSize = hiddenSize)



##################################
    # Description: Function iterates to organize the Kohonen layer

    # inputs: all inputs
    # epochs: epochs to iterate for
    # path: Path to save SOM plots
    # labels: if ground truth is available for color coding SOM
    # observationID: if observation ids are available for recording

    # windowSize: windowSize in terms of time steps (1,2,3.... maximum length of time series) to be used by DTW (for project), not usefull in assignment and set to 0.
    # targets: target labels for plotting.
    # showPlot: call the plt.show() usually in an editor or jupyter notebook but not command prompt.
#################################
windowSize = int(len(inputs[1]) * wSize)
stats = SOM.iterate(inputs,epochs = epochs,windowSize = windowSize,k=1,randomInitilization=False)

### This function is specifically for demo and visualized RGB values. Will not work for usual data.
SOM.plotMap_RGB(inputs,windowSize=0, labels=inputs, path = 'RGB_figure')

####functions available for visualizing data
########################
# Description: Visualized data and superimposed color on each observation based on labels.

# INPUTS:
# inputs: Input observations for which the BMU location is desired
# labels: Categorical Labels for each observation in same order as inputs, such as [0,1,2,3]
# labelNames: Python dictionary that associates label to a name such as labelNames[0] = 'Class 1'
# path: filename or path at which the figure will be saved.

#plotMap(inputs,windowSize, labels=[], labelNames = {},path = 'plot_epoch'):
########################



########################
# Description: Color map based SOM, usually ideal for continuous labels that can not be binned. If labels are binned then use plotmap.

# INPUTS:
# inputs: Input observations for which the BMU location is desired
# labels: Labels for data in continuous values, such as [0.5,10,1.24]
# path: filename or path at which the figure will be saved.

#plotMap_cmap(inputs, windowSize, labels=[], path='plot_epoch'):
########################

########################
# Description: Color map based SOM. It will interpolate labels in the background using gaussian filter.

# INPUTS:
# inputs: Input observations for which the BMU location is desired
#background_labels: Background values for each observation such as [0.5,10,1.24] --- These will be interpolated throughout the SOM
# labels: Labels for data in continuous values, such as [0,1,2]
# labelNames: Python dictionary that associates label to a name such as labelNames[0] = 'Class 1'
# Sigma: Sigma value for interpolation using gaussian filter
# path: filename or path at which the figure will be saved.

#plotMap_cmap_interpolated(inputs, windowSize, labels=[], background_labels=[], label_names={}, sigma=4,path='plot_epoch'):
########################