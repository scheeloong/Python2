# -*- coding: utf-8 --
"""
Created on Fri Mar 13 09:21:02 2015

@author: Soon Chee Loong
"""

# Part 2 of Assignment 3 
# Note: You must create the necessary folders before running this file

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from PIL import Image # To open up images 
import scipy as sci

os.chdir('.')

# This takes in the principal component matrix and the shapes of the images 
# and displays the top 25 principal components (k = 25)
def displaySavComps(V, im_shape, quantity):
    '''Display quantity components in V'''
    figure()
    for i in range(quantity):
        plt.subplot(5, 5, i+1) # resize it to 5 by 5 when plotting! 
        plt.axis('off')
        gray()
        imshow(V[i,:].reshape(im_shape))
    savefig('display_save_' + str(quantity) + '_comps.png')  
    show()       

# Given X, whereby each row corresponds to an image
# and the columns corresponds to the dimension of each image 
# It calculates and returns the 
# i) Projection Matrix 
# ii) Variance 
# iii) Mean 
def pca(X):
    """    Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
        From: Jan Erik Solem, Programming Computer Vision with Python
        #http://programmingcomputervision.com/
    """
    
    # get dimensions
    num_data,dim = X.shape
    # Each image is stored as a row on X
    # => num_data => Number of images
    #    dim => Number of dimensions of each image     
    
    # center data
    mean_X = X.mean(axis=0) # get the mean 
    X = X - mean_X # Center the data 
    
    # Use Compact Trick 
    # If more pixels than number of images, use PCA 
    # (Since will be able to calculate eigenvectors and eigenvalues)
    if dim>num_data:
        # PCA - compact trick used
        # First, need find direction (slope) of maximum variance 
        # To do that, calculate covariance marix 
        M = dot(X,X.T) # compute covariance matrix
        # Then, look for eigenvectors of the covariance matrix
        # This means that these vectors already points to direction of longest variability
        # Note: The larger eigenvalues basically means that these eigenvectors 
        # are the"principal" ones, that ones that influences the most. 
        # Therefore: "Principal Components" = Eigenvectors of covariance matrix with largest eigenvalues
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        # Uses feature extraction
        # Project the original dimension images on eigenvectors 
        tmp = dot(X.T,EV).T # this is the compact trick
        # Sort by largest eigenvectors first 
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        # Sort by largest Variances first
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        # Divide each value with its variance 
        # so that PCA is not affected by large values 
        # note: divide by variance to reduce variability 
        for i in range(V.shape[1]):
            V[:,i] /= S
    # Use singular value decomposition 
    else:
        # PCA - SVD used
        U,S,V = linalg.svd(X) # calculate the singular value decomposition of X 
        V = V[:num_data] # only makes sense to return the first num_data
    
    # return the projection matrix, the variance and the mean
    return V,S,mean_X

# Takes into input the a string which represents the directory containing the images 
def get_face_matrix(img_dir, nameOfPerson, quantityStart, quantityEnd):
    # Sort the string of filenames by appending the directory in front of its name 
    # Where, filename is any file which ends with ".jpg" 
    im_files = sorted([img_dir + filename for filename in os.listdir(img_dir) if filename[-4:] == ".png" and nameOfPerson in filename ]) 
    #im_files = sorted([img_dir + filename for filename in os.listdir(img_dir) if filename[-4:] == ".jpg"])

    # Get only first 120 
    im_files = im_files[quantityStart:quantityEnd] 
    
    # Read the image from the first image and get its shape 
    im_shape = array(imread(im_files[0])).shape[:2] # open one image to get the size 
    # Read each individual image flattened() as a 1D array from the sorted images 
    # Therefore, each row of im_matrix contains information about 1 file. 
    im_matrix = array([imread(im_file).flatten() for im_file in im_files])
    # for each row of images in im_matrix, divide that row with it's norm+0.0001 
    # Basically, normalize each matrix so can compare them relatively 
    im_matrix = array([im_matrix[i,:]/(norm(im_matrix[i,:])+0.0001) for i in range(im_matrix.shape[0])])
    # return the row of images and its shape 
    return (im_matrix, im_shape)

#Readability 
#For this project, you should crop out the images of the faces, 
#convert them to grayscale, and
#resize them to 32x32 before proceeding further.

# Use these faces 
act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',
    'Andrea Anders', 'Ashley Benson',
   'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']


# ID  # Since we know faces of training set 
# Also arrange so males are first before female 
# 0 : eckhart
# 1 : sandler
# 2 : brody
# Females 
# 3 : anders
# 4 : benson
# 5 : applegate
# 6 : agron
# 7 : anderson

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
files_dir = "./croppedPngImages/" # Directory containing the letters. 


# Separate dataset into three non-overlapping parts
# With 8 actors
# Training set = 100 face images per actor
# Validation set = 10 face images per actor 
# Test Set = 10 face images per actor 

# Get 100 faces of each person 

# Get eckhart's face 
# Note: Remember to convert to lowercase 
#train1Im, train1Shape = get_face_matrix(files_dir, 'eckhart', 0, 100)
#valIm, val1Shape = get_face_matrix(files_dir, 'eckhart', 101, 110)
#testIm, test1Shape = get_face_matrix(files_dir, 'eckhart', 111, 120)

# Get first 100 to be training set
train1Im, train1Shape = get_face_matrix(files_dir, 'eckhart', 0, 100)
train2Im, train2Shape = get_face_matrix(files_dir, 'sandler', 0, 100)
train3Im, train3Shape = get_face_matrix(files_dir, 'brody', 0, 100)
train4Im, train4Shape = get_face_matrix(files_dir, 'anders', 0, 100)
train5Im, train5Shape = get_face_matrix(files_dir, 'benson', 0, 100)
train6Im, train6Shape = get_face_matrix(files_dir, 'applegate', 0, 100)
train7Im, train7Shape = get_face_matrix(files_dir, 'agron', 0, 100)
train8Im, train8Shape = get_face_matrix(files_dir, 'anderson', 0, 100)



# Get 111-120 to be validation set
val1Im, val1Shape = get_face_matrix(files_dir, 'eckhart', 110, 120)
val2Im, val2Shape = get_face_matrix(files_dir, 'sandler', 110, 120)
val3Im, val3Shape = get_face_matrix(files_dir, 'brody', 110, 120)
val4Im, val4Shape = get_face_matrix(files_dir, 'anders', 110, 120)
val5Im, val5Shape = get_face_matrix(files_dir, 'benson', 110, 120)
val6Im, val6Shape = get_face_matrix(files_dir, 'applegate', 110, 120)
val7Im, val7Shape = get_face_matrix(files_dir, 'agron', 110, 120)
val8Im, val8Shape = get_face_matrix(files_dir, 'anderson', 110, 120)


# Get 101-110 to be test set 
test1Im, test1Shape = get_face_matrix(files_dir, 'eckhart', 100, 110)
test2Im, test2Shape = get_face_matrix(files_dir, 'sandler', 100, 110)
test3Im, test3Shape = get_face_matrix(files_dir, 'brody', 100, 110)
test4Im, test4Shape = get_face_matrix(files_dir, 'anders', 100, 110)
test5Im, test5Shape = get_face_matrix(files_dir, 'benson', 100, 110)
test6Im, test6Shape = get_face_matrix(files_dir, 'applegate', 100, 110)
test7Im, test7Shape = get_face_matrix(files_dir, 'agron', 100, 110)
test8Im, test8Shape = get_face_matrix(files_dir, 'anderson', 100, 110)

# Group all training images from each person to a row of images, 
# Each row represents each person 
trainIm = vstack((train1Im,train2Im, train3Im, train4Im, train5Im, train6Im, train7Im, train8Im))
valIm = vstack((val1Im, val2Im, val3Im, val4Im, val5Im, val6Im, val7Im, val8Im))
testIm = vstack((test1Im, test2Im, test3Im, test4Im, test5Im, test6Im, test7Im, test8Im))

print trainIm.shape 
print valIm.shape
print testIm.shape 


#print trainIm.shape 

# For each image, make it's intensity be a value between 0 to 1 
for i in range(trainIm.shape[0]):
    trainIm[i,:] = trainIm[i,:]/255.0

# Calculate the average face and eigenfaces 
# Calculate the Principal Component Analysis 
eigenFace, S, averageFace = pca(trainIm) 
    # Get:
    # i) Principal Components Matrix
    # ii) Variance
    # iii) Mean 

# Save and Display the average face 
averageFaceShaped = averageFace.reshape(train1Shape[0], train1Shape[1])
gray()
sci.misc.imsave("averageFace.png", averageFaceShaped)
imshow(averageFaceShaped)

eigenFace = eigenFace[1:]

# Save and Display the top 25 principal components 
displaySavComps(eigenFace, train1Shape, 25) 
