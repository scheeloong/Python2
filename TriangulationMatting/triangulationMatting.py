# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:10:03 2015

@author: Soon Chee Loong
"""

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

def outputImages (compA, compB, backA, backB, newBack, saveDir, name, smaller): 
    
    # Image 1: Flower
    # ImgFCA: FlowerWithBackgroundA (Composite A)
    # ImgFCV: FlowerWithBackgroundB (Composite B)
    # ImgFBA: FlowerBackgroundA 
    # ImgFBB: FlowerBackgroundB 
    
    ImgWC = imread(newBack)
    ImgFCA = imread(compA)
    ImgFCB = imread(compB)
    ImgFBA = imread(backA)
    ImgFBB = imread(backB)

    # Temporary for rescaling images the first time of run    
    if smaller == True: 
        while ImgWC.shape[0]>960:
            ImgWC = imresize(ImgWC, .5)
            ImgFCA = imresize(ImgFCA, .5)
            ImgFCB = imresize(ImgFCB, .5)
            ImgFBA = imresize(ImgFBA, .5)
            ImgFBB = imresize(ImgFBB, .5)

        ImgWC = imresize(ImgWC, [480,320])    
        ImgFCA = imresize(ImgFCA, [480,320])    
        ImgFCB = imresize(ImgFCB, [480,320])    
        ImgFBA = imresize(ImgFBA, [480,320])    
        ImgFBB = imresize(ImgFBB, [480,320])    
        imsave(saveDir + 'background1.png',ImgFBA)
        imsave(saveDir + 'background2.png',ImgFBB)
        imsave(saveDir + 'composite1.png',ImgFCA)
        imsave(saveDir + 'composite2.png',ImgFCB)
        imsave(saveDir + 'newbackground.png',ImgWC)

        

    # Normalize all images by 255 to be in range 0 to 1 for all 3 color channels 
    ImgFCA = double(ImgFCA) / 255.0 # Convert double and resize to [0,1]
    ImgFCB = double(ImgFCB) / 255.0 # Convert double and resize to [0,1] 
    ImgFBA = double(ImgFBA) / 255.0 # Convert double and resize to [0,1]
    ImgFBB = double(ImgFBB) / 255.0 # Convert double and resize to [0,1]
    ImgWC = double(ImgWC) / 255.0 # Convert double and resize to [0,1]
    #imshow(ImgFCA)
    

    # To know how to save to see the output 
    '''
    imsave( + 'ImgFCA.png',ImgFCA)
    imsave("./outputImages/" + 'ImgFCB.png',ImgFCB)
    imsave("./outputImages/" + 'ImgFBA.png',ImgFBA)
    imsave("./outputImages/" + 'ImgFBB.png',ImgFBB)
    '''
    # Define a 6x4 matrix for A 
    A = np.zeros([ImgFCA.shape[0], ImgFCA.shape[1],6,4]) 
    # Define a 4x1 matrix for A 
    X = np.zeros([ImgFCA.shape[0], ImgFCA.shape[1], 4, 1])
    # Define a 6x1 matrix for B
    B = np.zeros([ImgFCA.shape[0], ImgFCA.shape[1], 6, 1])
    
    # Has RGB Component 
    OutputFlowerForeground =  np.zeros([ImgFCA.shape[0], ImgFCA.shape[1],3])
    OutputFlowerAlpha =  np.zeros([ImgFCA.shape[0], ImgFCA.shape[1]])
    OutputFlowerWindow =  np.zeros([ImgFCA.shape[0], ImgFCA.shape[1],3])
    for i in range(ImgFCA.shape[0]):
        for j in range(ImgFCA.shape[1]):
            # Insert identities 
            A[i,j,0,0] = 1
            A[i,j,1,1] = 1
            A[i,j,2,2] = 1
            A[i,j,3,0] = 1
            A[i,j,4,1] = 1
            A[i,j,5,2] = 1
            # Insert colors 
            A[i,j,0,3] = -ImgFBA[i,j,0] # Background1 Red component 
            A[i,j,1,3] = -ImgFBA[i,j,1] # Background1 Green component 
            A[i,j,2,3] = -ImgFBA[i,j,2] # Background1 Blue component 
            A[i,j,3,3] = -ImgFBB[i,j,0] # Background2 Red component 
            A[i,j,4,3] = -ImgFBB[i,j,1] # Background2 Green component 
            A[i,j,5,3] = -ImgFBB[i,j,2] # Background2 Blue component 
            B[i,j,0,0] = ImgFCA[i,j,0] - ImgFBA[i,j,0]
            B[i,j,1,0] = ImgFCA[i,j,1] - ImgFBA[i,j,1]
            B[i,j,2,0] = ImgFCA[i,j,2] - ImgFBA[i,j,2]
            B[i,j,3,0] = ImgFCB[i,j,0] - ImgFBB[i,j,0]
            B[i,j,4,0] = ImgFCB[i,j,1] - ImgFBB[i,j,1]
            B[i,j,5,0] = ImgFCB[i,j,2] - ImgFBB[i,j,2]
            # Compute the pseudo inverse for this pixel 
            apseudoInverse = np.linalg.pinv(A[i,j])
            # Compure the foreground and alpha values for this pixel 
            X[i,j, :] =   np.dot(apseudoInverse, B[i,j,:]) 
            OutputFlowerAlpha[i,j] = X[i,j,3,0]
            OutputFlowerAlpha[i,j] = np.clip(OutputFlowerAlpha[i,j],0 , 1)
            OutputFlowerForeground[i,j,0] = X[i,j,0,0] * OutputFlowerAlpha[i,j] 
            OutputFlowerForeground[i,j,0] = np.clip(OutputFlowerForeground[i,j,0], 0, 1)
            OutputFlowerForeground[i,j,1] = X[i,j,1,0] * OutputFlowerAlpha[i,j]
            OutputFlowerForeground[i,j,1] = np.clip(OutputFlowerForeground[i,j,1], 0, 1)
            OutputFlowerForeground[i,j,2] = X[i,j,2,0] * OutputFlowerAlpha[i,j]
            OutputFlowerForeground[i,j,2] = np.clip(OutputFlowerForeground[i,j,2], 0, 1)
    
            OutputFlowerWindow[i,j,0] = OutputFlowerForeground[i,j,0] + (1-OutputFlowerAlpha[i,j])* ImgWC[i,j,0]       
            OutputFlowerWindow[i,j,1] = OutputFlowerForeground[i,j,1] + (1-OutputFlowerAlpha[i,j])* ImgWC[i,j,1]       
            OutputFlowerWindow[i,j,2] = OutputFlowerForeground[i,j,2] + (1-OutputFlowerAlpha[i,j])* ImgWC[i,j,2]       
    
    # make the matrix large enough to handle every single pixel 
    imsave(saveDir + name + 'OutputComposite.png',OutputFlowerWindow)
    imsave(saveDir + name + 'OutputForeground.png',OutputFlowerForeground)
    gray() 
    imsave(saveDir + name + 'OutputAlpha.png',OutputFlowerAlpha)



print "Starting Assignment 4" 
outputImages('leaves-compA.jpg','leaves-compB.jpg', 'leaves-backA.jpg', 
'leaves-backB.jpg', 'window.jpg', './outputImages/' , 'leaves', False)

outputImages('flowers-compA.jpg','flowers-compB.jpg', 'flowers-backA.jpg', 
'flowers-backB.jpg', 'window.jpg', './outputImages/', 'flower', False)

outputImages('composite1.png','composite2.png', 'background1.png', 
'background2.png', 'newbackground.png', './outputImages/', 'newComposite', False)
