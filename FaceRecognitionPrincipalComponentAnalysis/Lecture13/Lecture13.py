# -*- coding: utf-8 --
"""
Created on Fri Mar 13 09:21:02 2015

@author: Soon Chee Loong
"""

# Part 1 of Assignment 3 
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
#Readability 
#For this project, you should crop out the images of the faces, 
#convert them to grayscale, and
#resize them to 32x32 before proceeding further.

# Use these faces 
act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',
    'Andrea Anders', 'Ashley Benson',
    'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']

#act = ['Aaron Eckhart'] # Removed, TEMP FOR DEBUGGING 
#------------------------------------------------------------------------------------------
# convert image to grayscale 
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# Have a timeout function to timeout and stop getting picture from a specific source
# if it takes too long or doesn't exist 
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/
    473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

# For every name in act 
for a in act:
    name = a.split()[1].lower() # Get the last name only and make them all lower case 
    i = 0 # each individual person starts from 0 
    # For each line in the faces_subset.txt file 
    for line in open("faces_subset.txt"):
        # Only execute on this line if the name is in this line
        if a in line:
            # Create the filenam by concatenating the strings: 
            # "nameFromActArray" + "indexNumber"+"."+
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            filenamePNG = name+str(i)+'.png' # new name 
                    # Split by spaces and access 4th element starting from 0th 
                    # This element is the entire link, 
                    # Split this link again by using '.' as the delimiter 
                    # Then add the file extension to the name 
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images
            #which take too long to download
            
            # Use timeout() as wrapper to only execute the function up till a certain time. 
            # Set time limit to 30 seconds for this picture. 
            # Retrieve the link and store into a folder called uncropped/ with the 
            # name of filename that was made. 
            timeout(testfile.retrieve, (line.split()[4],
                        "uncropped/"+filename), {}, 30) 
                        # download images but with different extensions

            # If the file was not created due to timeout, continue to next iteration
            # without executing below 
            if not os.path.isfile("uncropped/"+filename):
                continue # continue to next line in file 

            # Open the image that was just downloaded, 
            # Then save as png in a different folder 
            #(jpg, png), read it right away and save as png. 

            try:
                imRGB = array(Image.open("uncropped/"+filename)) # read image 
                sci.misc.imsave("pngImages/"+filenamePNG, imRGB)
            except IOError as e:
                continue # repeat this iteration on next line 
            # Crop and convert image to grayscale, 32x32
            try:
                # Obtain the face part of the image
                # Get the coordinates for the image 
                # The coordinates of the bounding box for a face in the image.
                # The format is x1,y1,x2,y2, where
                # (x1,y1) is the coordinate of the top-left corner of the bounding box and
                # (x2,y2) is that of the bottom-right corner, with (0,0) as the top-left corner of the image.  Assuming the image is represented as a Python Numpy array I, a face in I can be obtained as I[y1:y2, x1:x2].
                # Top Left 
                x1 = int(line.split()[5].split(',')[0])
                y1 = int(line.split()[5].split(',')[1])
                # Bottom right 
                x2 = int(line.split()[5].split(',')[2])
                y2 = int(line.split()[5].split(',')[3])
    
                #Reopen the png image
                #imRGBNew = Image.open("pngImages/"+filenamePNG)).convert('LA')         
                imRGBNew = array(Image.open("pngImages/"+filenamePNG))
                    
                # Cropped out the face 
                #imRGBCropped =  imRGB[y1:y2, x1:x2]
                imRGBCropped =  imRGBNew[y1:y2, x1:x2]
    #
                    
                # Convert to grayscale 
                imGrayCropped = rgb2gray(imRGBCropped)
        

                # Resize to 32X 32 
                while imGrayCropped.shape[0]>64:
                    imGrayCropped = imresize(imGrayCropped, .5)
                imGrayCropped = imresize(imGrayCropped, [32,32])
            except: 
                continue # Save the next image instead 
            
            # Finally, save it into the cropped folder 
            sci.misc.imsave("croppedPngImages/"+filenamePNG, imGrayCropped)
            print filenamePNG
            i += 1
           # if i > 145: 
            #    break # Only need at most 120 pictures for each name 
