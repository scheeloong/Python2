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



imRGB = array(Image.open("tempForConversion/anderson72.jpg")) # read image 
                
# Resize to 32X 32 
while imRGB.shape[0]>64:
    imRGB = imresize(imRGB, .5)
imRGB = imresize(imRGB, [32,32])

sci.misc.imsave("tempForReport/anderson72.png", imRGB)

imRGB = array(Image.open("tempForConversion/eckhart8.jpg")) # read image 
                
# Resize to 32X 32 
while imRGB.shape[0]>64:
    imRGB = imresize(imRGB, .5)
imRGB = imresize(imRGB, [32,32])

sci.misc.imsave("tempForReport/eckhart8.png", imRGB)
          
imRGB = array(Image.open("tempForConversion/eckhart11.jpg")) # read image 
                
# Resize to 32X 32 
while imRGB.shape[0]>64:
    imRGB = imresize(imRGB, .5)
imRGB = imresize(imRGB, [32,32])

sci.misc.imsave("tempForReport/eckhart11.png", imRGB)

imRGB = array(Image.open("tempForConversion/eckhart17.jpg")) # read image 
                
# Resize to 32X 32 
while imRGB.shape[0]>64:
    imRGB = imresize(imRGB, .5)
imRGB = imresize(imRGB, [32,32])

sci.misc.imsave("tempForReport/eckhart17.png", imRGB)

imRGB = array(Image.open("tempForConversion/eckhart21.jpg")) # read image 
                
# Resize to 32X 32 
while imRGB.shape[0]>64:
    imRGB = imresize(imRGB, .5)
imRGB = imresize(imRGB, [32,32])

sci.misc.imsave("tempForReport/eckhart21.png", imRGB)
imRGB = array(Image.open("tempForConversion/eckhart27.jpg")) # read image 
                
# Resize to 32X 32 
while imRGB.shape[0]>64:
    imRGB = imresize(imRGB, .5)
imRGB = imresize(imRGB, [32,32])

sci.misc.imsave("tempForReport/eckhart27.png", imRGB)
imRGB = array(Image.open("tempForConversion/eckhart40.jpg")) # read image 
                
# Resize to 32X 32 
while imRGB.shape[0]>64:
    imRGB = imresize(imRGB, .5)
imRGB = imresize(imRGB, [32,32])

sci.misc.imsave("tempForReport/eckhart40.png", imRGB)
                                        
imRGB = array(Image.open("tempForConversion/eckhart41.jpg")) # read image 
                
# Resize to 32X 32 
while imRGB.shape[0]>64:
    imRGB = imresize(imRGB, .5)
imRGB = imresize(imRGB, [32,32])

sci.misc.imsave("tempForReport/eckhart41.png", imRGB)
          