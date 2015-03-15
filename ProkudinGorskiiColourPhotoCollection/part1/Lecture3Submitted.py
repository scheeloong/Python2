
#Steps: 
# 6. Part B: Handle the larger images, 
#       size down find offset, size up
#       

# 7. Part C: Report!!

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 01:55:09 2015

@author: Soon Chee Loong
CSC320 Winter 2015 Assignment 1 
University of Toronto 
"""

# from Python Imaging Library 
# from PIL import Image
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

inputImage = imread('small.jpg')

print inputImage.shape 

imshow(inputImage)

dimension = inputImage.shape
numRows = dimension[0]
numCols = dimension[1]

# Crop off the side of the image first 

inputImage = inputImage[: , numCols/20: numCols-numCols/20]

#inputImage = inputImage[numRows/20: numRows-numRows/20 , numCols/20: numCols-numCols/20]


# Get the total dimension of this cropped image 
dimension = inputImage.shape
numRows = dimension[0]
numCols = dimension[1]
print inputImage.shape 


# Split this image into 3 dimensions 

# Get the rows and entire columns 
blue = inputImage[0:numRows/3, :]
green = inputImage[numRows/3: 2*numRows/3, : ]
red = inputImage[2*numRows/3: , :]
# resize shapes of green and red to fit blue 
green = green[0:blue.shape[0], :]
red = red[0:blue.shape[0], :]

print blue.shape
print green.shape
print red.shape
outputImage = zeros(blue.shape+(3,)) # Add the rgb components 
print outputImage.shape

outputImage[:, :, 0] = red       # set the red component  
outputImage[:, :, 1] = 0  # set the green component  
outputImage[:, :, 2] = 0  # set the blue component  

imsave('./RedComponentOnly', outputImage)

outputImage[:, :, 0] = 0     # set the red component  
outputImage[:, :, 1] = green  # set the green component  
outputImage[:, :, 2] = 0  # set the blue component  

imsave('./GreenComponentOnly', outputImage)

outputImage[:, :, 0] = 0     # set the red component  
outputImage[:, :, 1] = 0  # set the green component  
outputImage[:, :, 2] = blue  # set the blue component  

imsave('./BlueComponentOnly', outputImage)

outputImage[:, :, 0] = red    # set the red component  
outputImage[:, :, 1] = green  # set the green component  
outputImage[:, :, 2] = blue  # set the blue component 

imshow(outputImage) # show the blue component 
imsave('./ImageWithoutAlignment', outputImage)


# Create variables to store the offset for alignment

#Note: Need 2 for loops to try every possible 
        # Horizontal and vertical alignment 

# To store best possible alignment 

redImage =  outputImage[:, :, 0]
greenImage =  outputImage[:, :, 1]
blueImage =  outputImage[:, :, 2]

difference = redImage - blueImage
differenceImage = zeros(difference.shape+(3,)) # Add the rgb components 
differenceImage [:, :, 0] = difference
differenceImage [:, :, 1] = difference
differenceImage [:, :, 2] = difference

# If perfect alignment, should be close to 0 
imshow(differenceImage) # show difference component 

#--------------------------------------------------------
# Matching using SSD (Sum of Squared Difference) 
#--------------------------------------------------------
# Try to align green and blue first 

totalSumSSD1 = sys.float_info.max # initialize to largest value
offsetXSSD1 = 0 
offsetYSSD1 = 0 

totalSumNCC1 = sys.float_info.min # initialize to smallest value
offsetXNCC1 = 0 
offsetYNCC1 = 0

# Try alignment for alignments from (-10, -10) to (10, 10)
for indexI in range (-10, 11):
    for indexJ in range (-10,11):
        #print "Offset is %d , %d " %(indexI, indexJ)
        # Note: All shifts must be positive 
         
        # With reference to top image
        minRowShift = 0
        maxRowShift = 0
        minColShift = 0
        maxColShift = 0
          
        if indexI <= 0:
            maxRowShift = 0
            minRowShift = indexI *(-1)
        else:
            maxRowShift = indexI
            minRowShift = 0
          
        if indexJ <= 0:
            maxColShift = 0
            minColShift = indexJ *(-1)
        else:
            maxColShift = indexJ
            minColShift = 0
        A = greenImage
        B = blueImage 
        # Dimensions must be the same for both here 
        dimRow = A.shape[0]          
        dimCol = A.shape[1]    
          
        Aoverlap = A[0+minRowShift:dimRow-maxRowShift, 0+minColShift:dimCol-maxColShift]
        Boverlap = B[0+maxRowShift:dimRow-minRowShift, 0+maxColShift:dimCol-minColShift]
          
        # Calculate the SSD
        differenceSSD = Aoverlap - Boverlap 
        differenceSSD = differenceSSD * differenceSSD 
        sumSSD = differenceSSD.sum() 
        if sumSSD < totalSumSSD1:
            totalSumSSD1 = sumSSD
            offsetXSSD1 = indexI
            offsetYSSD1 = indexJ
        
        # Calculate the NCC       
        Ancc = Aoverlap.flatten()
        Bncc = Boverlap.flatten()
        sumNCC =  np.dot(Ancc,Bncc)/(np.linalg.norm(Ancc)*np.linalg.norm(Bncc))
        #print "sumNCC is %d" %(sumNCC)
        if sumNCC >= totalSumNCC1:
            totalSumNCC1 = sumNCC
            offsetXNCC1 = indexI
            offsetYNCC1 = indexJ
            
print "Aligning Green over Blue"
print "finalSumSSD is %d " %(totalSumSSD1)

print "finalOffsetSSD is %d , %d " %(offsetXSSD1, offsetYSSD1)


print "finalSumNCC is %d " %(totalSumNCC1)

print "finalOffsetNCC is %d , %d " %(offsetXNCC1, offsetYNCC1)

totalSumSSD2 = sys.float_info.max # initialize to largest value
offsetXSSD2 = 0 
offsetYSSD2 = 0 

totalSumNCC2 = sys.float_info.min # initialize to smallest value
offsetXNCC2 = 0 
offsetYNCC2 = 0

# Try alignment for alignments from (-10, -10) to (10, 10)
for indexI in range (-10, 11):
    for indexJ in range (-10,11):
        #print "Offset is %d , %d " %(indexI, indexJ)
        # Note: All shifts must be positive 
         
        # With reference to top image
        minRowShift = 0
        maxRowShift = 0
        minColShift = 0
        maxColShift = 0
          
        if indexI <= 0:
            maxRowShift = 0
            minRowShift = indexI *(-1)
        else:
            maxRowShift = indexI
            minRowShift = 0
          
        if indexJ <= 0:
            maxColShift = 0
            minColShift = indexJ *(-1)
        else:
            maxColShift = indexJ
            minColShift = 0
        A = redImage
        B = blueImage 
        # Dimensions must be the same for both here 
        dimRow = A.shape[0]          
        dimCol = A.shape[1]    
          
        Aoverlap = A[0+minRowShift:dimRow-maxRowShift, 0+minColShift:dimCol-maxColShift]
        Boverlap = B[0+maxRowShift:dimRow-minRowShift, 0+maxColShift:dimCol-minColShift]
          
        # Calculate the SSD
        differenceSSD = Aoverlap - Boverlap 
        differenceSSD = differenceSSD * differenceSSD 
        sumSSD = differenceSSD.sum() 
        if sumSSD < totalSumSSD2:
            totalSumSSD2 = sumSSD
            offsetXSSD2 = indexI
            offsetYSSD2 = indexJ
        
        # Calculate the NCC       
        Ancc = Aoverlap.flatten()
        Bncc = Boverlap.flatten()
        sumNCC =  np.dot(Ancc,Bncc)/(np.linalg.norm(Ancc)*np.linalg.norm(Bncc))
        #print "sumNCC is %d" %(sumNCC)
            #Note: It will always output 0 due to print limitation
            # but bottom does not always evaluate to true 
        if sumNCC >= totalSumNCC2:
            totalSumNCC2 = sumNCC
            offsetXNCC2 = indexI
            offsetYNCC2 = indexJ  

print "Aligning Red over Blue"
print "finalSumSSD is %d " %(totalSumSSD2)

print "finalOffsetSSD is %d , %d " %(offsetXSSD2, offsetYSSD2)


print "finalSumNCC is %d " %(totalSumNCC2)

print "finalOffsetNCC is %d , %d " %(offsetXNCC2, offsetYNCC2)


#Assuming will never shift larger than image size 

#------------------------------------------------------------
#Handle for SSD images
offsetXOne = offsetXSSD1
offsetYOne = offsetYSSD1
offsetXTwo = offsetXSSD2
offsetYTwo = offsetYSSD2
#------------------------------------------------------------

minRowShiftA = 0
maxRowShiftA = 0
minColShiftA = 0
maxColShiftA = 0

minRowShiftB = 0
maxRowShiftB = 0
minColShiftB = 0
maxColShiftB = 0

minRowShiftC = 0
maxRowShiftC = 0
minColShiftC = 0
maxColShiftC = 0

B = blueImage
A = greenImage
C = redImage 
shiftRow = 0 # 0 if A is green, 1 if A is red
            #  0 if C is red,  1 if C is green 
offsetXMin = 0
offsetXMax = 0
offsetXLeft = 0
offsetXRight = 0

shiftCol = 0 

dimRow = A.shape[0]          
dimCol = A.shape[1]    

# WIll need to interchange A and C 

# Handle rows 

# If both same sides or 0 
if (offsetXOne * offsetXTwo) >= 0:
    #Always make sure A is offset by more or equal to C
    #Always Assume A is offset by even more 
    if abs(offsetXOne) >= abs(offsetXTwo):
        A = greenImage
        C = redImage 
        shiftRow = 0
        offsetXMin = offsetXTwo
        offsetXMax = offsetXOne
    else:
        A = redImage
        C = greenImage 
        shiftRow = 1
        offsetXMin = offsetXOne
        offsetXMax = offsetXTwo        
    
    # Handle A and C row offsets
    if offsetXMax <= 0:
        maxRowShiftA = 0
        minRowShiftA = abs(offsetXMax)
        maxRowShiftC = abs(offsetXMax) - abs(offsetXMin) 
        minRowShiftC = abs(offsetXMin)            
    else: # the values are > 0
        maxRowShiftA = abs(offsetXMax)
        minRowShiftA = 0
        maxRowShiftC = abs(offsetXMin)
        minRowShiftC = abs(offsetXMax) - abs(offsetXMin) 
    minRowShiftB = maxRowShiftA
    maxRowShiftB = minRowShiftA

# If both opposite sides 
else:
    #Always make sure A is on left (-) side 
    if offsetXOne < offsetXTwo:
        A = greenImage
        C = redImage 
        shiftRow = 0
        offsetXLeft = offsetXOne
        offsetXRight = offsetXTwo
    else:
        A = redImage
        C = greenImage 
        shiftRow = 1
        offsetXLeft = offsetXTwo
        offsetXRight = offsetXOne       
        
    minRowShiftA = abs(offsetXLeft) + offsetXRight
    maxRowShiftA = 0
    minRowShiftC = 0
    maxRowShiftC = abs(offsetXLeft) + offsetXRight
    minRowShiftB = offsetXRight 
    maxRowShiftB = abs(offsetXLeft) 
#------------------------------------------------------------#
#Save values for rows 
#------------------------------------------------------------#
finalShiftRow = shiftRow 
finalMinRowShiftA = minRowShiftA
finalMaxRowShiftA = maxRowShiftA
finalMinRowShiftB = minRowShiftB
finalMaxRowShiftB = maxRowShiftB
finalMinRowShiftC = minRowShiftC
finalMaxRowShiftC = maxRowShiftC
#restore
# A = green
# B = blue
# C = red 
if shiftRow == 1:
            # 0 if A is green, 1 if A is red
            #  0 if C is red,  1 if C is green 
    finalMinRowShiftA = minRowShiftC
    finalMaxRowShiftA = maxRowShiftC
    finalMinRowShiftC = minRowShiftA
    finalMaxRowShiftC = maxRowShiftA
#------------------------------------------------------------#
        
        
#------------------------------------------------------------
#Handle for SSD images
offsetXOne = offsetXSSD1
offsetYOne = offsetYSSD1
offsetXTwo = offsetXSSD2
offsetYTwo = offsetYSSD2

#replace rows with columns 
offsetXOne = offsetYOne
offsetXTwo = offsetYTwo
#------------------------------------------------------------

minRowShiftA = 0
maxRowShiftA = 0
minColShiftA = 0
maxColShiftA = 0

minRowShiftB = 0
maxRowShiftB = 0
minColShiftB = 0
maxColShiftB = 0

minRowShiftC = 0
maxRowShiftC = 0
minColShiftC = 0
maxColShiftC = 0

B = blueImage
A = greenImage
C = redImage 
shiftRow = 0 # 0 if A is green, 1 if A is red
            #  0 if C is red,  1 if C is green 
offsetXMin = 0
offsetXMax = 0
offsetXLeft = 0
offsetXRight = 0

shiftCol = 0 

dimRow = A.shape[0]          
dimCol = A.shape[1]    

# WIll need to interchange A and C 

# Handle columns

# If both same sides or 0 
if (offsetXOne * offsetXTwo) >= 0:
    #Always make sure A is offset by more or equal to C
    #Always Assume A is offset by even more 
    if abs(offsetXOne) >= abs(offsetXTwo):

        A = greenImage
        C = redImage 
        shiftRow = 0
        offsetXMin = offsetXTwo
        offsetXMax = offsetXOne
    else:
        A = redImage
        C = greenImage 
        shiftRow = 1
        offsetXMin = offsetXOne
        offsetXMax = offsetXTwo        
    # Handle A and C row offsets
    if offsetXMax <= 0:
        maxRowShiftA = 0
        minRowShiftA = abs(offsetXMax)
        maxRowShiftC = abs(offsetXMax) - abs(offsetXMin) 
        minRowShiftC = abs(offsetXMin)           
    else: # the values are > 0
        maxRowShiftA = abs(offsetXMax)
        minRowShiftA = 0
        maxRowShiftC = abs(offsetXMin)
        minRowShiftC = abs(offsetXMax) - abs(offsetXMin) 
    minRowShiftB = maxRowShiftA
    maxRowShiftB = minRowShiftA

# If both opposite sides 
else:
    print "IN ELSE"
    #Always make sure A is on left (-) side 
    if offsetXOne < offsetXTwo:
        A = greenImage
        C = redImage 
        shiftRow = 0
        offsetXLeft = offsetXOne
        offsetXRight = offsetXTwo
    else:
        A = redImage
        C = greenImage 
        shiftRow = 1
        offsetXLeft = offsetXTwo
        offsetXRight = offsetXOne       
        
    minRowShiftA = abs(offsetXLeft) + offsetXRight
    maxRowShiftA = 0
    minRowShiftC = 0
    maxRowShiftC = abs(offsetXLeft) + offsetXRight
    minRowShiftB = offsetXRight 
    maxRowShiftB = abs(offsetXLeft)         
        
#------------------------------------------------------------#
#Save values for cols
#------------------------------------------------------------#
finalShiftCol = shiftRow 
finalMinColShiftA = minRowShiftA
finalMaxColShiftA = maxRowShiftA
finalMinColShiftB = minRowShiftB
finalMaxColShiftB = maxRowShiftB
finalMinColShiftC = minRowShiftC
finalMaxColShiftC = maxRowShiftC

#restore
# A = green
# B = blue
# C = red 
if shiftRow == 1:
            # 0 if A is green, 1 if A is red
            #  0 if C is red,  1 if C is green 
    finalMinColShiftA = minRowShiftC
    finalMaxColShiftA = maxRowShiftC
    finalMinColShiftC = minRowShiftA
    finalMaxColShiftC = maxRowShiftA
#------------------------------------------------------------#
# Done getting alignment from SSD
    
Aoverlap = greenImage[0+finalMinRowShiftA:dimRow-finalMaxRowShiftA, 0+finalMinColShiftA:dimCol-finalMaxColShiftA]
Boverlap = blueImage[0+finalMinRowShiftB:dimRow-finalMaxRowShiftB, 0+finalMinColShiftB:dimCol-finalMaxColShiftB]
Coverlap = redImage[0+finalMinRowShiftC:dimRow-finalMaxRowShiftC, 0+finalMinColShiftC:dimCol-finalMaxColShiftC]

print Aoverlap.shape
print Boverlap.shape
print Coverlap.shape

outputImageSSD = zeros(Boverlap.shape+(3,)) # Add the rgb components 
print outputImageSSD.shape

outputImageSSD[:, :, 0] = Coverlap    # set the red component  
outputImageSSD[:, :, 1] = Aoverlap     # set the green component  
outputImageSSD[:, :, 2] = Boverlap  # set the blue component  

imshow(outputImageSSD) # show the blue component 
imsave('./ImageSSD', outputImageSSD)



#--------------------------------------------------------------------------
# HANDLE NCC 
#--------------------------------------------------------------------------
offsetXSSD1 = offsetXNCC1
offsetYSSD1 = offsetYNCC1
offsetXSSD2 = offsetXNCC2
offsetYSSD2 = offsetYNCC2

#------------------------------------------------------------
#Handle for NNC images
offsetXOne = offsetXSSD1
offsetYOne = offsetYSSD1
offsetXTwo = offsetXSSD2
offsetYTwo = offsetYSSD2
#------------------------------------------------------------

minRowShiftA = 0
maxRowShiftA = 0
minColShiftA = 0
maxColShiftA = 0

minRowShiftB = 0
maxRowShiftB = 0
minColShiftB = 0
maxColShiftB = 0

minRowShiftC = 0
maxRowShiftC = 0
minColShiftC = 0
maxColShiftC = 0

B = blueImage
A = greenImage
C = redImage 
shiftRow = 0 # 0 if A is green, 1 if A is red
            #  0 if C is red,  1 if C is green 
offsetXMin = 0
offsetXMax = 0
offsetXLeft = 0
offsetXRight = 0

shiftCol = 0 

dimRow = A.shape[0]          
dimCol = A.shape[1]    

# WIll need to interchange A and C 

# Handle rows 

# If both same sides or 0 
if (offsetXOne * offsetXTwo) >= 0:
    #Always make sure A is offset by more or equal to C
    #Always Assume A is offset by even more 
    if abs(offsetXOne) >= abs(offsetXTwo):
        A = greenImage
        C = redImage 
        shiftRow = 0
        offsetXMin = offsetXTwo
        offsetXMax = offsetXOne
    else:
        A = redImage
        C = greenImage 
        shiftRow = 1
        offsetXMin = offsetXOne
        offsetXMax = offsetXTwo        
    
    # Handle A and C row offsets
    if offsetXMax <= 0:
        maxRowShiftA = 0
        minRowShiftA = offsetXMax * -1
        maxRowShiftC = abs(offsetXMax) - abs(offsetXMin) 
        minRowShiftC = offsetXMin            
    else: # the values are > 0
        maxRowShiftA = offsetXMax
        minRowShiftA = 0
        maxRowShiftC = offsetXMin
        minRowShiftC = abs(offsetXMax) - abs(offsetXMin) 
    minRowShiftB = maxRowShiftA
    maxRowShiftB = minRowShiftA

# If both opposite sides 
else:
    #Always make sure A is on left (-) side 
    if offsetXOne < offsetXTwo:
        A = greenImage
        C = redImage 
        shiftRow = 0
        offsetXLeft = offsetXOne
        offsetXRight = offsetXTwo
    else:
        A = redImage
        C = greenImage 
        shiftRow = 1
        offsetXLeft = offsetXTwo
        offsetXRight = offsetXOne       
        
    minRowShiftA = abs(offsetXLeft) + offsetXRight
    maxRowShiftA = 0
    minRowShiftC = 0
    maxRowShiftC = abs(offsetXLeft) + offsetXRight
    minRowShiftB = offsetXRight 
    maxRowShiftB = abs(offsetXLeft) 
#------------------------------------------------------------#
#Save values for rows 
#------------------------------------------------------------#
finalShiftRow = shiftRow 
finalMinRowShiftA = minRowShiftA
finalMaxRowShiftA = maxRowShiftA
finalMinRowShiftB = minRowShiftB
finalMaxRowShiftB = maxRowShiftB
finalMinRowShiftC = minRowShiftC
finalMaxRowShiftC = maxRowShiftC
#restore
# A = green
# B = blue
# C = red 
if shiftRow == 1:
            # 0 if A is green, 1 if A is red
            #  0 if C is red,  1 if C is green 
    finalMinRowShiftA = minRowShiftC
    finalMaxRowShiftA = maxRowShiftC
    finalMinRowShiftC = minRowShiftA
    finalMaxRowShiftC = maxRowShiftA
#------------------------------------------------------------#
        
        
#------------------------------------------------------------
#Handle for SSD images
offsetXOne = offsetXSSD1
offsetYOne = offsetYSSD1
offsetXTwo = offsetXSSD2
offsetYTwo = offsetYSSD2

#replace rows with columns 
offsetXOne = offsetYOne
offsetXTwo = offsetYTwo
#------------------------------------------------------------

minRowShiftA = 0
maxRowShiftA = 0
minColShiftA = 0
maxColShiftA = 0

minRowShiftB = 0
maxRowShiftB = 0
minColShiftB = 0
maxColShiftB = 0

minRowShiftC = 0
maxRowShiftC = 0
minColShiftC = 0
maxColShiftC = 0

B = blueImage
A = greenImage
C = redImage 
shiftRow = 0 # 0 if A is green, 1 if A is red
            #  0 if C is red,  1 if C is green 
offsetXMin = 0
offsetXMax = 0
offsetXLeft = 0
offsetXRight = 0

shiftCol = 0 

dimRow = A.shape[0]          
dimCol = A.shape[1]    

# WIll need to interchange A and C 

# Handle columns

# If both same sides or 0 
if (offsetXOne * offsetXTwo) >= 0:
    #Always make sure A is offset by more or equal to C
    #Always Assume A is offset by even more 
    if abs(offsetXOne) >= abs(offsetXTwo):
        A = greenImage
        C = redImage 
        shiftRow = 0
        offsetXMin = offsetXTwo
        offsetXMax = offsetXOne
    else:
        A = redImage
        C = greenImage 
        shiftRow = 1
        offsetXMin = offsetXOne
        offsetXMax = offsetXTwo        
    
    # Handle A and C row offsets
    if offsetXMax <= 0:
        maxRowShiftA = 0
        minRowShiftA = abs(offsetXMax)
        maxRowShiftC = abs(offsetXMax) - abs(offsetXMin) 
        minRowShiftC = abs(offsetXMin)       
    else: # the values are > 0
        maxRowShiftA = offsetXMax
        minRowShiftA = 0
        maxRowShiftC = offsetXMin
        minRowShiftC = abs(offsetXMax) - abs(offsetXMin) 
    minRowShiftB = maxRowShiftA
    maxRowShiftB = minRowShiftA

# If both opposite sides 
else:
    print "IN ELSE" 
    #Always make sure A is on left (-) side 
    if offsetXOne < offsetXTwo:
        print "IN ELSE IF"
        A = greenImage
        C = redImage 
        shiftRow = 0
        offsetXLeft = offsetXOne
        offsetXRight = offsetXTwo
    else:
        print "IN ELSE ELSE" 
        A = redImage
        C = greenImage 
        shiftRow = 1
        offsetXLeft = offsetXTwo
        offsetXRight = offsetXOne       
        
    minRowShiftA = abs(offsetXLeft) + offsetXRight
    maxRowShiftA = 0
    minRowShiftC = 0
    maxRowShiftC = abs(offsetXLeft) + offsetXRight
    minRowShiftB = offsetXRight 
    maxRowShiftB = abs(offsetXLeft)          
        
#------------------------------------------------------------#
#Save values for cols
#------------------------------------------------------------#
finalShiftCol = shiftRow 
finalMinColShiftA = minRowShiftA
finalMaxColShiftA = maxRowShiftA
finalMinColShiftB = minRowShiftB
finalMaxColShiftB = maxRowShiftB
finalMinColShiftC = minRowShiftC
finalMaxColShiftC = maxRowShiftC

#restore
# A = green
# B = blue
# C = red 
if shiftRow == 1:
            # 0 if A is green, 1 if A is red
            #  0 if C is red,  1 if C is green 
    finalMinColShiftA = minRowShiftC
    finalMaxColShiftA = maxRowShiftC
    finalMinColShiftC = minRowShiftA
    finalMaxColShiftC = maxRowShiftA
#------------------------------------------------------------#
# Done getting alignment from SSD 
Aoverlap = greenImage[0+finalMinRowShiftA:dimRow-finalMaxRowShiftA, 0+finalMinColShiftA:dimCol-finalMaxColShiftA]
Boverlap = blueImage[0+finalMinRowShiftB:dimRow-finalMaxRowShiftB, 0+finalMinColShiftB:dimCol-finalMaxColShiftB]
Coverlap = redImage[0+finalMinRowShiftC:dimRow-finalMaxRowShiftC, 0+finalMinColShiftC:dimCol-finalMaxColShiftC]

print "TESTTTDEBUG"

print "FinalShiftsA %d, %d, %d, %d" %(finalMinRowShiftA,finalMaxRowShiftA,finalMinColShiftA, finalMaxColShiftA  )
print "FinalShiftsB %d, %d, %d, %d" %(finalMinRowShiftB,finalMaxRowShiftB,finalMinColShiftB, finalMaxColShiftB  )
print "FinalShiftsC %d, %d, %d, %d" %(finalMinRowShiftC,finalMaxRowShiftC,finalMinColShiftC, finalMaxColShiftC  )

print greenImage.shape
print blueImage.shape
print redImage.shape 
print  0+finalMinColShiftA
print dimCol-finalMaxColShiftA

outputImageSSD = zeros(Boverlap.shape+(3,)) # Add the rgb components 
print outputImageSSD.shape

outputImageSSD[:, :, 0] = Coverlap    # set the red component  
outputImageSSD[:, :, 1] = Aoverlap     # set the green component  
outputImageSSD[:, :, 2] = Boverlap  # set the blue component  

imshow(outputImageSSD) # show the blue component 
imsave('./ImageNNC', outputImageSSD)


'''
# Practice with simple Example

A = redImage[0:3, 0:3]
B = blueImage[0:3, 0:3]

A[0,0] = 4
A[0,1] = 5
A[0,2] = 6
A[1,0] = 7
A[1,1] = 8
A[1,2] = 9
A[2,0] = 0
A[2,1] = 0
A[2,2] = 0

B[0,0] = 1
B[0,1] = 2
B[0,2] = 3
B[1,0] = 4
B[1,1] = 5
B[1,2] = 6
B[2,0] = 7
B[2,1] = 8
B[2,2] = 9

# A is on top of B 
# -1, -1 => A moves left and up 
# 1, 1 => A moves right and down 

totalSumSSD = sys.float_info.max # initialize to largest value
offsetXSSD = 0 
offsetYSSD = 0 

totalSumNCC = sys.float_info.min # initialize to smallest value
offsetXNCC = 0 
offsetYNCC = 0

for indexI in range (-1, 2):
    for indexJ in range (-1,2):
          print "Offset is %d , %d " %(indexI, indexJ)
          # Note: All shifts must be positive 
         
          # With reference to top image
          minRowShift = 0
          maxRowShift = 0
          minColShift = 0
          maxColShift = 0
          
          if indexI <=0:
              maxRowShift = 0
              minRowShift = indexI *(-1)
          else:
              maxRowShift = indexI
              minRowShift = 0
          
          if indexJ <=0:
              maxColShift = 0
              minColShift = indexJ *(-1)
          else:
              maxColShift = indexJ
              minColShift = 0
          # Dimensions must be the same for both here 
          dimRow = A.shape[0]          
          dimCol = A.shape[1]
          
          #print "A row col is %d , %d " %(dimArow, dimAcol)
          #print "B row col is %d , %d " %(dimBrow, dimBcol)

          
          Aoverlap = A[0+minRowShift:dimRow-maxRowShift, 0+minColShift:dimCol-maxColShift]
          Boverlap = B[0+maxRowShift:dimRow-minRowShift, 0+maxColShift:dimCol-minColShift]
          print Aoverlap.shape
          print Boverlap.shape
          
          # Calculate the SSD
          differenceSSD = Aoverlap - Boverlap 
          differenceSSD = differenceSSD * differenceSSD 
          sumSSD = differenceSSD.sum() 
          if sumSSD < totalSumSSD:
              totalSumSSD = sumSSD
              offsetXSSD = indexI
              offsetYSSD = indexJ
        
          # Calculate the NCC       
          Ancc = Aoverlap.flatten()
          Bncc = Boverlap.flatten()
          sumNCC =  np.dot(Ancc,Bncc)/(np.linalg.norm(Ancc)*np.linalg.norm(Bncc))
          #print "sumNCC is %d" %(sumNCC)

          if sumNCC >= totalSumNCC:
             totalSumNCC = sumNCC
             offsetXNCC = indexI
             offsetYNCC = indexJ
             
              
          for index1 in range(0,Aoverlap.shape[0]):
              for index2 in range(0,Aoverlap.shape[1]):
                  print "Aoverlap[%d, %d] = %d " %(index1, index2, Aoverlap[index1,index2])
          for index1 in range(0,Aoverlap.shape[0]):
              for index2 in range(0,Aoverlap.shape[1]):
                  print "Boverlap[%d, %d] = %d " %(index1, index2, Boverlap[index1,index2])
'''


