import os

###########################################################################
## Handout painting code.
###########################################################################
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
import matplotlib.image as mpimg
import scipy as sci

# Set threshold of printing to not a number 
np.set_printoptions(threshold = np.nan)  

# This function is used at the end of all computations
# to create the image 
# Save image with color 
def colorImSave(filename, array):
    # Resize to 3x current size using nearest neighbour 
    imArray = sci.misc.imresize(array, 3., 'nearest')
    # If black and white, save with color map 
    if (len(imArray.shape) == 2):
        sci.misc.imsave(filename, cm.jet(imArray))
    else: # length is 3 (numRow,numCol, 3)
        # Save it normally as already colored 
        sci.misc.imsave(filename, imArray)

# Used by paintStroke 
def markStroke(mrkd, p0, p1, rad, val):
    # Mark the pixels that will be painted by
    # a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1).
    # These pixels are set to val in the ny x nx double array mrkd.
    # The paintbrush is circular with radius rad>0

    #mrkd is an empty array 
    #p0 is the point of the first pixel
    #p1 is the piont of the ending pixel 
    #rad is the radius of the circular stroke
    #val = 1 always 
    
    sizeIm = mrkd.shape #(width, height, 3)
    sizeIm = sizeIm[0:2]; # (width, height)
    nx = sizeIm[1] # x = width 
    ny = sizeIm[0] # y = height 
    # flatten both points 
    p0 = p0.flatten('F')
    p1 = p1.flatten('F')
    # make sure the radius is at least 1 
    rad = max(rad,1)
    
    # Make Bounding box
    concat = np.vstack([p0,p1]) # stack both points vertically
    # bounding box upper left  
        # Get (minimum value of x's and y's) - rad 
    bb0 = np.floor(np.amin(concat, axis=0))-rad #minimum value vertically - rad
    # bounding box lower right 
        # bounding maximum value of x and y + rad 
    bb1 = np.ceil(np.amax(concat, axis=0))+rad # minimum value vertically + rad
    
    # Check for intersection of bounding box with image.
    intersect = 1
    # If the upper left is outside the image (bottom right of image)
    # or if the bottom right is outside the image (top left of image)
    if ((bb0[0] > nx) or (bb0[1] > ny) or (bb1[0] < 1) or (bb1[1] < 1)):
        # it means no intersection has occurred, so do nothing 
        intersect = 0 # no intersection , do nothing 
    # If there is intersection between bounding box of points and image 
    if intersect:
        # Crop bounding box such that limited to within image size 
        # Make sure x and y has a min. of 1 and max of nx and ny 
        # for both upper left and bottom right points 
        # upper left handling min and max 
        bb0 = np.amax(np.vstack([np.array([bb0[0], 1]), np.array([bb0[1],1])]), axis=1)
        bb0 = np.amin(np.vstack([np.array([bb0[0], nx]), np.array([bb0[1],ny])]), axis=1)
        # bottom right handling min and max
        bb1 = np.amax(np.vstack([np.array([bb1[0], 1]), np.array([bb1[1],1])]), axis=1)
        bb1 = np.amin(np.vstack([np.array([bb1[0], nx]), np.array([bb1[1],ny])]), axis=1)
        
        # Compute distance d(j,i) to segment in bounding box
        tmp = bb1 - bb0 + 1 # calculate distance (bottom right - upperleft) + [1,1]
        #invert x and y to get the size 
        szBB = [tmp[1], tmp[0]] # size of bounding box is distance for y and x
        # rewrite the absolute points p0 and p1 
        # as points q0, q1 with respect to bounding box's origin
        q0 = p0 - bb0 + 1
        q1 = p1 - bb0 + 1
        # calculate the distance between points within the bounding box itself 
        t = q1 - q0
        nrmt = np.linalg.norm(t) #normalize t as a scalar 
                        # by taking the sqrt of all the element'squared 
        # create a meshgride of from the size of the bounding box 
        [x,y] = np.meshgrid(np.array([i+1 for i in range(int(szBB[1]))]), np.array([i+1 for i in range(int(szBB[0]))]))
        # create a new array with the size of the bounding box 
        d = np.zeros(szBB)
        # fill the new array with infinity values 
        d.fill(float("inf"))
        
        # if nrmt == 0 => the values were all 0 to begin with 
        # this means q1 = q0, so the start and end point is the same
        if nrmt == 0:
            # Use distance to point q0 
            # since q0 = q1 
            # Get the distance from each point in bounding box to the single pixel 
            d = np.sqrt( (x - q0[0])**2 +(y - q0[1])**2)
            # check if the distance is less than the radius 
            idx = (d <= rad)

        # if qo != q1, means the start and end points are different 
        else:
            # Use distance to segment q0, q1
            t = t/nrmt # get value of t in range [0, 1] 
            n = [t[1], -t[0]] # get normal to t 
            # get the distance from each point to extend along the tangent line 
            # and store into an array called tmp 
            tmp = t[0] * (x - q0[0]) + t[1] * (y - q0[1])
            #print tmp 
            # Create another array for each element in tmp
            # to determine if it appears in the final output 
            # Draw only if the final value is less than normalize (radius) but more than 0 
            idx = (tmp >= 0) & (tmp <= nrmt) # a bitmap 
            #print idx 
            # Case 1 : tmp is within [0, nrmt]  
            if np.any(idx.flatten('F')):
                # each of these trues must be calculated in d to extend along normal line 
                d[np.where(idx)] = abs(n[0] * (x[np.where(idx)] - q0[0]) + n[1] * (y[np.where(idx)] - q0[1]))
            idx = (tmp < 0)
            # Case 2: tmp is < 0 
            if np.any(idx.flatten('F')): # use q0 and find sqrt
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q0[0])**2 +(y[np.where(idx)] - q0[1])**2)
            idx = (tmp > nrmt)
            # Case 3: tmp is > nrmt  
            if np.any(idx.flatten('F')): # use q1 and find sqrt 
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q1[0])**2 +(y[np.where(idx)] - q1[1])**2)
            # finally, make sure the value is only within the radius distance 
            #Pixels within crop box to paint have distance <= rad
            idx = (d <= rad)
        
        # Still within intersect's if statement 
        #Mark the pixels
        # For pixels which are within radius distance 
        if np.any(idx.flatten('F')): # flatten using column major order 
            # Convert the index to be an index for a single long array 
            # for all indices which passes the within radius distance 
            xy = (bb0[1]-1+y[np.where(idx)] + sizeIm[0] * (bb0[0]+x[np.where(idx)]-2)).astype(int)
            sz = mrkd.shape
            # make entire mrkd matrix into a long array 
            m = mrkd.flatten('F') # flatten using column major order 
            #mark that index in the long array to be val 
            m[xy-1] = val #mark it all with value val passed in 
            # reshape long array into matrices 
            mrkd = m.reshape(mrkd.shape[0], mrkd.shape[1], order = 'F')

            '''
            row = 0
            col = 0
            for i in range(len(m)):
                col = i//sz[0]
                mrkd[row][col] = m[i]
                row += 1
                if row >= sz[0]:
                    row = 0
            '''            
    return mrkd

def paintStroke(canvas, x, y, p0, p1, colour, rad):
    # Paint a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1)
    # on the canvas (ny x nx x 3 double array).
    # The stroke has rgb values given by colour (a 3 x 1 vector, with
    # values in [0, 1].  The paintbrush is circular with radius rad>0
    sizeIm = canvas.shape # (width, height, 3)
    sizeIm = sizeIm[0:2] # (width, height) 
    # Mark the pixels that will be painted by paintStroke 
    idx = markStroke(np.zeros(sizeIm), p0, p1, rad, 1) > 0
        
    
    # Paint
    # for all pixels that are marked as true (1)  
    if np.any(idx.flatten('F')):
        # Reshape the canvas 
        canvas = np.reshape(canvas, (np.prod(sizeIm),3), "F")
        # Get value as a single index array 
        xy = y[idx] + sizeIm[0] * (x[idx]-1)
        # change the color at that index to be the colour 
        canvas[xy-1,:] = np.tile(np.transpose(colour[:]), (len(xy), 1))
        # reshape the canvas       
        canvas = np.reshape(canvas, sizeIm + (3,), "F")
        #note: If canvas already had a color at that pixel, 
        # it simply overwrites it 
    return canvas

if __name__ == "__main__":

    for abc in range (2):
        
        # channel to range [0,1].
        if abc == 0:
            imRGB = array(Image.open('orchid.jpg')) # read image 
        else:
            # Read image and convert it to double, and scale each R,G,B
            # channel to range [0,1].
            imRGB = array(Image.open('personal.jpg')) # read image 
        
        imRGB = double(imRGB) / 255.0 # Convert double and 
                                      # Scale each rgb to range [0,1]
        plt.clf() # Clearn the plot
        plt.axis('off') # Turn off plot axis 
        
        sizeIm = imRGB.shape # Get the size of image, (height,width, 3)
        #print sizeIm
        # Get only the dimensions of image and ignore the 3 
        sizeIm = sizeIm[0:2] # (height, width) = (163, 197) 
        #print sizeIm
        # Set radius of paint brush and half length of drawn lines
        rad = 3 # radius of paint brush is 3
        halfLen = 10 # half the length of drawn lines is 10
                     # => length of drawn lines is 20 
        
        # Set up x, y coordinate images, and canvas.
        
        # let xParameter be the size of Image from 1 to width (cols)
        # let yParameter be the size of image from 1 to height (rows)
        #take the mesh grid
        #---------------------------------------------------------
        # if colDim = 3, rowDim  = 4 
        # => Print out x's rows rowDim times 
        # => Print out y's cols colDim times 
        #[x1, y1] = np.meshgrid ([1, 2, 3], [4, 5, 6,7])
        #print x1
        #[[1 2 3]
        # [1 2 3]
        # [1 2 3]
        # [1 2 3]]
        #print y1 
        #[[4 4 4]
        # [5 5 5]
        # [6 6 6]
        # [7 7 7]]
        #---------------------------------------------------------
        [x, y] = np.meshgrid(np.array([i+1 for i in range(int(sizeIm[1]))]), 
                            np.array([i+1 for i in range(int(sizeIm[0]))]))
    
        # Create the array large enough for the canvas 
                           # 163,     197,     3 
        canvas = np.zeros((sizeIm[0],sizeIm[1], 3))
        # Fill entire canvas with -1 which marks unpainted pixel,
        # -1 cause it is out of range 
        canvas.fill(-1) 
        # Random number seed
        np.random.seed(29645) # Since seed is constant, 
                              # will always output same image. 
        # Orientation of paint brush strokes
        # its basically, theta = 2*pi*aRandomNumber 
        theta = 2 * pi * np.random.rand(1,1)[0][0]
            #print np.random.rand(1,1) # create a random matrix of size 1x1 
            # [0][0] => Get that element in the size 1x1 area 
        
        # Set vector from center to one end of the stroke.
        # Get point of other end of line from origin depending on theta
        # note: It has radius 1 
        delta = np.array([cos(theta), sin(theta)])
           
           
        time.time() # Number of ticks since 12:00am, January 1, 1970
        time.clock()
            # Returns the current CPU time as a floating-point number of seconds. 
            # To measure computational costs of different approaches, 
            # the value of time.clock is more useful than that of time.time().
    
        # Make 500 strokes 
        k = 0 # Part1 changes: added 
        #for k in range(500): #Part1 changes: removed 
        while np.any((canvas == -1).flatten('F')): # Part1 changes: added 
            # finding a negative pixel
            # Randomly select stroke center
            cntr = np.floor(np.random.rand(2,1).flatten() * np.array([sizeIm[1], sizeIm[0]])) + 1        
                # get 2 random numbers between [0,1]
                #and make into 1D array 
                # multiply with image dimensions to get a random point in image 
                # + 1 to start at index 1 instead of 0
            # handle edge case for values more than image dimensions 
            cntr = np.amin(np.vstack((cntr, np.array([sizeIm[1], sizeIm[0]]))), axis=0)
                # vstack => Create a vertical stack of both arrays 
                # the array is the randon position on first row 
                # and actual image dimension on 2nd row
                # make sure to get the minimum of both, in case where random value
                # is larger than image dimension, make sure to get 
                # end of image dimension as point 
                # amin => returns minimum of an array for axis 0 (along cols) 
                                                        # axis 1 (along rows)
            
            # Grab colour from image at center position of the stroke.
            colour = np.reshape(imRGB[cntr[1]-1, cntr[0]-1, :],(3,1))
                # Grab color from center of pixel stroke as 1 row 
                # note: Handle indexing by deducting 1 
                # Reshape it into 1 column of colours with 3 rows 
            
            # Add the stroke to the canvas
            nx, ny = (sizeIm[1], sizeIm[0]) # get size of image 
            length1, length2 = (halfLen, halfLen)  #initialize both lengths to halfLen      
            #finally paint the canvas         
            canvas = paintStroke(canvas, x, y, cntr - delta * length2, cntr + delta * length1, colour, rad)
            #print imRGB[cntr[1]-1, cntr[0]-1, :], canvas[cntr[1]-1, cntr[0]-1, :]
            print 'stroke', k #print the stroke's iteration out 
            k = k+1# Part1 changes: added  
        print "done!"
        time.time() # get current time 
        
        canvas[canvas < 0] = 0.0 
        # clear the plot, turn off axis and show the canvas 
        plt.clf()
        plt.axis('off')
        # show the canvas that was painted 
        plt.imshow(canvas)
        # wait for 3 seconds on plot before doing anything 
        plt.pause(3)
    
        if abc == 0: 
            # Save the image array, canvas as 'output.png' with color 
            colorImSave('output.png', canvas)
        else:
            colorImSave('outputSpecial.png', canvas)
