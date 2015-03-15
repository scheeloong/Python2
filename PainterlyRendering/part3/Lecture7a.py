'''
Module for Canny edge detection
Requirements: 1.scipy.(numpy is also mandatory, but it is assumed to be
                      installed with scipy)
              2. Python Image Library (only for viewing the final image.)
'''
try:
    import Image
except ImportError:
    print 'PIL not found. You cannot view the image'
import os
 
from scipy import *
from scipy.ndimage import *
from scipy.signal import convolve2d as conv

def canny(im, sigma, thresHigh = 50,thresLow = 10):
    '''
        Takes an input image in the range [0, 1] and generate a gradient image
        with edges marked by 1 pixels.
    '''
    imin = im.copy() * 255.0

    # Create the gauss kernel for blurring the input image
    # It will be convolved with the image
    # wsize should be an odd number
    wsize = 5
    #----------------------------------------------------------------
    # 1. Smoothing to eliminate noise 
    #----------------------------------------------------------------
    # note: sigma is used in the gaussian filter 
    gausskernel = gaussFilter(sigma, window = wsize)
    # fx is the filter for vertical gradient
    # fy is the filter for horizontal gradient
    # Please note the vertical direction is positive X
    #----------------------------------------------------------------
    # 2. Find gradients with filters 
    #----------------------------------------------------------------
    fx = createFilter([0,  1, 0,
                       0,  0, 0,
                       0, -1, 0])
    fy = createFilter([ 0, 0, 0,
                       -1, 0, 1,
                        0, 0, 0])

    # valid => Convolve only where signal overlaps completely 
    imout = conv(imin, gausskernel, 'valid')
    # print "imout:", imout.shape
    # convolve the output of gaussian with fx 
    gradxx = conv(imout, fx, 'valid')
    # convolve the output of gaussian with fy 
    gradyy = conv(imout, fy, 'valid')

    # pretty place to stroe the result 
    gradx = np.zeros(im.shape)
    grady = np.zeros(im.shape)
    # pad x is the difference / 2 
    padx = (imin.shape[0] - gradxx.shape[0]) / 2.0
    pady = (imin.shape[1] - gradxx.shape[1]) / 2.0
    # store the values  
    gradx[padx:-padx, pady:-pady] = gradxx
    grady[padx:-padx, pady:-pady] = gradyy
    
    # Net gradient is the square root of sum of square of the horizontal
    # and vertical gradients

    # get the net gradiant 
    grad = hypot(gradx, grady)
    # calculate the direction 
    theta = arctan2(grady, gradx)
    # convert to degrees 
    theta = 180 + (180 / pi) * theta
    # Only significant magnitudes are considered. All others are removed
    # find points where gradiant is < 10     
    xx, yy = where(grad < 10)
    # make these points for theta and gradient 0 
    theta[xx, yy] = 0
    grad[xx, yy] = 0
    
    #----------------------------------------------------------------
    # 3. Non-Maximum Supression 
    #----------------------------------------------------------------
    # The angles are quantized. This is the first step in non-maximum
    # supression. Since, any pixel will have only 4 approach directions.
    # direction 1 : horizontal
    # direction 2 : vertical 
    # direction 3 : (+) diagonal  
    # direction 4 : (-) diagonal

    # if angle is close to 0 , 360 or 180, it is horizontal 
    x0,y0 = where(((theta<22.5)+(theta>157.5)*(theta<202.5)
                   +(theta>337.5)) == True)
    # if theta is close to 45, 225, it is (+) diagonal 
    x45,y45 = where( ((theta>22.5)*(theta<67.5)
                      +(theta>202.5)*(theta<247.5)) == True)
    # if theta is close to 90, 270, it is vertical 
    x90,y90 = where( ((theta>67.5)*(theta<112.5)
                      +(theta>247.5)*(theta<292.5)) == True)
    # if theta is close to 135, 315, it is (-) diagonal 
    x135,y135 = where(((theta>112.5)*(theta<157.5)
                        +(theta>292.5)*(theta<337.5)) == True)

    # update theta 
    theta = theta
    Image.fromarray(theta).convert('L').save('Angle map.jpg')
    # save the angle map 
    #change all the angles at the True locations to respective angles 
    theta[x0,y0] = 0
    theta[x45,y45] = 45
    theta[x90,y90] = 90
    theta[x135,y135] = 135
    # get the shape of theta for x and y 
    x,y = theta.shape       
    # create a new RGB image for these directions 
    temp = Image.new('RGB',(y,x),(255,255,255))
    # for each of these pixels, insert an
    for i in range(x):
        for j in range(y):
            # blue color for horizontal 
            if theta[i,j] == 0:
                temp.putpixel((j,i),(0,0,255))
            # red color for (+) diagonal 
            elif theta[i,j] == 45:
                temp.putpixel((j,i),(255,0,0))
            elif theta[i,j] == 90:
                temp.putpixel((j,i),(255,255,0))
            # 
            elif theta[i,j] == 135: # Changed from 45
                temp.putpixel((j,i),(0,255,0))
    retgrad = grad.copy()
    x,y = retgrad.shape

    # Perform non maximum suspression check 
    for i in range(x):
        for j in range(y):
            if theta[i,j] == 0:
                test = nms_check(grad,i,j,1,0,-1,0)
                if not test:
                    retgrad[i,j] = 0

            elif theta[i,j] == 45:
                test = nms_check(grad,i,j,1,-1,-1,1)
                if not test:
                    retgrad[i,j] = 0

            elif theta[i,j] == 90:
                test = nms_check(grad,i,j,0,1,0,-1)
                if not test:
                    retgrad[i,j] = 0
            elif theta[i,j] == 135:
                test = nms_check(grad,i,j,1,1,-1,-1)
                if not test:
                    retgrad[i,j] = 0
                    
    #----------------------------------------------------------------
    # 4. Double Thresholding 
    #----------------------------------------------------------------
    init_point = stop(retgrad, thresHigh)
    #----------------------------------------------------------------
    # 5. Hysteresis Tracking 
    #----------------------------------------------------------------
    # Hysteresis tracking. Since we know that significant edges are
    # continuous contours, we will exploit the same.
    # thresHigh is used to track the starting point of edges and
    # thresLow is used to track the whole edge till end of the edge.

    while (init_point != -1):
        #Image.fromarray(retgrad).show()
        # print 'next segment at',init_point
        retgrad[init_point[0],init_point[1]] = -1
        p2 = init_point
        p1 = init_point
        p0 = init_point
        p0 = nextNbd(retgrad,p0,p1,p2,thresLow)

        while (p0 != -1):
            #print p0
            p2 = p1
            p1 = p0
            retgrad[p0[0],p0[1]] = -1
            p0 = nextNbd(retgrad,p0,p1,p2,thresLow)

        init_point = stop(retgrad,thresHigh)

    # Finally, convert the image into a binary image
    x,y = where(retgrad == -1)
    retgrad[:,:] = 0
    retgrad[x,y] = 1.0
    return retgrad

def createFilter(rawfilter):
    '''
        This method is used to create an NxN matrix to be used as a filter,
        given a N*N list
    '''
    # calculate the order 
    # order = sqrt(length of the filter)
    order = pow(len(rawfilter), 0.5) # len(rawfilter) = N^2
    order = int(order) # make it an integer 
    # create an array out of the raw filter
    filt_array = array(rawfilter)
    # reshape the array into NxN matrix 
    outfilter = filt_array.reshape((order,order))
    return outfilter

def gaussFilter(sigma, window = 3):
    '''
        This method is used to create a gaussian kernel to be used
        for the blurring purpose. inputs are sigma and the window size
    '''
    # create a kernel of size WxW, W = window 
    kernel = zeros((window,window))
    # divide the window by 2 
    c0 = window // 2 #divide with floor 

    for x in range(window):
        for y in range(window):
            # get the magnitude of the differences 
            r = hypot((x-c0),(y-c0))
            # Calculate the sigma for noise 
            val = (1.0/2*pi*sigma*sigma)*exp(-(r*r)/(2*sigma*sigma))
            kernel[x,y] = val
    return kernel / kernel.sum() #return normalize kernel values 
 
def nms_check(grad, i, j, x1, y1, x2, y2):
    '''
        Method for non maximum supression check. A gradient point is an
        edge only if the gradient magnitude and the slope agree

        for example, consider a horizontal edge. if the angle of gradient
        is 0 degress, it is an edge point only if the value of gradient
        at that point is greater than its top and bottom neighbours.
    '''
    try:
        # if it is more than neighbours, return 1
        if (grad[i,j] > grad[i+x1,j+y1]) and (grad[i,j] > grad[i+x2,j+y2]):
            return 1
        # else, return 0 
        else:
            return 0
    except IndexError:
        return -1
     
def stop(im, thres):
    '''
        This method is used to find the starting point of an edge.
    '''
    # Find location of x and y where image is more than threshold given
    X,Y = where(im > thres)
    try:
        y = Y.min()
    except:
        return -1
    # Return all points that are aboave the threshold 
    X = X.tolist()
    Y = Y.tolist()
    index = Y.index(y)
    x = X[index]
    return [x,y]
   
def nextNbd(im, p0, p1, p2, thres):
    '''
        This method is used to return the next point on the edge.
    '''
    kit = [-1,0,1]
    X,Y = im.shape
    # Will loop through 6 pixels
    # does not include case where:
    # i = -1, j = 1
    # i = 1, j = -1
    # i = 0, j = 0 
    for i in kit:
        for j in kit:
            if (i+j) == 0:
                continue
            x = p0[0]+i
            y = p0[1]+j

            if (x<0) or (y<0) or (x>=X) or (y>=Y):
                continue
            if ([x,y] == p1) or ([x,y] == p2):
                continue
            # Return next point larger than threshold 
            if (im[x,y] > thres): #and (im[i,j] < 256):
                return [x,y]
    return -1

# Apply summer and next year stay 