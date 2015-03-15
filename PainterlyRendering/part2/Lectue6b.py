# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 16:23:03 2015

@author: Soon Chee Loong
"""

if __name__ == "__main__":

    print "Here" 
    """        
    print np.random.rand(2,1).flatten()  # generate 2 random numbers 
    print np.array([sizeIm[1], sizeIm[0]])
    print cntr 
    print np.where(canvas == -1)
    """
    
    #TODO: Get points of all unpainted pixels into array 
    #TODO: randomly select one of these points 
    #TODO: Convert this randomly selected point into actual indices 
    lala = np.zeros((2,3, 3)) # create a 3d matrix of 2,3,3 
    lala.fill(-1)
    print lala 
    print 'next' 
    #problem is each color will have 3 different pixels! 
    lala[0][0][0] = -2 
    lala[0][0][1] = -3 
    lala[0][0][2] = -4 
    lala[0][1][0] = -5 
    lala[0][1][1] = -6 
    lala[0][1][2] = -7 
    lala[1][0][0] = -8 
    lala[1][0][1] = -9 
    lala[1][0][2] = -10 
    baba = lala.copy() 
    print baba
    baba[0][0][0] = 10
    print 'caca'
    print lala 
    
    print'caca3'
    k1 = 2
    print k1
    k2 = k1
    print k2 
    k2 = 4
    print k2 
    print k1
    

    #lala[3][5][2] = -2 
    #lala[4][6][2] = -2 
    print 'nextnext1'

    print lala 
    print 'nextnext2'

    print (lala==-2)
    print (np.where(lala==-2))
    de = np.where(lala==-2)
    print len(de[0]) # must be de[0] for num elements 
    """

    for k in range(500):
        index = np.round(np.random.rand(1,1).flatten() * np.array(len(de[0])))      
        print index 
        index = int(index)
        print index 
        width = de[0][index]
        height = de[1][index]
        #print "Kaka"
        print width
        print height 
        

    """
    """
    print 'nextnextnext' 
    lalaFlatten = lala.flatten()
    print lalaFlatten
    print lalaFlatten[lalaFlatten==-2] # this gives all elements satisfying condition 
    a = np.where(lalaFlatten == -2)
    print 'caca' 
    print len(a[0]) # this is correct 
    index = np.floor(np.random.rand(1,1).flatten() * np.array(len(a[0])))      
    print index 
    index = int(index)
    print index 
    print a[0][index]
    lalaFlatten[index] = -3 # temporary as -3 to look for it 
    #randomly select an index in a 
    #print a.shape
    lalaOri = np.reshape(lalaFlatten,(2,3,3))
    print lalaOri 
    babaSizeIm = sizeIm 
    print 'bababa' 
    #print babaSizeIm[canvas == -2]
    print babaSizeIm
    """
    
    
    
    
    
    
    