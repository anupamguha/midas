'''
Created on Mar 27, 2013

@author: algomorph
'''
from core import *
import cv2
import argparse

class GradientTool(AbstractEdgeTool):
    op_name = 'gradient'
    '''
    classdocs
    '''
    def __init__(self,raw_gradients, edges, type, start = -1, end = sys.maxint, verbose = 0):
        '''
        Constructor
        '''
        super(GradientTool,self).__init__(raw_gradients, raw_gradients, edges,start,end,verbose=verbose, loadImages=True, loadEdges=False)
        self.gradType = type
        self.windowsUsed = False
        
    def processScharrChannel(self,channel,delta,ddepth,scale):
        gradX = cv2.Scharr(channel,ddepth,1,0,scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
        gradX *= gradX
        gradY = cv2.Scharr(channel,ddepth,0,1,scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
        gradY *= gradY
        #gradX = cv2.convertScaleAbs(gradX)
        #gradY = cv2.convertScaleAbs(gradY)
        grad = np.sqrt(gradX + gradY)
        #return cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
        return cv2.convertScaleAbs(grad) 
    
    def processSobelChannel(self,channel,delta,ddepth,scale):
        gradX = cv2.Sobel(channel,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
        gradX *= gradX
        gradY = cv2.Sobel(channel,ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
        gradY *= gradY
        grad = np.sqrt(gradX + gradY)
        return cv2.convertScaleAbs(grad) 
    
    def run(self):
        gradTypeChanFuncMap = {
                               "sobel":self.processSobelChannel,
                               "scharr":self.processScharrChannel
                               }
        processChannel = gradTypeChanFuncMap[self.gradType]
        for wi in self.images:
            img = wi.image
            #img = cv2.cvtColor(wi.image,cv2.COLOR_BGR2LAB)
            if len(img.shape) == 3:
                sobelB = processChannel(img[:,:,0], 1, cv2.CV_32F, 1)
                sobelG = processChannel(img[:,:,1], 1, cv2.CV_32F, 1)
                sobelR = processChannel(img[:,:,2], 1, cv2.CV_32F, 1)
                gradient = np.empty((sobelB.shape[0],sobelB.shape[1],3),np.uint8)
                gradient[:,:,0] = sobelB
                gradient[:,:,1] = sobelG
                gradient[:,:,2] = sobelR
            else:
                gradient = processChannel(img,1,cv2.CV_32F,1)
            #gradient = cv2.cvtColor(gradient,cv2.COLOR_LAB2BGR)
            cv2.imwrite(wi.pathToResult,gradient)
        
        
                
parser = argparse.ArgumentParser(description='Label Tool\n A tool for computing various color gradients.')
parser.add_argument("--photos","-p",default="images/photos",metavar="path to original photos")
parser.add_argument("--gradients","-g",default="images/gradients",metavar="path to the gradients")
parser.add_argument("--type","-t",default="sobel",metavar="type of gradient to find")
parser.add_argument("--start","-s",type=int,default=0,metavar="number of the first image to be loaded")
parser.add_argument("--end","-e",type=int,default=sys.maxint,metavar="number of the last image to be loaded")
parser.add_argument("--verbose","-v",type=int,default=0,metavar="verbosity level") 