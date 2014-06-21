'''
Created on Jul 28, 2013

@author: Gregory Kramida
'''
import numpy as np
import cv2

def find_intersections(bw_raster):
    '''
    for a thresholded B/W image,
    finds locations of all 8-connected
    intersections, i.e. any pixel surrounded by more than 3 pixels
    '''
    intersectionKernel = np.array([[1, 1,1],
                                   [1,10,1],
                                   [1, 1,1]],dtype='uint8')
    connectivityImg = cv2.filter2D(bw_raster / 255,-1,intersectionKernel, borderType = cv2.BORDER_CONSTANT)
    intersectionImg = connectivityImg > 12
    return np.array(np.nonzero(intersectionImg)).T

def filter_out_corners(bw_raster):
    '''
    filters out the central pixel 
    for the following cases:
     
    010 010 000 000
    110 011 110 011
    000 000 010 010
    '''
    filter2D = np.array([[0,1,0],
                         [2,0,2],
                         [0,1,0]])
    binaryCopy = bw_raster / 255
    cornerResult = cv2.filter2D(binaryCopy, -1, filter2D, borderType = cv2.BORDER_CONSTANT)
    copy = np.copy(bw_raster)
    copy[cornerResult == 3] = 0
    return copy

def remove_points(points,bw_raster):
    for point in points:
        bw_raster[point[0],point[1]] = 0

def find_endpoints(bw_raster):
    '''
    finds chain/edge endpoints for a B/W edge raster that is thinned, 8-connected,
    and where edges don't have intersections/branches/protrusions
    @param bw_raseter: the B/W edge raster that is thinned, 8-connected, and intersectionless
    @return: a numpy array of endpoints, where each row is a 2-d point coordinate
    '''
    endpointValues = [259,3,5,7,
                      9,13,17,25,
                      33,49,65,97,
                      129,193,257,385]
    endptKernel = np.array([[  2, 4,  8],
                            [256, 1, 16],
                            [128,64, 32]],np.float32)
    endptVariations = cv2.filter2D(bw_raster.astype(np.bool).astype(np.uint16),-1,endptKernel,borderType = cv2.BORDER_CONSTANT)
    endptMap = endptVariations == 1 #single-pixel edges, usually pitch black
    for endptValue in endpointValues:
        endptMap = endptMap | (endptVariations == endptValue)
    endpoints = np.array(np.nonzero(endptMap)).T
    return endpoints

def find_protrusions(bw_raster):
    '''
    Finds all endpoints that are single-pixel protrusions for B/W edges
    @param bw_raster: the thinned, 8-connected B/W raster with possible intersections
    @return: a numpy array of protrusions, as well as a numpy array of filtered enpoints without the protrusions 
    '''
    endpoints = find_endpoints(bw_raster)
    neighbor_filter = np.array([[1,1,1],
                                [1,0,1],
                                [1,1,1]],dtype='uint8')
    protrusions = []
    filtered_endpoints = []
    height = bw_raster.shape[0]
    width = bw_raster.shape[1]    
    for endpoint in endpoints:
        (endpoint_y,endpoint_x) = tuple(endpoint)
        is_protrusion = False
        for i in range(max(endpoint_y-1,0),min(endpoint_y+2,height)):
            for j in range(max(endpoint_x-1,0),min(endpoint_x+2,width)):
                if(bw_raster[i,j]==1) and (is_protrusion == 0):
                    result = cv2.bitwise_and(bw_raster[max(i-1,0):min(i+1,height),
                                                       max(j-1,0):min(j+1,width)], 
                                             neighbor_filter)
                    if(np.count_nonzero(result) > 2):
                        bw_raster[tuple(endpoint)]=0
                        is_protrusion = True;
        if not is_protrusion:
            filtered_endpoints.append(endpoint)
        else:
            protrusions.append(endpoint)
    return np.array(protrusions), np.array(filtered_endpoints)