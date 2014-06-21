'''
Created on Jul 28, 2013

@author: Gregory Kramida
'''
import numpy as np
import cv2
from utils.geom import is_point_between, distance, pixel_distance
from bw.preprocessing import find_endpoints
from bw.line_iter import LineIterator


def chain(bw_raster, endpoints = None, verbose = 0):
    '''
    @summary: Chains all edges in given binary bw_raster. 
    Finds simple chain keypoints for edges in a B/W edge bw_raster 
    that is thinned, 8-connected, and where edges don't have
    intersections/branches/protrusions (have only two endpoints
    or are closed loops wih no endpoints)
    Returns the results as a list of numpy arrays, where each
    array represents all the keypoints of a single unique edge.
    @param bw_raster: the binary bw_raster to chain the edges of 
    '''
    if type(endpoints) == type(None):
        endpoints = find_endpoints(bw_raster)
    
    directions =   [( 0, 1),#e
                    (-1, 1),#se
                    (-1, 0),#s
                    (-1,-1),#sw
                    ( 0,-1),#w
                    ( 1,-1),#nw
                    ( 1, 0),#n
                    ( 1, 1)]#ne

    
    #directionFilter = np.array( [[3,2,1],
    #                             [4,0,0],
    #                             [5,6,7]])
    directionFilter = np.array( [[4,3,2],
                                 [5,0,1],
                                 [6,7,8]])
    binary = bw_raster / 255
    dirImg = cv2.filter2D(binary,-1,directionFilter,borderType = cv2.BORDER_CONSTANT)
    unprocessed = binary.astype(np.bool)
    chain_bank = []
    #loop through all endpoints
    for endptNda in endpoints:
        #set the cursor to the endpoint at first
        curNda = np.copy(endptNda)
        cur = tuple(curNda)
        #if this is a "second" enpoint, skip the thing altogether
        if not unprocessed[cur]:
            continue
        #previous direction is something weird that doesn't share any bits with other directions
        prevDir = 9
        backDir = 0
        keypoints = []
        #if this is the end of an edge (current pixel is black), quit traversing
        ix = 0
        end_reached = False
        while(not end_reached):
            dirVal = dirImg[cur]
            ix += 1
            nextDir = dirVal - backDir - 1
            if(nextDir != prevDir):
                #add a new keypoint if the direction changes
                keypoints.append(cur)
            if dirVal == backDir:
                #add last keypoint regardless of direction change
                if (nextDir == prevDir):
                    keypoints.append(cur)
                end_reached = True
            backDir = 1 + (nextDir + 4) % 8
            unprocessed[cur] = False
            curNda += directions[nextDir]  
            cur = tuple(curNda)
            prevDir = nextDir
        #filter out single-point "chains"
        if(len(keypoints) > 2 or (len(keypoints) == 2 and not keypoints[0] == keypoints[1])):
            chain_bank.append(np.vstack(keypoints))

    #now, attend to the loops
    while(unprocessed.sum() != 0):
        nonBlack = np.transpose(np.nonzero(unprocessed))
        curNda = nonBlack[0]
        cur = tuple(nonBlack[0])
        
        backDir = -1
        foundNeighbor = False
        while(foundNeighbor == False):
            backDir +=1
            if(unprocessed[tuple(curNda + directions[backDir])]):
                foundNeighbor = True
        backDir +=1        
        prevDir = 9
        
        keypoints = []
        #if this is a "second" enpoint, skip the thing alltogether
        #if this is the end of an edge (current pixel is black), quit traversing
        ix = 0
        while(unprocessed[cur]):
            dirVal = dirImg[cur]
            ix += 1
            nextDir = dirVal - backDir - 1
            if(nextDir != prevDir):
                keypoints.append(cur)
            backDir = 1 + (nextDir + 4) % 8
            unprocessed[cur] = False
            curNda += directions[nextDir]   
            cur = tuple(curNda)
            prevDir = nextDir
        if(len(keypoints) != 0): 
            curNda += directions[backDir]
            #append the last point if it's not there yet (was skipped)
            if(not np.array_equal(curNda, keypoints[len(keypoints) - 1])):
                keypoints.append(curNda)
            chain_bank.append(np.vstack(keypoints))

    return chain_bank

    
def find_chain_length(chain):
    '''
    Finds the total length of the edge chain, by summing up the distances between
    centers of the chain's keypoints
    @param chain: a chain of keypoints
    @return: the total length of the chain as a double
    '''
    chain_len = 0.
    #traverse segments
    for ixKeyPoint in xrange(0,len(chain)-1):
        pt1 = chain[ixKeyPoint]
        pt2 = chain[ixKeyPoint+1]
        weight = distance(pt1,pt2)
        chain_len += weight
    return chain_len

def find_chain_segment_lengths(chain):
    lengths = np.empty((len(chain)-1),dtype=np.int32)
    for ixKpt in xrange(0,len(chain)-1):
        pt1 = chain[ixKpt]
        pt2 = chain[ixKpt+1]
        lengths[ixKpt] = pixel_distance(pt1,pt2)
    return lengths 
         
def simplify_chain(chain, verbose = 0):
    ixPt = 1
    simplified = []
    
    curPt = chain[0]
    simplified.append(curPt)
    #traverse the chain
    while(ixPt < len(chain)-1):
        #one hop point, potentially after skipping
        oneHopPt = chain[ixPt]
        #the very next point after the one hop point
        twoHopPt = chain[ixPt+1]
        if(not is_point_between(oneHopPt,curPt,twoHopPt,0.17814)):
            #if point is between twoHopPt and curPt, skip it
            curPt = twoHopPt
            ixPt+=2
        else:
            #otherwise, keep it
            simplified.append(oneHopPt)
            curPt = oneHopPt
            ixPt+=1
            
    #add the last point
    simplified.append(chain[len(chain)-1])
    if(verbose > 0):
        print "Reduction: {0:.3%}".format(1.0 - float(len(simplified))/len(chain))
    return np.array(simplified)

def chain_strength(chain,raster):
    count = 0
    edgeLen = 0.
    for ixKeyPoint in xrange(0,len(chain)-1):
        pt1 = chain[ixKeyPoint]
        pt2 = chain[ixKeyPoint+1]
        weight = distance(pt1,pt2)
        edgeLen += weight
        lIt = LineIterator(pt1,pt2)
        for ixPx in xrange(0, lIt.count):
            pos = lIt.pos()
            count += raster[pos]
            lIt.next()
    count += raster[pt2]
    return count/edgeLen


def chain_map(chain, function, arg):
    for ixPt in xrange(0,len(chain)-1):
        pt1 = chain[ixPt]
        pt2 = chain[ixPt+1]
        lIt = LineIterator(pt1,pt2)
        for ixPx in xrange(0, lIt.count):
            pos = lIt.pos()
            function(pos,arg)
            lIt.next()
    function((pt2[0],pt2[1]),arg)