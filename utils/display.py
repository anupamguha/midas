'''
Created on Mar 8, 2013
File for cv2 display procedures 
@author: Algomorph
'''
import cv2
import numpy as np
import matplotlib.cm as cm
from utils.helpers import isnone

arrow = np.array([[0,50],
                  [-20,30],
                  [-10,30],
                  [-10,0],
                  [10,0],
                  [10,30],
                  [20,30]]);

def draw_arrow(raster,position,rotation,color):
    transformedArrow = (position + arrow.dot(rotation)).astype('int32')
    shp = transformedArrow.shape
    cv2.fillPoly(raster, np.fliplr(transformedArrow.astype(np.int32)).reshape((1,shp[0],shp[1])), color)

def overlay_color(raster1,raser2):
    channels0 = cv2.split(raster1)
    channels1 = cv2.split(raser2)
    merged = []
    for iChan in range(0,3):
        chan0 = channels0[iChan]
        chan1 = channels1[iChan]
        mergedChan = chan0 + chan1
        mergedChan[mergedChan > 255] = 255
        merged.append(mergedChan) 
    return cv2.merge(np.array(merged))

def overlay_gray(raster1,raster2):
    chan0 = raster1 * 1.0
    chan1 = raster2 * 1.0
    mergedChan = chan0 + chan1
    max = mergedChan.max()
    mergedChan = np.uint8(255. * mergedChan / max)
    
    return cv2.cvtColor(mergedChan,cv2.COLOR_BAYER_GR2BGR)#cv2.merge(np.array(merged))#

def random_BGR_color():
    iColor = np.random.randint(0, 0xFFFFFF)
    return (iColor & 0xff,(iColor>>8) & 0xff, (iColor>>16) & 0xff)

def drawContinousEdges(dimensions,edgesPx,color="grade", highlightIndices = None, 
                       saveUnder = None, display = True, saveGreyscale = False):
    windowName = "Edge preview"
    image=np.zeros((dimensions[0],dimensions[1],3),np.uint8)
    for epx in edgesPx:
        draw_point_series(epx,image,color)
    (ix1,ix2,ix3) = (0,1,2)
    if(highlightIndices):
        (ix1,ix2,ix3) = highlightIndices    
        (red,green,blue) = ((0,0,255),(0,255,0),(255,0,0))
        #white = (255,255,255)
        #grey = (128,128,128)
        draw_point_series(edgesPx[ix1],image,red)
        draw_point_series(edgesPx[ix2],image,green)
        draw_point_series(edgesPx[ix3],image,blue)
    if(saveUnder):
        if(saveGreyscale):
            cv2.imwrite(saveUnder,cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
        else:   
            cv2.imwrite(saveUnder,image)
    if(display):
        cv2.imshow(windowName,image)
        ch = 0xFF & cv2.waitKey(0)
        return (ch == 27), image
    return False, image

def draw_point_series(points, img, color = "random", cmap = None):
    if(color == "random"):
        colorArr = np.array([random_BGR_color() for i in xrange(0,len(points)/2)])
    elif(color == "grade"):
        if not cmap:
            cmap = cm.get_cmap("gist_earth_r",len(points))
        halfLen = len(points)/2 + 1
        colorArr = [ (cmap(float(i)/halfLen)[0:3]) for i in xrange(0,halfLen)]
        colorArr2 = [colorArr[i] for i in xrange(halfLen-1,0,-1)]
        colorArr.extend(colorArr2)
        colorArr = np.array(colorArr)
        colorArr *= 255
    elif (type(color) == np.ndarray):
        colorArr = color    
    else:
        colorArr = np.array([color] * len(points))
    for iPt in xrange(0,len(points)):
        point = points[iPt]
        clr = colorArr[iPt]
        x = int(point[0])
        y = int(point[1])
        if(x >= 0 and y >= 0 and x < img.shape[1] and y < img.shape[0]):
            img[y,x] = clr

def draw_pixel(pos,arg):
    '''
    Used in map functions to draw a single pixel
    @param arg: an iterable whose first element is the image and second - the color 
    '''
    image = arg[0]
    color = arg[1]
    image[pos] = color

def draw_quad(vertices,raster, colors = None):
    if(isnone(colors)):
        colors = [(244,21,65),(45,243,245),(255,0,255),(34,21,125)]
    '''
    draws the given quadrilaterals on the given raster
    @param vertices:  four points signifying the quadrilaterals
    @param offset: offset (if any)
    @param raster: the raster image where to draw the quad
    @param colors: colors to draw the sides with. By default: blue, yellow, violet, and burgundy
    '''
    cv2.line(raster,tuple(vertices[0]),tuple(vertices[1]),colors[0])#blue side
    cv2.line(raster,tuple(vertices[1]),tuple(vertices[2]),colors[1])#yellow side
    cv2.line(raster,tuple(vertices[2]),tuple(vertices[3]),colors[2])#violet side
    cv2.line(raster,tuple(vertices[3]),tuple(vertices[0]),colors[3])#burgundy side

def draw_crossbars(crossbars):
    allEnds = crossbars.reshape((-1,2))
    minima = allEnds.min(axis=0)
    maxima = allEnds.max(axis=0)
    endRange = maxima - minima
    endsDisp = allEnds - minima
    disp = np.zeros((endRange[1],endRange[0],3),dtype=np.uint8)
    
    for ixPair in xrange(0,len(endsDisp),3):
        pt1 = endsDisp[ixPair]
        pt2 = endsDisp[ixPair+2]
        cv2.line(disp,tuple(pt1),tuple(pt2),(255,255,255))
    return disp
    
def draw_seg_sample(image, sample_length, sample_breadth, destination_raster,
                    rectified_pts, quad, factor = 8):
    
    maxima = quad.max(axis=0)
    minima = quad.min(axis=0)
    
    ranges = maxima - minima
    size = (max(ranges[1],sample_breadth), max(ranges[0], sample_length))
    
    # source window
    source_window = image[minima[0]:minima[0] + size[0] + 1,
                          minima[1]:minima[1] + size[1] + 1]
    
    # source points translated to the source window coordinates
    # subtract the minima from the quad
    source_window_pts = np.fliplr((quad - [minima] * 4).astype(np.float32))
    offset = factor / 2   
    origin_scaled = cv2.resize(source_window,(source_window.shape[0]*factor,source_window.shape[1]*factor))
    
    source_window_pts *= factor
    source_window_pts = source_window_pts.astype(np.int32)
    
    draw_quad(source_window_pts+offset, origin_scaled)
    
    for point in source_window_pts:
        cv2.circle(origin_scaled, tuple(point+offset),offset, (0,128,244))
    

    destination_scaled = cv2.resize(destination_raster,(destination_raster.shape[1]*factor,destination_raster.shape[0]*factor))
    rectified_pts *= factor
    rectified_pts = rectified_pts.astype(np.int32)
    
    for point in rectified_pts:
        cv2.circle(destination_scaled+offset,tuple(point+offset), offset, (0,128,244))
    
    draw_quad(rectified_pts,  destination_scaled)
    return origin_scaled, destination_scaled

def draw_edge_samples(sampleL,sampleR,disp_scale_factor):
    #resize samples
    disp_sample_left = cv2.resize(sampleL,(0,0),fx=disp_scale_factor,fy=disp_scale_factor)
    disp_sample_right = cv2.resize(sampleR,(0,0),fx=disp_scale_factor,fy=disp_scale_factor)
    
    if(len(sampleL.shape) == 3):
        vertLine = np.empty((max(disp_sample_left.shape[0],disp_sample_right.shape[0]),1,3),dtype='uint8')
        vertLine[:] = (23,156,134)
        diff = vertLine.shape[0] - disp_sample_left.shape[0]
        if(diff > 0):
            disp_sample_left = np.append(disp_sample_left,np.zeros((diff,disp_sample_left.shape[1],3),dtype=np.uint8), axis=0)
                
        disp = np.append(disp_sample_left,vertLine,axis=1)
        
        diff = vertLine.shape[0] - disp_sample_right.shape[0]
        if(diff > 0):
            disp_sample_right = np.append(disp_sample_right,np.zeros((diff,disp_sample_right.shape[0],3),dtype=np.uint8), axis=0)

        disp = np.append(disp,disp_sample_right,axis=1)
    else:
        vertLine = np.empty((max(disp_sample_left.shape[0],disp_sample_right.shape[0]),1),dtype='uint8')
        vertLine[:] = 240
        diff = vertLine.shape[0] - disp_sample_left.shape[0]
        if(diff > 0):
            disp_sample_left = np.append(disp_sample_left,np.zeros((diff,disp_sample_left.shape[1]),dtype=np.uint8), axis=0)
                
        disp = np.append(disp_sample_left,vertLine,axis=1)
        
        diff = vertLine.shape[0] - disp_sample_right.shape[0]
        if(diff > 0):
            disp_sample_right = np.append(disp_sample_right,np.zeros((diff,disp_sample_right.shape[0]),dtype=np.uint8), axis=0)
        disp = np.append(disp,disp_sample_right,axis=1)
        
    return disp
