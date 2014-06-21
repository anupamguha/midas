'''
Created on Mar 9, 2013
Module for geometric operations
@author: Gregory Kramida
'''
import numpy as np
import math

rmat90 = np.array([[0, -1],
                   [1, 0]],
                  dtype='float32')


def within_bounds(point,bounds):
    return point[0] > 0 and point[1] > 0 and bounds[0] > point[0] and bounds[1] > point[1]

def signed_angle(vec1, vec2):
    '''
    Finds signed angle between two 2D vectors
    @param vec1: the first y, x vector
    @param vec2: the second y, x vector
    '''
    perp_dot_product = vec1[1]*vec2[0] - vec1[0]*vec2[1]
    return math.atan2(perp_dot_product,vec1.dot(vec2))

def integer_distance(p1,p2): 
    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    return int(np.round(math.sqrt(dx*dx + dy*dy)))

def pixel_distance(p1,p2):
    return max(abs(p2[0]-p1[0]),abs(p2[1]-p1[1]))

def distance(p1,p2):
    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    return math.sqrt(float(dx*dx + dy*dy))

def mid_point(p1,p2):
    return (p1[0] + (p2[0]-p1[0])/2, p1[1]+(p2[1]-p1[1])/2)

def mid_point_list(p1,p2):
    return [p1[0] + (p2[0]-p1[0])/2, p1[1]+(p2[1]-p1[1])/2]

def segment_length(seg):
    return distance(seg[0],seg[1])

def segment_midpoint(seg):
    return mid_point(seg[0],seg[1])

def random_BGR_color():
    iColor = np.random.randint(0, 0xFFFFFF)
    return (iColor & 0xff,(iColor>>8) & 0xff, (iColor>>16) & 0xff)

def normal(pt1,pt2):
    vec = pt2-pt1
    rvec = rmat90.dot(vec)
    veclen=distance(pt1,pt2)
    if(veclen == 0):
        return np.array([0,0],dtype='float32')
    return rvec/veclen

def norm(vec):
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])

def rotate90(vec):
    return rmat90.dot(vec)
def sign(val):
    return math.copysign(1,val)

def line_circle_intersection(p1,p2,center,radius):
    cx = center[1]
    cy = center[0]
    x1 = p1[1]
    y1 = p1[0]
    x2 = p2[1]
    y2 = p2[0]
    
    dx = x2-x1
    dy = y2-y1
    
    fx = cx-x1
    fy = cy-y1
    
    dr_sq = dx*dx+dy*dy
    temp = (dx*fy-dy*fx)
    discr = radius*radius*(dr_sq)-temp*temp
    negB = dx*fx+dy*fy
    if(discr < 0):
        return []
    elif(discr == 0):
        t = negB/dr_sq
        y=y1 + t*dy
        x=x1 + t*dx
        return [[y,x]]
    else:
        discr_root = math.sqrt(discr)
        t = (negB - discr_root)/dr_sq
        y=y1 + t*dy
        x=x1 + t*dx
        i1 = [y,x]
        t = (negB + discr_root)/dr_sq
        y=y1 + t*dy
        x=x1 + t*dx
        i2 = [y,x]
        return [i1,i2]

def segment_normal(segment):
    '''
    Calculates the counter-clockwise normal to the segment
    @param segment: an indexable set of two points
    @return: a counter-clockwise normal to the segment
    '''
    p1 = segment[0]
    p2 = segment[1]
    return normal(p1,p2)

def find_line_circle_intersections(pt1,pt2,center,radius):
    x1=pt1[1]
    x2=pt2[1]
    y1=pt1[0]
    y2=pt2[0]
    cx = center[1]
    cy = center[0]
    
    dx=x2-x1
    dy=y2-y1
    
    a = dx*dx + dy*dy
    if(a < 1e-10):
        #no real solutions - line goes past the circle
        return []
    b = 2 * (dx * (x1-cx) + dy * (y1-cy))
    c = (x1 - cx) * (x1 - cx) + (y1 - cy) * (y1-cy) - radius*radius
    
    det = b*b - 4 *a*c #from quadratic formula
    if(det < 0):
        #no real solutions - line goes past the circle
        return []
    elif det == 0:
        #one solution - line is touching the circle
        t = -b / (2*a)
        return np.array([[y1+t*dy, x1+t*dx]])
    else:
        t = (-b + math.sqrt(det)) / (2 * a)#pos. sq. root solution
        int_one = [y1+t*dy,x1+t*dx]
        t = (-b - math.sqrt(det)) / (2 * a)#neg. sq. root solution
        int_two = [y1+t*dy,x1+t*dx]
        #two solutions - line is crossing the circle
        return 

def distance_along_segment(seg,pt):
    p1 = seg[0]
    p2 = seg[1]
    d1 = distance(pt,p1)
    d2 = distance(pt,p2)
    if(d1+d2 == 0.):
        return 0.5
    
    return(d1 / (d1+d2))

def interpolate(pt1,pt2,frac):
    vec = pt2-pt1
    return pt1 + vec * frac

def is_point_between(target,pt1,pt2, threshold = 0.01):
    '''
    @summary: finds out whether the target point is between the two other points or not, i.e. on the segment formed by pt1 and pt2 
    @param target: the target point 
    @param pt1: the first point that composes the segment to check
    @param pt2: the second point that composes the segment to check
    @param threshold: the threshold for distance comparison
    '''
    y1 = pt1[0]; x1 = pt1[1]
    y2 = pt2[0]; x2 = pt2[1]
    
    xPt = target[1];
    yPt = target[0];

    xD1 = xPt - x1;
    xD2 = xPt - x2;
    yD1 = yPt - y1;
    yD2 = yPt - y2;

    sumDist = math.sqrt(float(xD1*xD1 + yD1*yD1)) + math.sqrt(float(xD2*xD2 + yD2*yD2));
    dst = distance(pt1,pt2)
    return abs(dst - sumDist) > threshold

def is_point_on_segment(pt, seg, threshold = 0.01):
    return is_point_between(pt,seg[0],seg[1],threshold)

def detect_consecutive_intersections(ordered_segments):
    
    segments = np.empty(((len(ordered_segments)-1),2,2,2),ordered_segments.dtype)
    
    for ixSrc in xrange(0,len(ordered_segments)-1):
        segments[ixSrc,0] = ordered_segments[ixSrc]
        segments[ixSrc,1] = ordered_segments[ixSrc+1]
    
    return detect_intersections(segments)

def find_consecutive_intersections(ordered_segments):
    
    segments = np.empty(((len(ordered_segments)-1),2,2,2),ordered_segments.dtype)
    
    for ixSrc in xrange(0,len(ordered_segments)-1):
        segments[ixSrc,0] = ordered_segments[ixSrc]
        segments[ixSrc,1] = ordered_segments[ixSrc+1]
    
    return find_intersections(segments)

def detect_intersections(segments):
    transp = np.transpose(segments.reshape((-1)))
    y1 = transp[0];x1 = transp[1];
    y2 = transp[2];x2 = transp[3];
    
    y3 = transp[4];x3 = transp[5];
    y4 = transp[6];x4 = transp[7];
    
    xD2 = x4-x3;
    yD3 = y1-y3;
    yD2 = y4-y3;
    xD3 = x1-x3;
    xD1 = x2-x1;
    yD1 = y2-y1;
    
    len1 = math.sqrt(float(xD1*xD1 + yD1*yD1));
    len2 = math.sqrt(float(xD2*xD2 + yD2*yD2));
    
    dot = (xD1*xD2+yD1*yD2);
    deg = dot/(len1 + len2);
    
    if (abs(deg) == 1.0):
        return False
    
    
    div = yD2*xD1 - xD2*yD1;
    
    #both horizontal, cannot intersect
    if div == 0:
        return False
    
    #compute offset factor from the first point to the second
    ua = (xD2*yD3 -yD2*xD3) / div;

    x5 = x1 + ua*xD1;
    y5 = y1 + ua*yD1;

    xD1 = x5 - x1;
    xD2 = x5 - x2;
    yD1 = y5 - y1;
    yD2 = y5 - y2;

    sumDist1 = math.sqrt(float(xD1*xD1 + yD1*yD1)) + math.sqrt(float(xD2*xD2 + yD2*yD2));

    xD1 = x5 - x3;
    xD2 = x5 - x4;
    yD1 = y5 - y3;
    yD2 = y5 - y4;

    sumDist2 = math.sqrt(float(xD1*xD1 + yD1*yD1)) + math.sqrt(float(xD2*xD2 + yD2*yD2));

    if(abs(len1 - sumDist1)>0.01 or abs(len2 - sumDist2)>0.01):
        return False

    return True

def find_intersections(segments):
    
    transp = np.transpose(segments.reshape((-1)))
    
    y1 = transp[0];x1 = transp[1];
    y2 = transp[2];x2 = transp[3];
    
    y3 = transp[4];x3 = transp[5];
    y4 = transp[6];x4 = transp[7];
    
    x4_x3 = x4-x3;
    y1_y3 = y1-y3;
    y4_y3 = y4-y3;
    x1_x3 = x1-x3;
    x2_x1 = x2-x1;
    y2_y1 = y2-y1;
    
    numeratorA = x4_x3 * y1_y3 - y4_y3 * x1_x3
    numeratorB = x2_x1 * y1_y3 - y2_y1 * x1_x3
    denominator = y4_y3 * x2_x1 - x4_x3 * y2_y1
    
    uA = numeratorA / denominator
    uB = numeratorB / denominator
    
    xi = x1 + x2_x1 * uA
    yi = y1 + y2_y1 * uB
    
    
    intersections = np.transpose(np.array([yi,xi]))
    #areParallel = denominator == 0
    epsilon = 1e-1
    intersect = (uA > epsilon) and ((uA - 1.0) < epsilon) and (uB >= epsilon) \
    and ((uB - 1.0) < epsilon) and (denominator != 0)
    #coincide = (numeratorA == 0) and (numeratorB == 0) and areParallel
        
    return intersections, intersect#, areParallel, coincide
    
