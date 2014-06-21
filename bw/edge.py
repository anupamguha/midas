'''
Created on Dec 13, 2012

@author: Gregory Kramida & Anupam Guha
'''
import numpy as np
import math
from bw.line_iter import LineIterator
from bw.chain import find_chain_segment_lengths, chain_map, find_chain_length
from utils.geom import normal, segment_normal, segment_midpoint, distance, distance_along_segment,\
    segment_length, mid_point, mid_point_list
import utils.display as di
from label import BELONGING_LABELS
from bw.label import BELONGING_COLOR_BY_LABEL
from utils.helpers import isnone

def calc_keypoint_normal(chain,target_kpt_index,influence_range, seg_lengths = None):
    '''
    Calculates the normal to the chain at the given keypoint.
    The normal is calculated by taking receding influence as moving away from the keypoint.
    One exception is the case when there are only two keypoints. There, the normal is calculated
    directy by taking the normal to the only segment.
    @param chain: a chain of keypoints representing contigous segments
    @param target_kpt_index: index of the target keypoint
    @param influence_range: influence range - i.e. how much of the chain to look at when calculating the normal.
    @param seg_lengths: an optional array of chain's segment lengths (if precomputed)
    @return: the normal at the keypoint in the chain with the specified index.  
    '''
    
    if len(chain) == 2:
        return normal(chain[0],chain[1])
    
    ix_previous = target_kpt_index
    run_length = 0
    nrm = np.zeros(2,dtype=np.float32)
    
    if isnone(seg_lengths):
        seg_lengths = find_chain_segment_lengths(chain)
    
    while (ix_previous > 0 and run_length < influence_range):
        cur_length = seg_lengths[ix_previous-1]
        cur_normal = normal(chain[ix_previous-1],chain[ix_previous])#normal of the hind-seg
        start = float(1 + run_length)
        end = float(1 + run_length + cur_length)
        cur_influence = (start * math.log(start/influence_range) - end * math.log(start/influence_range) - start + end)/influence_range
        nrm += (cur_normal * cur_influence)
        run_length += cur_length
        ix_previous-=1
        
    ix_next = target_kpt_index
    run_length = 0
    
    while (ix_next < len(chain)-1 and run_length < influence_range):
        cur_length = seg_lengths[ix_next]
        cur_normal = normal(chain[ix_next],chain[ix_next+1])#normal of the hind-seg
        start = float(1 + run_length)
        end = float(1 + run_length + cur_length)
        cur_influence = (start * math.log(start/influence_range) - end * math.log(start/influence_range) - start + end)/influence_range
        nrm += (cur_normal * cur_influence)
        run_length += cur_length
        ix_next+=1
    
    vecLen = math.sqrt(nrm[0]*nrm[0] + nrm[1]*nrm[1])
    return nrm / vecLen


class Edge(object):
    def __init__(self,chain):
        '''
        Constructor
        @summary: builds a new edge
        @param chain: a raw opencv contour traced around a b/w edge (sub-output of the cv2.findContours function) 
        '''
        self.chain = chain
        self.length = find_chain_length(self.chain)
        mdpt, mdpt_normal = self.point_along_edge(0.5)
        self.normal_flipped = False
        #flip normal to follow left-ish for the up-clockwise side, right-ish for down-counter-clockwise side
        if(mdpt_normal[1] > 0):
            mdpt_normal = -mdpt_normal
            self.normal_flipped = True
        self.midpoint = mdpt
        self.midpoint_normal = mdpt_normal
        self._label = BELONGING_LABELS.UNKNOWN
        self.samples = None
        self.sample_pair_count = None
        self.features = None
        self.prediction = None
        self.ch = 0
        
    def __getitem__(self,index):
        return self.chain[index]
        
    @property
    def label(self):
        '''
        Label of the current edge. One of %s.
        ''' % (BELONGING_COLOR_BY_LABEL.keys())
        return self._label
    
    @label.setter
    def label(self, value):
        self._label = value
    
    @label.deleter
    def label(self):
        self._label = BELONGING_LABELS.UNKNOWN
    
    @property
    def belongs_left(self):
        '''Whether this edge belongs to the left or not'''
        return self._label in [BELONGING_LABELS.LEFT, BELONGING_LABELS.BOTH]
    
    @property
    def belongs_right(self):
        '''Whether this edge belongs to the right'''
        return self._label in [BELONGING_LABELS.RIGHT, BELONGING_LABELS.BOTH]
    
    @belongs_left.setter
    def belongs_left(self,value):
        if(value):
            if(self._label == BELONGING_LABELS.RIGHT):
                self._label = BELONGING_LABELS.BOTH
            else:
                self._label = BELONGING_LABELS.LEFT
        else:
            if(self._label == BELONGING_LABELS.BOTH or 
               self._label == BELONGING_LABELS.RIGHT):
                self._label = BELONGING_LABELS.RIGHT
            else:
                self._label = BELONGING_LABELS.NONE
                
    @belongs_right.setter
    def belongs_right(self,value):
        if(value):
            if(self._label == BELONGING_LABELS.LEFT):
                self._label = BELONGING_LABELS.BOTH
            else:
                self._label = BELONGING_LABELS.RIGHT
        else:
            if(self._label == BELONGING_LABELS.BOTH or 
               self._label == BELONGING_LABELS.LEFT):
                self._label = BELONGING_LABELS.LEFT
            else:
                self._label = BELONGING_LABELS.NONE

    @belongs_left.deleter 
    def belongs_left(self):
        if(self._label == BELONGING_LABELS.LEFT):
            self._label = BELONGING_LABELS.UNKNOWN
        elif(self._label == BELONGING_LABELS.BOTH):
            self._label = BELONGING_LABELS.RIGHT
            
    @belongs_right.deleter 
    def belongs_right(self):
        if(self._label == BELONGING_LABELS.RIGHT):
            self._label = BELONGING_LABELS.UNKNOWN
        elif(self._label == BELONGING_LABELS.BOTH):
            self._label = BELONGING_LABELS.LEFT
    
    def simple_normal_at_point(self,first_segment_keypoint_index,target_point):
        '''
        If a keypoint use avg of normals of prev and next segment, otherwise just normal to current segment
        @param first_segment_keypoint_index: index of the first keypoint the target point lies on
        @param target_point: a point that has to lie on the segment referenced by the previous parameter
        @return local (counter-clockwise) normal to the given point on the segment & edge
        '''
        keypoints = self.chain
        if(len(keypoints) < 2):
            raise ValueError("Edge normal can only be calculated for edges with at least two points")
        cur_seg = [keypoints[first_segment_keypoint_index],keypoints[first_segment_keypoint_index+1]]
        ccur = segment_midpoint(cur_seg)
        dist = distance(target_point,ccur)
        if(dist*2 == segment_length(cur_seg) and first_segment_keypoint_index > 0 and first_segment_keypoint_index < len(keypoints)-2):
            prevSeg = [keypoints[first_segment_keypoint_index-1],keypoints[first_segment_keypoint_index]]
            nextSeg = [keypoints[first_segment_keypoint_index+1],keypoints[first_segment_keypoint_index+2]]
            norm1 = segment_normal(prevSeg)
            norm2 = segment_normal(nextSeg)
            mid1 = segment_midpoint(prevSeg) 
            mid2 = segment_midpoint(nextSeg)
            tp1 = mid_point(norm1,norm2)
            tp2 = mid_point(mid1,mid2)
            return normal(np.array(tp1),np.array(tp2))
        return segment_normal(cur_seg)
        
        
    def three_seg_normal_at_point(self,first_segment_keypoint_index,target_point):
        '''
        Finds the local normal at the given point on the edge based on the normals to the segment
        the point lies on, the next segment's normal, the previous segment's normal, and how far is the
        point from the midpoints of all three of these segments.
        @param first_segment_keypoint_index: index of the first keypoint of the segent the provided point lies on
        @param target_point: a point that has to lie on the segment referenced by the previous parameter
        @return local (counter-clockwise) normal to the given point on the segment & edge
        '''
        keypoints = self.chain
        if(len(keypoints) < 2):
            raise ValueError("Edge normal can only be calculated for edges with at least two points")
        if(len(keypoints) == 2):
            return segment_normal([keypoints[first_segment_keypoint_index],keypoints[first_segment_keypoint_index+1]])
        cur_seg = [keypoints[first_segment_keypoint_index],keypoints[first_segment_keypoint_index+1]]
        cur_seg_normal = segment_normal(cur_seg)
        cur_seg_mdpt = segment_midpoint(cur_seg)
        cur_seg_dist = distance(target_point,cur_seg_mdpt)
        if(first_segment_keypoint_index == 0):
            next_seg = [keypoints[first_segment_keypoint_index+1],keypoints[first_segment_keypoint_index+2]]
            other_seg_normal = segment_normal(next_seg)
            other_seg_center = segment_midpoint(next_seg)
            other_seg_dist = distance(other_seg_center,cur_seg[1]) + distance(target_point,cur_seg[1])
        elif(first_segment_keypoint_index == len(keypoints)-2):
            prev_seg = [keypoints[first_segment_keypoint_index-1],keypoints[first_segment_keypoint_index]]
            other_seg_normal = segment_normal(prev_seg)
            other_seg_center = segment_midpoint(prev_seg)
            other_seg_dist = distance(other_seg_center,cur_seg[0]) + distance(target_point,cur_seg[0])
        else:
            frac = distance_along_segment(cur_seg,target_point)
            if(frac <=0.5):
                prev_seg = [keypoints[first_segment_keypoint_index-1],keypoints[first_segment_keypoint_index]]
                other_seg_normal = segment_normal(prev_seg)
                other_seg_center = segment_midpoint(prev_seg)
                other_seg_dist = distance(other_seg_center,cur_seg[0]) + distance(target_point,cur_seg[0])
            else:
                next_seg = [keypoints[first_segment_keypoint_index+1],keypoints[first_segment_keypoint_index+2]]
                other_seg_normal = segment_normal(next_seg)
                other_seg_center = segment_midpoint(next_seg)
                other_seg_dist = distance(other_seg_center,cur_seg[1]) + distance(target_point,cur_seg[1])
        
        total = other_seg_dist+cur_seg_dist
        cur_seg_weight = cur_seg_dist/total
        other_seg_weight = other_seg_dist/total
        nrm = cur_seg_weight * cur_seg_normal + other_seg_weight * other_seg_normal 
        return (nrm / math.sqrt(nrm[0]*nrm[0]+nrm[1]*nrm[1]))

        
   
    def get_first_edge_color(self,raster):
        '''
        Returns color of the first keypoint of this edge in the given raster
        @param raster: the raster where this edge occurs
        @return: the color of the pixel at the first keypoint on the given image
        '''
        return raster[self.chain[0][0],self.chain[0][1]]
   
    def find_average_normal(self):
        '''
        Finds the average (counter-clockwise) normal for the whole edge
        '''
        cumuWeight = 0.
        nrm = np.zeros(2)
        #traverse segments
        for ixKeyPoint in xrange(0,len(self.chain)-1):
            pt1 = self.chain[ixKeyPoint]
            pt2 = self.chain[ixKeyPoint+1]
            weight = distance(pt1,pt2)
            cumuWeight += weight
            nrm += (normal(pt1,pt2)*weight)
        return (nrm / math.sqrt(nrm[0]*nrm[0]+nrm[1]*nrm[1])),cumuWeight
    
        
    def point_along_edge(self,fraction):
        '''
        Finds the point along the edge at the given fraction of the total edge's lengtj
        @param fraction: the fraction of the edge length, a value from 0.0 to 1.0
        @return: the point that lays on this edge at the given fraction of its length
        '''
        traveled_distance = 0.
        at = self.length * fraction
        for ixKeyPoint in xrange(0,len(self.chain)-1):
            pt1 = self.chain[ixKeyPoint]
            pt2 = self.chain[ixKeyPoint+1]
            segment_length = distance(pt1,pt2)
            #append current segment length to the total travelled distance
            traveled_distance += segment_length
            if(traveled_distance > at):
                lIt = LineIterator(pt1,pt2)
                #stop fraction of the way down this segment
                for ixPx in xrange(0, lIt.count+1 / 2):
                    lIt.next()
                local_normal = self.three_seg_normal_at_point(ixKeyPoint, lIt.pos())
                return lIt.pos(), local_normal
            
    
    def draw(self,raster,color = None):
        '''
        @summary: draws the current edge with the given color (or the default color if none is given)
        @param raster: the raster where to draw the edge
        @param color: the color with which to draw the edge
        '''
        if (color == None):
            if(len(raster.shape) == 3):
                color = (0,128,255)
            else:
                color = 128
        chain_map(self.chain, di.draw_pixel, (raster,color))
        
            
    def draw_side_arrow(self,side,raster):
        '''
        @summary: draws an arrow on the raster on the given side of this edge
        Draws an arrow along the normal to the middle of this edge
        @param side: can be either LEFT or RIGHT
        @param raster: the raster where to draw the arrow 
        '''
        if side == BELONGING_LABELS.LEFT:
            nrm = self.midpoint_normal
            color = (0,255,0)
        elif side == BELONGING_LABELS.RIGHT:
            nrm = -self.midpoint_normal
            color = (255,0,0)
        else:
            raise ValueError("The side should be either '%s' or '%s'." %
                             (BELONGING_LABELS.LEFT,BELONGING_LABELS.RIGHT))
        rotmat = np.array([[nrm[1],-nrm[0]],
                           [nrm[0],nrm[1]]])
        midpt = self.midpoint
        
        di.draw_pixel(midpt,(raster,(0,0,255)))
        di.draw_arrow(raster, midpt + nrm * 10, rotmat, color)

