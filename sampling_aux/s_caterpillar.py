'''
Created on Aug 7, 2013

@author: Algomorph
'''
import cv2
import numpy as np

from utils.display import draw_seg_sample, draw_crossbars, draw_quad
from bw.edge import calc_keypoint_normal
from utils.enum import enum
from bw.chain import find_chain_segment_lengths
from bw.label import BELONGING_LABELS
from bw.line_iter import LineIterator
import utils.ascii as ascii
import utils.geom as ugeom
import sampling_aux.s_core as sc
from numpy.core.numeric import dtype

CROSSBAR_TYPE = enum(DISJUNCT=0, LEFT_CONV=1, RIGHT_CONV=2)
POSITION = enum(BEFORE=-1, AFTER=1)

class CaterpillarSampler(sc.AbstractSampler):
    mode = "caterpillar"
    def __init__(self,parent_tool):
        super(CaterpillarSampler,self).__init__(parent_tool)
        #determine what to display based on verbosity level
        self._display_subsamples = self.parent.verbose > 2
        self._display_crossbars = self.parent.verbose > 3
        self._crossbar_range = self.parent.sample_range + self.parent.blur_offset
        
            
    def rectify_side_sample_fan(self, image, quad,crossbar_type, 
                                segment_length, dest_raster):
        
        #destination will have the samples ordered from (and the edge flowing from)
        #left to right
        if(crossbar_type == CROSSBAR_TYPE.LEFT_CONV):
            get_direction = lambda conv_point, edge_pos, dist: (conv_point - edge_pos).astype(np.float32) / dist
            get_sample_pos = lambda edge_pos, conv_point, direction, cur_dist: np.int32(edge_pos + direction * cur_dist)
            #for left convergence, take any point on the left
            conv_point = quad[0]
            #allways iterate through points on the side of the edge, in the upwards direction
            #index 3 yields bottom center (at edge, right side of the quad)
            #index 2 yields top center (at edge, right side of the quad)
            edge_px_iter = LineIterator(quad[3],quad[2])
        elif(crossbar_type == CROSSBAR_TYPE.RIGHT_CONV):
            get_direction = lambda conv_point, edge_pos, dist: (edge_pos - conv_point).astype(np.float32) / dist
            get_sample_pos = lambda edge_pos, conv_point, direction, cur_dist: np.int32(conv_point + direction * cur_dist)
            #for right convergence, take any point on the right
            conv_point = quad[3]
            #index 0 yields the bottom center (left side of quad)
            #index 1 yields the top center (left side of quad)
            edge_px_iter = LineIterator(quad[0],quad[1])    
        else:
            raise ValueError("Given crossbar type is not supported by fanning rectification.")
        for i_edge_px in range(0, segment_length):
            edge_pos = edge_px_iter.pos()
            #find the distance between them
            dist = ugeom.distance(conv_point, edge_pos)
            #recompute the direction from convergence point to the current point on edge
            direction = get_direction(conv_point,edge_pos,dist)
            #determine how much to increment the distance
            step = dist / self._crossbar_range
            cur_dist = 0
            for i_samp_pixel in range(0,self._crossbar_range):
                #sample stripe from left
                sample_pos = get_sample_pos(edge_pos,conv_point, direction, cur_dist)
                dest_raster[i_samp_pixel,i_edge_px] = image[sample_pos[0],sample_pos[1]]
                cur_dist += step
            edge_px_iter.next()
            
        if self._display_sample_areas:
            draw_quad(np.fliplr(quad), self._display_image, 
                      colors=[(0,21,65),(40,191,255),(0,0,255),(40,34,117)])
        #display the sample if need be
        if self._display_subsamples:
            # points in the rectified sample's window
            rectified_pts = np.array([[0, self._crossbar_range - 1],
                                  [segment_length - 1, self._crossbar_range - 1],
                                  [segment_length - 1, 0],
                                  [0, 0]], dtype=np.float32)
            original, transformed = draw_seg_sample(image,segment_length,
                                                    self._crossbar_range, 
                                                    dest_raster, rectified_pts,
                                                    quad,factor=self.parent.display_scale_factor * 2)
            self.parent.imshow("Original Subsample",original, wait=False)
            ch = self.parent.imshow("Modified Subsample",transformed, wait=True)
            if ch == ascii.CODE.ESC:
                #the user chooses to stop viewing subsamples during this run
                self._display_subsamples = False
                
    def rectify_side_sample_projection(self, image, quad,segment_length, dest_raster):
        #opencv requires x,y coordinates whereas we're getting y,x
        #flip them
        quad_flipped = np.fliplr(quad)
        # points in the rectified sample's window
        rectified_pts = np.array([[0, self._crossbar_range - 1],
                                  [segment_length - 1, self._crossbar_range - 1],
                                  [segment_length - 1, 0],
                                  [0, 0]], dtype=np.float32)
        #obtain the transform
        transform = cv2.getPerspectiveTransform(quad_flipped.astype(np.float32), rectified_pts)
        #we can't use the return value here, as it will re-initialize the output array,
        #whereas we need to use the existing slice
        cv2.warpPerspective(image, transform, 
                            (dest_raster.shape[1],
                             dest_raster.shape[0]),
                            dst=dest_raster)
        
        if self._display_sample_areas:
            draw_quad(quad_flipped, self._display_image)
        #display the sub-sample if need be
        if self._display_subsamples:
            original, transformed = draw_seg_sample(image,segment_length,
                                                    self._crossbar_range, 
                                                    dest_raster, rectified_pts,
                                                    quad,factor=self.parent.display_scale_factor * 2)
            self.parent.imshow("Original Subsample",original,  wait=False)
            ch = self.parent.imshow("Modified Subsample",transformed,  wait=True)
            if ch == ascii.CODE.ESC:
                #the user chooses to stop viewing subsamples during this run
                self._display_subsamples = False
    

    def find_crossbar(self, edge, segment_lengths, target_kpt_index, sensitivity):
        chain = edge.chain
        cur_normal = calc_keypoint_normal(chain, target_kpt_index, sensitivity, segment_lengths)
        left_end = chain[target_kpt_index] + cur_normal * self._crossbar_range
        mid = chain[target_kpt_index]
        right_end = chain[target_kpt_index] - cur_normal * self._crossbar_range
        return np.array([left_end, mid, right_end])

        
    def find_crossbars_nudge(self, edge, kpt_norm_sensitivity):
        
        kpts = edge.chain
        skip_range = self._crossbar_range / 6
        
        seg_lengths = find_chain_segment_lengths(kpts)
        #find first crossbar
        cbar = self.find_crossbar(edge, seg_lengths, 0, 
                                 kpt_norm_sensitivity)
        crossbars = [cbar]
        prev_kpt = kpts[0]
        run_length = 0
        
        #loop through all the rest except for the last one
        for ix_kpt in xrange(1,len(kpts)-1):
            kpt = kpts[ix_kpt]
            run_length += seg_lengths[ix_kpt-1]
            #skip over the ones which are too close
            if(run_length < skip_range):
                continue
            run_length = 0
            #find the crossbar
            cbar = self.find_crossbar(edge, seg_lengths, ix_kpt, 
                                 kpt_norm_sensitivity)
            #store it
            crossbars.append(cbar)
            prev_kpt = kpt
        
        ix_kpt = len(kpts)-1#last point
        kpt = kpts[ix_kpt]
        while(ugeom.distance(prev_kpt, kpt) < skip_range and len(crossbars) > 2):
            #this time, remove the previous, because we want to keep the last one
            crossbars.pop()
            prev_kpt = crossbars[len(crossbars)-1][1]
            
        #append the last crossbar
        cbar = self.find_crossbar(edge, seg_lengths, ix_kpt, 
                                 kpt_norm_sensitivity)
        crossbars.append(cbar)
         
        
        adjustments = np.inf
        crossbars = np.asarray(crossbars, dtype=np.int32)
        while(adjustments > 0):
            adjustments = 0
            if self._display_crossbars:
                disp = draw_crossbars(crossbars)
                ch = self.parent.imshow("Crossbars", disp)
                if(ch == ascii.CODE.ESC):
                    self._display_crossbars = False
            for ix_cbar in xrange(0,len(crossbars)-1):
                cbar_a = crossbars[ix_cbar]
                cbar_b = crossbars[ix_cbar+1]
                intersect = ugeom.detect_intersections(np.array([[cbar_a[0], cbar_a[1]],
                                                        [cbar_b[0], cbar_b[1]]],
                                                       dtype=np.float32))
                if(intersect and not np.array_equal(cbar_a[0], cbar_b[0])):
                    #intersection on left side
                    '''
                    #old method
                    temp = cbar_a[0]
                    cbar_a[0] = cbar_b[0]
                    cbar_b[0] = temp
                    '''
                    #new method
                    mpt = ugeom.mid_point_list(cbar_a[0], cbar_b[0])
                    cbar_a[0] = mpt
                    cbar_b[0] = mpt
                    adjustments += 1
                    
                intersect = ugeom.detect_intersections(np.array([[cbar_a[1], cbar_a[2]],
                                                        [cbar_b[1], cbar_b[2]]],
                                                       dtype=np.float32))
                if(intersect and not np.array_equal(cbar_a[2], cbar_b[2])):
                    #intersection on right side
                    '''
                    #old method
                    temp = cbar_a[2]
                    cbar_a[2] = cbar_b[2]
                    cbar_b[2] = temp
                    '''
                    #new method
                    mpt = ugeom.mid_point_list(cbar_a[2], cbar_b[2])
                    cbar_a[2] = mpt
                    cbar_b[2] = mpt
                    adjustments += 1
                    
        for ix_cbar in xrange(0,len(crossbars)-1):
            cbar_a = crossbars[ix_cbar]
            cbar_b = crossbars[ix_cbar+1]
            #check if both subsample quads are convex
            #Note:positive angles are clockwise because y-axis is flipped
            #right side
            vec_away = cbar_a[2]-cbar_a[1]
            vec_side = cbar_b[2]-cbar_a[2]
            signed_angle_rb = ugeom.signed_angle(vec_away, vec_side)
            if(signed_angle_rb > 0):
                #left subsample is concave on bottom side
                #fix the top point
                if(ix_cbar + 2 < len(crossbars)):
                    #top crossbar isn't the last one
                    cbar_b[2] = ugeom.mid_point_list(crossbars[ix_cbar+2][2],cbar_a[2])
                else:
                    #form a proper parallelogram
                    cbar_b[2] = cbar_b[1] + vec_away
            vec_away = cbar_b[2]-cbar_b[1]
            vec_side = cbar_a[2]-cbar_b[2]
            signed_angle_rt = ugeom.signed_angle(vec_away, vec_side)
            if(signed_angle_rt < 0):
                #left subsample is concave on top side
                if(ix_cbar > 0):
                    #top crossbar isn't the last one
                    cbar_a[2] = ugeom.mid_point_list(crossbars[ix_cbar-1][2], cbar_b[2])
                else:
                    #form a proper parallelogram
                    cbar_a[2] = cbar_a[1] + vec_away
            #left side
            vec_away = cbar_a[0]-cbar_a[1]#vec from edge to outside along bottom crossbar
            vec_side = cbar_b[0]-cbar_a[0]#vec on the far side
            signed_angle_lb = ugeom.signed_angle(vec_away, vec_side)
            if(signed_angle_lb < 0):
                #left subsample is concave on bottom side
                #fix the top point
                if(ix_cbar + 2 < len(crossbars)):
                    #top crossbar isn't the last one
                    cbar_b[0] = ugeom.mid_point_list(crossbars[ix_cbar+2][0],cbar_a[0])
                else:
                    #form a proper parallelogram
                    cbar_b[0] = cbar_b[1] + vec_away
            vec_away = cbar_b[0]-cbar_b[1]
            vec_side = cbar_a[0]-cbar_b[0]#vec on the far side
            signed_angle_lt = ugeom.signed_angle(vec_away, vec_side)
            if(signed_angle_lt > 0):
                #left subsample is concave on top side
                if(ix_cbar > 0):
                    #top crossbar isn't the last one
                    cbar_a[0] = ugeom.mid_point_list(crossbars[ix_cbar-1][0], cbar_b[0])
                else:
                    #form a proper parallelogram
                    cbar_a[0] = cbar_a[1] + vec_away
                
        if self._display_crossbars:
            disp = draw_crossbars(crossbars)
            ch = self.parent.imshow("Crossbars", disp)
            if(ch == ascii.CODE.ESC):
                self._display_crossbars = False
            
        return crossbars    

    
    def fan_crossbars(self, crossbar_stack, collapse_to=0):
        np_stack = np.array(crossbar_stack)
        if(collapse_to == -1):
            target_bar = np_stack[0]
        elif(collapse_to == 1):
            target_bar = np_stack[len(crossbar_stack) - 1]
        else:
            # use midpoints of all bars to make the "middle bar"
            target_bar = np_stack.mean(axis=0)
        mins = np_stack.min(axis=0)
        maxs = np_stack.max(axis=0)
        rangeXL = maxs[0][0] - mins[0][0]
        rangeYL = maxs[0][1] - mins[0][1]
        rangeXR = maxs[2][0] - mins[2][0]
        rangeYR = maxs[2][1] - mins[2][1]
        areaL = rangeXL * rangeYL
        areaR = rangeYR * rangeXR
        
        if(areaL < areaR):
            mpt = target_bar[0]
            for ixCbar in range(0, len(crossbar_stack)):
                cbar = np_stack[ixCbar]
                # crossbars.append(cbar)
                cbar[0] = mpt
                #crossbars.append(cbar)
                crossbar_stack[ixCbar] = cbar
            return CROSSBAR_TYPE.LEFT_CONV
        else:
            mpt = target_bar[2]
            for ixCbar in range(0, len(crossbar_stack)):
                cbar = np_stack[ixCbar]
                #crossbars.append(cbar)
                cbar[2] = mpt
                #crossbars.append(cbar)
                crossbar_stack[ixCbar] = cbar
            return CROSSBAR_TYPE.RIGHT_CONV
        
    def _add_cbar_if_necessary(self,crossbar, fan_stack, cbar_type, where = POSITION.AFTER):
        '''
        Helper method for checking if a crossbar intersect with the first or last
        crossbar in a given "fan" stack and adding it to the stack (whilst
        altering it's left-most or right-most coordinate in accordance with the
        stacks nature, as dictated by cbar_type
        @param crossbar: crossbar. This would be a crossbar coming either directly
                         before or directly after the fan stack.
        @param fan_stack: a "fan" stack of crossbars with one identical coordinate
        @param cbar_type: either CROSSBAR_TYPE.LEFT_CONV or CROSSBAR_TYPE.RIGHT_CONV -
                          determines which side do the fan stack crossbars have in common
        @param where: either POSITION.BEFORE or POSITION.AFTER
        '''
        if(where == 1):
            stack_cb_to_check = fan_stack[len(fan_stack)-1]
        elif(where == -1):
            stack_cb_to_check = fan_stack[len(fan_stack)-1]
        else:
            raise ValueError("where can only be POSITION.BEFORE or POSITION.AFTER")
        
        intersect = ugeom.detect_intersections(np.array([[stack_cb_to_check[0], stack_cb_to_check[2]],
                                                        [crossbar[0], crossbar[2]]],
                                                       dtype=np.float32))
        if(intersect):
            #it intersects the first/last crossbar in the stack, now modify and add it
            if(cbar_type == CROSSBAR_TYPE.LEFT_CONV):
                mpt = stack_cb_to_check[0]
                crossbar = np.vstack(mpt, crossbar[1],crossbar[2])
                fan_stack.append()
            elif(cbar_type == CROSSBAR_TYPE.RIGHT_CONV):
                mpt = stack_cb_to_check[2]
                crossbar = np.vstack(crossbar[0], crossbar[1],mpt)
            else:
                raise ValueError("cbar_type can only be CROSSBAR_TYPE.LEFT_CONV or CROSSBAR_TYPE.RIGHT_CONV")
            fan_stack
            return True
        else:
            return False
        
    def find_crossbars_fan(self, edge, kpt_norm_sensitivity):
        crossbar_range = self._crossbar_range
        kpts = edge.chain
        skip_range = crossbar_range / 3
        seg_lengths = find_chain_segment_lengths(kpts)
        previous_crossbar = self.find_crossbar(edge, seg_lengths, 0, 
                                          kpt_norm_sensitivity)  
        crossbars = [previous_crossbar]
        inters_crossbar_stack = None
        index_kpt = 1
        
        fan_cbar_hash = {}
        
        need_check = True
        
        # look through the first few crossbars looking for intersections
        while index_kpt < len(kpts) and need_check:
            
            crossbar = self.find_crossbar(edge, seg_lengths, index_kpt, kpt_norm_sensitivity)
            
            # if this crossbar is intersecting the first one or the ones following it
            # in a series of intersections, ignore until you find one that doesn't
            if (ugeom.distance(previous_crossbar[1], crossbar[1]) < skip_range):
                index_kpt += 1
                continue
            
            intersect = ugeom.detect_intersections(np.array([[previous_crossbar[0], previous_crossbar[2]],
                                                        [crossbar[0], crossbar[2]]],
                                                       dtype=np.float32))
            if(not intersect):
                if(inters_crossbar_stack):
                    self.fan_crossbars(inters_crossbar_stack, collapse_to= -1)
                    crossbars += inters_crossbar_stack
                    inters_crossbar_stack = None
                # if it doesn't
                need_check = False
            else:
                crossbars.pop()
                inters_crossbar_stack = [previous_crossbar, crossbar]
                index_kpt += 1
        
        previous_crossbar = crossbars[len(crossbars) - 1]
        
        startKpt = index_kpt
        # traverse remaining keypoints
        for index_kpt in xrange(startKpt, len(kpts)):
            # find crossbars
            crossbar = self.find_crossbar(edge, seg_lengths, index_kpt, kpt_norm_sensitivity)
            
            if(ugeom.distance(previous_crossbar[1], crossbar[1]) < skip_range):
                continue
            
            intersect = ugeom.detect_intersections(np.array([[previous_crossbar[0], 
                                                              previous_crossbar[2]],
                                                    [crossbar[0], crossbar[2]]],
                                                       dtype=np.float32))
            if(intersect):
                if(inters_crossbar_stack):
                    inters_crossbar_stack.append(crossbar)
                else:
                    crossbars.pop()
                    inters_crossbar_stack = [previous_crossbar, crossbar]
            else:
                if(inters_crossbar_stack):
                    self.fan_crossbars(inters_crossbar_stack)
                    crossbars += inters_crossbar_stack
                    inters_crossbar_stack = None
                crossbars.append(crossbar)
            previous_crossbar = crossbar
        
        if(inters_crossbar_stack):
            self.fan_crossbars(inters_crossbar_stack, collapse_to=1)
            crossbars += inters_crossbar_stack
            
        
        #debug block - check for persistent intersections
        for ix_cbar in xrange(0,len(crossbars)):
            crsSrc = crossbars[ix_cbar]
            for ix_other_cbar in xrange(ix_cbar+1,len(crossbars)):
                crsDst = crossbars[ix_other_cbar]
                inter = ix_cbar != ix_other_cbar and (ugeom.detect_intersections(np.array([[crsSrc[0],crsSrc[1]],
                                                        [crsDst[0],crsDst[1]]],
                                                       dtype=np.float32)) or 
                                                      ugeom.detect_intersections(np.array([[crsSrc[1],crsSrc[2]],
                                                        [crsDst[1],crsDst[2]]],
                                                       dtype=np.float32)))
                if(inter):
                    print "Found intersection. Crossbar %d intersects crossbar %d."% (ix_cbar, ix_other_cbar)
                    #print "%d: %s" % (ix_cbar, str(crsSrc))
                    #print "%d: %s" % (ix_other_cbar, str(crsDst))
                    #print "" 
        
            
        return np.asarray(crossbars)
 
    def sample(self, edge, bordered_image):
        super(CaterpillarSampler, self).sample(edge,bordered_image)
        sampling_range= self.parent.sample_range
        blur_offset = self.parent.blur_offset
        verbosity = self.parent.verbose
        border_thickness = self.parent.border_thickness
        # TODO: cut off blur_offset at the very last stage
        crossbar_range = sampling_range + blur_offset
        #this is for determining how far to look back and forward along the edge
        #when determining the edge's normal at a given keypoint
        kpt_norm_sensitivity = 3 * crossbar_range
        
        crossbars = self.find_crossbars_nudge(edge, kpt_norm_sensitivity)
        
        if(verbosity > 1):
            print "Number of crossbars in edge: %d" % len(crossbars)
        
        # debug block to see  crossbars
        if(self._display_crossbars):
            cbar_image = draw_crossbars(crossbars.astype(np.int32))
            ch = self.parent.imshow("Crossbars", cbar_image)
            #if the user hit escape, he no longer wishes to see crossbars for subsequent images.
            self._display_crossbars = ch != ascii.CODE.ESC 
        
        # indices to keep track of how much is filled
        cur_length = 0
        
        # initialize the sample window to a size twice the edge length.
        # this way we know the sample won't need to be grown
        begLength = int(edge.length * 2);  
    
        if (len(bordered_image.shape) == 3):
            sample_right = np.empty((crossbar_range, begLength, 3), dtype=np.uint8)
            sample_left = np.empty((crossbar_range, begLength, 3), dtype=np.uint8)
            keypoint_color = (0, 255, 0)
        elif(len(bordered_image.shape) == 2):
            sample_right = np.empty((crossbar_range, begLength), dtype=np.uint8)
            sample_left = np.empty((crossbar_range, begLength), dtype=np.uint8)
            keypoint_color = 255
        else:
            raise ValueError("Can only sample BGR or greyscale images.")
        
        for i_seg in xrange(1, len(crossbars)):  
            if(self.parent.verbose > 2):
                print "Subsampling using crossbars %d and %d" % (i_seg-1, i_seg)
            # moving "up" through the crossbars
            bottom_crossbar = crossbars[i_seg - 1]
            top_crossbar = crossbars[i_seg]
            
            bottom_keypt = bottom_crossbar[1]
            top_keypt = top_crossbar[1]
            
            distance_points = ugeom.distance(top_keypt, bottom_keypt)
            
            distance_center = int(round(distance_points))
            # 1-px length vector in the derection from top to bottom keypoints
            #offset_vector = ((bottom_keypt - top_keypt) / distance_points).round().astype(np.int32)
            offset_vector = [0,0]
            
            bl = bottom_crossbar[0]
            br = bottom_crossbar[2]
            tl = top_crossbar[0] + offset_vector
            tr = top_crossbar[2] + offset_vector
            
            if(np.array_equal(bl, top_crossbar[0])):
                cb_type = CROSSBAR_TYPE.LEFT_CONV
                if(self.parent.verbose > 2):
                    print "Left Conv"
            elif(np.array_equal(br, top_crossbar[2])):
                cb_type = CROSSBAR_TYPE.RIGHT_CONV
                if(self.parent.verbose > 2):
                    print "Right Conv"
            else:
                if(self.parent.verbose > 2):
                    print "Disjunct"
                cb_type = CROSSBAR_TYPE.DISJUNCT
            
            bc = bottom_keypt
            tc = top_keypt + offset_vector
            
            # tl------tc------tr           1        2
            # [       +        ]
            # [       +        ]
            # [       +edge    ]
            # [       +        ]
            # [       +        ]
            # bl------bc------br           0        3
            points_left = np.array([bl, tl, tc, bc], dtype=np.int32) + border_thickness
            points_right = np.array([bc, tc, tr, br], dtype=np.int32) + border_thickness
            #distance_center -= 1
            
            segment_length = distance_center
            
            # pick out a window from the sample
            sub_sample_left = sample_right[:, cur_length:cur_length + segment_length] 
            sub_sample_right = sample_left[:, cur_length:cur_length + segment_length]
            
            
            # rectify the trapezoids
            #left side
            if(cb_type == CROSSBAR_TYPE.LEFT_CONV):
                self.rectify_side_sample_fan(bordered_image, points_left,
                                        cb_type, segment_length, sub_sample_left)
            else:
                self.rectify_side_sample_projection(bordered_image,
                                                     points_left,segment_length,
                                                     sub_sample_left)
            #right side   
            if(cb_type == CROSSBAR_TYPE.RIGHT_CONV):
                self.rectify_side_sample_fan(bordered_image, points_right,
                                        cb_type, segment_length, sub_sample_right)
                    
            else:
                self.rectify_side_sample_projection(bordered_image,
                                                     points_right, segment_length,
                                                     sub_sample_right)
                    
            
            # update cur heights
            cur_length += (segment_length - 1)
            
        #re-orient so that left is belonging and right is not 
        sample_left, sample_right = self.flip_samples(edge, 
                                                    sample_left[:, 0:cur_length], 
                                                    sample_right[:, 0:cur_length])
    
        # View the sample crossbars
        if(self._display_sample_areas):
            sh = self._display_image.shape
            #cut off the borders for display
            display_image = self._display_image[border_thickness:sh[0] - border_thickness, 
                                        border_thickness:sh[1] - border_thickness]
            edge.draw(display_image)
            edge.draw_side_arrow(BELONGING_LABELS.LEFT, display_image)
            for crsbr in crossbars:
                kpt = crsbr[1]
                display_image[kpt[0], kpt[1]] = keypoint_color
            
            ch = self.parent.imshow("Sample areas on original image",display_image) 
            if(ch == ascii.CODE.ESC):
                self._display_sample_areas = False
            
            
        # return windows into the sample spaces up to the current heights
        return sample_left, sample_right
