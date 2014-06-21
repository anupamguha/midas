'''
Created on Aug 7, 2013

@author: Algomorph
'''
import sampling_aux.s_core as sc
import numpy as np
import utils.geom as ugeom
#from sympy.geometry import Point, Segment, Circle, Line
import cv2
class SovietTetrisSampler(sc.AbstractSampler):
    mode = "soviet_tetris"
    def __init__(self,parent_tool):
        super(SovietTetrisSampler,self).__init__(parent_tool)
        
    def find_box_base(self, edge, segment_index, edge_pixel):
        '''
        Find base of a tetris box aligned with the edge at the specified edge_pixel
        '''
        sampling_range = self.parent.sample_range
        border_thickness = self.parent.border_thickness
        #border_thickness = self.parent.border_thickness
        #find pixels on edge at 1/2 sampling range cartesian distance from edge pixel
        edge_pixel = np.array(edge_pixel)
        half_range = float(sampling_range) /2
        ix_back_kpt = segment_index
        cur_dist = 0
        #look back to find the point on the edge intersecting
        #with the circle centered at edge_pixel and having half_range radius 
        while ix_back_kpt >=0 and cur_dist < half_range:
            back_pt = edge.chain[ix_back_kpt]
            cur_dist = ugeom.distance(edge_pixel, back_pt)
            ix_back_kpt -=1 #go further back the edge
        
        if(cur_dist > half_range):
            #slightly overshot, meaning there's an intersection
            #of the circle from edge_pixel w/ radius of half_range with
            #the segment whose keypoint back_pt is
            ix_back_kpt += 1#undo last change
            other_seg_endpt = edge.chain[ix_back_kpt+1]#next kepoint toward edge_pixel
            #calculate intersection with circle
            #note that coordinates are flipped (circle.x is actually the y-coord)
            intersections = ugeom.line_circle_intersection(back_pt, 
                                                           other_seg_endpt, 
                                                           edge_pixel, 
                                                           half_range)
            if(len(intersections) > 1):
                int1 = intersections[0]
                int2 = intersections[1]
                dist1 = ugeom.distance(back_pt,int1)
                dist2 = ugeom.distance(back_pt,int2)
                #select one closer to the back_pt keypoint
                if dist1 < dist2:
                    back_pt = np.array(int1)
                else:
                    back_pt = np.array(int2)
            else:
                back_pt = np.array([float(x) for x in intersections[0]])
        #otherwise, we just keep the back_pt at the keypoint
            
        
        #now, look forward:
        cur_dist = 0
        ix_fore_kpt = segment_index + 1
        while ix_fore_kpt < len(edge.chain) and cur_dist < half_range:
            fore_pt = edge.chain[ix_fore_kpt]
            cur_dist = ugeom.distance(edge_pixel, fore_pt)
            ix_fore_kpt +=1 #go further back the edge
        
        if(cur_dist > half_range):
            #slightly overshot, meaning there's an intersection
            #of the circle from edge_pixel w/ radius of half_range with
            #the segment whose keypoint fore_pt is
            ix_fore_kpt -= 1#undo last change
            other_seg_endpt = edge.chain[ix_fore_kpt-1]#previous keypoint
            #calculate intersection with circle
            #note that coordinates are flipped (circle.x is actually the y-coord)
            intersections = ugeom.line_circle_intersection(fore_pt, 
                                                           other_seg_endpt, 
                                                           edge_pixel, 
                                                           half_range)
            if(len(intersections) > 1):
                #two intersections - segment goes through the circle (rare long segment case)
                int1 = intersections[0]
                int2 = intersections[1]
                dist1 = ugeom.distance(fore_pt,int1)
                dist2 = ugeom.distance(fore_pt,int2)
                #select one closer to the fore_pt keypoint
                if dist1 < dist2:
                    fore_pt = np.array(int1)
                else:
                    fore_pt = np.array(int2)
            else:
                #a single intersection
                fore_pt = np.array(intersections[0])
        #otherwise, we just keep the fore_pt at the keypoint
        parallel_seg = np.array([back_pt,fore_pt])
        #a_1 = a.dot(b_hat) -- see Wikipedia article on vector projection
        #a_2 = a - a1
        #b
        delta_vec = fore_pt - back_pt
        #b_hat
        delta_dir = delta_vec / ugeom.norm(delta_vec)
        #a
        back_to_edge = edge_pixel - back_pt
        #a_1
        proj_onto_parallel = delta_dir * back_to_edge.dot(delta_dir)
        parallel_closest_pt = back_pt + proj_onto_parallel
        base_center = ugeom.mid_point_list(edge_pixel, parallel_closest_pt)
        if(self.parent.verbose > 2):
            cv2.circle(self._display_image, tuple(edge_pixel[::-1].astype(np.int) + border_thickness),
                       int(half_range), (23,67,123))
            #display base guide
            cv2.line(self._display_image,
                     tuple(fore_pt[::-1].astype(np.int) + border_thickness),
                     tuple(back_pt[::-1].astype(np.int) + border_thickness),
                     (134,234,59))
        #-a_2 = a1 - a
        base_normal = ugeom.normal(back_pt, fore_pt)
        return base_center, base_normal
        
        
        
        

    def sample_edge_pixel_block(self, bordered_image, edge, segment_index, 
                                edge_pixel):
        '''
        samples samplinRange-wide blocks along edge normals to the provided edge pixel
        spanning from blur_offset to blur_offset+sample_range on each side
        '''
        sampling_range = self.parent.sample_range
        display_image = self._display_image
        blur_offset = self.parent.blur_offset
        border_thickness = self.parent.border_thickness
        base_center, normal = self.find_box_base(edge, segment_index, edge_pixel)
        #normal = edge.simple_normal_at_point(segment_index,edge_pixel)
        #normal = edge.local_normal_at_point(segment_index,edge_pixel)
        base_center = np.array(base_center) + border_thickness
        tangent = ugeom.rmat90.dot(normal)
        
        if(len(bordered_image.shape) == 3):
            #three channels
            #allocate boxes for samples
            sampleL = np.zeros((sampling_range,sampling_range,3))
            sampleR = np.zeros((sampling_range,sampling_range,3))
            #color for display
            colorL = (0,128,255)
            colorC = (0,255,255)
            colorR = (0,255,128)
            
            if(edge.normal_flipped):
                temp = colorR
                colorR = colorL
                colorL = temp
        else:
            #gotta be greyscale
            sampleL = np.zeros((sampling_range,sampling_range))
            sampleR = np.zeros((sampling_range,sampling_range))
            #color for display
            colorL = 80
            colorC = 128
            colorR = 120
            
            if(edge.normal_flipped):
                temp = colorR
                colorR = colorL
                colorL = temp
        
        col_range = range(-sampling_range/2+1,sampling_range/2+1)
        
        #display routine - draw box from which the sample is taken
        if self._display_sample_areas: 
            firstBasePos = base_center + tangent * col_range[0]
            lastBasePos = base_center + tangent * col_range[len(col_range)-1]
            #draw edge center
            display_image[int(firstBasePos[0]),int(firstBasePos[1])] = colorC
            display_image[int(lastBasePos[0]),int(lastBasePos[1])] = colorC
            #side bars - left
            for ixSamplePx in range(-sampling_range+1,1):
                samplePos = np.int32(firstBasePos + normal * (ixSamplePx - blur_offset))
                display_image[samplePos[0],samplePos[1]] = colorL
                samplePos = np.int32(lastBasePos + normal * (ixSamplePx - blur_offset))
                display_image[samplePos[0],samplePos[1]] = colorL
            #side bars - right
            for ixSamplePx in range(0,sampling_range):
                samplePos = np.int32(firstBasePos + normal * (ixSamplePx + blur_offset))
                display_image[samplePos[0],samplePos[1]] = colorR
                samplePos = np.int32(lastBasePos + normal * (ixSamplePx + blur_offset))
                display_image[samplePos[0],samplePos[1]] = colorR
            #middle range
            for i_samp_col in col_range:
                basePos = base_center + tangent * i_samp_col
                #center
                display_image[int(basePos[0]),int(basePos[1])] = colorC
                #left inside
                samplePos = np.int32(basePos - normal * ((sampling_range-1) + blur_offset))
                display_image[samplePos[0],samplePos[1]] = colorL
                #left outside
                samplePos = np.int32(basePos - normal * blur_offset)
                display_image[samplePos[0],samplePos[1]] = colorL
                
                #right inside
                samplePos = np.int32(basePos + normal * blur_offset)
                display_image[samplePos[0],samplePos[1]] = colorR
                #right outside
                samplePos = np.int32(basePos + normal * ((sampling_range-1) + blur_offset))
                display_image[samplePos[0],samplePos[1]] = colorR
                
        #the negative bound's magnintude will be greater by 1 than the positive bound for odd
        #sampling ranges
        i_col = 0 
        for i_samp_col in col_range:
            basePos = base_center + tangent * i_samp_col
            ixRow = 0
            
            #sample a bar for left box
            for ixSamplePx in range(-sampling_range+1,1):
                samplePos = np.int32(basePos + normal * (ixSamplePx - blur_offset))
                sampleL[ixRow,i_col] = bordered_image[samplePos[0],samplePos[1]]
                ixRow+=1
            
            #sample a bar for right box
            for ixSamplePx in range(0,sampling_range):
                samplePos = np.int32(basePos + normal * (ixSamplePx + blur_offset))
                sampleR[ixSamplePx,i_col] = bordered_image[samplePos[0],samplePos[1]]
            i_col+=1
        
           
        return True, sampleL, sampleR

    def sample(self, edge,bordered_image):
        super(SovietTetrisSampler, self).sample(edge,bordered_image)
        return self.sample_edge_pixels(edge,bordered_image, 
                                       self.sample_edge_pixel_block, 
                                       self.parent.step)