'''
Created on Aug 7, 2013

@author: Algomorph
'''

import sampling_aux.s_core as sc
import numpy as np
from utils.helpers import isnone


class HedgehogSampler(sc.AbstractSampler):
    mode = "hedgehog"
    def __init__(self,parent_tool):
        super(HedgehogSampler,self).__init__(parent_tool)
    def sample_edge_pixel_line(self, bordered_image, edge, segment_index, 
                                edge_pixel):
        '''
        samples pixel-wide stripes along edge normals to the provided edge pixel
        spanning from blur_offset to blur_offset+sample_range on each side
        '''
        sampling_range = self.parent.sample_range
        blur_offset = self.parent.blur_offset
        display_image = self._display_image
        normal = edge.three_seg_normal_at_point(segment_index,edge_pixel)
        edge_pixel = np.array(edge_pixel) + self.parent.border_thickness
        if(len(bordered_image.shape) == 3):
            #three channels
            #allocate boxes for samples
            sampleL = np.zeros((sampling_range,3))
            sampleR = np.zeros((sampling_range,3))
            #color for display
            colorL = (0,128,255)
            colorR = (0,255,128)
            
            if(edge.normal_flipped):
                temp = colorR
                colorR = colorL
                colorL = temp
            
        else:
            
            #gotta be greyscale
            sampleL = np.zeros(sampling_range)
            sampleR = np.zeros(sampling_range)
            #color for display
            colorL = 80
            colorC = 128
            colorR = 120
            
            if(edge.normal_flipped):
                temp = colorR
                colorR = colorL
                colorL = temp
            
        
        if self._display_sample_areas:
            i_row = 0
            for i_samp_pixel in range(-sampling_range+1,1):
                #draw left stripe
                sample_pos = np.int32(edge_pixel + normal * (i_samp_pixel - blur_offset))
                display_image[sample_pos[0],sample_pos[1]] = colorL
                i_row+=1
            for i_samp_pixel in range(0,sampling_range):
                #draw right stripe
                sample_pos = np.int32(edge_pixel + normal * (i_samp_pixel + blur_offset))
                display_image[sample_pos[0],sample_pos[1]] = colorR
        
        i_row = 0
        for i_samp_pixel in range(-sampling_range+1,1):
            #sample stripe from left
            sample_pos = np.int32(edge_pixel + normal * (i_samp_pixel - blur_offset))
            sampleL[i_row] = bordered_image[sample_pos[0],sample_pos[1]]
            i_row+=1
        for i_samp_pixel in range(0,sampling_range):
            #sample stripe from right
            sample_pos = np.int32(edge_pixel + normal * (i_samp_pixel + blur_offset))
            sampleR[i_samp_pixel] = bordered_image[sample_pos[0],sample_pos[1]]
                
        return True, sampleL, sampleR
    
    def sample(self, edge,bordered_image):
        super(HedgehogSampler, self).sample(edge,bordered_image)
        return self.sample_edge_pixels(edge,bordered_image, 
                                       self.sample_edge_pixel_line, 
                                       self.parent.step)