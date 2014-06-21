import numpy as np

from bw.line_iter import *
import utils.ascii as ascii
import abc
from bw.label import BELONGING_LABELS

class AbstractSampler:
    __metaclass__ = abc.ABCMeta
    def __init__(self,parent_tool):
        '''
        @param parent_tool: an instance of the SamplingTool class 
        '''
        self.parent = parent_tool
        self._display_sample_areas = self.parent.verbose > 1
        self._display_image = None
        

    def flip_samples(self,edge,sample_left,sample_right):
        if((edge.normal_flipped and (not edge.label or edge.label == 'RIGHT'))
            or (not edge.normal_flipped and edge.label == 'LEFT')):
            #rotate right side 90 degrees clockwise, transpose left side
            #transpose of left side is equivalent to 90 degree clockwise rotation and fliplr
            return np.rot90(sample_right,3), np.transpose(sample_left,(1,0,2)) 
        else:
            #rotate left side 90 degrees counterclockwise,
            #rotate right side 90 degrees counterclockwise and flip columns from left to right
            return np.rot90(sample_left), np.fliplr(np.rot90(sample_right))
    
    def sample_edge_pixels(self, edge,bordered_image,
                           pixel_sampling_func,step = 1):
         
        sampleHash = {}
        samplesL = []
        samplesR = []
        kpts = edge.chain
        border_thickness = self.parent.border_thickness
        #for contigous sampling set step to 1
        i_sample = 0
        for ix_kpt in xrange(0,len(kpts)-1):
            pt1 = kpts[ix_kpt]
            pt2 = kpts[ix_kpt+1]
            lit = LineIterator(pt1,pt2)
            for index_pixel in xrange(0, lit.count+1):
                pixel_pos = lit.pos();
                if(i_sample % step == 0):
                    if((pixel_pos[0],pixel_pos[1]) not in sampleHash):
                        retval, sampleL, sampleR = pixel_sampling_func(bordered_image,edge,ix_kpt,pixel_pos)
                        if(retval):
                            samplesL.append(sampleL)
                            samplesR.append(sampleR)
                            sampleHash[(pixel_pos[0],pixel_pos[1])] = True
                i_sample+=1
                lit.next()
        
        if(self._display_sample_areas):
            sh = self._display_image.shape
            self._display_image = self._display_image[border_thickness:sh[0]-border_thickness,
                                                      border_thickness:sh[1]-border_thickness]
            if(len(sh) == 3):
                color = (0,0,255)
            else:
                color = 128
            edge.draw(self._display_image,color)
            edge.draw_side_arrow(BELONGING_LABELS.LEFT, self._display_image)
            ch = self.parent.imshow("Sample on bodered bordered image",self._display_image)
            if ch == ascii.CODE.ESC:
                self._display_sample_areas = False
        
        return self.flip_samples(edge,np.hstack(samplesL).astype(np.uint8), np.hstack(samplesR).astype(np.uint8))

    @abc.abstractmethod
    def sample(self, edge, bordered_image):
        if(self._display_sample_areas):
            self._display_image = np.copy(bordered_image)
        else:
            self._display_image = None
