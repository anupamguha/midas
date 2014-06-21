'''
Created on Feb 17, 2013

@author: algomorph
'''

import core
from sampling_aux.s_tetris import SovietTetrisSampler
from sampling_aux.s_hedgehog import HedgehogSampler
from sampling_aux.s_caterpillar import CaterpillarSampler
import cv2
import numpy as np
import utils.display as di
import sys
import argparse
import utils.ascii as ascii
import pandas as pd
import os

class SampleTool(core.AbstractEdgeTool):
    op_name = 'sample'
    
    '''
    This tool samples pixels and saves that in the form of little numpy arrays
    inside the cells of the output pandas table.
    @param sampling_source: which images to sample from
    @param data_path: path to the dataframe where to save the samples
    @param step: for certain sampling modes, how sampling interval in pixels along the edge
    @param mode: sampling mode
    @param display_scaling: how much to scale the display for samples and subsamples
    @param sample_range: how far away to sample from the edge
    @param blur_offset: how much of a boundary to leave off the edge for blur     
    '''
    def __init__(self,
                 sampling_source,
                 data_path,
                 output,
                 step,
                 mode,
                 display_scaling,
                 sample_range,
                 column_suffix,
                 blur_offset,
                 start = -1, 
                 end = sys.maxint, 
                 verbose=0,
                 no_save=False,
                 save_to_edge=False):
        '''
        Constructor
        '''
        super(SampleTool,self).__init__(sampling_source,
                                        data_path,
                                        start=start,
                                        end=end,
                                        verbose=verbose,
                                        save_result=not no_save)
        
        self.column_suffix = column_suffix
        self.sample_range = sample_range
        self.step = step
        self.mode = mode
        self.save_path = output
        self.display_scale_factor = display_scaling
        self.blur_offset = blur_offset
        Sampler = self.mode_hash[mode]
        self.sampler = Sampler(self)
        self.save_to_edge = save_to_edge

        
    mode_hash = {
                 CaterpillarSampler.mode:   CaterpillarSampler,
                 SovietTetrisSampler.mode:   SovietTetrisSampler,
                 HedgehogSampler.mode:       HedgehogSampler,
                }
    
    def process_image(self,image):
        '''
        Samples pixels around the edges of a single given image
        @type image: core.Image or its subclass
        @param image: the image to sample from
        '''
        edge_sample_window_name = "Edge samples"       
        self.border_thickness = (self.sample_range + self.blur_offset) << 1
        bordered = cv2.copyMakeBorder(image.raster, 
                                     (self.border_thickness),  
                                     (self.border_thickness), 
                                     (self.border_thickness), 
                                     (self.border_thickness), 
                                     cv2.BORDER_REPLICATE)
        ix = 0
        ch = 0
        for edge in image.edges:
            if(self.verbose > 0):
                print "sampling edge %d" % (ix)
                
            #get samples for this edge
            sample_owned,sample_unowned = self.sampler.sample(edge, bordered)
            
            if(self.verbose > 1):
                #alternative flags: cv2.WINDOW_NORMAL | 0x00000010
                disp = di.draw_edge_samples(sample_owned,np.fliplr(sample_unowned),self.display_scale_factor)
                ch = self.imshow(edge_sample_window_name,disp, wait=True)
                if(ch == ascii.CODE.ESC):
                    break
            ix+=1
            edge.samples=[sample_owned,sample_unowned]
        
        if(self.verbose > 1 and ch == ascii.CODE.ESC):
            self._termination_requested = True        

    def store_result(self, data):
        if self.save_to_edge:
            expr = lambda image, edge_ix: image.edges[edge_ix].samples[0]
            self._store_edge_property(data, core.COLUMNS.SAMPLE_OBJECT, expr, dtype=np.object)
            expr = lambda image, edge_ix: image.edges[edge_ix].samples[1]
            self._store_edge_property(data, core.COLUMNS.SAMPLE_BG, expr, dtype=np.object)
        else:   
            sample_range = self.sample_range
            #count up total samples
            sample_count = 0
            for image in self.images:
                for edge in image.edges:
                    sample_count += edge.samples[0].shape[0] / sample_range
            sample_count *= 2
            #generate dataframe columns separately
            image_num = np.zeros((sample_count,),dtype=np.int32)
            edge_ix = np.zeros((sample_count,),dtype=np.int32)
            is_foreground = np.zeros((sample_count,),dtype=np.bool)
            raster = np.zeros((sample_count,),dtype=np.object)
            ix_sample = 0
            ix_edge = 0
            for image in self.images:
                for edge in image.edges:
                    foreground_samples = edge.samples[0]
                    background_samples = edge.samples[1]
                    n_samples = foreground_samples.shape[0] / sample_range
                    foreground_samples = np.split(foreground_samples, n_samples, axis=0)
                    background_samples = np.split(background_samples, n_samples, axis=0)
                    for i_sample in xrange(0,n_samples):
                        fg_sample = foreground_samples[i_sample]
                        bg_sample = background_samples[i_sample]
                        image_num[ix_sample] = image.number
                        edge_ix[ix_sample] = ix_edge
                        is_foreground[ix_sample] = True
                        raster[ix_sample] = fg_sample
                        ix_sample += 1
                        image_num[ix_sample] = image.number
                        edge_ix[ix_sample] = ix_edge
                        is_foreground[ix_sample] = False
                        raster[ix_sample] = bg_sample
                        ix_sample += 1
                    ix_edge += 1 
            samples = pd.DataFrame(image_num, columns=["image_num"])
            samples["edge_ix"] = edge_ix
            samples["is_foreground"] = is_foreground
            samples["raster"] = raster
            if(os.path.isfile(self.save_path)):
                ans = raw_input("Output file exists. overwrite (Y/n)? ").lower()
                answers = ["yes","y","no","n"]
                while ans not in answers:
                    ans = raw_input("Come again (Y/n)? ").lower()
                if ans == "no" or ans == "n":
                    return
            samples.to_pickle(self.save_path)
        
parser = argparse.ArgumentParser(description="Sampling Tool\n A tool that retrieves the pixels around each edge in accordance to the specified sampling mode.")
parser.add_argument("--sampling_source","-ss",default="images/1/photos",
                    help="path to images to use as sampling source")
parser.add_argument("--data_path","-pe",default="features/1/man_edges/edges.pd",
                    help="path to the file with edge and belonging data")
parser.add_argument("--output","-o",default="features/1/man_edges/samples.pd",
                    help="path to the file where to save the edge data")
parser.add_argument("--step","-st",type=int,default=1,
                    help="step at which to gather samples (1 for continuous)")
parser.add_argument("--mode","-m",default="caterpillar",
                    metavar="SAMPLING_MODE",
                    choices=SampleTool.mode_hash.keys(),
                    help="sampling mode, can be one of %s" % str(SampleTool.mode_hash.keys()) )
parser.add_argument("--display_scaling","-ds",type=int,default=1,
                    help="integer scaling factor used to display the samples at higher verbosity levels")
parser.add_argument("--sample_range","-sr",type=int,default=16,
                    help="range of sampling")
parser.add_argument("--column_suffix","-cs",default="",
                    help="Suffix to append to the column. Do not worry about putting mode here!")
parser.add_argument("--blur_offset","-bo",type=int,default=3,
                    help="offset for the blur")
parser.add_argument("--start","-s",type=int,default=0,
                    help="number of the first image to be loaded")
parser.add_argument("--end","-e",type=int,default=sys.maxint,
                    help="number of the last image to be loaded")
parser.add_argument("--verbose","-v",type=int,default=0,help="verbosity level")
parser.add_argument("--no-save","-ns",action="store_true",default=False,
                    help="skip storing results")
parser.add_argument("--save-to-edge","-se",action="store_true",default=False,
                    help="save to the orinal edge dataset instead of creating a new sample dataset")

        
