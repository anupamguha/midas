'''
Created on Feb 2, 2013

@author: Gregory Kramida
'''
import cv2
import sys
import utils.ascii as ascii
from bw.label import *
import numpy as np
import utils.display as disp
import pickle as pk
import argparse
import core
from utils.helpers import isnone

LEFT = BELONGING_LABELS.LEFT
RIGHT = BELONGING_LABELS.RIGHT
BOTH = BELONGING_LABELS.BOTH
NONE = BELONGING_LABELS.NONE
UNKNOWN = BELONGING_LABELS.UNKNOWN
sides = [LEFT,RIGHT]

class LabelTool(core.AbstractEdgeTool):
    op_name = 'label'
    
    '''
    A tool for labeling edge belonging
    '''
    def __init__(self,photos,data_path,
                 start = -1, end = sys.maxint, 
                 verbose = 0, no_save = False,
                 resizable_windows = False):
        '''
        Constructor
        '''
        super(LabelTool,self).__init__(photos,
                                       data_path,
                                       start=start,
                                       end=end,
                                       verbose=verbose,
                                       save_result=not no_save)
        if(self.data is None):
            raise ValueError("Could not load data file. Please make sure the file exists at " + data_path + ", and that there is permission to access it.")
            
        self._resizable_windows = resizable_windows
        self.cur_edge_side_ix = 0
        self._start_from_last = False
        self.cur_image = None
     
    def current_edge_arrow_raster(self):
        if (self.cur_edge_side_ix >= len(self.cur_image.edges) << 1):
            return None
        raster = np.copy(self.cur_image.raster)
        edge = self.cur_image.edges[self.cur_edge_side_ix >> 1]
        side = sides[self.cur_edge_side_ix % 2]
        edge.draw(raster)
        edge.draw_side_arrow(side, raster)
        return raster
    
    def log_belonging(self, val):
        edge = self.cur_image.edges[self.cur_edge_side_ix >> 1]
        if(val == None):
            del edge.label
        elif(self.cur_edge_side_ix % 2 == 0):
            edge.belongs_left = val
        else:
            edge.belongs_right = val
    
    def gen_labeled_image(self):
        disp = np.copy(self.cur_image.raster)
        for edge in self.cur_image.edges:
            color = BELONGING_COLOR_BY_LABEL[edge.label]
            edge.draw(disp,color)
            if edge.label == BOTH:
                edge.draw_side_arrow(LEFT,disp)
                edge.draw_side_arrow(RIGHT,disp)
            elif edge.label == LEFT:
                edge.draw_side_arrow(LEFT,disp)
            elif edge.label == RIGHT:
                edge.draw_side_arrow(RIGHT,disp)
        return disp
    
    def store_result(self, data):
        expr = lambda image, edge_ix: image.edges[edge_ix].label
        #store chains
        self._store_edge_property(data, core.COLUMNS.LABEL, expr, dtype=np.str)
        
        
    def process_image(self, image):
        ch = 0xFF
        win_name = "Label Tool"
        flags = None 
        if (self._resizable_windows):
            flags = 2
        self.cur_image = image
        if self._start_from_last:
            #happens on going to previous edge
            self._start_from_last = False
            self.cur_edge_side_ix = (len(image.edges) << 1) - 1
        else:
            self.cur_edge_side_ix = 0;
        disp = self.current_edge_arrow_raster()
        while(not isnone(disp) and not self._go_back_requested):
            ch = self.imshow(win_name, disp,flags = flags)
            if(self.verbose > 1):
                print "Key pressed: %d" % ch
            if(ch == ascii.CODE.ESC):
                #escape means "terminate the labeler"
                break;
            if(ch == ascii.CODE.e or ch == ascii.CODE.y):
                #e or y means "yes"
                self.log_belonging(True)
                self.cur_edge_side_ix+=1
            elif (ch == ascii.CODE.r or ch == ascii.CODE.n):
                #r or n means "no"
                self.log_belonging(False)
                self.cur_edge_side_ix+=1
            elif (ch == ascii.CODE.u or ch == ascii.CODE.w):
                #u or w means "mark as unknown"
                self.log_belonging(None)
            elif (ch == ascii.CODE.SPACEBAR):
                #space means "skip edge side"
                self.cur_edge_side_ix +=1 #((self.cur_edge_side_ix >> 1) << 1) + 2
            elif (ch == 83 or ch == ascii.CODE.NUMPAD_6):
                #right arrow means "skip (rest of) the current image"
                break
            elif (ch == 81  or ch == ascii.CODE.NUMPAD_4):
                #left arrow means "redo the previous image"
                self._go_back_requested = True
            elif (ch == ascii.CODE.s):
                #'s' means "save labeling result"
                self.store_result(self.data)
                self.data.to_pickle(self._data_path)
            elif (ch == 8):
                #backspace means "go back to the previous edge/side"
                self.cur_edge_side_ix-=1
                if(self.cur_edge_side_ix < 0):
                    self._go_back_requested = True
                    self._start_from_last = True 
            if(ch == ascii.CODE.v):
                #v = view current result
                disp = self.gen_labeled_image()
            elif(ch == ascii.CODE.h):
                #h = display saved result
                print "Currently not supported."
            else:
                disp = self.current_edge_arrow_raster()
            
        if(ch == ascii.CODE.v):
            #display current result
            ch = self.imshow(win_name, disp, flags = flags)
        if(ch == ascii.CODE.ESC):
            self._termination_requested = True
        

parser = argparse.ArgumentParser(description='''Label Tool
                                                An interactive visual tool to manually label edge belonging.''')
parser.add_argument("--photos","-p",default="images/2/photos",
                    help="path to original photos")
parser.add_argument("--data_path","-d",default="edges/2/manual/edge.pd",
                    help="path to the processed edges")
parser.add_argument("--start","-s",type=int,default=0,
                    help="number of the first image to be loaded")
parser.add_argument("--end","-e",type=int,default=sys.maxint,
                    help="number of the last image to be loaded")
parser.add_argument("--verbose","-v",type=int,default=0,help="verbosity level")
parser.add_argument("--no-save","-ns",action="store_true",default=False,
                    help="skip storing results")
parser.add_argument("--resizable_windows","-rs",action="store_true",default=False,
                    help="use resizable windows")