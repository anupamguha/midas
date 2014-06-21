'''
Created on Feb 24, 2013

@author: Gregory Kramida
'''
import core
import cv2
import argparse
import sys
import bw.preprocessing as prep
import bw.chain as ch
import numpy as np
import cvex
from utils.enum import enum
from bw.edge import Edge

#add as necessary
BASIC_EDGE_PROP = enum(
                        LENGTH="length"
                        )

class EdgeDetectTool(core.AbstractEdgeTool):
    op_name = 'detect_edges'
    '''
    A tool for preprocessing B/W edge images and extracting the edges from them
    '''
    def __init__(self,
                 images_dir,
                 data_path,
                 length_threshold,
                 start = -1, 
                 end = sys.maxint, 
                 verbose = 0):
        '''
        Constructor
        '''
        super(EdgeDetectTool,self).__init__(images_dir, 
                                      data_path,
                                      start,
                                      end,
                                      verbose=verbose, 
                                      load_images=True)
        self.length_threshold = length_threshold
    

    def process_image(self, image):
        raster = image.raster
        if(len(raster.shape) > 2):
            #assume BGR image, convert to gray
            grey_raster = cv2.cvtColor(raster,cv2.COLOR_BGR2GRAY)
        else:
            grey_raster = raster
        #TODO: insert cvex thinning here
        #threshold for good measure:
        thr_raster = cv2.threshold(grey_raster, 128, 255, cv2.THRESH_BINARY)[1]
        #thin
        bw_raster = prep.filter_out_corners(cvex.thinZhangSuen(thr_raster))
        its = prep.find_intersections(bw_raster)
        if(len(its) > 0):
            raise ValueError("Given image contains intersections at:\n %s" % str(its))
        #make sure there are no corners:
        bw_raster = prep.filter_out_corners(bw_raster)
        #find endpoints and protrusions
        protrusions, endpoints = prep.find_protrusions(bw_raster)
        #remove the protrusions
        prep.remove_points(protrusions, bw_raster)
        #chain
        chain_set = ch.chain(bw_raster, endpoints)
        if(self.verbose > 0):
            print "Discovered %d edges." % (len(chain_set))
        #reset image edges
        image.edges = []
        #simplify each chain and create an edge
        for chain in chain_set:
            simplified_chain = ch.simplify_chain(chain, self.verbose - 1)
            image.edges.append(Edge(simplified_chain))
    
    def store_result(self, data):
        expr = lambda image, edge_ix: image.edges[edge_ix].chain
        #store chains
        self._store_edge_property(data, core.COLUMNS.CHAIN, expr, dtype=np.object)
        expr = lambda image, edge_ix: image.edges[edge_ix].length
        #store edge lengths
        self._store_edge_property(data, BASIC_EDGE_PROP.LENGTH, expr, dtype=np.float64)
            
                        
parser = argparse.ArgumentParser(description = "Edge Tool\n"+
                                 " A tool for extracting strong edges from images, gradients, or edge detector results.")
parser.add_argument("--images_dir","-im",default="images/2/man_edges",
                    help="directory of the source images")
parser.add_argument("--data_path","-d",default="edges/2/manual/edge.pd",
                    help="path to pandas data frame to store results in")
parser.add_argument("--length_threshold","-lt",type=int,default=25,
                    help="minimum length threshold for a single edge's length for the edge to be considered")
parser.add_argument("--start","-s",type=int,default=0,
                    help="number of the first image to be loaded")
parser.add_argument("--end","-e",type=int,default=sys.maxint,
                    help="number of the last image to be loaded")
parser.add_argument("--verbose","-v",type=int,default=0,
                    help="verbosity level")
        
