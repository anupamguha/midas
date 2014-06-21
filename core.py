'''
Created on Jul 27, 2013

@author: Algomorph
'''
import abc
import sys
import os
import cv2
import numpy as np
import pandas as pd
import utils.data_management as dm
import utils.ascii as ascii
import math

from utils.helpers import isnone
from utils.enum import enum
from bw.edge import Edge

COLUMNS = enum(
               IMAGE_NUM = "image_num",
               EDGE_IX = "edge_ix",
               CHAIN = "chain",
               LABEL = "label",
               SAMPLE_OBJECT = "sample_object",
               SAMPLE_BG = "sample_bg",
               SAMPLE_PAIR_COUNT = "sample_counts"
               )

WINDOW_NORMAL = 0
if(hasattr(cv2, "CV_WINDOW_NORMAL")):
    WINDOW_NORMAL = cv2.CV_WINDOW_NORMAL
if(hasattr(cv2, "WINDOW_NORMAL")):
    WINDOW_NORMAL = cv2.WINDOW_NORMAL

WINDOW_AUTOSIZE = 0
if(hasattr(cv2, "CV_WINDOW_AUTOSIZE")):
    WINDOW_AUTOSIZE = cv2.CV_WINDOW_AUTOSIZE
if(hasattr(cv2, "WINDOW_NORMAL")):
    WINDOW_AUTOSIZE = cv2.WINDOW_AUTOSIZE
 

class Image:
    def __init__(self, raster, number):
        self.raster = raster
        self.number = number
        self.edges = []
        

class AbstractEdgeTool:
    '''
    A class for an abstract tool
    '''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,
                 images_dir,
                 data_path,
                 start = -1, 
                 end = sys.maxint,
                 verbose = 0, 
                 load_images = True,
                 ImageClass = Image,
                 EdgeClass = Edge,
                 save_result = True,
                 load_samples = False):
        '''
        Constructor
        @param images_dir: where to load the images from. Every file should contain a unique number in its name.
        @param data_path: where to load/save the data from
        @param start: beginning of the number range of which images to load
        @param end: end of the number range of which images to load
        @param load_images: whether to load images at all
        @param ImageClass: class to use when constructing the images; should have an initializer mimiking that of the {@link Image} class
        '''
        self._windows_used = False
        self._store_subset = False
        self.data = None
        self._images_dir = images_dir
        self._data_path = data_path
        if not os.path.isdir(self._images_dir):
            if((os.path.isfile(self._images_dir) 
               or os.path.islink(self._images_dir))
               and load_images):
                print "Error: data path cannot be an existing file or link. It should either be an existing directory or not exist at all."
            os.makedirs(self._images_dir)
        self._start_image_num = start
        self._end_image_num = end
        self.images = []
        self.image_by_number = {}
        self.verbose = verbose
        self._ImageClass = ImageClass
        self._EdgeClass = EdgeClass
        self._load_images(self._images_dir,
                          self._start_image_num,
                          self._end_image_num,
                          load_images)
        self._load_data(self._data_path)
        self._window_names = {}
        self._termination_requested = False
        self._save_result = save_result
        self._load_samples = load_samples
        
        
    @property
    def data_loaded(self):
        return type(self.data) != type(None)
    
    @property
    def used_cv2_windows(self):
        '''
        Whether any cv2 windows were used.
        '''
        return self._windows_used
    
    '''
    TODO: figure out whether any of this code can actually be used
    def _gen_sample_df(self,edge_df,store_raster):
        if not (COLUMNS.SAMPLE_OBJECT in edge_df.columns and
                COLUMNS.SAMPLE_BG in edge_df.columns):
            raise ValueError("Input dataframe should contain columns %s and %s." 
                             % (COLUMNS.SAMPLE_OBJECT, COLUMNS.SAMPLE_BG))
        sorted_edge_data = edge_df[[COLUMNS.IMAGE_NUM,COLUMNS.EDGE_IX,
                                    COLUMNS.LABEL,COLUMNS.SAMPLE_OBJECT,COLUMNS.SAMPLE_BG]
                                   ].sort(columns=[COLUMNS.IMAGE_NUM, COLUMNS.EDGE_IX])
        if COLUMNS.SAMPLE_PAIR_COUNT in edge_df.columns:
            n_samples = edge_df[COLUMNS.SAMPLE_PAIR_COUNT].sum()
        else:
            n_samples = 0
            
        for image_num, edge_ix, label, sample_owned, sample_unowned in sorted_edge_data.iterrows():
            pass
    ''' 
        
    def imshow(self,window_name,image,flags = None, wait=True, save_on_s=True):
        '''
        @summary: shows the given image in a named window
        If None is passed as the image argument, simply creates the window.
        @param image: the image to show in he window
        @param window_name: the name of the window to use
        @param flags: the flags to use during window creation (if any)
        @param wait: whether to wait for the user to press a key before returning
        @param save_on_s: whether to allow the user to save the image in the current
                          directory under filename equivalent to window_name by
                          pressing the "s" key on the keyboard (only works if wait=True)
        @return: if wait is set to True, returns the character ascii code produced
                 by the user's keystroke. Otherwise, returns 0.
        '''
        if(flags):
            cv2.namedWindow(window_name,flags)
        else:
            cv2.namedWindow(window_name)
        
        if not isnone(image):
            cv2.imshow(window_name,image)
        self._windows_used = True
        self._window_names[window_name] = True
        if wait and not isnone(image):
            ch = 0xFF & cv2.waitKey()
            if save_on_s and (ch == ascii.CODE.s and ch == ascii.CODE.S):
                cv2.imwrite(window_name+".png", image)
            return ch
        else:
            return 0

    def __del__(self):
        if(self._windows_used):
            for window_name in self._window_names.keys():
                cv2.destroyWindow(window_name)
                
    def run(self):
        '''
        Issued when the tool is run on all input
        '''
        self._termination_requested = False
        self._repeat_requested = False
        self._go_back_requested = False
        ix_image = 0
        #traverse all images
        while(ix_image < len(self.images)):
            image = self.images[ix_image]
            if(self.verbose > 0):
                print "Processing image %d" % image.number
            #process the image
            self.process_image(image)
            
            #check for termination or requests to go to previous image
            if self._termination_requested:
                break;
            if self._go_back_requested:
                ix_image = max(ix_image - 1, 0)
                self._go_back_requested = False
            #check whether a repeat is requested
            elif not self._repeat_requested:
                ix_image += 1
            else:
                self._repeat_requested = False
                 
        if not self._termination_requested:
            if(self._save_result):
                if(self.verbose > 0):
                    print "Storing results."
                #save result unless specified not to
                if not self.data_loaded:
                    self.data = self._generate_initial_edge_df()
                
                self.store_result(self.data)
                self.data.to_pickle(self._data_path)
            else:
                if(self.verbose > 0):
                    print "Skipping result save."
    @abc.abstractmethod
    def process_image(self,image):
        pass
    
    @abc.abstractmethod
    def store_result(self, data):
        pass
            
    def _load_images(self,path_to_images,start_num,end_num,load_rasters):
        '''
        Loads all the images at the given path
        '''
        if(self.verbose > 0):
            print "Loading images."
        im_names = dm.get_raster_names(path_to_images)
        filtered_names = []
        filtered_numbers = []
        for name in im_names:
            num = dm.parse_num_from_name(name)
            if(num <= end_num and num >= start_num):
                filtered_names.append(name)
                filtered_numbers.append(num)
        
        if load_rasters:
            rasters, numbers = dm.load_numbered_rasters_from_dir(path_to_images, 
                                                                 names=filtered_names,
                                                                 numbers=filtered_numbers,
                                                                 verbose = self.verbose - 1)
        else:
            numbers = filtered_numbers
            rasters = [None]*len(numbers)
            
        for ixImage in xrange(0,len(rasters)):
            raster = rasters[ixImage]
            number = numbers[ixImage]
            image = self._ImageClass(raster,number)
            self.images.append(image)
            self.image_by_number[number] = image
            
    @property
    def edge_count(self):
        count = 0
        for image in self.images:
            count += len(image.edges)
        return count
            
    @property
    def image_numbers(self):
        """Numbers of all the images currently being used"""
        numbers = self.image_by_number.keys()
        return np.sort(numbers)
            
    def _generate_initial_edge_df(self):
        '''
        Generates a data frame out of the existing images and their edges.
        @return a data frame with two columns, one representing the number of the image and one - the index of the edge for that image.
        '''
        edge_count = self.edge_count
        if(edge_count == 0):
            #terminate if there are no edges
            return None
        init_data = np.zeros((edge_count,2), dtype=np.uint16)
        cum_ix_edge = 0
        for image in self.images:
            ix_edge = 0
            for edge in image.edges:
                init_data[cum_ix_edge,0] = image.number
                init_data[cum_ix_edge,1] = ix_edge
                cum_ix_edge += 1
                ix_edge += 1
        data = pd.DataFrame(data=init_data, columns=[COLUMNS.IMAGE_NUM,COLUMNS.EDGE_IX])
        return data
    
    def _store_edge_property(self, data, column_name, expression, dtype = np.float32):
        '''
        Stores a single edge property at the specified column in the dataframe
        '''
        if(self._store_subset):
            if not column_name in data.columns:
                #the column doesn't exist yet for the whole data. Make one and fill it with blanks first.
                blank_series = np.empty((data.index.size),dtype=dtype)
                data[column_name] = blank_series    
            subset = data.ix[(data[COLUMNS.IMAGE_NUM] >= self._start_image_num) 
                         & (data[COLUMNS.IMAGE_NUM] <= self._end_image_num)]
        else:
            #we're storing the output for the entire set of images already in data
            subset = data
        
        
        local_edge_count = self.edge_count
        if(len(subset) != local_edge_count):
            raise ValueError("Images in the data file seem not to have the same number of edges (%d) as gleaned from the current images (%d)." % 
                             (len(subset), local_edge_count))
        #column = np.empty((local_edge_count),dtype=dtype)
        cum_ix_edge = 0
        for image in self.images:
            ix_edge = 0
            for edge in image.edges:
                data.loc[cum_ix_edge,column_name] = expression(image,ix_edge)
                cum_ix_edge += 1
                ix_edge += 1
        #subset[column_name] = column
        
        #if(self._store_subset):
        #    #we have to store only a subset of the data, since we didn't load all images
        #    data[column_name].ix[subset.index[0]:subset.index[len(subset.index)-1]] = subset[column_name]

    def _image_numbers_correspond(self,data):
        '''
        checks whether the image numbers in the given dataset correspond 
        to the image numbers of the images on-hand
        @return true if the image numbers fully match, false otherwise
        '''
        data_im_numbers = np.sort(np.unique(data[COLUMNS.IMAGE_NUM].values))
        return np.array_equal(data_im_numbers, self.image_numbers)
    
    def _load_data(self,path_to_data):
        self.data = dm.load_data_from_path(path_to_data, create_if_missing = False)
        data = self.data
        if(self.data_loaded and self.images and len(self.images) > 0):
            #try loading existing edge info
            if(COLUMNS.EDGE_IX in data.columns and COLUMNS.IMAGE_NUM in data.columns):
                if not self._image_numbers_correspond(data):
                    self._store_subset = True
                    
                if(COLUMNS.CHAIN in data.columns):
                    essential_column_list = [COLUMNS.IMAGE_NUM,
                                             COLUMNS.EDGE_IX,
                                             COLUMNS.CHAIN,
                                             COLUMNS.LABEL,
                                             COLUMNS.SAMPLE_OBJECT,
                                             COLUMNS.SAMPLE_BG,
                                             COLUMNS.SAMPLE_PAIR_COUNT]
                    
                    
                    columns_present = list(data.columns)
                    essent_cols_present = list(np.intersect1d(essential_column_list, 
                                                         columns_present, True)) 
                    
                    ix_chain = essent_cols_present.index(COLUMNS.CHAIN)
                    ix_image_num = essent_cols_present.index(COLUMNS.IMAGE_NUM)
                    ix_label = -1
                    ix_owned = -1
                    ix_unowned = -1
                    ix_pair_ct = -1
                    if(COLUMNS.LABEL in essent_cols_present):
                        ix_label = essent_cols_present.index(COLUMNS.LABEL)
                    if(COLUMNS.SAMPLE_OBJECT in essent_cols_present):
                        ix_owned = essent_cols_present.index(COLUMNS.SAMPLE_OBJECT)
                    if(COLUMNS.SAMPLE_BG in essent_cols_present):
                        ix_unowned = essent_cols_present.index(COLUMNS.SAMPLE_BG)
                    if(COLUMNS.SAMPLE_PAIR_COUNT in essent_cols_present):
                        ix_pair_ct = essent_cols_present.index(COLUMNS.SAMPLE_PAIR_COUNT)
                    
                    sorted_edge_data=data[essent_cols_present].sort(columns=[COLUMNS.IMAGE_NUM,
                                                                               COLUMNS.EDGE_IX])
                    last_image_num = -1
                    cur_image = None #no negative image numbers allowed                        
                    for row_ix, row in sorted_edge_data.iterrows():
                        image_num = row.ix[ix_image_num]
                        chain = row.ix[ix_chain]
                        if(image_num != last_image_num):
                            #if image number not in ones currently loaded, skip loading
                            if not image_num in self.image_by_number:
                                continue
                            #fetches the next image if the image number changes
                            cur_image = self.image_by_number[image_num]
                            cur_image.edges = []#reset image edges
                            last_image_num = cur_image.number
                        edge = self._EdgeClass(chain)
                        if(ix_label != -1):
                            edge.label = row.ix[ix_label]
                        if(ix_owned != -1 and ix_unowned != -1):
                            edge.samples = [row.ix[ix_owned],row.ix[ix_unowned]]
                        if(ix_pair_ct != -1):
                            edge.sample_pair_count = ix_pair_ct
                        cur_image.edges.append(edge)
