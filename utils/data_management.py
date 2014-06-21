'''
Created on Jun 11, 2013

@author: Gregory Kramida
'''
import cv2
import os
import re
import numpy as np
import pandas as pd
from utils.helpers import isnone

IMREAD_UNCHANGED = -1;#it usually had the value -1
if(hasattr(cv2,'IMREAD_UNCHANGED')):
    IMREAD_UNCHANGED = getattr(cv2,'IMREAD_UNCHANGED')
elif(hasattr(cv2,'CV_LOAD_IMAGE_UNCHANGED')):
    IMREAD_UNCHANGED = getattr(cv2,'CV_LOAD_IMAGE_UNCHANGED')

def parse_num_from_name(name):
    num_match = re.search("\d+", name)
    if not num_match:
        return None
    return int(num_match.group(0))

def get_raster_names(directory):
    names = os.listdir(directory)
    names.sort()
    raster_names = []
    for name in names:
        if(re.search(r'[^.]*.png',name)):
            raster_names.append(name)
    return raster_names;
    

def load_rasters_from_dir(directory, names = None):
    rasters = []
    if(not names):
        names = get_raster_names(directory)
        
    for name in names:
        rasters.append(cv2.imread(directory + os.path.sep + name,IMREAD_UNCHANGED))
    return rasters

def load_numbered_rasters_from_dir(directory, names = None, numbers = None, verbose = 0):
    images = []
    
    if(not names):
        names = get_raster_names(directory);
    ix_name = 0
    
    if(isnone(numbers)):
        have_numbers = False
        numbers = []
    else:
        have_numbers=True 
    
    for name in names:
        if(have_numbers):
            num = numbers[ix_name]
        else:
            num = parse_num_from_name(name)
            numbers.append(num)
        if(verbose > 0):
            print "Loading raster %d" % num
        images.append(cv2.imread(directory + os.path.sep + name,IMREAD_UNCHANGED))
        
        ix_name += 1
    return images, numbers

def save_rasters_to_dir(directory, images, names=None):
    if(not names):
        names = get_raster_names(directory)
    for ix in xrange(0,len(names)):
        cv2.imwrite(directory + os.path.sep + names[ix],images[ix])

def load_data_from_path(path, create_if_missing = True):
    '''
    Loads a pandas dataframe from the given path or creates a new one if 
    non exist and the corresponding argument is given
    @param path: path to the pandas data frame file to load
    @param create_if_missing: creates a new data frame instead of loading if the file is missing
    @return: a new dataframe or None if the file is missing, the existing one if it's not
    '''
    if os.path.isdir(path):
        raise ValueError("The given path, '%s', is a folder/directory. Please, specify path to a pandas dataframe file instead." % path)
    df = None
    if os.path.isfile(path):
        df = pd.read_pickle(path)
    elif create_if_missing:
        df = pd.DataFrame();
    return df

def split_data_for_training(dataAndLabels, training_ratio = 0.7, 
                         validation_ratio = 0.15, 
                         labelColumn='label', excludeCols = [], shuffle = True):
    
    #check ratio validity
    if (training_ratio <= 0 or validation_ratio <= 0 or 
        training_ratio+validation_ratio >= 1.0):
        raise ValueError("""Invalid ratio set for training & validation: %(f) and %(f).
                         They should be positive and their sum should be between 0 and 1.0.""")
    if isinstance(dataAndLabels,pd.DataFrame):
        #exclude extra columns
        excludeCols.append(labelColumn)
        dataCols = np.setdiff1d(dataAndLabels.columns,excludeCols,True)
        if shuffle:
            dataAndLabels.reindex(np.random.permutation(dataAndLabels.index))
        #split into data/labels
        data = dataAndLabels[dataCols].values
        labels = dataAndLabels[labelColumn].values
    else:
        #assume an indexable structure (e.g. tuple or array) with (data,labels) 
        data = dataAndLabels[0]
        labels = dataAndLabels[1]
        if shuffle:
            idxs = np.random.permutation(np.arange(len(labels)))
            data = data[idxs]
            labels = labels[idxs]
    
    #derive counts
    train_count = len(data) * training_ratio
    val_count = len(data) * validation_ratio
    test_offset = train_count + val_count
    #split into training/validation/testing
    train_data = data[0:train_count]
    train_labels = labels[0:train_count]
    val_data = data[train_count:test_offset]
    val_labels = labels[train_count:test_offset]
    test_data = data[test_offset:]
    test_labels = labels[test_offset:]
    return train_data,train_labels,val_data,val_labels,test_data,test_labels