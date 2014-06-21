import core
import sys
import os
import re
import cv2
import argparse
import numpy as np
import math
import pandas as pd
import scipy.optimize as so
import scipy.stats as st
import scipy.io as sio
import webbrowser as web
from bw.chain import chain
import time

class ExtractTool(core.AbstractEdgeTool):
    op_name = 'extract'
    outputDType = np.float32
    colorConversions = {
                   'BGR2LAB':cv2.COLOR_BGR2LAB,
                   'BGR2LUV':cv2.COLOR_BGR2LUV,
                   'BGR2HSV':cv2.COLOR_BGR2HSV_FULL,
                   'BGR2HLS':cv2.COLOR_BGR2HLS_FULL,
                   'BGR2YUV':cv2.COLOR_BGR2YUV,
                   'BGR2GRAY':cv2.COLOR_BGR2GRAY,
                   'None':None
                   }
    
    '''
    classdocs
    '''
    def __init__(self,
                 edge_images_dir,
                 data_path,
                 mode,
                 feature,
                 feature_postfix,
                 fitting_function,
                 color_conversion,
                 metlab,
                 no_jokes,
                 flip_right,
                 multiprocessing,
                 start = 0, 
                 end = sys.maxint, 
                 verbose = 0,
                 no_save = False):
        super(ExtractTool,self).__init__(images_dir=edge_images_dir,
                                         data_path=data_path,
                                         start=start,
                                         end=end, 
                                         verbose=verbose,
                                         load_images=False,
                                         save_result=not no_save)
        
        self.fitFunc = ExtractTool.string_to_lambda(fitting_function)
        self.feature = feature
        self.mode = mode
        self.multiprocessing = multiprocessing
        self.matlab = metlab
        self.jokes = not no_jokes
        self.flipRight = flip_right
        self.featurePostfix = feature_postfix
        self.cvtColor = ExtractTool.colorConversions[color_conversion]
    
    def extractCount(self,sample):
        kptBank = chain(sample)
        return float(len(kptBank)) / sample.shape[0]
    
    def extractStupidity(self,sample):
        #we don't need to extract stupidity, we already have that
        return 1000000000
    
    def extractKeypoints(self,sample):
        kptBank = chain(sample)
        combKptCount = 0
        for kpts in kptBank:
            combKptCount += len(kpts) 
        return combKptCount / sample.shape[0]
    
    def extractExpFit(self,sample):
        samplingRange = sample.shape[1]
        sampleLen = sample.shape[0]
        
        sampleFl = sample.astype(np.float32)
        totals = np.array([0.0,0.0,0.0])
        
        for iPerpProfile in range(0,sample.shape[0]):
            profile = sampleFl[iPerpProfile,:]
            if(len(profile.shape) == 2):
                for iChannel in range(0,3):
                    y = np.log(profile[:,iChannel])
                    cor = -st.pearsonr(np.arange(0,samplingRange),y)[0]
                    if(math.isnan(cor)):
                        cor = 0.0
                    totals[iChannel] += cor
            else:
                y = np.log(profile)
                cor = -st.pearsonr(np.arange(0,samplingRange),y)[0]
                if(math.isnan(cor)):
                    cor = 0.0 
                totals[0] += cor
        totals /= sampleLen    
        
        if(len(sample.shape) == 3):
            return np.array(totals,dtype=ExtractTool.outputDType)
        else:
            return ExtractTool.outputDType(totals[0])
        
    def extractRowFit(self, sample):
        return self.extractRowFitAndConsistency(sample)

    def extractRowConsistency(self,sample):
        return self.extractRowFitAndConsistency(sample, True)
    
    def extractSampleGradientDirection(self, sample):
        #average of gradient direction for every pixel per sample      
        sampleFl = sample.astype(np.float32)
        gradarr = np.gradient(sampleFl)
        dy = np.asarray(gradarr[0]).reshape(-1)
        dx = np.asarray(gradarr[1]).reshape(-1)
        dirs = [np.arctan2(y,x) for y,x in zip(dy,dx)]
        avgdir = sum(dir)/float(len(dirs))
        return avgdir
        
        
        
    
    def extractRowFitAndConsistency(self, sample, consistency=False):
        samplingRange = sample.shape[0]
        sampleLen = sample.shape[1] 
        sampleFl = sample.astype(np.float32)
        
        fitFunc = self.fitFunc
        pixelIndices = np.arange(-samplingRange+1,1)
           
        profileIndices = np.arange(sampleLen)
        
        if(len(sample.shape) == 3):    
            coeffs = np.empty((sampleLen,3,2),dtype=np.float32)
            var = np.empty((sampleLen,3,2),dtype=np.float32)
            for iPerpProfile in range(0,sample.shape[0]):
                profile = sampleFl[iPerpProfile,:]
                if(len(profile.shape) == 2):
                    for iChannel in range(0,3):
                        y = profile[:,iChannel]
                        x = pixelIndices
                        coeffs[iPerpProfile,iChannel],pcov = so.curve_fit(fitFunc, x, y)
                        var[iPerpProfile,iChannel] = (pcov[0,0],pcov[1,1])

            if(consistency):
                cor0a = st.pearsonr(profileIndices,coeffs[:,0,0])[0]
                cor0b = st.pearsonr(profileIndices,coeffs[:,0,1])[0]
                
                cor1a = st.pearsonr(profileIndices,coeffs[:,1,0])[0]
                cor1b = st.pearsonr(profileIndices,coeffs[:,1,1])[0]
                
                cor2a = st.pearsonr(profileIndices,coeffs[:,2,0])[0]
                cor2b = st.pearsonr(profileIndices,coeffs[:,2,1])[0]
            
                return np.abs(np.array([cor0a, cor0b, cor1a, cor1b, cor2a, cor2b],dtype=ExtractTool.outputDType))
            else:
                meanVar = var.mean(axis=0)
                return np.array([meanVar[0,0],meanVar[0,1],
                                 meanVar[1,0],meanVar[1,1],
                                 meanVar[2,0],meanVar[2,1]],dtype=ExtractTool.outputDType)
        else:
            coeffs = np.empty((sampleLen,2))
            var = np.empty((sampleLen,2))
            for iPerpProfile in range(0,sample.shape[0]):
                profile = sampleFl[iPerpProfile]
                y = profile
                x = pixelIndices
                coeffs[iPerpProfile],pcov = so.curve_fit(fitFunc, x, y)
                var[iPerpProfile] = (pcov[0,0],pcov[1,1])
            if(consistency):#TODO: make consistency also use the so.curve_fit function
                cora = st.pearsonr(np.arange(0,sampleLen),coeffs[:,0])[0]
                corb = st.pearsonr(np.arange(0,sampleLen),coeffs[:,1])[0]
                return np.abs(np.array[cora,corb],dtype=ExtractTool.outputDType)
            else:
                meanVar = var.mean(axis=0)
                return np.array([meanVar[0],meanVar[1]],ExtractTool.outputDType)
            
    def extractSampleFit(self,sample):
        return self.extr_samp_fit_complete(sample)
    
    def extractSampleCoeffs(self,sample):
        return self.extr_samp_fit_complete(sample, True)
    
    def extractSampleFitAndCoeffs1Channel(self,sample,fitFunc,extrCoeffs=False):
        sampleLen = sample.shape[1]
        sampleFlRow = sample.astype(np.float32).reshape(-1)
        indices = np.tile(np.arange(sampleLen),sampleLen)
        #coefficients
        coeffs,pcov = so.curve_fit(fitFunc,indices,sampleFlRow)
        if(extrCoeffs):
            return coeffs.reshape(-1)
        else:
            #number of entries along the diagonal
            numCoeffs = pcov.shape[0]
            variances = []
            for iCoeff in range(numCoeffs):
                #collect variances from the diagonal of the covariance
                variances.append(pcov[iCoeff,iCoeff])
            return np.array(variances,dtype=np.float32)
        
    @staticmethod
    def string_to_lambda(strFunc):
        '''
        @summary: Converts a string representation of a multivariate function into a lambda function.
        The function can only contain numpy mathematical operators or functions,
        _^_ as alternative for _ to the power of _, e^(_) or e^_ for np.exp(_), and
        single-letter variables containing numpy arrays. A lowercase 'x' must be present, and will
        be used as the first parameter. 
        @param strFunc: the string representation to convert
        @return: a lambda expression from the string function representation
        '''
        if(strFunc.find("x") == -1):
            raise ValueError("function string should contain \"x\"")
        paramRegex = re.compile(r"[a-df-wy-z]",re.I)
        paramChars = re.findall(paramRegex, strFunc)
        expRegex = re.compile(r"e(?:\^|\*\*)(?:\s*|\()((?<!\()(?:[a-z]|-?\d.?\d*)|(?:[^()]|[^()]*\([^()]*\)[^()]*)*(?=\)))",re.I)
        eRegex = re.compile(r"(?<!\.)e(?!x)")
        #replace e^_, e**_ with np.exp(_), 
        #where _ can be either a single digit/letter or a parenthetic expression
        #(a maximum of depth one nested parenthesis set is allowed inside the parenthetic expresssion).
        #Also, replace the remaining e-s with math.e and "^" with **
        strFunc = re.sub(eRegex,"math.e",re.sub(expRegex,r"np.exp(\1)",strFunc)).replace("^","**")
        evalStr =  "lambda x, " + ", ".join(paramChars) + ": " + strFunc
        return eval(evalStr)
        
        
        
    def extr_samp_fit_complete(self,sample,extrCoeffs=False):
        fitFunc = self.fitFunc
        if(len(sample.shape) == 3):
            results = []
            for iChannel in range(3):
                #calculate result for each channel separately 
                results.append(self.extractSampleFitAndCoeffs1Channel(sample[:,:,iChannel],
                                                                      fitFunc,extrCoeffs=extrCoeffs))
            #flatten and return
            return np.array(results,dtype=np.float32).reshape(-1)
        else:
            #calculate result for the only channel
            return self.extractSampleFitAndCoeffs1Channel(sample,fitFunc,extrCoeffs=extrCoeffs)
    
     
    def extr_samp_mean(self,sample):
        return sample.mean(axis=(0,1)).astype(ExtractTool.outputDType);
    
    def extr_samp_std(self,sample):
        return sample.std(axis=(0,1));
    
    def extr_length_std(self,sample):
        '''
        @summary: extracts length-wise standard deviation
        '''
        return sample.std(axis=0).reshape(-1).astype(ExtractTool.outputDType)
            
    
    featureTypesHash = {
                    'exp_fit': extractExpFit,
                    'samp_fit':extractSampleFit,
                    'samp_coeff':extractSampleCoeffs,
                    'row_fit': extractRowFit,
                    'row_cons': extractRowConsistency,
                    'length_std': extr_length_std,
                    'samp_mean': extr_samp_mean,
                    'count':extractCount,
                    'keypoints':extractKeypoints,
                    'grad_dir':extractSampleGradientDirection,
    }
    
    def edge_extract(self, feature_func, sampleL, sampleR, label = 1):
        return [label], [feature_func(self,sampleL) - feature_func(self,sampleR)]
    
    def side_extract(self, feature_func, sampleL, sampleR, labels = None):
        if labels == None:
            #default labels assume left sample always "belongs"
            labels = [1,0]
        return labels, [feature_func(self,sampleL),feature_func(self,sampleR)]
    
    def block_extract(self, feature_func, sampleL, sampleR, labels = None):
        #process each block individually
        blockResults = []
        blockWidth = sampleL.shape[1]
        side_block_ct = sampleL.shape[0] / blockWidth
        blockCount = side_block_ct << 1
        if(labels == None):
            # default labels assume left block always  "belongs",
            # and this is the order blocks will be processed: 
            # 1, 2,
            # 3, 4 ...
            # hence, the order 1,0,1,0...
            labels = [(i+1) % 2 for i in range(0,blockCount)]
        curLy = 0
        curRy = 0
        nextLy = blockWidth 
        nextRy = blockWidth
        for ix_block_row in xrange(0,side_block_ct):
            left_block = sampleL[curLy:nextLy]
            right_block = sampleR[curRy:nextRy]
            blockResults.append(feature_func(self,left_block))
            blockResults.append(feature_func(self,right_block))    
            curLy = nextLy
            curRy = nextRy
            nextLy += blockWidth
            nextRy += blockWidth
        return labels, blockResults
        
    extractModesHash = {
                 'edge': edge_extract,
                 'side': side_extract,
                 'soviet_tetris': block_extract                 
    }
    
    def store_result(self, data):
        expr = lambda image, edge_ix: image.edges[edge_ix].features[0]
        pass
    
    def process_image(self,image):
        '''
        Performs the actual feature extraction with optional output.
        '''
        featureVals = []
        labels = []
        edgeIndices = []
        imageNumbers = []
        
        extractFunc = ExtractTool.extractModesHash[self.mode]
        extractFeatureFunc = ExtractTool.featureTypesHash[self.feature]
        
        start = time.clock()
        for item in self.images:
            if(self.verbose > 0):
                print ("Processing samples for image %d") % item.number
            
            #load lengths from the text file
            lengths = np.loadtxt(self._pathToOriginals + os.path.sep +\
                                 item.name.replace(".png", ".len.txt"),
                                 np.int32)
            

            if(len(lengths.shape) == 0):
                lengths = np.array([lengths])
            
            item.loadEdgeLabels()
            samples = item.image
            
            #perform color conversion if necessary
            if(self.cvtColor != None):
                if(len(samples.shape) != 3):
                    print ("Color conversion requires 3 channel images,"+\
                           "got %d channels. Skipping.") % len(samples.shape)
                    self.cvtColor = None
                else:
                    samples = cv2.cvtColor(samples,self.cvtColor)
            
            if(len(lengths) > 0):
                #sampling range is 1/2 the total sample image width
                samplingRange = samples.shape[1] >> 1
                
            #flip the right sample (as required) 
            #to make all processing procedures equivalent for left/right of edge
            if(self.flipRight):
                samples[:,samplingRange:samples.shape[1]] = \
                np.fliplr(samples[:,samplingRange:samples.shape[1]])
                
            curLength = 0
            
            for ixEdge in xrange(0,len(lengths)):
                if(self.verbose > 1):
                    print ("Processing edge %d") % ixEdge
                
                #lengths should be the same for both orig. item samples and pb samples
                lengthL = lengths[ixEdge]
                lengthR = lengths[ixEdge]
                
                sampleL = samples[curLength:curLength + lengthL,0:samplingRange]
                sampleR = samples[curLength:curLength + lengthR,samplingRange:samples.shape[1]]

                #extract features
                labelset,resultset = extractFunc(self,extractFeatureFunc, sampleL, sampleR)
                #accumulate results
                for ixResult in xrange(0,len(labelset)):
                    
                    featureVals.append(resultset[ixResult])
                    labels.append(labelset[ixResult])
                    edgeIndices.append(ixEdge)
                    imageNumbers.append(item.number)
        end = time.clock()
        
        if(self.verbose > 0):
            print "Total processing time: %f s" % (end - start)
                
        #if output path is specified, store it
        if(self._pathToOutput):
            doSave = True          
            #generate MATLAB output path
            matFilePath = re.sub(r"\..*$", ".mat", self._pathToOutput)
            if(matFilePath.find(".mat") == -1):
                #still no extension, means pandas path had no extension.
                #strap on extensions to both files
                self._pathToOutput += ".pd"
                matFilePath += ".mat"
            
            if os.path.isfile(self._pathToOutput):
                #already have pandas file 
                features = pd.load(self._pathToOutput)
                if(len(features) != len(labels)):
                    print ("""Save failed: the number of records in file (%d)
                           does not correspond to the label count (%d)""" 
                           % (len(features),len(labels)))
                    doSave = False
            else:
                features = pd.DataFrame(labels,columns=['label'])
                features['edge_ix'] = edgeIndices
                features['image_num'] = imageNumbers
            
            if(doSave):
                featureVals = np.array(featureVals)
                #break down into channels
                if(len(featureVals.shape) == 2):
                    for ix in xrange(featureVals.shape[1]):
                        features[(self.feature +\
                                  self.featurePostfix + "_" + str(ix))] =\
                                  np.nan_to_num(featureVals[:,ix])
                else:
                    features[(self.feature + self.featurePostfix)] = np.nan_to_num(featureVals)
                features.save(self._pathToOutput)
                if(self.matlab):
                    sio.savemat(matFilePath, 
                                {"columns":np.array(features.columns,dtype=np.object),
                                 "features":features.values},
                                 long_field_names=True)
                    if(self.jokes):
                        print "ExtractTool has detected your insolence in using the unworthy MATLAB!"
                        print ("You will now be redirected to a web page that "+\
                               "describes how stupid you are to do this...")
                        countDown = 3
                        while(countDown >= 0):
                            print "Countdown: %d" % countDown
                            time.sleep(1)
                            countDown-=1  
                        web.open('http://abandonmatlab.wordpress.com/',autoraise=True)
                        sys.exit()
                
        
parser = argparse.ArgumentParser(description="Extract Tool\n"+\
                                 " A tool for extracting features from pixel"+\
                                 " samples surrounding edges.")
parser.add_argument("--edge_images_dir","-ei",default="images/manual/edges",
                    help="path to the folder with edge images")
parser.add_argument("--data_path","-d",default="images/manual/edge.pd",
                    help="path to the data to load/store results")
parser.add_argument("--mode","-m",default="edge",metavar="MODE",
                    choices=ExtractTool.extractModesHash.keys(), 
                    help="extraction mode, one of %s" %\
                    (ExtractTool.extractModesHash.keys()))
parser.add_argument("--feature","-f",default="rowFit",metavar="FEATURE",
                    choices=ExtractTool.featureTypesHash.keys(),
                    help=("type of feature to extract. Can be one of " 
                             + str(ExtractTool.featureTypesHash.keys())))
parser.add_argument("--feature_postfix","-fp",default="",
                    help=("optional prefix for the feature"))
parser.add_argument("--fitting_function","-ff",default="a*e^x + b",
                    help="function to fit row- or sample-wise "+\
                    "(only used with fit/consistency/coefficient features).")
parser.add_argument("--color_conversion","-c",default='None',metavar="COLOR_CONVERSION",
                    choices=ExtractTool.colorConversions.keys(),
                    help=("color conversion for 3-channel samples. Cane be one of " 
                             + str(ExtractTool.colorConversions.keys())))
parser.add_argument("--metlab","-mtlb",action="store_true",default=False,
                    help="store output in matlab .mat format as well as pandas.")
parser.add_argument("--no-jokes","-nj",action="store_true",default=False,
                    help="bypass jokes.")

parser.add_argument("--flip-right","-fr",action="store_true",default=False,
                    help="flip the right side of the sample (only use if right" +\
                    "side was sampled without flipping left-right).")
parser.add_argument("--multiprocessing","-mp",action="store_true",default=False,
                    help="use multiprocessing.")
parser.add_argument("--start","-s",type=int,default=0,
                    help="number of the first image to be loaded.")
parser.add_argument("--end","-e",type=int,default=sys.maxint,
                    help="number of the last image to be loaded.")
parser.add_argument("--verbose","-v",type=int,default=1,help="verbosity level.")
parser.add_argument("--no-save","-ns",action="store_true",default=False,
                    help="skip storing results")

    
