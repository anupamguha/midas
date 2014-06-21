import numpy as np
import pandas as pd
import math
from pyfann.libfann import * 


def prepData(df):
    return df[df.columns[(df.columns != 'rating') & (df.columns != 'isgood')]], df['rating']

class NeuralNet(object):
    
    def __init__(self, df = None, labelColumn = 'label', excludeColumns = None,
                 trainingRatio = 0.6, validationRatio = 0.2, precision = np.float64):
        if(excludeColumns == None):
            excludeColumns = []
        self._data = None
        self.setRatios(trainingRatio, validationRatio)
        self._precision = np.float64
        self.model = neural_net()
        self._ftrd = None
        self._fvld = None
        self._fted = None
        self._bestModel = None
        self._bestValidationError = np.Inf
        
        self._failLimit = 2
        self._fails = 0
        self._lastError = np.inf
        
        #prepare data if ready
        if(type(df) != type(None)):
            self.setData(df, labelColumn, excludeColumns)
        else:
            self._labelColumn = labelColumn
            self._excludeColumns = excludeColumns
    
    def __del__(self):
        if(self._ftrd != None):
            self._ftrd.destroy_train()
        self.model.destroy()
    
    def setRatios(self, trainingRatio, validationRatio):
        if (trainingRatio <= 0 or validationRatio < 0 or 
            trainingRatio+validationRatio > 1.0):
            raise ValueError("Invalid ratio set for training & validation: %(f) and %(f)."
                             "They should be positive and their sum should be in (0,1.0]")
        self._trainingRatio = trainingRatio
        self._validationRatio = validationRatio
        if(type(self._data) != type(None)):
            self._prepareData()
            
    def setData(self, df, labelColumn = 'label', excludeColumns = None):
        
        #check type
        if(type(df) != pd.DataFrame):
            raise TypeError("Data should be a Pandas DataFrame object. Got: %s" 
                             % str(type(df)))
        
        self._data = df
        self._labelColumn = labelColumn
        self._excludeColumns = excludeColumns
        if not labelColumn in self._data.columns:
                raise ValueError("Given label column (%s) is not amongst the data columns: %s."
                                 % (labelColumn, str(self._data.columns)))
        self._prepareData()
        
    @property
    def bitFailLimit(self):
        return self.model.get_bit_fail_limit()
    
    @bitFailLimit.setter
    def bitFailLimit(self,value):
        return self.model.set_bit_fail_limit(value)
        
    @property
    def precision(self):
        return self._precision
    
    @precision.setter
    def precision(self,value):
        updateNecessary = False 
        if(self._precision != value and self._data):
            updateNecessary = True
        self._precision = value
        if(updateNecessary and type(self._data) != type(None)):
            self._prepareData()
        
    def _prepareData(self):
        excl = list(self._excludeColumns)
        excl.append(self._labelColumn)
        dataCols = np.setdiff1d(np.array(self._data.columns),excl)
        dataDF = self._data[dataCols]
        data = dataDF.values.astype(self.precision)
        labels = self._data[self._labelColumn].values.astype(np.int32)
        trainingCount = len(data) * self._trainingRatio
        validationCount = len(data) * self._validationRatio
        testingOffset = trainingCount + validationCount
        
        idxs = np.arange(len(labels))
        np.random.shuffle(idxs)
        data = data[idxs]
        labels = labels[idxs]
        
        #split into training/validation/testing
        self._trainingData = data[0:trainingCount]
        self._trainingLabels = labels[0:trainingCount]
        self._validationData = data[trainingCount:testingOffset]
        self._validationLabels = labels[trainingCount:testingOffset]
        self._testingData = data[testingOffset:]
        self._testingLabels = labels[testingOffset:]
        
        #translate into FANN format
        if(self._ftrd != None):
            self._ftrd.destroy_train()
        if(self._fvld != None):
            self._fvld.destroy_train()    
        if(self._fted != None):
            self._fted.destroy_train()
            
        self._ftrd = training_data()
        self._ftrd.set_train_data(self._trainingData,
                            self._trainingLabels.reshape((len(self._trainingLabels),1)))
        #self._ftrd.shuffle_train_data();
        if(len(self._validationData) > 0):
            self._fvld = training_data()
            self._fvld.set_train_data(self._validationData,
                                self._validationLabels.reshape((len(self._validationLabels),1)))
        
        if(len(self._testingData) > 0):
            self._fted = training_data()
            self._fted.set_train_data(self._testingData,
                                self._testingLabels.reshape((len(self._testingLabels),1)))
        
    @property
    def trainingSet(self):
        return self._ftrd
    
    @property
    def validationSet(self):
        return self._fvld
    
    @property
    def testingSet(self):
        return self._fted
    
    @staticmethod
    def asList(data):
        l = [];
        if(type(data) == pd.DataFrame or type(data) == pd.Series):
            for (index,example) in data.iterrows():
                ex = np.float32(example)
                ex[np.isnan(ex)] = 0.0
                l.append(list(ex))
        elif(type(data) == np.ndarray):
            return data.tolist()
        elif(type(data) == list):
            return data
        return l
    
    def _test(self,dataset,labels):
        testResult = self.predict(dataset)
        error = 1.0 - float(np.equal(testResult,labels).sum()) / len(labels)
        return error
    
    def _trainEpoch(self, verbose = 1):
        mseTrain = self.model.train_epoch(self._ftrd)
        errorTrain = self._test(self._trainingData,self._trainingLabels)
        if(verbose > 0):
            print "Training error: {0:.6%}".format(errorTrain) 
        if(len(self._validationData) > 0):
            mseVal = self.model.test_data(self._fvld)
            errorVal = self._test(self._validationData,self._validationLabels)
            if(verbose > 0):
                print "Validation error: {0:.6%}".format(errorVal)
            if errorVal > self._lastError:
                self._fails += 1
                if(self._fails == self._failLimit):
                    if(verbose > 0):
                        print "Validation error decresed %d times in a row, stopping." % self._failLimit
                    return False;
            elif errorVal < self._bestValidationError:
                self._bestValidationError = errorVal
            else:
                self._fails = 0
                self._lastError = errorVal
        return True
                
        
    def train(self,numHiddenNeurons = None, learningRate = 0.0001,
              maxIterations = 100, desiredError = 0.00000001,  
              iterationsBetweenReports = 10, verbose = 1, failLimit = 5):
        
        
        if(type(self._data) == type(None)):
            print "No data loaded"
            return
        
        self._failLimit = failLimit
        
        connectionRate = 1
        
        
        numInput = len(self._trainingData[0])
        numOutput = 1
        
        if(not numHiddenNeurons):
            numHiddenNeurons = int((0.4 * numInput))+1
            if(verbose > 0):
                print "Using %d hidden neurons." % numHiddenNeurons
        
        
        self.model.destroy()
        self.model = neural_net()
        self.model.create_sparse_array(connectionRate, (numInput, numHiddenNeurons, numOutput))
        self.model.set_learning_rate(learningRate)
        self.model.set_activation_function_output(SIGMOID_SYMMETRIC_STEPWISE)
        #self.model.train_on_data(self._ftrd, maxIterations, iterationsBetweenReports, desiredError)
        for iter in xrange(0,maxIterations):
            if(verbose > 0):
                print "Training epoch %d." % iter
            cont = self._trainEpoch(verbose = verbose)
            if not cont:
                break
        
        if(len(self._testingData) > 0):
            error = self._test(self._testingData, self._testingLabels)
            success = 1.0 - error
            if(verbose > 0):
                print "Test data: {0:.4%} success, {1:.4%} error.".format(success,error);
        
        
    def predict(self,inputSet):
        if(type(inputSet) == pd.DataFrame):
            excl = list(self._excludeColumns)
            excl.append(self._labelColumn)
            dataCols = np.setdiff1d(np.array(self._data.columns),excl)
            inputArr = inputSet[dataCols].values
        else:
            inputArr = inputSet
        out = np.zeros(len(inputArr),dtype=np.int32)
        for ix in xrange(0,len(inputArr)):
            entry = inputArr[ix]
            out[ix] = int(round(self.model.run(entry)[0]))
        return out
        
if __name__ == '__main__':
    rambo = pd.load('features/everything.pd')
    nn = NeuralNet(rambo,'label',['edge_ix','image_num'],0.8,0.0)
    nn.train(maxIterations = 2)

    