import cPickle
import gzip
import os
import sys
import time
import matplotlib.pyplot
import numpy as np
import pandas as pd
import utils.dataManagement as dm
from utils.NeuralNet import NeuralNet, THEAN0_FLOAT_X

import argparse

import theano
import theano.tensor as Tensor

def convertToTheano(data,labels):
    data = theano.shared(np.asarray(data,dtype=THEAN0_FLOAT_X))
    labels = theano.shared(np.asarray(labels,dtype=np.int32))
    return data, labels 

def loadDataPickle(data):
    datafile = gzip.open(data, 'rb')
    trainingSet, validationSet, testingSet = cPickle.load(datafile)
    datafile.close()

    trainingData, trainingLabels = trainingSet
    validationData, validationLabels = validationSet
    testingData, testingLabels = testingSet

    trainingData, trainingLabels = convertToTheano(trainingData, trainingLabels)
    validationData, validationLabels = convertToTheano(validationData, validationLabels)
    testingData,testingLabels = convertToTheano(testingData,testingLabels)

    partitionedData = \
        [(trainingData, trainingLabels), 
         (validationData, validationLabels),
         (testingData, testingLabels)]
    return partitionedData


def loadDataPandas(datasetPath, trainingRatio, validationRatio, 
                     labelColumn, excludeCols, randomize):
    
    dataAndLabels = pd.load(datasetPath)
    trainingData,trainingLabels,validationData,validationLabels,testingData,testingLabels=\
    dm.splitDataForTraining(dataAndLabels, trainingRatio, validationRatio, 
                            labelColumn, excludeCols, shuffle = randomize)

    trainingData, trainingLabels = convertToTheano(trainingData, trainingLabels)
    validationData, validationLabels = convertToTheano(validationData, validationLabels)
    testingData,testingLabels = convertToTheano(testingData,testingLabels)
    
    return [(trainingData,trainingLabels),
            (validationData,validationLabels),
            (testingData,testingLabels)]

def optimizeNeuralNet(learningRate=0.0001, lambda1=0.00, lambda2=0.00001, 
                      maxEpochs=1000, datasetPath='features/everything.pd', 
                      batchSize=250, nHiddenNeurons=None, usePandas = True,
                      trainingRatio=0.7,validationRatio=0.15,
                      labelColumn='label',excludeColumns=['edge_ix','image_num'],
                      verbose=1, useRMSerror = False, randomizeData = True):
    trainErrorList = []
    validationErrorList = []
    testErrorList = []
    if not usePandas:
        allData = loadDataPickle(datasetPath)
        if(verbose > 0):
            print "Data unpickled"
        trainingInput, trainingLabels = allData[0]
        validationInput, validationLabels = allData[1]
        testingInput, testingLabels = allData[2]
    else:
        allData = loadDataPandas(datasetPath, trainingRatio, validationRatio, 
                                 labelColumn, excludeColumns, 
                                 randomize=randomizeData)
        trainingInput, trainingLabels = allData[0]
        validationInput, validationLabels = allData[1]
        testingInput, testingLabels = allData[2]
        if(verbose > 0):
            print "Data unpanded"

    trainingBatchCount = trainingInput.get_value().shape[0] / batchSize
    validationBatchCount = validationInput.get_value().shape[0] / batchSize
    testingBatchCount = testingInput.get_value().shape[0] / batchSize

    batchIndex = Tensor.lscalar()
    data = Tensor.matrix('data')
    labels = Tensor.ivector('labels')
    randomVal = np.random.RandomState()
    
    if(nHiddenNeurons == None):
        nHiddenNeurons = int(0.4 * trainingInput.get_value().shape[1])+1
        if(verbose > 0):
            print "Setting number of neurons in the hidden layer to %i" % nHiddenNeurons
    
    if(verbose > 0):
        print "Setting the learning rate to %f" % learningRate
        print "Setting the batch size to %d" % batchSize
        print "Setting the L2 regularization weight to %f" % lambda2

    #output dimensions should be one greater than the other thing
    learner = NeuralNet(randomVal=randomVal, data=data, inputDimensions=trainingInput.get_value().shape[1],
                     hiddenDimensions=nHiddenNeurons, outputDimensions=np.max(testingLabels.get_value())+1)

    #use negative log-likelihood  as the loss function
    lossFunction = learner.negativeLogLikelihood(labels) + \
    lambda1 * learner.normL1 + lambda2 * learner.normL2
    
    
    testingFunction = theano.function(inputs=[batchIndex],
            outputs=learner.predictionAccuracy(labels),
            givens={
                data: testingInput[batchIndex * batchSize:(batchIndex + 1) * batchSize],
                labels: testingLabels[batchIndex * batchSize:(batchIndex + 1) * batchSize]})

    validationFunction = theano.function(inputs=[batchIndex],
            outputs=learner.predictionAccuracy(labels),
            givens={
                data: validationInput[batchIndex * batchSize:(batchIndex + 1) * batchSize],
                labels: validationLabels[batchIndex * batchSize:(batchIndex + 1) * batchSize]})

    gradientParameters = []
    if(useRMSerror):
        for param, meanSquare in zip(learner.parameters,learner.meanSquare):
            gparam = Tensor.grad(lossFunction, param)
            meanSquare = Tensor.mul(meanSquare, 0.9) + Tensor.mul(Tensor.pow(gparam, 2), 0.1)
            gparam = Tensor.div_proxy(gparam, Tensor.add(Tensor.sqrt(meanSquare), 1e-8))
            gradientParameters.append(gparam)  
    else:
        for param in learner.parameters:
            gparam = Tensor.grad(lossFunction, param)
            gradientParameters.append(gparam)
        
    learningUpdates = []
    for parameters, gradientParameters in zip(learner.parameters, gradientParameters):
        learningUpdates.append((parameters, parameters - learningRate * gradientParameters))   

    trainingFunction = theano.function(
            inputs=[batchIndex], 
            outputs=lossFunction,
            updates=learningUpdates,
            givens={
                    data: trainingInput[batchIndex * batchSize:(batchIndex + 1) * batchSize],
                    labels: trainingLabels[batchIndex * batchSize:(batchIndex + 1) * batchSize]
                    }
           )

    #todo: this is exactly the same as validationFunction. Is this necessary? Just use that instead?
    trainErrorFunction = theano.function(inputs=[batchIndex],
            outputs=learner.predictionAccuracy(labels),
            givens={
                data: trainingInput[batchIndex * batchSize:(batchIndex + 1) * batchSize],
                labels: trainingLabels[batchIndex * batchSize:(batchIndex + 1) * batchSize]})

    trainingLoops = 2400000
    validationFrequency = min(batchSize, trainingLoops / 2)
    if verbose > 0:
        print "Setting validation frequency to %i iterations." % validationFrequency
    bestParams = None
    lowestValidationError = np.inf
    testError = np.inf
    iteration = 0
    stopTraining = False
    currentEpoch = 0
    
    if verbose > 0:
        print "Training with batchSize stochastic gradient descent and Neural Network"
    
    while (currentEpoch < maxEpochs) and (not stopTraining):
        currentEpoch += 1
        
        for batchIndex in xrange(trainingBatchCount):
            loss = trainingFunction(batchIndex)

            if (iteration + 1) % validationFrequency == 0:
                validationErrors = [validationFunction(i) for i in xrange(validationBatchCount)]
                currentValidationError = np.mean(validationErrors)
                
                if(verbose > 2):
                    trainErrors = [trainErrorFunction(i)for i in xrange(trainingBatchCount)]
                    trainError = np.mean(trainErrors)
                    sumNan = np.isnan(learner.hiddenLayer.weight.get_value()).sum()
                    print('Epoch {0:d}, batch {1:d}: training error = {2:.4%}, validation error = {3:.4%}'
                          .format(currentEpoch-1, batchIndex, trainError, currentValidationError))
                    if(sumNan > 0):
                        print "%d NaN encountered in weights." % sumNan
                elif(verbose > 1):
                    print('Epoch {0:d}, batch {1:d}: validation error = {2:.4%}'
                          .format(currentEpoch-1, batchIndex, currentValidationError))
                    
                    
                if currentValidationError < lowestValidationError:
                    lowestValidationError = currentValidationError

                    testErrors = [testingFunction(i) for i in xrange(testingBatchCount)]
                    testError = np.mean(testErrors)

                    trainErrors = [trainErrorFunction(i)for i in xrange(trainingBatchCount)]
                    trainError = np.mean(trainErrors)

                    trainErrorList.append(trainError)
                    validationErrorList.append(currentValidationError)
                    testErrorList.append(testError)
                    bestParams = learner.hiddenLayer.weight
                    np.save("weights",bestParams)
                    if(verbose > 0):
                        print("New best model: epoch {0:d} with validation error = {1:.4%} and test error = {2:.4%}"
                              .format(currentEpoch, lowestValidationError, testError))
                    
            iteration +=1   
            if iteration > trainingLoops:
                stopTraining = True
                break
        

    print('Gradient descent optimization finished with validation error = {0:.4%} and test error = {1:.4%}'
          .format(lowestValidationError, testError))
    
    np.save("weights",bestParams)
    return bestParams, trainErrorList, validationErrorList, testErrorList

def visualizeWeights(weights):
    weights = weights.T
    for index, classWeights in enumerate(weights):
        classWeights = classWeights.reshape(-1)
        c = matplotlib.pyplot.get_cmap('gray')
        matplotlib.pyplot.imshow(classWeights, c)
        matplotlib.pyplot.savefig('receptive_field'+np.str(index)+".png", dpi=150)
        
def visualizeErrors(errors_array, split):
    x = []
    for i in xrange(len(errors_array)):
        x.append(i)
    matplotlib.pyplot.plot(x,errors_array)
    matplotlib.pyplot.savefig("error_" + split + ".png")
    if split == "test":
        matplotlib.pyplot.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net.')
    parser.add_argument("--datafilePath","-d",default="features/everything.pd",metavar="path to the data file")
    parser.add_argument("--verbose","-v",type=int,default=1,metavar="verbosity level")
    parser.add_argument("--batchSize","-bs",type=int,default=1000,metavar="minibatch size")
    parser.add_argument("--hiddenNodeNum","-hn",type=int,default=-1,metavar="number of hidden nodes")
    parser.add_argument("--learningRate","-l",type=float,default=0.0001,metavar="the rate of gradient updates")
    parser.add_argument("--no-shuffle","-ns",action="store_true",default=False)
    parser.add_argument("--use-rms","-rms",action="store_true",default=False)
    parser.add_argument("--l2-weight","-l2",type=float,default=0.0001,metavar="weight of L2 regularizer")
    args = parser.parse_args()
    #weights, trainErrors, validationErrors, testErrors = optimize_load_data_pickle_msgd()
    if(args.datafilePath.endswith(".pd")):
        usePandas = True
    elif(args.datafilePath.endswith(".gz")):
        usePandas = False
    weights, trainErrors, validationErrors, testErrors = optimizeNeuralNet(datasetPath=args.datafilePath,
                                                                           verbose=args.verbose,
                                                                           usePandas=usePandas,
                                                                           nHiddenNeurons=None if args.hiddenNodeNum < 0 else args.hiddenNodeNum,
                                                                           learningRate=args.learningRate,
                                                                           randomizeData=not args.no_shuffle,
                                                                           useRMSerror=args.use_rms,
                                                                           batchSize=args.batchSize,
                                                                           lambda2=args.l2_weight)
    #visualizeErrors(trainErrors, "train")
    #visualizeErrors(validationErrors, "validate")
    #visualizeErrors(testErrors, "test")
    #visualizeWeights(weights.get_value())
