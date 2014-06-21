'''
Created on Jul 4, 2013

@author: Gregory Kramida
'''
import theano
import theano.tensor as T
import numpy as np

def softmax(parameter):
    e = T.exp(np.array(parameter) / 1.0)
    softmax_result = e / T.sum(e)
    return softmax_result

def tanh(parameter):
    e = T.exp(np.array(parameter)*(-2))
    tanh_result = (1 - e)/(1 + e)
    return tanh_result

def sigmoid(parameter):
    e = T.exp(np.array(parameter)*(-1))
    sigmoid_result = 1/(1 + e)
    return sigmoid_result

THEAN0_FLOAT_X = np.dtype(np.double).name
if(hasattr(theano.config,'floatX')):
    THEAN0_FLOAT_X = getattr(theano.config,'floatX')

class OutputLayer(object):
    def __init__(self, data, inputDimensions, outputDimensions):
        self.weight = theano.shared(value=np.zeros((inputDimensions, outputDimensions), dtype=THEAN0_FLOAT_X))
        self.bias = theano.shared(value=np.zeros((outputDimensions,),dtype=THEAN0_FLOAT_X))
        self.class_conditional_probability = T.nnet.softmax((T.dot(data, self.weight) + self.bias))
        self.class_prediction = T.argmax(self.class_conditional_probability, axis=1)
        self.Norm_L1 = abs(self.weight).sum()
        self.Norm_L2 = abs(self.weight ** 2).sum()
        self.parameters = [self.weight, self.bias]
        self.meanSquarePerWeight = theano.shared(value=np.zeros((inputDimensions, outputDimensions), dtype=THEAN0_FLOAT_X))
        self.meanSquarePerBias = theano.shared(value=np.zeros((outputDimensions,), dtype=THEAN0_FLOAT_X))
        self.meanSquare = [self.meanSquarePerWeight, self.meanSquarePerBias]
       
        
    def changeValue(self, data):
        self.class_conditional_probability =  softmax(T.dot(data, self.weight) + self.bias)
        self.class_prediction =  T.argmax(self.class_conditional_probability, axis=1)

    def negativeLogLikelihood(self, y):
        return -T.mean(T.log(self.class_conditional_probability)[T.arange(y.shape[0]), y])

    def predictionAccuracy(self, y):
        return T.mean(T.neq(self.class_prediction, y))


class HiddenLayer(object):
    """
    Typical hidden layer of a MLP: units are fully-connected and have
    sigmoidal activation function. weight matrix W is of shape (n_in,n_out)
    and the bias vector b is of shape (n_out,).

    NOTE : The nonlinearity used here is tanh

    Hidden unit activation is given by: tanh(dot(input,W) + b)

    :type randomGen: numpy.random.RandomState
    :param randomGen: a random number generator used to initialize weights

    :type input: theano.tensor.dmatrix
    :param input: a symbolic tensor of shape (n_examples, n_in)

    :type inputDimensions: int
    :param inputDimensions: dimensionality of input

    :type outputDimensions: int
    :param outputDimensions: number of hidden units

    :type activationFunction: theano.Op or function
    :param activationFunction: Non linearity to be applied in the hidden
                          layer
    """
    def __init__(self, randomGen, data, inputDimensions,
                 outputDimensions, weight=None, bias=None,
                 activationFunction=T.tanh):
        self.data = data

        self.meanSquarePerWeight = theano.shared(value=
                                                 np.zeros((inputDimensions, outputDimensions), 
                                                          dtype=THEAN0_FLOAT_X))
        self.meanSquarePerBias = theano.shared(value=
                                               np.zeros((outputDimensions,), 
                                                        dtype=THEAN0_FLOAT_X))
        if weight is None:
            bound = np.sqrt(0.6 / (inputDimensions + outputDimensions))
            weightValues = np.asarray(randomGen.uniform(
                                        low=-bound,high=bound,
                                        size=(inputDimensions, outputDimensions)),
                                        dtype=THEAN0_FLOAT_X)
            #weightValues = np.load("weights.npy")
            #np.save("weights",weightValues);
            if activationFunction == sigmoid:
                weightValues *= 4
            weight = theano.shared(value=weightValues, name='weight')

        if bias is None:
            biasValues = np.zeros((outputDimensions,), dtype=THEAN0_FLOAT_X)
            bias = theano.shared(value=biasValues, name='bias')

        self.weight = weight
        self.bias = bias

        linearOutput = T.dot(data, self.weight) + self.bias
        self.output = (linearOutput if activationFunction is None
                       else activationFunction(linearOutput))
        self.parameters = [self.weight, self.bias]
        self.meanSquare = [self.meanSquarePerWeight, self.meanSquarePerBias]
        
class NeuralNet(object):
    def __init__(self, randomVal, data, inputDimensions, hiddenDimensions, outputDimensions):
        self.hiddenLayer = HiddenLayer(randomGen=randomVal, data=data,
                                       inputDimensions=inputDimensions, 
                                       outputDimensions=hiddenDimensions,
                                       activationFunction=T.tanh)

        self.logisticRegressionLayer = OutputLayer(
            data=self.hiddenLayer.output,
            inputDimensions=hiddenDimensions,
            outputDimensions=outputDimensions)

        self.normL1 = abs(self.hiddenLayer.weight).sum() \
                + abs(self.logisticRegressionLayer.weight).sum()

        self.normL2 = (self.hiddenLayer.weight ** 2).sum() \
                    + (self.logisticRegressionLayer.weight ** 2).sum()

        self.negativeLogLikelihood = self.logisticRegressionLayer.negativeLogLikelihood
        self.predictionAccuracy = self.logisticRegressionLayer.predictionAccuracy
        self.parameters = self.hiddenLayer.parameters + self.logisticRegressionLayer.parameters
        self.meanSquare = self.hiddenLayer.meanSquare + self.logisticRegressionLayer.meanSquare