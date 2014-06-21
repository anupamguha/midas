import numpy as np
import pandas as pd
from sets import Set
import time

class DecisionTreeNode:
    '''
    Creates a decision tree node
    @param left - the left subtree (if not leaf)
    @param right - the right subtree (if not leaf)
    @param isLeaf - whether this is a leaf or an internal node
    @param value - if the node is a leaf, it's the final value. If not, it's the value on which to split.
    @param feature - if not leaf, the feature on which to split 
    '''
    def __init__(self, left, right, isLeaf, value, feature, depth):
        self.isLeaf = isLeaf
        self.value = value
        self.depth = depth
        if not isLeaf:
            self.left = left
            self.right = right
            self.feature = feature
    
    def __repr__(self):
        if(self.isLeaf):
            return "Leaf node with label %d at depth %d" % (self.value,self.depth) 
        return "Node with split on feature %s at value %f" % (self.feature,self.value)

class DecisionTree:
    '''
    Creates an instance of DecisionTree.
    @param depth: the maximum depth of the decision tree, at the bottom of 
    which there can be nothing but leaves
    @param traindat: DataFrame object with the training examples 
    @param labelColumn: name of the column containing the labels in the training examples
    @param useInformationGain: when set to true, the decision tree will use information gain as a scoring function. Alternatively, it will use majority scoring. 
    '''
    def __init__(self, trainData, depth, labelColumn, useInformationGain = False, excludeCols = None, verbose = 0):
        self.depth = depth
        self.labelColumn = labelColumn
        self.useInformationGain = useInformationGain
        self.__usedSplits = Set();
        self.verbose = verbose
        if useInformationGain:
            self.__score = self.__informationGain
        else:
            self.__score = self.__majorityVote
            
        if excludeCols:
            cols = np.setdiff1d(trainData.columns,excludeCols,True)
            trainData = trainData[cols]
        
        labels = trainData[labelColumn].values
        if tuple(np.unique(labels)) == (0,1):
            if(verbose > 0):
                print "Renumbering labels from (0,1) to (-1,1)..."
            #convert 0, 1 to -1, 1
            labels = labels * 2 - 1
            trainData[labelColumn] = trainData[labelColumn] * 2 - 1
            if(verbose > 0):
                print "Done."
        
        if(verbose > 0):
            print "Training..."
            start = time.clock()
        self.root = self.__train(trainData, 0)
        if(verbose > 0):
            end = time.clock()
            print "Training complete. Total training wall time: {0:.3f} s.".format(end-start)
        
        featureColumns = list(trainData.columns)
        featureColumns.remove(labelColumn)
        trainDataResult = self.classify(trainData[featureColumns]).values
        
        errors = np.abs((trainDataResult - labels)) / 2
        posCount = np.count_nonzero(labels + 1)
        negCount = len(labels) - posCount
        if(self.verbose > 0):
            print "Positive error rate on training data: %s" % str(float(errors[labels == 1].sum()) / posCount)
            print "Negative error rate on training data: %s" % str(float(errors[labels == -1].sum()) / negCount)
        
        
        
        

    '''
    Calculates entropy of a given set given the total # of elements in the set 
    and the number of positive examples within it
    '''
    def __entropy(self, positiveCount, total):
        if total == 0 or positiveCount == 0 or positiveCount == total:
            return 0
        positiveRatio = float(positiveCount)/total;
        negativeRatio = 1.0 - positiveRatio
        return -positiveRatio * np.log2(positiveRatio) - negativeRatio * np.log2(negativeRatio)
    
    '''
    Groups the elements within the DataFrame with the specified clause and
    returns a Series of counts of examples falling within each group
    '''
    def __counts(self, dataset, groupByClause):
        return self.__fillMissing(dataset.groupby(groupByClause, sort = True).size())
    
    '''
    Used on a series of counts keyed by tuples, where each tuple has length two and
    the last value is -1 or 1 (i.e. positive or negative example). If there is a key
    with value [x, y], where x is any value and y is -1/1, makes sure that the resulting
    series has this key and it's value *AS WELL AS* key [x,-y] with the value 0.  
    '''
    def __fillMissing(self, counts):
        missingKeys = [(row[0],-row[1]) for row in counts.index if (row[0],-row[1]) not in counts.index]
        counts = counts.append(pd.Series([0]* len(missingKeys),index=missingKeys))
        return counts.sort_index()
    '''
    Computes the greatest majority score for the given split point parameters.
    NOTE: it's not exactly the same as in the book. Here I made sure that the maximum sum of opposite
    votes on both sides constitutes the score. 
    Consider a split where [-1,-1,-1,1,-1] are on the left side and [-1,-1,-1] are on the right.
    In this case, if we go by the book, the majority vote on the left would be 4 (for -1) and on the 
    right - 3 (also for -1). But this split would be worse than, say, if the right side was changed to
    [-1,1,1], since in the first case, the majority vote is for the same label. 
    We really care about the sum of *differing* votes from either side. For the given example,
    it is max(4,4), or 4 instead of 7. If the right side is changed to [-1,1,1], it would be max(6,2) = 6.
    '''
    def __majorityVote(self, lessThanPos,lessThanTotal,greaterThanPos,greaterThanTotal,dummyVal,total):
        if greaterThanTotal == 0:
            return 0
        return max(lessThanPos + greaterThanTotal-greaterThanPos, greaterThanPos + lessThanTotal-lessThanPos)
    
    '''
    Computes the information score for the given split point parameters
    '''
    def __informationGain(self, lessThanPos, lessThanTotal, greaterThanPos, greaterThanTotal, generalEntropy, total):
        return generalEntropy \
            - float(lessThanTotal)/total * self.__entropy(lessThanPos, lessThanTotal) \
            - float(greaterThanTotal)/total * self.__entropy(greaterThanPos, greaterThanTotal)
    '''
    Finds the best information score / greatest majority score and the split point 
    that produces it for a specified continuous feature in the given (DataFrame) dataset.
    Assumes datset contains both spam and non-spam examples.
    '''
    def __findSplit(self, dataset, feature):
        labelCounts = dataset.groupby(self.labelColumn).size()
        total = labelCounts.sum()
        if self.useInformationGain:
            generalEntropy = self.__entropy(labelCounts[1], total) 
        else:
            generalEntropy = None
        counts = self.__counts(dataset,[feature, self.labelColumn])
        countDict = {}
        negCt = 0
        posCt = 0
        # aggregate the counts of positives and negatives that are <= each potential split point 
        for key in counts.index:
            if(key[0] in countDict):
                subdict = countDict[key[0]]
            else:
                subdict = {}
                countDict[key[0]] = subdict
            if(key[1] == -1):
                negCt += counts[key]
                subdict[-1] = negCt
            else:
                posCt += counts[key]
                subdict[1] = posCt
        bestScore = -1;
        bestSplit = 0;
        #all values are the same, disregard
        if len(countDict) == 1:
            return (bestScore,bestSplit,None)
        #find the greatest score
        for key in countDict:
            countPair = countDict[key]
            greaterThanPos = labelCounts[1] - countPair[1]
            lessThanTotal = countPair[-1] + countPair[1]
            greaterThanTotal = total - lessThanTotal
            #calculate score
            score = self.__score(countPair[1],lessThanTotal,greaterThanPos,greaterThanTotal,generalEntropy,total)
            #record if it's the best so far
            if score > bestScore:
                bestScore = score
                bestSplit = key
        countPair = countDict[bestSplit]
        splitCounts = {
                       'above':countPair,
                       'below':{
                                -1:labelCounts[-1] - countPair[-1],
                                1:labelCounts[1] - countPair[1],
                                }
                       }
        
        return bestScore, bestSplit, splitCounts
    
    '''
    Splits the dataset at the given value of the given feature, 
    returns two parts of the original dataset with feature values below (<=) and above split point
    '''
    def __split(self, dataset, feature, splitPoint):
        return (dataset[dataset[feature] <= splitPoint],dataset[dataset[feature] > splitPoint])
   
    def __train(self, dataset, depth):
        labelCounts = dataset.groupby(self.labelColumn).size()
        '''
        if depth-level is reached or the data is unambiguous, return the label
        with the greatest count
        '''
        if depth == self.depth or labelCounts.size == 1:
            return DecisionTreeNode(None,None,True, labelCounts.idxmax(),None,depth)
        splitFeature = "";
        bestSplit = 0;
        bestScore = -1;
        bestSplitCounts = None;
        #find best feature to split on and the best split point for that feature
        featureScores = []
        for feature in dataset.columns:
            if feature != self.labelColumn:
                score,split,splitCounts = self.__findSplit(dataset, feature)
                if(self.verbose > 1):
                    featureScores.append((score,split,splitCounts,feature))
                if score > bestScore:
                    bestScore = score
                    splitFeature = feature
                    bestSplit = split
                    bestSplitCounts = splitCounts
        #no clue, examples are entirely equal in quality but different in labels.
        if bestScore <= 0.0:
            return DecisionTreeNode(None,None,True,-1,None,depth)
        
        if(self.verbose > 0):
            print "Label counts at current place in tree:"
            print labelCounts
            print "Splitting on feature \"{0}\" at value {1} with score {2} at depth {3}"\
            .format(splitFeature,bestSplit,bestScore, depth)
            print "  Split counts: %s" %str(bestSplitCounts)
            aboveTotal = bestSplitCounts['above'][-1]+bestSplitCounts['above'][1]
            belowTotal = bestSplitCounts['below'][-1]+bestSplitCounts['below'][1]
            print "  Split percentages: (below: [1:{3:.2%},-1:{2:.2%}], above: [1:{1:.2%},-1:{0:.2%} ])"\
            .format(float(bestSplitCounts['above'][-1])/aboveTotal,
                    float(bestSplitCounts['above'][1])/aboveTotal,
                    float(bestSplitCounts['below'][-1])/belowTotal,
                    float(bestSplitCounts['below'][1])/belowTotal)
            if(self.verbose > 1):
                #sort on score
                featureScores.sort(key=lambda entry: entry[0], reverse=True)
                print "    Listing candidates in descending order:"
                if(self.verbose > 2):
                    for (score,split,splitCounts,feature) in featureScores:
                        print '''
    Best split for feature \"{0}\" is at value {1}
    with score {2} at this depth, {3}"\
                              '''.format(feature,split,score,depth)
                        print "    Split counts: %s" %str(splitCounts)
                        aboveTotal = splitCounts['above'][-1]+splitCounts['above'][1]
                        belowTotal = splitCounts['below'][-1]+splitCounts['below'][1]
                        print '''
    Split percentages: (below: [1:{3:.2%},-1:{2:.2%}],
                        above: [1:{1:.2%},-1:{0:.2%} ])
                              '''.format(float(splitCounts['above'][-1])/aboveTotal,
                                        float(splitCounts['above'][1])/aboveTotal,
                                        float(splitCounts['below'][-1])/belowTotal,
                                        float(splitCounts['below'][1])/belowTotal)
                else:
                    print "    Listing candidates in descending order:"
                    print "    Feature                value      score      below( 1)  below(-1)  above( 1)  above(-1)"
                    for (score,split,splitCounts,feature) in featureScores:
                        aboveTotal = splitCounts['above'][-1]+splitCounts['above'][1]
                        belowTotal = splitCounts['below'][-1]+splitCounts['below'][1]
                        belowPos = float(splitCounts['below'][1])/belowTotal
                        belowNeg = float(splitCounts['below'][-1])/belowTotal
                        abovePos = float(splitCounts['above'][1])/aboveTotal
                        aboveNeg = float(splitCounts['above'][-1])/aboveTotal
                        print "{0:>25}  {1:>9.6g} {2:>12.6g}   {3:>7.2%}    {4:>7.2%}    {5:>7.2%}    {6:>7.2%}"\
                        .format(feature, split, score, belowPos, belowNeg, abovePos, aboveNeg)
            (leftData,rightData) = self.__split(dataset,splitFeature,bestSplit)
    
            left = self.__train(leftData,depth+1)
            right = self.__train(rightData,depth+1)
            return DecisionTreeNode(left,right, False, bestSplit, splitFeature,depth)
    
    def __classifySingleVerbose(self,node,example):
        if node.isLeaf:
            print "Reached leaf node with value %d." % node.value               
            return node.value
        print "Node %s: %f, comparing with value %f" % (node.feature,node.value, example[node.feature])
        if example[node.feature] > node.value:
            print "Going right."
            return self.__classifySingleVerbose(node.right, example)
        else:
            print "Going left."
            return self.__classifySingleVerbose(node.left, example)
        
    def __classifySingle(self,node,example):
        if node.isLeaf:               
            return node.value
        if example[node.feature] > node.value:
            return self.__classifySingle(node.right, example)
        else:
            return self.__classifySingle(node.left, example)
    '''
    Classifies a given example or set of examples.
    @param data: either a simple example (Series-like) or a DataFrame containing examples 
    '''
    def classify(self,data,verbose = 0):
        if(verbose > 0):
            classifyFunc = self.__classifySingleVerbose
        else: 
            classifyFunc = self.__classifySingle
        if data.__class__.__name__ == "DataFrame":
            outDict = {}
            for rowIndex in data.index:
                outDict[rowIndex] = classifyFunc(self.root,data.ix[rowIndex])
            return pd.Series(outDict)
        else:
            return classifyFunc(self.root, data)
    