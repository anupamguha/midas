'''
Created on Jul 28, 2013

@author: Gregory Kramida
'''
from copy import copy
import numpy as np

#TODO: revise this to work with new chaining
def whittle(points, verbose = 0):
    '''
    A general de-branching algorithm that assumes there are intersections
    in the keypoint set, i.e. it's no just one contigous chain,
    but a chain with multiple branches. It finds the longest chain and all
    the additional branches.
    @param keyponits: a set of 8-connected points to consider. 
    @param verbose: verbosity level.
    @return: a numpy array of the longest chain, and a list of numpy arrays representing the branches
    '''
    deJaVu = {}
    filtered = []
    for pt in points:
        tpt = tuple(pt)
        if tpt not in deJaVu:
            filtered.append(tpt)
            deJaVu[tpt] = True
    
    #find endpoints
    endpoints = []
    orig = np.array(filtered)
    
    for cur in filtered:
        nei = orig[np.abs(orig - cur).max(axis=1) == 1]
        if(len(nei) == 1):
            endpoints.append(cur)
    #if there are no endpoints, it must be a "clean loop". Grab any point and start processing.  
    if(len(endpoints) == 0):
        #grab a random point, go around the loop
        nei = [filtered[0]]
        truePoints = []
        stack = copy(filtered);
        while(len(nei) > 0 and len(stack) > 1):
            cur = tuple(nei[0])
            stack.remove(cur)
            truePoints.append(cur)
            ndSt = np.array(stack)
            nei = ndSt[np.abs(ndSt - cur).max(axis=1) == 1]
        truePoints.append(stack[0])
        #no branches to worry about, return
        return np.array(truePoints), []
        
    #filter short branches from endpoints, until two remain
    stack = copy(filtered)
    
    branches = [[] for _ in xrange(len(endpoints))]
    
    endpoints_orig = copy(endpoints)
    
    #append initial endpoints
    ix = 0
    for pt in endpoints:
        (branches[ix]).append(pt)
        ix+=1
    
    endsRemaining = len(endpoints_orig)
    branchesToRemove = {}
    stemTerminals = {}
    removed = {}
    goodEndIx = 0
    
    while endsRemaining > 2:
        lookahead = {} 
        for ix in xrange(0,len(endpoints)):
            if ix in branchesToRemove: 
                continue
            
            cur = endpoints[ix] #put underneath the next if statement
            
            #if someone else must have removed the cursor pixel, this is a terminal branch
            if(cur in removed):
                branches[ix].pop()
                branchesToRemove[ix] = True
                endsRemaining-=1;
                continue
            # if this is the last pixel in the stack, we must be at
            # a juncture of 3 or more branches w/ equal length.
            # In this case, simply mark the current branch 
            if(len(stack) == 1):
                branchesToRemove[ix] = True
                endsRemaining-=1
                continue

            stack.remove(cur)
            removed[cur] = True
            ndSt = np.array(stack)
                
            nei = ndSt[np.abs(ndSt - cur).max(axis=1) == 1]
            #if neighbors are 0, someone must have removed the connecting link
            #if neighbors are > 1, we're at a junction
            if(len(nei) == 1 and not tuple(nei[0]) in lookahead):
                tnei = tuple(nei[0])
                endpoints[ix] = tnei
                branches[ix].append(tnei)
                goodEndIx = ix
                lookahead[tnei] = True
            else:
                #check whether the terminal should belong to the stem (i.e. a "stem" terminal)
                #or a branch. This is done by seeing if the two neighbors of the terminal
                #are next to each other (8-connected). If they are, the terminal belongs to the branch.
                if (len(nei) > 1 and not np.abs(nei[0] - nei[1]).max() == 1):
                    stemTermBranches = []
                    struckLoop = False
                    neIx = 0
                    nonStemIx = -1
                    #check whether the stem termials
                    while neIx < len(nei) and not struckLoop:
                        tnei = tuple(nei[neIx])
                        if tnei in stemTerminals:
                            stemTermBranches.append(stemTerminals[tnei])
                            #all but one neighbors are "stem" terminals,
                            #we must have struck either a loop or a case
                            #where two branches are blocking each-other
                            if(len(stemTermBranches) == len(nei)-1):
                                struckLoop = True;
                        else:
                            nonStemIx = neIx
                        neIx+=1
                    if(struckLoop):
                        if(nonStemIx == -1):
                            #if it's never been assigned, it must be the last
                            nonStemIx = len(nei)-1
                        for loopBranch in stemTermBranches:
                            del branchesToRemove[loopBranch]
                            tneiOther = endpoints[loopBranch]
                            del stemTerminals[tneiOther]
                            branches[loopBranch].append(tneiOther)
                            endsRemaining+=1
                        #add the terminal back to the branch
                        #it should already be in endpoints[struckLoop]
                        tnei = tuple(nei[nonStemIx])
                        endpoints[ix] = tnei
                        branches[ix].append(tnei)
                        goodEndIx = ix
                        lookahead[tnei] = True
                        continue
                    else:
                        branches[ix].pop()
                        stack.append(cur)
                        stemTerminals[cur] = ix
                        del removed[cur]
                branchesToRemove[ix] = True
                endsRemaining-=1;
    separatedBranches = []
    
    for ixBr in branchesToRemove.keys():
        branch = branches[ixBr]
        separatedBranches.append(np.array(branch))
        for pt in branch:
            if(pt not in filtered):
                if(verbose > 0):
                    print "Trying to remove a branch point twice"
                    print pt
                continue
            filtered.remove(pt)
                
    stack = copy(filtered)
    
    
    nei = [endpoints_orig[goodEndIx]]
    truePoints = []
    stemDisc = False
    while(len(stack) > 1):
        cur = tuple(nei[0])
        if(cur not in stack):
            if(verbose > 0):
                print "Original good endpoint lost"
                np.save('orig',orig)
                print endpoints_orig[goodEndIx]
                print cur
                print nei
                print stack 
            break
        stack.remove(cur)
        truePoints.append(cur)
        ndSt = np.array(stack)
        nei = ndSt[np.abs(ndSt - cur).max(axis=1) == 1]
        if(len(nei) == 0):
            stemDisc = True
            if(verbose > 0):
                print "Discontinuity in stem"
                stack = [stack[0]]
                print endpoints_orig[goodEndIx]
                #print stack
                print cur
                #print truePoints
            break
    if(not stemDisc and len(stack) > 0):
        truePoints.append(stack[0])
    
    return np.array(truePoints), separatedBranches