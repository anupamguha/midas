#!/usr/bin/env python
'''
Created on Feb 2, 2013

@author: algomorph
'''
import sys
import labeling
import sampling
import edge_detection as edge
import gradient

opToolMapping = {
                 labeling.LabelTool.op_name         :(labeling,     labeling.LabelTool),
                 sampling.SampleTool.op_name        :(sampling,     sampling.SampleTool),
                 edge.EdgeDetectTool.op_name        :(edge,         edge.EdgeDetectTool),
                 gradient.GradientTool.op_name      :(gradient,     gradient.GradientTool)
                 }
def printUsage():
    print "Usage: python eb.py <operation> <operation-specific-arguments>\nRun \"python eb.py <operation>\" for operation-specific usage.";

if __name__ == '__main__':
    argLen = len(sys.argv) - 1
    if(argLen < 1):
        printUsage()
    else:
        op = sys.argv[1]
        activeModule = opToolMapping[op][0]
        makeTool = opToolMapping[op][1]
        parser = activeModule.parser
        if(makeTool):
            tool = None
            args = vars(parser.parse_args(sys.argv[2:]))
            tool = makeTool(**args)
            tool.run()
        else:
            printUsage()
            
