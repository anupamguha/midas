'''
Created on Jul 28, 2013

@author: Gregory Kramida
'''
from utils.enum import enum

BELONGING_COLORS = enum(BOTH = (0,128,255), 
                        LEFT = (0,255,128), 
                        RIGHT = (255,255,0), 
                        NONE = (171,104,56), 
                        UNKNOWN = (255,255,255))

BELONGING_LABELS = enum(BOTH = "BOTH", 
                        LEFT = "LEFT", 
                        RIGHT = "RIGHT", 
                        NONE = "NONE", 
                        UNKNOWN = "UNKNOWN")

BELONGING_COLOR_BY_LABEL = {
                            BELONGING_LABELS.BOTH:BELONGING_COLORS.BOTH,
                            BELONGING_LABELS.LEFT:BELONGING_COLORS.LEFT,
                            BELONGING_LABELS.RIGHT:BELONGING_COLORS.RIGHT,
                            BELONGING_LABELS.NONE:BELONGING_COLORS.NONE,
                            BELONGING_LABELS.UNKNOWN:BELONGING_COLORS.UNKNOWN
                            }