'''
Created on Mar 9, 2013

@author: algomorph
'''

class LineIterator(object):
    def __init__(self,p1,p2):
        self.steep = abs(p2[0] - p1[0]) > abs(p2[1] - p1[1])
        if(self.steep):
            x0, y0 = p1[0],p1[1]
            x1, y1 = p2[0],p2[1]
        else:
            x0, y0 = p1[1],p1[0]
            x1, y1 = p2[1],p2[0]
            
        if x0 > x1:
            self._xstep = -1
        elif x0 < x1:
            self._xstep = 1
        else:
            self._xstep = 0
            
        if y0 < y1: 
            self._ystep = 1
        elif y0 > y1:
            self._ystep = -1
            
        self.p1 = p1
        self.p2 = p2
        
        self._deltaX = abs(x1 - x0)
        self._deltaY = abs(y1 - y0)
        self._error = 0
        self._y = y0
        self._x = x0
        self.count = abs(x1 - x0) + 1
        self._ix = 1;
        
        
    def next(self):
        if(self._ix < self.count): 
            self._error += self._deltaY
            if (self._error << 1) >= self._deltaX:
                self._y += self._ystep
                self._error -= self._deltaX
            self._x += self._xstep
    
    def pos(self):
        if(self.steep):
            return (self._x,self._y)
        else:
            return (self._y,self._x)