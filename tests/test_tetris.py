'''
Created on Aug 29, 2013

@author: Algomorph
'''
import unittest
import cv2
import sampling_aux.s_tetris as tetr
import sampling as samp
from bw.chain import chain, simplify_chain
from bw.edge import Edge
import utils.geom as ugeom
import numpy as np
import math

class TestTetris(unittest.TestCase):
    def setUp(self):
        self.sampling_range = 40
        self.sampleTool = samp.SampleTool('tetris','tetris/test.pd',self.sampling_range,
                                    tetr.SovietTetrisSampler.mode,1,self.sampling_range,'',
                                    0,no_save=True)
        self.sampler = tetr.SovietTetrisSampler(self.sampleTool)

    def test_tetris1(self):
            sampler = self.sampler
            raster = cv2.imread('tetris/test_tetris_01.png')
            bw_raster = cv2.cvtColor(raster,cv2.COLOR_BGR2GRAY)
            edges = chain(bw_raster)
            simp_edges = []
            for ch in edges:
                s_edge = simplify_chain(ch)
                simp_edges.append(s_edge)
                
            self.assertTrue(len(simp_edges) == 1, "Too many edges found")
            ch = simp_edges[0]
            self.assertTrue(len(ch) == 4, "Too many keypoints found")
            
            edge = Edge(ch)
            center_seg = ugeom.mid_point_list(ch[0],ch[1])
            center, normal = sampler.find_box_base(edge, 0, center_seg)
            self.assertTrue(np.array_equal(np.array(center,dtype=np.int32),
                                           np.array(center_seg,dtype=np.int32)),
                            "base center incorrect for seg 0")
            self.assertTrue(np.array_equal(normal,ugeom.normal(ch[0], ch[1])),
                            "base normal incorrect for seg 0")
            centerP1, normalP1 = sampler.find_box_base(edge, 1, ch[1])
            centerP2, normalP2 = sampler.find_box_base(edge, 1, ch[2])
            centerP1 = np.array(centerP1)
            centerP2 = np.array(centerP2)
            normalP1 = np.array(normalP1)
            normalP2 = np.array(normalP2)
            #45-degree angle - y coordinate of p1 and x coordinate of p2
            #should have equal magnitude but ooposite sign
            self.assertTrue(np.array_equal(-normalP1[::-1],normalP2),"normals of diagonal points don't relate properly");
            self.assertLess(ugeom.distance(centerP1, centerP2),2.0,"centers of diagonal points are too far apart");
            center_seg = ugeom.mid_point_list(ch[2],ch[3])
            center, normal = sampler.find_box_base(edge, 2, center_seg)
            self.assertLess(ugeom.distance(np.array(center,dtype=np.float32),
                                 np.array(center_seg,dtype=np.float32)), 1.0,
                            "base center incorrect for seg 2")
            normal_seg = ugeom.normal(ch[2], ch[3])
            angle_normal_seg = math.degrees(math.atan2(*normal_seg))
            angle_derived = math.degrees(math.atan2(*normal))
            self.assertLess(np.abs(angle_derived - angle_normal_seg), 5.0,
                             "normal for seg 2 is more than 5 degrees from segment normal")
    def test_tetris2(self):
        sampler = self.sampler
        raster = cv2.imread('tetris/test_tetris_02.png')
        bw_raster = cv2.cvtColor(raster,cv2.COLOR_BGR2GRAY)
        edges = chain(bw_raster)
        simp_edges = []
        for ch in edges:
            s_edge = simplify_chain(ch)
            simp_edges.append(s_edge)
        ch = simp_edges[0]
        edge = Edge(ch)
        self.assertEqual(len(edge.chain), 103, "incorrect jagged edge chain length")
        point = [66,62]
        center, normal = sampler.find_box_base(edge, 51, point)
        self.assertTrue(np.array_equal(np.array(center,dtype=np.int32),
                                       np.array(point,dtype=np.int32)),
                        "incorrect center point for jagged segment")
        self.assertTrue(np.array_equal(normal,ugeom.normal(ch[0], ch[len(ch)-1])),
                        "base normal incorrect for jagged segment")
                        
        
    
    
    
        
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()