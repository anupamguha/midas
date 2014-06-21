'''
Created on Jul 25, 2013

@author: algomorph
'''
import core
import numpy as np
import utils.data_management as dm
import unittest
import bw.edge as et
import bw.chain as ch
import cv2


class TestSimplify(unittest.TestCase):
    def test_simplify(self):
        origEdgeImages = dm.load_rasters_from_dir('../images/manual/edges')
        redr_edge_images = []
        simp_edge_images = []
        edgeSets = []
        simp_edge_sets = []
        for img in origEdgeImages:
            edges = ch.chain(img)
            edgeSets.append(edges)
            simplified_set = []
            edge_image = np.zeros_like(img)
            simp_edge_image = np.zeros_like(img)
            for chain in edges:
                edge = et.Edge(chain)
                edge.draw(edge_image, 255)
                simp_chain = ch.simplify_chain(chain,1)
                simp_edge = et.Edge(simp_chain)
                simp_edge.draw(simp_edge_image, 255)
                simplified_set.append(simp_chain)
            redr_edge_images.append(edge_image)
            simp_edge_images.append(simp_edge_image)
            simp_edge_sets.append(simplified_set)
            redr_matches_orig = np.array_equal(img, edge_image)
            if not redr_matches_orig:
                bad_pixels = np.argwhere(np.not_equal(img,edge_image).astype(np.uint8))
                copy = cv2.cvtColor(np.copy(img),cv2.COLOR_GRAY2BGR)
                for pixel in bad_pixels:
                    copy[pixel[0],pixel[1]] = (0,0,255)
                for chain in edges:
                    for pt in chain:
                        copy[pt[0],pt[1]] = (255,128,2)
                cv2.imwrite('bad_pixels.png',copy)
            self.assertTrue(redr_matches_orig,
                            "The redrawn edge image does not match the original image."+
                            " Percentage of unmatched pixels: %f" % 
                            (float(np.not_equal(img,edge_image).sum())/img.size))
            simp_matches_orig = np.array_equal(img, simp_edge_image) 
            if not simp_matches_orig:
                bad_pixels = np.argwhere(np.not_equal(img,simp_edge_image).astype(np.uint8))
                copy = cv2.cvtColor(np.copy(img),cv2.COLOR_GRAY2BGR)
                for pixel in bad_pixels:
                    if(simp_edge_image[pixel[0],pixel[1]] != 0):
                        copy[pixel[0],pixel[1]] = (0,0,255)
                for simp_chain in simplified_set:
                    for pt in simp_chain:
                        copy[pt[0],pt[1]] = (255,128,2)
                cv2.imwrite('bad_pixels.png',copy)    
            self.assertTrue(simp_matches_orig,
                            "The simplified edge image does not match the original image."+
                            " Percentage of unmatched pixels: %f" % 
                            (float(np.not_equal(img,simp_edge_image).sum())/img.size))
    
        
if __name__ == '__main__':
    unittest.main()