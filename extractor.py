#!/usr/bin/env python
import sys
import argparse
import pandas as pd
import cv2
import numpy as np
import utils.gradient as gd

color_conversions = {
                   'BGR2LAB':cv2.COLOR_BGR2LAB,
                   'BGR2LUV':cv2.COLOR_BGR2LUV,
                   'BGR2HSV':cv2.COLOR_BGR2HSV_FULL,
                   'BGR2HLS':cv2.COLOR_BGR2HLS_FULL,
                   'BGR2YUV':cv2.COLOR_BGR2YUV,
                   'BGR2GRAY':cv2.COLOR_BGR2GRAY,
                   'None':None
                   }

def grad(sample):
	'''
	Compute average direction & magnitude of the gradient for a 3-channel raster
	'''
	grad_mag,grad_dir = gd.grad_sobel(sample)
	mag_means = (grad_mag[:,:,0].mean(),grad_mag[:,:,1].mean(),grad_mag[:,:,2].mean())
	mag_stds = (grad_mag[:,:,0].std(),grad_mag[:,:,1].std(),grad_mag[:,:,2].std())
	dir_means = (grad_dir[:,:,0].mean(),grad_dir[:,:,1].mean(),grad_dir[:,:,2].mean())
	dir_stds = (grad_dir[:,:,0].std(),grad_dir[:,:,1].std(),grad_dir[:,:,2].std())
	return mag_means + mag_stds + dir_means + dir_stds

#add feature extraction functions to this hash
feature_extract_functions = {
	'grad':grad,
}

feature_columns = {
	'grad':['gdm_mean0',
			'gdm_mean1',
			'gdm_mean2',
			'gdm_std0',
			'gdm_std1',
			'gdm_std2',
			'gdd_mean0',
			'gdd_mean1',
			'gdd_mean2',
			'gdd_std0',
			'gdd_std1',
			'gdd_std2']
}

parser = argparse.ArgumentParser(description="Sample Feature Extractor\n"+\
                                 " A tool for extracting features from pixel"+\
                                 " samples surrounding edges.")
parser.add_argument("--samples","-s",default="features/2/man_edges/samples.pd",
                    help="path to the pandas dataframe file containing images")
parser.add_argument("--color_conversion","-c",default='BGR2HSV',metavar="COLOR_CONVERSION",
                    choices=color_conversions.keys(),
                    help=("color conversion for 3-channel samples. Cane be one of "
                             + str(color_conversions.keys())))
parser.add_argument("--feature","-f",default=feature_extract_functions.keys()[0],metavar="FEATURE",
                    choices=feature_extract_functions.keys(),
                    help=("type of feature to extract. Can be one of "
                             + str(feature_extract_functions.keys())))
parser.add_argument("--verbose","-v",type=int,default=1,metavar="verbosity level")

if __name__ == "__main__":
	args = parser.parse_args(sys.argv[1:])
	path = args.samples
	verbose = args.verbose
	samples = pd.read_pickle(path)
	n_samples = len(samples)

	#take care of color conversions
	color_conversion = color_conversions[args.color_conversion]
	if color_conversion is None:
		prep_color = lambda sample: sample
	else:
		prep_color = lambda sample: cv2.cvtColor(sample,color_conversion)

	#choose which function to use
	extract_feature = feature_extract_functions[args.feature]
	feature_cols = feature_columns[args.feature]
	features = np.zeros((n_samples,len(feature_cols)), dtype=np.float64)

	for ix_row, (image_num,edge_ix,is_foreground,raster) in samples.iterrows():
		sample = prep_color(raster)
		features[ix_row] = extract_feature(sample)
		if(verbose > 0):
			frac_done = float(ix_row+1) / n_samples
			print '{:.3%} done'.format(frac_done),
			sys.stdout.flush()
			print "\r",
	print '{:.3%} done'.format(1.0)
	ix_col = 0


	for col in feature_cols:
		samples[col] = features[:,ix_col]
		ix_col += 1
	samples.to_pickle(path)




