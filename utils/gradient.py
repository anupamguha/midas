import cv2
import numpy as np

def get_channel_gradient_scharr(channel):
    ddepth = cv2.CV_32F
    delta = 0
    scale = 1
    grad_x = cv2.Scharr(channel,ddepth,1,0,scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
    grad_x_sq = grad_x * grad_x
    grad_y = cv2.Scharr(channel,ddepth,0,1,scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    grad_y_sq = grad_y * grad_y
    grad_mag = np.sqrt(grad_x_sq + grad_y_sq)
    grad_dir = np.arctan2(grad_x,grad_y)
    return grad_mag, grad_dir
    
def get_channel_gradient_sobel(channel):
    ddepth = cv2.CV_32F
    delta = 0
    scale = 1
    grad_x = cv2.Sobel(channel,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
    grad_x_sq = grad_x * grad_x
    grad_y = cv2.Sobel(channel,ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    grad_y_sq = grad_y * grad_y
    grad_mag = np.sqrt(grad_x_sq + grad_y_sq)
    grad_dir = np.arctan2(grad_x,grad_y)
    return grad_mag, grad_dir

def grad_sobel(raster):
    c0m,c0d = get_channel_gradient_sobel(raster[:,:,0])
    c1m,c1d = get_channel_gradient_sobel(raster[:,:,1])
    c2m,c2d = get_channel_gradient_sobel(raster[:,:,2])
    grad_mag = np.dstack([c0m,c1m,c2m])
    grad_dir = np.dstack([c0d,c1d,c2d])
    return grad_mag, grad_dir
