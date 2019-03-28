

import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage

from sklearn import cluster
from sklearn.cluster import KMeans

from skimage import data, io, color, filters
from skimage import exposure
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte, random_noise


# creates a mask given the height and width of an image, center of the circle, and radius
def create_circular_mask(h, w, center=None, radius=None):
    Y, X = np.ogrid[0:h, 0:w]
    # get distance from center for each pixel
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    # get pixels within the circle
    mask = dist_from_center <= (radius)
    return mask

def image_preprocessing(image, plot = False):
    # set pixel values to between -1 and 1
    pic = np.interp(image, (image.min(), image.max()), (-1, +1))

    # change exposure, so the light from the well edge is prominent
    image = exposure.equalize_adapthist(pic, clip_limit=0.03)

    # set grey pixels to white to make the edge of the well cleaner
    white_px = np.asarray(1.0)
    black_px = np.asarray(0.3)
    (row, col) = image.shape

    for r in range(row):
        for c in range(col):
            px = image[r][c]
            if px > black_px:
                image[r][c] = white_px

    
    # use canny algorithm for edge detection
    edges = canny(image, sigma=1, low_threshold = .5, high_threshold= .7)

    # use hough algorithm to detect circles
    # look for radii between 175 and 240
    hough_radii = np.arange(175,240, 1)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 10 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=10)
    # select the smallest circle (I want the inner edge of the well)
    val, idx = min((val, idx) for (idx, val) in enumerate(radii))
    
    y_coord = cy[idx]
    x_coord = cx[idx]
    radius = radii[idx]
    
    # calculate circle perimeter from the center and the radius
    #circy, circx = circle_perimeter(y_coord, x_coord, radius)

    #perim_im = pic.copy()
    #perim_im[circy, circx] = (0)    

    
    # get dimensions of the images
    h, w = image.shape[:2]
    # get pixels within the circle area
    mask = create_circular_mask(h, w, center = [x_coord, y_coord], radius = radius)
    masked_img = pic.copy()
    masked_img[~mask] = 1
    

    # reduce noise 
    sigma = 0.155
    masked_img_scale = np.interp(masked_img, (masked_img.min(), masked_img.max()), (0, +1))
    bilateral_img1 = denoise_bilateral(masked_img_scale, sigma_color=0.05, sigma_spatial=15,
                                      multichannel=False)
    #bilateral_img = np.interp(bilateral_img1, (bilateral_img1.min(), bilateral_img1.max()), (0, +255))

    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                           sharex=True, sharey=True)

        plt.gray()
        
        ax[0, 0].imshow(pic)
        ax[0, 0].axis('off')
        ax[0, 0].set_title('Original')                       

        ax[0, 1].imshow(edges)
        ax[0, 1].axis('off')
        ax[0, 1].set_title('Canny edge detection')
        
        ax[1, 0].imshow(bilateral_img1)
        ax[1, 0].axis('off')
        ax[1, 0].set_title('Perimeter')    
        
        ax[1, 1].imshow(bilateral_img)
        ax[1, 1].axis('off')
        ax[1, 1].set_title('Masked and smoothed')  
        fig.tight_layout()

        plt.show()

    return bilateral_img1