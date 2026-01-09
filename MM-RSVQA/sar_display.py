

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal 

def extract(im, bbox):
    
    ind_row = np.arange(bbox[1],bbox[3]+1)
    ind_col = np.arange(bbox[0],bbox[2]+1)
    ind_row_int = ind_row.astype(int)
    ind_col_int = ind_col.astype(int)
    
    extract = im[ind_row_int,:]
    extract = extract[:,ind_col_int]
    
    return extract

def threshSAR(im,thresh = 3, exp = 1):
    
    im = return2zeros(im)
    val_max  = computeTreshMax(im,thresh)
    im = applyTreshMax(im,val_max)
    im = im**exp
    #print(im)
    return im

def return2zeros(im):
    
    im = abs(im)
    val_min = np.amin(im)
    im = im-val_min
    
    return im

def computeTreshMax(im,thresh):
    
    mean = np.mean(im)
    std = np.std(im)
    val_max = mean + thresh*std
    
    return val_max

def computeTreshMin(im,thresh):
    
    mean = np.mean(im)
    std = np.std(im)
    val_min = max(0,mean - thresh*std)
    
    return val_min
    
def applyTreshMax(im,val_max):
    
    mask=im<val_max
    im=im*mask+(1-mask)*val_max
    im = im/val_max
    
    return im

def applyTreshMin(im,val_min):
    
    mask=im>val_min
    im=im*mask+(1-mask)*val_min
    
    return im
    
def imShow(im,thresh = 3, exp = 1, colorbar=False, xFontSize=16, yFontSize=16):
    
    im_thresh = threshSAR(im,thresh,exp)
    
    fig, ax = plt.subplots()
    h = ax.imshow(im_thresh, cmap = 'gray')
    ax.tick_params(axis='x', labelsize=xFontSize)
    ax.tick_params(axis='y', labelsize=yFontSize)
    if colorbar == True:
        cbar = fig.colorbar(h, ax = ax)
        cbar.set_label(label='amplitude',size = 16)
    
    return fig, ax, im_thresh

def imCompareSameTresh(im1,im2,thresh=3,exp=1):
    
    im1_thresh = threshSAR(im1,thresh,exp)
    im2_thresh = threshSAR(im2,thresh,exp)
    
    im_rgb = np.zeros((im1.shape[0],im1.shape[1],3))
    im_rgb[:,:,0] = im1_thresh
    im_rgb[:,:,1] = (im1_thresh+im2_thresh)/2
    im_rgb[:,:,2] = im2_thresh
    
    return im_rgb

def imCompare3SameTresh(im1,im2,im3, thresh=3,exp=1):
    
    im1_thresh = threshSAR(im1,thresh,exp)
    im2_thresh = threshSAR(im2,thresh,exp)
    im3_thresh = threshSAR(im3,thresh,exp)
    
    im_rgb = np.zeros((im1.shape[0],im1.shape[1],3))
    im_rgb[:,:,0] = im1_thresh
    im_rgb[:,:,1] = im2_thresh
    im_rgb[:,:,2] = im3_thresh
    
    return im_rgb

def imCompareSameTreshMinMax(im1,im2,threshMax, threshMin,exp):
    
    im1 = return2zeros(im1)
    im2 = return2zeros(im2)
    
    val_min_1 = computeTreshMin(im1,threshMin)
    val_min_2 = computeTreshMin(im2,threshMin)
    im1 = applyTreshMin(im1,val_min_1)-val_min_1
    im2 = applyTreshMin(im2,val_min_2)-val_min_2
    
    val_max_1  = computeTreshMax(im1,threshMax)
    val_max_2  = computeTreshMax(im2,threshMax)
    im1 = (applyTreshMax(im1,val_max_1))**exp
    im2 = (applyTreshMax(im2,val_max_2))**exp
    
    im_rgb = np.zeros((im1.shape[0],im1.shape[1],3))
    im_rgb[:,:,0] = im1
    im_rgb[:,:,1] = (im1+im2)/2
    im_rgb[:,:,2] = im2
    
    return im_rgb
     
     
def imCompareSameDynamic(im1,im2,threshMax,threshMin,exp):
    
    im1 = return2zeros(im1)
    im2 = return2zeros(im2)
    
    val_min = computeTreshMin(im1,threshMin)
    im1 = applyTreshMin(im1,val_min)-val_min
    im2 = applyTreshMin(im2,val_min)-val_min
    
    val_max  = computeTreshMax(im1,threshMax)
    im1 = (applyTreshMax(im1,val_max)/val_max)**exp
    im2 = (applyTreshMax(im2,val_max)/val_max)**exp
    
    im_rgb = np.zeros((im1.shape[0],im1.shape[1],3))
    im_rgb[:,:,0] = im1
    im_rgb[:,:,1] = (im1+im2)/2
    im_rgb[:,:,2] = im2
    
    return im_rgb
    
def imInterfoHSV(im_complex,thresh = [0, 1]):
    
    
    color = np.angle(im_complex)
    im_coherence = np.abs(im_complex)
    imRGB = imHSV(color, im_coherence, thresh = thresh,range_color = [-np.pi, np.pi], range_saturation = [0,1])
    
    return imRGB
        

def imHSV(color, saturation, thresh = [0, 1], **kwargs):
    range_color = kwargs.get('range_color', [np.min(color), np.max(color)])
    range_saturation = kwargs.get('range_saturation', [np.min(saturation), np.max(saturation)])
    
    if color.shape !=saturation.shape: 
        error_string = "Color image should have the same size than the Saturation image"
        raise ValueError(error_string)
    else:
        [Ny, Nx] = color.shape
        
    extend = thresh[1]-thresh[0]
    color = (color - range_color[0])/(range_color[1]-range_color[0])
    color[color<0] = 0
    color[color>1] = 1
    saturation = (saturation - range_saturation[0])/(range_saturation[1]-range_saturation[0])
    saturation[saturation<0] = 0
    saturation[saturation>1] = 1
    tresh_saturation = thresh[0]+extend*saturation
    
    mat_hsv = np.ones([Ny, Nx, 3])
    mat_hsv[:,:,0] = color
    mat_hsv[:,:,1] = tresh_saturation
#    mat_hsv[:,:,2] = tresh_saturation
    im_rgb = mcolors.hsv_to_rgb(mat_hsv)
    
    return im_rgb

def imPolarTresh(imPolar, thresh = 3, exp = 1):
    
    im = return2zeros(imPolar)
    val_max  = computeTreshMax(im,thresh)
    
    Npolar = imPolar.shape[2]
    for ind in range(Npolar):
        im[:,:,ind] = applyTreshMax(im[:,:,ind],val_max)**exp
    
    return im

def boxcarFilter(imIn,windowShape):
    '''
    Local estimation using the maximum likelyhood estimation of a rayleigh distribution :
    mu = sqrt(sum(abs(pixel)**2)/N) where N is the number of pixels used in the sum
    '''
    element = np.ones(windowShape) # filtering kernel, just a rectangle

    imConv = signal.convolve2d(np.abs(imIn)**2, element, mode = 'same') 
    #since the mode is 'same', the output image has the same size than the input

    denomIm = np.ones(imIn.shape) 
    denomConv = signal.convolve2d(denomIm, element, mode = 'same')
    #since the mode is 'same', the border of the image are average less than the middle 
    #this line count the number of pixels used in the averaging

    imFilter = np.sqrt(imConv/denomConv)

    return imFilter
    


def interfero(im1,im2, windowShape):
    
    element = np.ones(windowShape)
    cov = signal.convolve2d(im1*np.conj(im2), element, mode = 'same')
    denom1 = signal.convolve2d(im1*np.conj(im1), element, mode = 'same')
    denom2 = signal.convolve2d(im2*np.conj(im2), element, mode = 'same')
    
    interfero = cov/np.sqrt(denom1*denom2)
    
    return interfero

def geoTransform(imIn, orbit_direction, look_direction):
    
    if (look_direction == 'R'):
        if (orbit_direction == 'A'):
            imOut = np.flipud(imIn)
        elif (orbit_direction == 'D'):
            imOut = np.fliplr(imIn)
        else:
            raise ValueError('this type of orbit direction is unknown. Only Ascending and Descending orbits are taken into account')
    elif (look_direction == 'L'):
        imOut = imIn
    else:
        raise ValueError('this type of look direction is unknown. Only Right and Left looking sensors are taken into account')
    
    return imOut


def imShowGeo(imIn, orbit_direction, look_direction, resA = 1, resR = 1, thresh = 3, exp = 1, colorbar=False, xFontSize=16, yFontSize=16,):
    imOut = geoTransform(imIn, orbit_direction, look_direction)
    fig, ax = imShow(imOut, thresh = thresh, exp = exp, colorbar = colorbar, xFontSize = xFontSize, yFontSize = yFontSize)
    ax.set_aspect(resA/resR)
    
    return fig, ax

def flightTransform(imIn, orbit_direction, look_direction):
    
    if (look_direction == 'R'):
        if (orbit_direction == 'A'):
            imOut = np.rot90(imIn, 3)
        elif (orbit_direction == 'D'):
            imOut = np.rot90(imIn)
        else:
            raise ValueError('this type of orbit direction is unknown. Only Ascending and Descending orbits are taken into account')
    elif (look_direction == 'L'):
        if (orbit_direction == 'A'):
            imOut = np.rot90(imIn)
        elif (orbit_direction == 'D'):
            imOut = np.rot90(imIn,3)
        else:
            raise ValueError('this type of orbit direction is unknown. Only Ascending and Descending orbits are taken into account')

    else:
        raise ValueError('this type of look direction is unknown. Only Right and Left looking sensors are taken into account')
    
    return imOut

def imShowFlight(imIn, orbit_direction, look_direction, resA = 1, resR = 1, thresh = 3, exp = 1, colorbar=False, xFontSize=16, yFontSize=16):
    imOut = flightTransform(imIn, orbit_direction, look_direction)
    fig, ax = imShow(imOut, thresh = thresh, exp = exp, colorbar = colorbar, xFontSize = xFontSize, yFontSize = yFontSize)
    ax.set_aspect(resR/resA)
    
    return fig, ax    


   

