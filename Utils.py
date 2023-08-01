import numpy as np
import torch
import math
import sys
sys.path.append('../')
from scipy.io import savemat
import os
import scipy.misc
#from spectrum import fftshift
#from tensorflow.python.ops.signal.helper import fftshift
#from tensorflow.python.ops.signal.helper import ifftshift 
#from tensorflow.python import roll as _roll
#from tensorflow.python.framework import ops
#from tensorflow.python.util.tf_export import tf_export


def fft2c(img):
    """ Centered fft2 """
    return np.fft.fft2(img) / np.sqrt(img.shape[-2]*img.shape[-1])

def ifft2c(img):
    """ Centered ifft2 """
    return np.fft.ifft2(img) * np.sqrt(img.shape[-2]*img.shape[-1])

def np_mriAdjointOp(rawdata, coilsens, mask):
    """ Adjoint MRI Cartesian Operator """
    mask = np.expand_dims( mask.astype(np.float32), axis=1)
    return np.sum(ifft2c(rawdata * mask)*np.conj(coilsens), axis=1)

def np_mriForwardOp(img, coilsens, mask):
    """ Forward MRI Cartesian Operator """
#    mask = np.expand_dims( mask.astype(np.float32), axis=1)
#    img = np.expand_dims( img, axis=1)
#    
    return fft2c(coilsens * img)*mask

def make_blocks_vectorized(x,d):
    if len(x.shape) == 3:
        p,m,n = x.shape
        return x.reshape(-1,m//d,d,n//d,d).permute(1,3,0,2,4).reshape(-1,d,d)
    if len(x.shape) == 4:
        p,ch,m,n = x.shape
        return x.reshape(-1,ch,m//d,d,n//d,d).permute(0,2,4,1,3,5).reshape(-1,ch,d,d)

def unmake_blocks_vectorized(x,d,m,n):     #corpd_target,30,300,300
    if len(x.shape) == 3:
        return x.reshape(m//d,n//d,-1,d,d).permute(2,0,3,1,4).reshape(-1,m,n)
    if len(x.shape) == 4:
        return x.reshape(-1,m//d,n//d,18,d,d).permute(0,3,1,4,2,5).reshape(-1,18,m,n)

def psnr(imag1, imag2):
    mse = np.mean( ( abs(imag1) - abs(imag2) ) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = abs(imag1).max()
    relative_error = np.linalg.norm( abs(imag1) - abs(imag2), 'fro' )/np.linalg.norm( abs(imag1), 'fro')
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)), relative_error  

def mriForwardOp(img, sampling_mask):
    # centered Fourier transform
    Fu = torch.fft.fft2(img) #torch.Tensor(data, requires_grad=True)
    # apply sampling mask
    partial_kspace = torch.complex(torch.real(Fu) * sampling_mask, torch.imag(Fu) * sampling_mask)
    return partial_kspace

def mriAdjointOp(f, sampling_mask):
    # apply mask and perform inverse centered Fourier transform
    Finv = torch.fft.ifft2(torch.complex(torch.real(f) * sampling_mask, torch.imag(f) * sampling_mask))
    return Finv 

def removeFEOversampling(src):
    """ Remove Frequency Encoding (FE) oversampling.
        This is implemented such that they match with the DICOM images.
    """
    assert src.ndim >= 2
    nFE, nPE = src.shape[-2:]
    if nPE != nFE:
        return np.take(src, np.arange(int(nFE*0.25)+1, int(nFE*0.75)+1), axis=-2)
    else:
        return src
    
def removePEOversampling(src):
    """ Remove Phase Encoding (PE) oversampling. """
    nPE = src.shape[-1]
    nFE = src.shape[-2]
    PE_OS_crop = (nPE - nFE) / 2

    if PE_OS_crop == 0:
        return src
    else:
        return np.take(src, np.arange(int(PE_OS_crop)+1, nPE-int(PE_OS_crop)+1), axis=-1)

def removeFE(src):
    assert src.ndim >= 2
    nFE, nPE = src.shape[-2:]
    return np.take(src, np.arange(int(nFE*0.25)+1, int(nFE*0.75)+1), axis=-2)

def removePE(src):
    nPE = src.shape[-1]
    nFE = src.shape[-2]
    PE_OS_crop = (nPE - nFE) / 2

    return np.take(src, np.arange(int(PE_OS_crop)+1, nPE-int(PE_OS_crop)+1), axis=-1)


def saveAsMat(img, filename, matlab_id, mat_dict=None):
    """ Save mat files with ndim in [2,3,4]

        Args:
            img: image to be saved
            file_path: base directory
            matlab_id: identifer of variable
            mat_dict: additional variables to be saved
    """
    assert img.ndim in [2, 3, 4]

    img_arg = img.copy()
#    if img.ndim == 3:
#        img_arg = np.transpose(img_arg, (1, 2, 0))
#    elif img.ndim == 4:
#        img_arg = np.transpose(img_arg, (2, 3, 0, 1))

    if mat_dict == None:
        mat_dict = {matlab_id: img_arg}
    else:
        mat_dict[matlab_id] = img_arg

    dirname = os.path.dirname(filename) or '.'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    savemat(filename, mat_dict)

    
def _normalize(img):
    """ Normalize image between [0, 1] """
    tmp = img - np.min(img)
    tmp /= np.max(tmp)
    return tmp

def contrastStretching(img, saturated_pixel=0.004):
    """ constrast stretching according to imageJ
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm"""
    values = np.sort(img, axis=None)
    nr_pixels = np.size(values)
    lim = int(np.round(saturated_pixel*nr_pixels))
    v_min = values[lim]
    v_max = values[-lim-1]
    img = (img - v_min)*(255.0)/(v_max - v_min)
    img = np.minimum(255.0, np.maximum(0.0, img))
    return img


def getContrastStretchingLimits(img, saturated_pixel=0.004):
    """ constrast stretching according to imageJ
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm"""
    values = np.sort(img, axis=None)
    nr_pixels = np.size(values)
    lim = int(np.round(saturated_pixel*nr_pixels))
    v_min = values[lim]
    v_max = values[-lim-1]
    return v_min, v_max

def normalize(img, v_min, v_max, max_int=255.0):
    """ normalize image to [0, max_int] according to image intensities [v_min, v_max] """
    img = (img - v_min)*(max_int)/(v_max - v_min)
    img = np.minimum(max_int, np.maximum(0.0, img))
    return img


def imsave(img, filepath, normalize=True):
    """ Save an image. """
    path = os.path.dirname(filepath) or '.'
    if not os.path.exists(path):
        os.makedirs(path)

    if img.dtype == np.complex64 or img.dtype == np.complex128:
        print('img is complex! Take absolute value.')
        img = np.abs(img)

    if normalize:
        img = _normalize(img)
        img *= 255.0
    scipy.misc.imsave(filepath, img.astype(np.uint8))
    