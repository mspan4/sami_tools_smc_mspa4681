"""This package contains a number of routines to test SAMI and other data reduced 
using the 2dFdr pipeline.  These were written with SAMI data specifically in mind, 
but in most cases can be used for data from HERMES, 2dF/AAOmega, SPIRAL/AAOmega etc.
This includes some functions taken from other SAMI packages (e.g. sami.utils.other.find_fibre_table
so that there are no major dependenceies other than standard python packages.
"""

import matplotlib
import pylab as py
import numpy as np
import numpy.ma as ma
import scipy as sp
import scipy.ndimage as nd
import scipy.ndimage.filters as filters
import astropy.io.fits as pf
import astropy.time as at
#import pyfits as pf
import math
import subprocess
import os
import os.path
import sys
import colorsys
import glob
import time
import shutil
import re
#import spectres

import scipy.stats as stats
import scipy.interpolate as interp
import scipy.signal as signal
from scipy.interpolate import griddata
from scipy import special
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
from matplotlib._png import read_png

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, AnchoredOffsetbox

import astropy.convolution as convolution
from astropy.convolution.kernels import CustomKernel


##############################################################################
# script to centroid the 5577 sky line to check sky subtraction and
# wavelength calibration
#
def centroid_5577(infile,doplot=True,return_results=False):


    # open file:
    hdulist = pf.open(infile)

    lam5577=5577.346680
    
    # get data and variance
    im = hdulist[0].data
    var = hdulist['VARIANCE'].data

    # get array sizes:
    (ys,xs) = im.shape
    print('input array sizes:',ys,' x ',xs)

    nfib = ys
    npix = xs
    
    # try and get the sky spectum, but if its 
    # not there, assume this is an arc frame and so 
    # set sky = 0
    try:
        sky = hdulist['SKY'].data
    except KeyError:
        print("SKY extension not found!")
        sky = np.zeros(xs)

    # get wavelength info:
    primary_header=hdulist['PRIMARY'].header
    crval1=primary_header['CRVAL1']
    cdelt1=primary_header['CDELT1']
    crpix1=primary_header['CRPIX1']
    naxis1=primary_header['NAXIS1']
    x=np.arange(naxis1)+1
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    lam=L0+x*cdelt1

    # find the pixel closest to 5577 sky line:
    i_sky5577 = int((lam5577-L0)/cdelt1)
    print('sky line near pixel: ',i_sky5577)
    
    # get binary table info:
    fib_tab_hdu=find_fibre_table(hdulist)
    table_data = hdulist[fib_tab_hdu].data
    types=table_data.field('TYPE')
    
    # next add back in the sky spectrum to the data and then
    im_sky = im + sky

    # define arrays for storage:
    fibnum=np.zeros(nfib)
    line_cent=np.zeros(nfib) 
    
    
    # define the median background around the sky line, ignoring the core
    # of the line:

    j1=i_sky5577-20
    j2=i_sky5577+20
    j1_in=i_sky5577-5
    j2_in=i_sky5577+5

    #py.figure(2)
        
    for i in range(nfib):
        background = (np.nanmedian(im_sky[i,j1:j1_in])+np.nanmedian(im_sky[i,j2_in:j2]))/2.0

        
        #py.plot(lam[j1:j2],im_sky[i,j1:j2]-background)

        tmp_sky =im_sky[i,:]-background

        # calculate centroid:
        cent = np.sum(tmp_sky[j1_in:j2_in]*lam[j1_in:j2_in])/np.sum(tmp_sky[j1_in:j2_in])
        #print i,background,cent
        fibnum[i] = i+1
        line_cent[i] = cent

    if (doplot): 
        py.figure(1)
        py.plot(fibnum,line_cent-lam5577,label='all fibres')
        for i in range(nfib):
            if ((types[i] == 'S')):
                py.plot(fibnum[i],line_cent[i]-lam5577,'o',color='r',label='sky fibres')
        py.xlabel('fibre number')
        py.ylabel('Wavelength offset (Angstroms)')

    if (return_results):
        return line_cent,types
    else:
        return
            
    
#############################################################################

def find_fibre_table(hdulist):
    """Returns the extension number for FIBRES_IFU, MORE.FIBRES_IFU FIBRES or MORE.FIBRES,
    whichever is found. Modified from SAMI versiuon that only uses FIBRES_IFU.
    Raises KeyError if neither is found."""

    extno = None
    try:
        extno = hdulist.index_of('FIBRES')
    except KeyError:
        pass

    if extno is None:
        try:
            extno = hdulist.index_of('MORE.FIBRES')
        except KeyError:
            pass

    if extno is None:            
        try:
            extno = hdulist.index_of('FIBRES_IFU')
        except KeyError:
            pass
        
    if extno is None:
        try:
            extno = hdulist.index_of('MORE.FIBRES_IFU')
        except KeyError:
            raise KeyError("Extensions 'FIBRES_IFU' and "
                           "'MORE.FIBRES_IFU' both not found")
    return extno
