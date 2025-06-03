
import matplotlib
import pylab as py
import numpy as np
import numpy.ma as ma
import numpy.polynomial.polynomial as polynomial
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
from scipy import fftpack
from matplotlib.colors import LogNorm
#import spectres

import scipy.stats as stats
import scipy.special as special
import scipy.interpolate as interp
import scipy.optimize as optimize
import scipy.signal as signal
from scipy.interpolate import griddata
from scipy import special
from scipy.optimize import least_squares
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
#from matplotlib._png import read_png

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, AnchoredOffsetbox

import astropy.convolution as convolution
from astropy.convolution.kernels import CustomKernel
import astropy.nddata as nddata

from .sami_stats_utils import gaus2d,lorentzian,gaussian,polyfitr,clipped_mean_rms,gaussian_filter_nan

#################################################################################
#
def back_sub(imfile,backfile):

    hdulist1 = pf.open(backfile)
    backim = hdulist1[0].data    

    hdulist2 = pf.open(imfile)
    im = hdulist2[0].data

    sub = im-backim
    
    hdu_out = pf.PrimaryHDU(sub)
    slfile=imfile.replace('im.fits','sub.fits')
    hdu_out.writeto(slfile,overwrite=True)
 
#################################################################################
# Read in a reduced flat and do some simple stats on it (plus a few figures):
#
def fibflat_stats(imfile):

    hdulist = pf.open(imfile)
    im = hdulist[0].data    

    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.hist(np.ravel(im),bins=100,range=[0.8,1.2],histtype='step',label=imfile)
    py.legend(prop={'size':8})
    ax1.set(xlabel='fibre flat value',ylabel='Number')
    # get stats:
    stdev = np.nanstd(np.ravel(im))
    p05 = np.nanpercentile(np.ravel(im),5.0)
    p95 = np.nanpercentile(np.ravel(im),95.0)
    print('frame: ',imfile,'   std dev:',stdev)
    print('frame: ',imfile,'   5th-95th percentile range: ',p05,p95,p95-p05)
    
 
    
##################################################################################
# generate a model image with many PSFs...
#
def gen_model_im(imsize=1000,sigma_noise=2.0,scale=1.0e5):

    # make a large image:
    im = np.zeros((imsize,imsize))

    # generate 
    psf = psf_model_a(a=2,b=0.08,maxval=10,doplot=False,return_psf=True,write_psf=False)

    psfys,psfxs = np.shape(psf)
    print('psf size: ',psfys,psfxs)
    
    # loop through image and place a psf down at certain locations:
    for i in range(1+40,imsize-40,5):
        
        ycent = float(i)
        xcent = float(imsize)/2
        ix1 = int(xcent)-int(psfxs/2)
        ix2 = ix1+int(psfxs)
        iy1 = int(ycent)-int(psfys/2)
        iy2 = iy1+int(psfys)
        #print(ix1,ix2,iy1,iy2)

        imtmp = im[iy1:iy2,ix1:ix2]

        im[iy1:iy2,ix1:ix2]=im[iy1:iy2,ix1:ix2]+psf
        
        print(np.shape(imtmp))

    noise_im = np.random.normal(0.0,sigma_noise,(imsize,imsize))
    im = im*scale+noise_im
    
    fig1=py.figure(1)
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    ax1.imshow(im,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)

    hdu_out = pf.PrimaryHDU(im)
    hdu_out.writeto('im_multi_psf.fits',overwrite=True)
        

##################################################################################
# generate a model psf:
#
def psf_model_a(a=0.01,b=0.5,maxval=0.1,doplot=True,verbose=True, return_psf=False,write_psf=True):

    xs = 2000
    ys = 2000
    xstart = -40
    xend   = 40
    pixsize = float(xend-xstart)/float(xs)
    print('pixel size: ',pixsize)
    print('1/pixel size: ',1/pixsize)
    x1 = np.linspace(xstart,xend,xs)
    y1 = np.linspace(xstart,xend,ys)
    x, y = np.meshgrid(x1, y1) # get 2D variables instead of 1D
    # get the gaussian:
    z = gaus2d(x, y)

    # multiply by sinc function in each direction separately:
    ysinc = (np.sinc(a*x1)*np.sinc(b*x1))**2
    yysinc = np.tile(ysinc,(xs,1))
    xxsinc = np.transpose(yysinc)

    # multiply gaussian by sinc**2 functions:
    psf1 = yysinc * xxsinc

    # now assume the fibre core (pior to diffraction) is a gaussian with sigma of 1.
    # we need to convolve the two together so we get
    
    psf = nd.gaussian_filter(psf1,int(1/pixsize))
    
    # now resample to single pixels:
    #psf_resample = sami_stats_utils.block_mean(psf,int(1/pixsize))
    psf_resample = nddata.block_reduce(psf,int(1/pixsize))    
    
    if (doplot):
        fig1=py.figure(1)
        fig1.clf()

    # generate a scaled and noisy vesion:
    yss,xss = np.shape(psf_resample)
    sigma_noise=2.0
    noise_im = np.random.normal(0.0,sigma_noise,(yss,xss))
    psf_sim = (psf_resample * 1.0e5)  + noise_im
    
    if (doplot):
        ax1 = fig1.add_subplot(231)
        ax1.imshow(z,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        ax2 = fig1.add_subplot(232)
        ax2.plot(x1,ysinc,'-',color='r')

        ax3 = fig1.add_subplot(233)
        cax3 = ax3.imshow(psf,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu,vmin=0.0,vmax=maxval)
        cbar3 = fig1.colorbar(cax3)
    
        ax4 = fig1.add_subplot(234)
        cax4 = ax4.imshow(np.log10(psf),origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu,vmin=-10,vmax=1.0)
        cbar4 = fig1.colorbar(cax4)

        ax5 = fig1.add_subplot(235)
        cax5 = ax5.imshow(np.log10(psf_resample),origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu,vmin=-10,vmax=1.0)
        cbar5 = fig1.colorbar(cax5)

        ax6 = fig1.add_subplot(236)
        #cax6 = ax6.imshow(psf_resample,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu,vmin=0,vmax=maxval)
        cax6 = ax6.imshow(psf_sim,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu,vmin=-2,vmax=10)
        cbar6 = fig1.colorbar(cax6)

        py.draw()

    if (write_psf):
        hdu_out = pf.PrimaryHDU(psf)
        hdu_out.writeto('psf_test.fits',overwrite=True)
        hdu_out = pf.PrimaryHDU(psf_resample)
        hdu_out.writeto('psf_resample_test.fits',overwrite=True)
        hdu_out = pf.PrimaryHDU(psf_sim)
        hdu_out.writeto('psf_sim_test.fits',overwrite=True)

    if (return_psf):
        return psf_resample
    else:
        return
    

##############################################################################
# read in data from 2dfdr trace scattered light fitting
#
def plot_2dfdr_sl_trace_line(gappixfile,dopdf=True):


    # read the columns in the files
    gapnum, flux, xval, yval, mflux, scont, sflux = np.loadtxt(gappixfile, usecols=(0, 2, 4, 5, 6, 7, 8), unpack=True)

    # get the number of gaps:
    ngap1 = np.min(gapnum)
    ngap2 = np.max(gapnum)
    print('first gap is number ',ngap1)
    print('last gap is number ',ngap2)

    # random scaling of mflux:
    #mflux = mflux *2000
    
    # start plotting:
    fig1 = py.figure(1)

    gax = np.linspace(1,14,14)
    ggamma = np.zeros(14)
    ggamma_err = np.zeros(14)
    glorsum = np.zeros(14)
    ggamma_m = np.zeros(14)
    ggamma_err_m = np.zeros(14)
    glorsum_m = np.zeros(14)
    scales = np.zeros(14)
    scales_sum = np.zeros(14)
    scales_y = np.zeros(14) 

    
    
    # loop through each gap:
    for igap in range(int(ngap1),int(ngap2+1)):
        ax1 = fig1.add_subplot(211)
        ax2 = fig1.add_subplot(212)
        print(igap)
        idx1 = np.where(gapnum==igap)

        xmin=np.min(xval[idx1])
        xmax=np.max(xval[idx1])
        xax = np.linspace(xmin,xmax,100)

        scales_y[igap-1]=np.nanmedian(yval[idx1])
        
        # bin the data and 2d model:
        nbins=40
        binsize=(xmax-xmin)/nbins
        xbin = np.linspace(xmin+binsize/2,xmax-binsize/2,nbins)

        # data first:
        bin_means, bin_edges, bin_number = stats.binned_statistic(xval[idx1],sflux[idx1], statistic='mean', bins=nbins,range=[xmin,xmax])
        bin_stddev, bin_edges, bin_number = stats.binned_statistic(xval[idx1],sflux[idx1], statistic='std', bins=nbins,range=[xmin,xmax])
        bin_count, bin_edges, bin_number = stats.binned_statistic(xval[idx1],sflux[idx1], statistic='count', bins=nbins,range=[xmin,xmax])
        bin_stddev = bin_stddev/np.sqrt(bin_count)
        # now model:
        bin_means_m, bin_edges, bin_number = stats.binned_statistic(xval[idx1],mflux[idx1], statistic='mean', bins=nbins,range=[xmin,xmax])
        bin_stddev_m, bin_edges, bin_number = stats.binned_statistic(xval[idx1],mflux[idx1], statistic='std', bins=nbins,range=[xmin,xmax])
        bin_count_m, bin_edges, bin_number = stats.binned_statistic(xval[idx1],mflux[idx1], statistic='count', bins=nbins,range=[xmin,xmax])
        bin_stddev_m = bin_stddev_m/np.sqrt(bin_count_m)

        # do a simple lorentzian fit.  Fit to the binned data to reduce the scatter:
        pin = (10.0,1850.0,30.0)
        #(poptl2,cov2)=sp.optimize.curve_fit(lorentzian,xbin,bin_means,p0=pin)
        (poptl2,cov2)=sp.optimize.curve_fit(gaussian,xbin,bin_means,p0=pin)
        #lmod_data = lorentzian(xax,*poptl2)
        lmod_data = gaussian(xax,*poptl2)
        ggamma[igap-1] = np.abs(poptl2[2])
        ggamma_err[igap-1] = cov2[2,2]
        glorsum[igap-1] = poptl2[0]/(np.pi*ggamma[igap-1])
        #cont = poptl2[3]

        # also fit to 2d model:
        #(poptl2,cov2)=sp.optimize.curve_fit(lorentzian,xbin,bin_means_m,p0=pin)
        #lmod_mod = lorentzian(xax,*poptl2)
        (poptl2,cov2)=sp.optimize.curve_fit(gaussian,xbin,bin_means_m,p0=pin)
        lmod_mod = gaussian(xax,*poptl2)
        ggamma_m[igap-1] = np.abs(poptl2[2])
        ggamma_err_m[igap-1] = cov2[2,2]
        glorsum_m[igap-1] = poptl2[0]/(np.pi*ggamma[igap-1])
        #cont_m = poptl2[3]

        print('fitted sigma:',ggamma[igap-1])
        print('fitted sigma (model):',ggamma_m[igap-1])
        #print(cont,cont_m)
        
        # calculate the average scaling between the pixel model and the data:
        #av_scale = np.nanmean((bin_means-cont)/bin_means_m)
        av_scale = np.nanmean((bin_means)/bin_means_m)
        av_scale = np.sum(sflux[idx1])/np.sum(mflux[idx1])
        print('average scaling:',av_scale)

        # do LSQ for av_scale:
        x0=av_scale
        result = least_squares(fun_min, x0,args=(bin_means,bin_means_m,bin_stddev))
        print(result.x)
        av_scale=result.x
        scales[igap-1] = av_scale

        # or try summing the flux and looking at difference:
        dsum = np.nansum(sflux[idx1])
        msum = np.nansum(mflux[idx1])
        print('summed fluxes:',dsum,msum,dsum/msum)
        scales_sum[igap-1] = dsum/msum 

        
        # plot 
        ax1.errorbar(xbin,bin_means,bin_stddev,fmt='o',color='r')
        ax1.errorbar(xbin,bin_means_m*av_scale,bin_stddev_m,fmt='o',color='b')


        #ax1.plot(xval[idx1],flux[idx1]-cont,'.',color='k')
        ax1.plot(xval[idx1],sflux[idx1],'.',color='k')
        ax1.plot(xax,lmod_data,'-',color='r')
        ax1.plot(xax,lmod_mod*av_scale,'-',color='b')
        
        ax1.plot(xval[idx1],mflux[idx1]*av_scale,'.',color='g')

        ax2.plot(xval[idx1],sflux[idx1]-mflux[idx1]*av_scale,'.',color='g')
        
        py.draw()
        yn = input("Continue?")
        fig1.clf()

    fig2 = py.figure(2)
    ax1 = fig2.add_subplot(221)
    ax1.errorbar(scales_y,ggamma,ggamma_err,fmt='o',color='b')
    ax1.errorbar(scales_y,ggamma_m,ggamma_err_m,fmt='o',color='r')

    
    
    # get median gamma (or sigma):
    medgam = np.nanmedian(ggamma)
    medgam_cent = np.nanmedian(ggamma[3:10])
    py.axhline(medgam,color='b',linestyle=':')
    print('median gamma:',medgam)
    medgam_m = np.nanmedian(ggamma_m)
    py.axhline(medgam_m,color='r',linestyle=':')
    print('median gamma (from model):',medgam_m)

    # plot some simple models:
    yax = np.linspace(1,4096,4096)
    #ptest1 = medgam_cent*(1.0-6*((yax-2048)/4096)**4)
    #ax1.plot(yax,ptest1,'-',color='g')
    z = np.polyfit(scales_y,ggamma,4)
    p = np.poly1d(z)
    ax1.plot(yax,p(yax),'-',color='g')
    print('best fit sigma poly parameters:',z)

    
    
    ax2 = fig2.add_subplot(222)
    ax2.plot(gax,glorsum,'o',color='b')
    
    ax3 = fig2.add_subplot(223)
    ax3.plot(scales_y,scales,'o',color='b')
    ax3.plot(scales_y,scales_sum,'o',color='g')

    # plot  
    z = np.polyfit(scales_y, scales, 4)
    p = np.poly1d(z)
    print(z)
    ax3.plot(scales_y,p(scales_y),'-',color='r')
    medcent=np.nanmedian(scales[4:8])
    ptest1 = medcent*(1.0+20*((yax-2048)/4096)**4)
    ptest2 = medcent+3.0e5*((yax-2048)/4096)**6
    ptest3 = medcent*(1.0+5*((yax-2048)/4096)**2)
    yaxs = (yax-2048)/4096
    ss = 3.5

    # do LSQ for symmetric quartic:
    x0=[medcent,5.0,12.0]
    scales_y_norm = (scales_y-2048)/4096
    result = least_squares(quart_min, x0,args=(scales_y_norm,scales))
    print(result.x)
    x1 = result.x

    ptest4 = x1[0]*(1.0 + x1[1]*yaxs*yaxs + x1[2]*yaxs**4)
    

    ax3.plot(yax,ptest1,'-',color='g')
    ax3.plot(yax,ptest2,'-',color='c')
    ax3.plot(yax,ptest3,'-',color='m')


    ax3.plot(yax,ptest4,'-',color='y')
    
    
##############################################################################
# simple function to minimize:
#
def fun_min(x,d,m,dsig):

    res = (d-x*m)/dsig

    return res

##############################################################################
# symmetric quadratic to minimize:
#
def quart_min(p,x,y):

    model = p[0] * (1.0 + p[1]*x*x + p[2] * x**4)
    res = y-model

    return res
        
    
##############################################################################
# read in data from 2dfdr trace scattered light fitting
#
# example (old):
# sami_dr_smc.sami_2dfdr_scatlight.plot_2dfdr_sl_trace('12apr10046_outdir/gap_pixvals.dat','12apr10046_outdir/gap_fit.dat','12apr10046_outdir/gap_bin.dat','12apr10046_outdir/background.fits','12apr10046im.fits',nsum=10)
#
# new:
# sami_dr_smc.sami_2dfdr_scatlight.plot_2dfdr_sl_trace('12apr10046_outdir/gap_pixvals.dat','12apr10046_outdir/gap_fit.dat','12apr10046_outdir/gap_bin.dat','12apr10046_outdir/background.fits','12apr10046im.fits',nsum=10)
#
# def plot_2dfdr_sl_trace(gappixfile,gapfitfile,gapbinfile,backfile,imfile,dopdf=True,colstart=0,nsum=1):
#
# New usage:
# sami_dr_smc.sami_2dfdr_scatlight.plot_2dfdr_sl_trace('04apr10027',nsum=10)     

def plot_2dfdr_sl_trace(frame,dopdf=True,colstart=0,nsum=10):


    # convert from frame to all other files:
    gappixfile = frame+'_outdir/gap_pixvals.dat'
    gapfitfile = frame+'_outdir/gap_fit.dat'
    gapbinfile = frame+'_outdir/gap_bin.dat'
    backfile = frame+'_outdir/background.fits'
    imfile = frame+'im.fits'
    
    # read the columns in the files
    # values read are
    # gapnum - number of gap
    # flux - flux value of individual pixels
    # xval - x coordinate of individual pixels
    gapnum, flux, xval, yval = np.loadtxt(gappixfile, usecols=(0, 2, 4, 5), unpack=True)
    #
    gapnumfit, xvalfit, yvalfit, fluxfit = np.loadtxt(gapfitfile, usecols=(0, 1, 2, 3), unpack=True)
    #
    gapnumb, xvalb, yvalb, fluxb,sigmab = np.loadtxt(gapbinfile, usecols=(0, 2, 3, 4, 5), unpack=True)

    # read in the background image:
    hdulist = pf.open(backfile)
    backim = hdulist[0].data
    # get the dimensions of the data:
    (ys,xs) = np.shape(backim)
    print('ys x xs',ys,xs)
    hdulist.close()

    # read in the im file:
    hdulist = pf.open(imfile)
    im = hdulist[0].data
    # get the dimensions of the data:
    (ys,xs) = np.shape(im)
    print('ys x xs',ys,xs)
    hdulist.close()

    # get the number of gaps:
    ngap1 = np.min(gapnum)
    ngap2 = np.max(gapnum)
    print('first gap is number ',ngap1)
    print('last gap is number ',ngap2)

    # start plotting:
    fig1 = py.figure(1)

    # loop through each gap:
    for igap in range(int(ngap1),int(ngap2+1)):
        ax1 = fig1.add_subplot(111)
        print(igap)
        idx1 = np.where(gapnum==igap)
        idx2 = np.where(gapnumfit==igap)
        idx3 = np.where(gapnumb==igap)
        
        ax1.plot(xval[idx1],flux[idx1],'.',color='k',label='individual gap pixels',zorder=1)
        #ax1.plot(xval[idx1],sval[idx1],'.',color='r')
        ax1.errorbar(xvalb[idx3],fluxb[idx3],sigmab[idx3],fmt='o',color='g',label='mean values',zorder=2)
        ax1.plot(xvalfit[idx2],fluxfit[idx2],'-',color='b',label='fit to mean',zorder=3)

        title = 'gap: '+str(igap)
        
        ax1.legend(loc='upper right',prop={'size':8})
        ax1.set(title=title,xlabel='x pixels',ylabel='counts')
        
        py.draw()
        yn = input("Continue?")
        fig1.clf()

    # now plot cuts on background image:
    fig2 = py.figure(2)
    xax = np.linspace(1,ys,ys)
    cont = True
    icol = colstart
    while (cont):
        ax2 = fig2.add_subplot(111)
        print(icol)
        idx1 = np.where(xvalfit==(icol+1))

        istart = max(icol-int(nsum/2),0)
        iend = min(icol+int(nsum/2),xs)
        immean = np.nanmean(im[:,istart:iend],axis=1)
        ax2.plot(xax,immean,'-',color='b')
        #ax2.plot(xax,im[:,icol],'-',color='b')
        ax2.plot(xax,backim[:,icol],'-',color='k')
        ax2.plot(yvalfit[idx1],fluxfit[idx1],'.',color='r')
        py.title('column '+str(icol))
        ymin = np.nanmin(immean,0)
        ymax = np.nanmax(fluxfit[idx1])*2.0
        ax2.set(xlim=[0.0,ys],ylim=[ymin,ymax])
        py.draw()
        yn = input("Continue? (return = y), n=quit, number = skip to col:")
        if ((yn == 'n') | (yn == 'N')):
            cont = False
        elif re.match(r'^[0-9]+$',yn):
            icol = int(yn)
        else:
            icol = icol+1
            

            
        fig2.clf()
        
    
    
    
##############################################################################
# attempt at FFT filtering:
#
def fft_filt(imfile):

 #  first open the files   
    hdulist_im = pf.open(imfile)
    
    # get image:
    
    #im = hdulist_im[0].data
    im = np.require(hdulist_im[0].data, dtype=np.float32)

    fig1=py.figure(1)
    ax1 = fig1.add_subplot(121)
    cax1=ax1.imshow(im,origin='lower',interpolation='nearest')

    im_fft = fftpack.fft2(im)

    ax2 = fig1.add_subplot(122)
    cax1=ax1.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    cbar=fig1.colorbar(cax1)
    


##############################################################################
# Funtion to do a simple convolution of the image for scattered light
# subtraction.  Main aim is to remove enough for frames with bright objects
# that regular scattered light modelling in 2dfdr will work okay.  This means
# that the flux in the gaps should be relatively flat.
#
def scat_light_conv(infile,tlmfile,sigma=20.0,scale=0.1,nsum=10,minsep=3.0,interactive=True,doplot=True,offset=0.0):

    # first open the files   
    hdulist_im = pf.open(infile)
    
    # get image and variance:
    im = hdulist_im[0].data
    im_var = hdulist_im['VARIANCE'].data

    # get the dimensions of the data:
    (ys,xs) = np.shape(im)
    print('image size: ',ys,xs)

    # define an array with SL image:
    im_sl = np.zeros((ys,xs))
    
    # get TLM file:
    hdulist_tlm = pf.open(tlmfile)
    tlm = hdulist_tlm[0].data
    prf = hdulist_tlm['SIGMAPRF'].data
    (ytlms,xtlms) = tlm.shape
    print('TLM image size: ',ytlms,xtlms)

    # location of slices to plot:
    #slice_loc = [200,700,1200,1700,1900]
    #
    slice_loc = [290,670,870,1470,1900]
    nslice = np.size(slice_loc)


    # define an array of all gap pixels, with non-gaps set tp NaN:
    #gaploc=[0,63,126,189,252,315,378,441,504,567,630,693,756,819]
    #gaploc=[63,126,189,252]
    gaploc=[126,189]
    ngaps = np.size(gaploc)

    gap_mask = np.zeros((ys,xs))
    gap_mask[:] = np.nan

    for loc in gaploc:
        # for this gap identify the tlm and sigma around the gaps:
        for ix in range(xs):
            # for the first element in the list (assuming it is an end gap):
            if (loc == 0):
                tlm1 = 0.0
                tlm2 = tlm[loc,ix]
                sig1 = 0.0
                sig2 = prf[loc,ix]
            # for the last element in the list (assuming its an end gap):
            elif (loc == 819):
                tlm1 = tlm[loc-1,ix]
                tlm2 = ys
                sig1 = prf[loc-1,ix]
                sig2 = 0.0
            else:
                tlm1 = tlm[loc-1,ix]
                tlm2 = tlm[loc,ix]
                sig1 = prf[loc-1,ix]
                sig2 = prf[loc,ix]

            # now loop over y pixels in suitable range:
            for iy in range(int(tlm1),int(tlm2)):
                yy = float(iy) + 0.5
                # identify pixels in the gap:
                if ((abs(tlm1 - yy) > (minsep*sig1)) & (abs(tlm2 - yy) > (minsep*sig2))):
                    if (not np.isnan(im[iy,ix])):
                        gap_mask[iy,ix] = 1.0
                
    im_mask = gap_mask * im
    
    if (doplot):
        fig1 = py.figure(1)
        ax11 = fig1.add_subplot(1,3,1)
        ax12 = fig1.add_subplot(1,3,2)
        ax13 = fig1.add_subplot(1,3,3)

        fig3 = py.figure(3)
        ax31 = fig3.add_subplot(1,3,1)
        ax32 = fig3.add_subplot(1,3,2)
        # plot original and convolved image:
        vmin = 0.0
        vmax = np.nanpercentile(im_mask,90.0)
        ax31.cla()
        ax31.imshow(im_mask,vmin=vmin,vmax=vmax,origin='lower',interpolation='nearest')

        fig2 = py.figure(2)
        ax2 = []
        for i in range(nslice):
            ax2.append(fig2.add_subplot(nslice,1,i+1))

        xax = np.linspace(1.0,float(ys),ys)

        
    # interactive loop:
    while True:

    # no-interactive loop:
    #for j in range(20):
    #    sigma = 10.0+float(j)*4.0

        
        # convolve image with filter:
        im_conv = gaussian_filter_nan(im,(sigma,sigma))

        # find the robust best scaling.  This is done by finding the median
        # ratio between the model and the good pixels in gaps:
        ratio = (im_conv)/(im_mask-offset)
        med_ratio = np.nanmedian(ratio)
        #if (doplot):
        #    vmin = 0.0
        #    vmax = np.nanpercentile(im_mask,90.0)
        #    ax32.cla()
        #    ax32.imshow(im_conv,vmin=vmin,vmax=vmax,origin='lower',interpolation='nearest')

        scale = 1/med_ratio

        model = (im_conv*scale) + offset
            
        # plot original and convolved image:
        vmin = 0.0
        vmax = np.nanpercentile(im,90.0)
        ax11.cla()
        ax12.cla()
        ax13.cla()
        ax11.imshow(im,vmin=vmin,vmax=vmax,origin='lower',interpolation='nearest')
        ax12.imshow(model,vmin=vmin,vmax=vmax,origin='lower',interpolation='nearest')
        ax13.imshow(im-model,vmin=vmin,vmax=vmax,origin='lower',interpolation='nearest')

        # calculate the median absolute difference:
        med_abs_diff = np.nanmedian(np.absolute(im_mask-model))

        print('sigma,scale,median_abs_deviation:',sigma,scale,med_abs_diff)
        
        # plot slices at various columns
        for i in range(nslice):
            colnum = slice_loc[i]
            if (nsum == 1):
                y = im[:,colnum]
                y_conv = model[:,colnum]
                n1 = colnum
                n2 = colnum
            else:
                n1 = int(np.rint(colnum-nsum/2))
                n2 = int(np.rint(colnum+nsum/2))
                y = np.nanmean(im[:,n1:n2],axis=1)
                y_conv = np.nanmean(model[:,n1:n2],axis=1)
                
            ax2[i].plot(xax,y)
            ax2[i].plot(xax,y_conv)
            ax2[i].set(xlim=[0.0,1300.0],ylim=[np.nanmin(y_conv)-3.0,np.nanmax(y_conv)+3.0])
            outstr='cols {0:3d}-{1:3d}'.format(n1,n2)
            ax2[i].text(0.05,0.85,outstr, horizontalalignment='left',verticalalignment='center', transform=ax2[i].transAxes)

        # for this model determine how well it matches to the data:
            
        # get input
        py.draw()
        if (interactive):
            str = input('New sigma,offset?')
            str_c = str.split(',')
            sigma = float(str_c[0])
            offset = float(str_c[1])
        

    
##############################################################################
# Test a model for the SAMI scattered light that is based on the convolved
# original image.  This should be a good approximation for the scattering
# wings of the PSF as long as we convolve with the right model.  We will
# also likely need to put a cut off at the edge of the detector.
#
#
def scat_light_conv_model(imfile,tlmfile,sigma,scale,colnum=500,rownum=15,nsum=1,gamma=5.0,profile='gaussian',maxiter=5,interactive=False):

    """Python function to convolve an input *im.fits file to model the scattered
    light in AAOmega.

    """     
    # define time:
    time_start = time.time()

    nsig_step = 10
    sig_step = 5.0
    # start at 10 for arc:
    sig_step_start = 20.0
    # best fit single sigma for an arc:
    #nsig_step = 1
    #sig_step = 1.0
    #sig_step_start = 16.4
    
    #  first open the files   
    hdulist_im = pf.open(imfile)
    
    # get image:
    im = hdulist_im[0].data
    im_var = hdulist_im['VARIANCE'].data

    # get the dimensions of the data:
    (ys,xs) = np.shape(im)
    print('image size: ',ys,xs)

    im_sl_best = np.zeros((ys,xs))
    
    # get TLM file:
    hdulist_tlm = pf.open(tlmfile)
    tlm = hdulist_tlm[0].data
    prf = hdulist_tlm['SIGMAPRF'].data
    (ytlms,xtlms) = tlm.shape
    print('TLM image size: ',ytlms,xtlms)

    # do a medium-scale gaussian filter and use this to define a mask for
    # the emission lines
    V = im.copy()
    V[im!=im]=0
    VV = nd.filters.gaussian_filter(V,5.0)
    
    W = 0*im.copy()+1
    W[im!=im] = 0
    WW = nd.filters.gaussian_filter(W,5.0)

    im_filt = VV/WW

    im_filt_diff = im-im_filt

    #
    im_flag = np.where((im_filt_diff>50.0),np.NAN,im)
    
    # filter
    fsize = 7
    kern = np.ones((fsize,fsize))/float(fsize*fsize)
    im_flag_ex = nd.convolve(im_flag,kern)
    im_flag2 = im_flag_ex
    

    
    minsep=5.0
    doiter=True
    im_conv = np.zeros((ys,xs))
    im_conv_scale = np.zeros((ys,xs))

    # define a mask that defines the gaps
    im_mask = np.zeros((ys,xs))
    im_mask_gapnum = np.zeros((ys,xs))
    im_mask[:,:] = np.NAN 
    im_mask_gapnum[:,:] = np.NAN 
    gaploc=[0,63,126,189,252,315,378,441,504,567,630,693,756,819]
    #gaploc=[378]
    ngaps = np.size(gaploc)
    gaplineloc=np.zeros(ngaps)
    
    # go through all pixels and figure out if its in a gap:
    ngood=0
    # bright blue arc line loc:
    x1 = 1300
    x2 = 1500
    # 5577 line loc:
    #x1 = 1750
    #x2 = 1950

    ng = 0
    for loc in gaploc:
        # find the pixel with the highest flux in the fibre next to the gap:
        peakflux = 0.0
        pealflux_x = (x1+x2)/2
        for ix in range(x1,x2,1):
            tloc = loc 
            if (loc == gaploc[-1]):
                tloc = gaploc[-1]-1
            f = im[int(tlm[tloc,ix]),ix]
            if (f > peakflux):
                peakflux = f
                peakflux_x = ix

        print('peak flux found:',peakflux,peakflux_x)
                
        for ix in range(xs):
            # for the first element in the list (assuming it is an end gap):
            if (loc == 0):
                tlm1 = 0.0
                tlm2 = tlm[loc,ix]
                sig1 = 0.0
                sig2 = prf[loc,ix]
            elif (loc == gaploc[-1]):
                tlm1 = tlm[loc-1,ix]
                tlm2 = ys
                sig1 = prf[loc-1,ix]
                sig2 = 0.0
            else:
                tlm1 = tlm[loc-1,ix]
                tlm2 = tlm[loc,ix]
                sig1 = prf[loc-1,ix]
                sig2 = prf[loc,ix]

            if (ix == peakflux_x):
                gaplineloc[ng] = ((tlm1+sig1) + (tlm2-sig2))/2.0
                
            #tlm1 = tlm[loc-1,ix]
            #tlm2 = tlm[loc,ix]
            #sig1 = prf[loc-1,ix]
            #sig2 = prf[loc,ix]

            
            # Now loop over y pixels in a suitable range:
            for iy in range(int(tlm1),int(tlm2)):
                yy = float(iy) + 0.5
                if ((abs(tlm1 - yy) > (minsep*sig1)) & (abs(tlm2 - yy) > (minsep*sig2))):
                    if (not np.isnan(im[iy,ix])):
                        if ((ix>x1) & (ix < x2) & (abs(ix-peakflux_x) > 5)):
                            im_mask[iy,ix]=1.0
                            im_mask_gapnum[iy,ix] = loc
                            ngood=ngood+1
            
        ng=ng+1
                            
    print('number of good pixels: ',ngood)


    # now step through different values of sigma:

    fit_qual = np.zeros((nsig_step,ngaps))
    fit_sigma = np.zeros((nsig_step,ngaps))
    fit_amp = np.zeros((nsig_step,ngaps))
    qual_best = 1.0e10
    sigma_best = -99.9
    
    for iss in range(nsig_step):
        sigma = sig_step_start + sig_step * float(iss)
        print('optimizing for sigma = ',sigma)

        doiter = True
        niter = 0
        while (doiter):

            # subtract convolved image from data:
            imsub = im - im_conv_scale

            sig2 = sigma/2.0
        
            # convolve the image with a Gaussian.  We need to filter out
            # the NaNs in the image, and this is a method that will do that:
            if (profile == 'gaussian'):
                V = imsub.copy()
                V[imsub!=imsub]=0
                VV = nd.filters.gaussian_filter(V,sigma)
                VV2 = nd.filters.gaussian_filter(V,sig2)

                W = 0*imsub.copy()+1
                W[imsub!=imsub] = 0
                WW = nd.filters.gaussian_filter(W,sigma)
                WW2 = nd.filters.gaussian_filter(W,sig2)

                im_conv1 = VV/WW
                im_conv2 = VV2/WW2

                im_conv=im_conv1 - 0.15 * im_conv2
    
            # alternative using lorentzian filter:
            elif (profile == 'lorentzian'):
                kern = lorentz2D_kernel(401,gamma)
                lkernel = convolution.kernels.CustomKernel(kern)
                im_conv = convolution.convolve_fft(imsub, lkernel,allow_huge=True)

            else:
                print('error, profile not recognized: ',profile)
    
            # set up axes:
            yp = np.linspace(1,ys,ys)

            yps = (yp - float(ys)/2)/float(ys)
            ypim =np.transpose(np.tile(yps,(xs,1)))

            print('ypim size:',np.shape(ypim))

            # do a least squares minimization of the function that will do the subtraction
            # a key point is that we need to define a mask image that is only okay for
            # the gaps, otherwise the fit fails. 
            x0=[0.1]
            #x0=[0.1,0.0,1.0]
            #x0=[0.1,0.0,1.0,0.0,0.0]
            # do the least squares fit for each gap individually:
            ng=0
            for loc in gaploc:
                idx = np.where((np.isfinite(im_mask) & (im_mask_gapnum == loc)))
                result = least_squares(scat_res,x0,args=(np.ravel(im[idx]),np.ravel(im_conv[idx]),np.ravel(ypim[idx])))
                xfit=result.x
                print(xfit)
                # get the vector of residuals at the best solution and get a chisq value:
                npt = np.size(result.fun)
                fitgood = np.sum((result.fun)**2)/npt
                print('fit quality: ',fitgood)
                fit_qual[iss,ng] = fitgood
                fit_sigma[iss,ng] = sigma
                fit_amp[iss,ng] = xfit
                ng=ng+1
            
#        yy = (yp-ys/2)/ys
                yy = yps
        #yscale_fit = xfit[0] * (1.0 + xfit[1] * yy + xfit[2] * yy**2  + xfit[3] * yy**3 + xfit[4] * yy**4)
        # quadratic:
            #yscale_fit = xfit[0] * (1.0 + xfit[1] * yy + xfit[2] * yy**2)
            # constant:
                yscale_fit = xfit[0]

            if (interactive):
                fig1=py.figure(1)
                ax1=fig1.add_subplot(1,1,1)
                ax1.plot(yps,yscale_fit)
        
            # new convolved image with varying scale:
            #im_conv_scale = im_conv * xfit[0] * (1.0 + xfit[1] * ypim + xfit[2] * ypim**2 + xfit[3] * ypim**3 + xfit[4] * ypim**4)
            #im_conv_scale = im_conv * xfit[0] * (1.0 + xfit[1] * ypim + xfit[2] * ypim**2)
            im_conv_scale = im_conv * xfit[0]

            if (fitgood < qual_best):
                qual_best = fitgood
                sigma_best = sigma
                im_sl_best = im_conv_scale
        
            xp = range(xs)

            #plot a slice through the images in the Y direction:
            if (interactive):
                fig2=py.figure(2)
                fig3=py.figure(3)

                # define data to plot:
                if (nsum ==1):
                    flux = im[:,colnum]
                    fluxsub = imsub[:,colnum]
                    flux_conv = im_conv[:,colnum]
                    flux_conv_scale = im_conv_scale[:,colnum]
                    flux_r = im[rownum,:]
                    fluxsub_r = imsub[rownum,:]
                    flux_conv_r = im_conv[rownum,:]
                    flux_conv_scale_r = im_conv_scale[rownum,:]
                else:
                    n1 = int(colnum-nsum/2)
                    n2 = int(colnum+nsum/2)
                    nr1 = int(rownum-nsum/2)
                    nr2 = int(rownum+nsum/2)
                    flux = np.average(im[:,n1:n2],axis=1)
                    fluxsub = np.average(imsub[:,n1:n2],axis=1)
                    flux_conv = np.average(im_conv[:,n1:n2],axis=1)
                    flux_conv_scale = np.average(im_conv_scale[:,n1:n2],axis=1)
                    flux_r = np.average(im[nr1:nr2,:],axis=0)
                    fluxsub_r = np.average(imsub[nr1:nr2,:],axis=0)
                    flux_conv_r = np.average(im_conv[nr1:nr2,:],axis=0)
                    flux_conv_scale_r = np.average(im_conv_scale[nr1:nr2,:],axis=0)
        
                ax21=fig2.add_subplot(1,1,1)
                ax21.plot(yp,flux)
                ax21.plot(yp,flux_conv)
                ax21.plot(yp,flux_conv_scale)
        
                ax31=fig3.add_subplot(2,1,1)
                ax31.plot(xp,flux_r)
                ax31.plot(xp,flux_conv_r)
                ax31.plot(xp,flux_conv_scale_r)
                ax32=fig3.add_subplot(2,1,2)
                ax32.plot(xp,flux_r-flux_conv_scale_r)

                py.draw()
                yn = input("Continue?")
        #fig1.clf()
     
            niter=niter+1
            if (niter > maxiter):
                doiter = False
            

    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(111)

    cols=iter(py.cm.rainbow(np.linspace(0,1,ngaps)))
    best_fit_gaps = np.zeros(ngaps)
    for ig in range(ngaps):
        for iss in range(nsig_step):
            print(ig,iss,fit_sigma[iss,ig],fit_amp[iss,ig],fit_qual[iss,ig])            

        # fit for the nest solution for each gap:
        ppar = np.polyfit(fit_sigma[:,ig],fit_qual[:,ig],4)
        pp = np.poly1d(ppar)

        xxx = np.linspace(fit_sigma[0,ig],fit_sigma[-1,ig],1000)
        idx_min = np.argmin(pp(xxx))
        minqual = xxx[idx_min]
        #only the min for a quadratic:
        #minqual = -1.0 * ppar[1] / (2.0*ppar[0])
        best_fit_gaps[ig] = minqual
        
        print('minqual value of sigma:',minqual)
        c = next(cols)    
        ax1.plot(fit_sigma[:,ig],fit_qual[:,ig],'o',color=c,label=str(gaploc[ig]))
        ax1.plot(xxx,pp(xxx),'-',color=c)
        ax1.axvline(minqual,linestyle='--',color=c)
    
        
    py.legend(prop={'size':8})

    # plot the 
    xgap = np.linspace(1,14,ngaps)
    fig2 = py.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.plot(gaplineloc,best_fit_gaps,'o')
    print(xgap)
    print(gaplineloc)
    print(best_fit_gaps)
    ppar = np.polyfit(gaplineloc,best_fit_gaps,4)
    print(ppar)
    pp = np.poly1d(ppar)
    xxx = np.linspace(gaplineloc[0],gaplineloc[-1],1000)
    ax2.plot(xxx,pp(xxx),'-')
    ppar = np.polyfit(gaplineloc,best_fit_gaps,2)
    print(ppar)
    pp = np.poly1d(ppar)
    ax2.plot(xxx,pp(xxx),'-')
        
    hdu_out = pf.PrimaryHDU(im_sl_best)
    slfile=imfile.replace('im.fits','conv.fits')
    hdu_out.writeto(slfile,overwrite=True)

    hdu_out = pf.PrimaryHDU(im-im_sl_best)
    slfile=imfile.replace('im.fits','sub.fits')
    hdu_out.writeto(slfile,overwrite=True)
    
    hdu_out = pf.PrimaryHDU(im_flag2)
    hdu_out.writeto('im_filt_diff.fits',overwrite=True)


#####################################################################
# function to minimize for scattering profile as a function of slit location.
# this takes in two images and returns the sum of the residuals
#
def scat_res(x,im,slim,ypim):

    #fig4=py.figure(4)
    #ax4=fig4.add_subplot(111)
    #xax = np.linspace(1,xs,xs)
    #res = im - (x[0]* (1.0 + x[1]*ypim + x[2]*ypim**2 + x[3]*ypim**3 + x[4]*ypim**4)) * slim
    # std quadratic:
    #res = im - (x[0]* (1.0 + x[1]*ypim + x[2]*ypim**2)) * slim
    # simple constant:
    res = im - x[0] * slim

    return res
    
#####################################################################
# function to minimize for scattering profile as a function of slit location.
# this takes in two images and returns the sum of the residuals
#
def scat_res_old(x,im,slim,ypim,ys,xs,im_mask):

    #fig4=py.figure(4)
    #ax4=fig4.add_subplot(111)
    #xax = np.linspace(1,xs,xs)
    res = 0
    for i in range(ys):
#        for j in range (xs):
        y = (float(i)-ys/2)/ys
#            if (np.isfinite(im_mask[i,j]):
                
        rrr = im[i,:] - (x[0]* (1.0 + x[1] * y + x[2] * y * y)) * slim[i,:] * im_mask[i,:]
        res_row = np.nansum(im[i,:] - (x[0]* (1.0 + x[1] * y + x[2] * y * y)) * slim[i,:] * im_mask[i,:])
        if (np.isfinite(res_row) & (res_row != 0)):
            res = res + res_row
            print(i,rrr)
            #ax4.plot(xax,im[i,:])
            #ax4.plot(xax,slim[i,:])
            #py.draw()
            #print(i,res_row)
            #yn = input("Continue?")

        
    print(res,x)
    return res
    

##############################
def lorentz2D(x,y,gamma):
    """2D Lorentz window (same as Cauchy function)
    """
    lor = gamma/(x**2+y**2+gamma**2)**1.5/(2*math.pi)

    return lor

#############################
def lorentz2D_kernel(n,gamma):

    kern = np.zeros([n,n])

    cent = int(n/2)

    for i in range(n):
        for j in range(n):

            x = float(i-cent)
            y = float(j-cent)
            kern[i,j] = lorentz2D(x,y,gamma)

    tot = np.sum(kern)
    kern = kern / tot
            
    return kern

###############################################################################
def rational(x, p, q):
    """
    The general rational function description.
    p is a list with the polynomial coefficients in the numerator
    q is a list with the polynomial coefficients (except the first one)
    in the denominator
    The zeroth order coefficient of the denominator polynomial is fixed at 1.
    Numpy stores coefficients in [x**2 + x + 1] order, so the fixed
    zeroth order denominator coefficent must comes last. (Edited.)
    """
    return np.polyval(p, x) / np.polyval(q + [1.0], x)


################################################################################
# prototype scattered ligh model:
#
#def sl_poly_erf(x,p0,p1,p2,p3,q1,q2,q3):
def sl_poly_erf(x,*args):

    #val1 = rational(x,[p0,p1,p2,p3],[q1,q2,q3])
#    print(args)
    a = args[0]
    b = args[1]
    val1 = polynomial.polyval(x,args[2:-1])

    #val1 = np.where((x < erf_start1),polynomial.polyval(erf_start1,args),val1)
    #val1 = np.where((x > erf_start2),polynomial.polyval(erf_start2,args),val1)

    #val2 = c+special.erf((x-erf_start1)/sl_sigma)*val1
    val2 = a+b*x+special.erf((x-erf_start1)/sl_sigma1)*val1*special.erf((erf_start2-x)/sl_sigma2)
    
    return val2
    

################################################################################
# prototype scattered ligh model:
#
def sl_poly_erf2(x,*args):

    val1 = polynomial.polyval(x,args)

    val2 = special.erf((x-erf_start1)/sl_sigma)*val1*special.erf((erf_start2-x)/sl_sigma)
    
    return val2
    

################################################################################

def trace_sl(frame,tlmfile,arcfile,doplot=True,verbose=True,nsum=1,profile='gaussian',sigma=50.0,gamma=5.0,scale=0.1):
    """Function to trace the scattered light profile along each gap between fibre bundles.  The aim
    is to see how well we can fit this (e.g. using a spline) to build a more robust constraint on the
    scattered light model.  The physical constraint is that the scattered light needs to vary approximately
    smoothly in the wavelength direction.  We aim to make sure of this property.

    Based on plot_optex and fit_sl
    frame name without extension, e.g. 06mar20055 the tlm file to use and
    a column number to plot.
    Usage:
    sami_2dfdr_reduction_tests.trace_sl('06mar10065','06mar10045tlm_orig.fits','06mar10048red.fits',nsum=10)

    """

    # define order of spline (between 1 and 5, typically 3):
    kx=3
    ky=1

    # column to slice for wavelength shifts.  Ideally this should be either
    # at the location of a particularly strong feature (e.g. 5577) or in the middle
    # of the detector.  This seams like a reasonable compromise for typical setups:
    lslicecol = 1500
    
    # Minsep in terms of sigma for us to consider as gaps:
    minsep = 4.0

    # binsize (in x direction) for binning up a gap for testing:
    binsize = 10
    
    # plotting offset for gaps:
    doff = 5.0
    
    # open files:
    imfile = frame+'im.fits'
    hdulist_im = pf.open(imfile)
    hdulist_tlm = pf.open(tlmfile)
    hdulist_arc= pf.open(arcfile)

    # get data:
    im = hdulist_im[0].data
    tlm = hdulist_tlm[0].data
    prf = hdulist_tlm['SIGMAPRF'].data

    # get variance:
    var = hdulist_im['VARIANCE'].data
    
    # get WAVELA image and SHIFTS from arc:
    wavela = hdulist_arc['WAVELA'].data
    shifts = hdulist_arc['SHIFTS'].data
    
    # get header keywords:
    primary_header=hdulist_im['PRIMARY'].header
    gain = primary_header['RO_GAIN']
    ron = primary_header['RO_NOISE']
    obstype = primary_header['OBSTYPE']
    detector = primary_header['DETECTOR']
    spectid = primary_header['SPECTID']  # BL or RD
    
    # Note y is first axis in python array, x is second
    (ys,xs) = im.shape
    # Note y is first axis in python array, x is second
    (ytlms,xtlms) = tlm.shape

    # get the wavelength shifts:
    wslice = np.zeros(ytlms)
    tslice = np.zeros(ytlms)
    # ideally transform this to the true wavelength (using SHIFTS array), but
    # not yet implemented.
    wslice = wavela[:,lslicecol]
    tslice = tlm[:,lslicecol]

    # fit polynomial to wslice and tslice:
    coeff = polyfitr(tslice,wslice,2,5.0)
    pol = np.poly1d(coeff)
    mslice=pol(tslice)

    yslice = np.arange(ys)+0.5
    myslice=pol(yslice)
    minlam = np.min(myslice)

    # estimate pixsize from difference between first and last value in WAVELA:
    # note that the units of the wavalength in the WAVELA array are nm not Angstroms!
    ytlmcent = int(ytlms/2)
    pixsize = (wavela[ytlmcent,xtlms-1] - wavela[ytlmcent,0])/(float(xs)-1.0)
    if (verbose):
        print('Pixel size (in nm): ',pixsize)

    # transform the shift to pixels:
    dslice = (myslice - minlam) / pixsize

    # get the max shift:
    maxshift = int(math.ceil(np.max(dslice)))

    if (verbose):
        print('max shift:',maxshift)
    
    # generate a shifted image (mostly for testing, not actually used):
    im_shift = np.zeros((ys,xs+int(maxshift)))
    for i in range(ys):
        ishift = int(round(dslice[i]))
        im_shift[i,0+ishift:xs+ishift] = im[i,:]


    # plot the polynomial vs wavelength:
    if (doplot):
        py.figure(1)
        py.plot(tslice,wslice,label='data')
        py.plot(yslice,myslice,'--',label='polynomial fit')
        py.xlabel('pixel')
        py.ylabel('wavelength (nm)')
        py.legend(prop={'size':8})
    
        py.figure(2)
        py.subplot(1,3,1)
        py.imshow(im,vmin=0.0,vmax=np.percentile(im,90.0),origin='lower',interpolation='nearest')
        py.colorbar()
        py.subplot(1,3,2)
        py.imshow(im_shift,vmin=0.0,vmax=np.percentile(im,90.0),origin='lower',interpolation='nearest')
        py.colorbar()
        
    # define an output image for the SL model:
    sl_image = np.zeros((ys,xs))
    sl_image_1d = np.zeros((ys,xs))
    sl_image_fn = np.zeros((ys,xs))
    sl_image_bs = np.zeros((ys,xs))
    # define to be NaN unless we fill it with something:
    sl_image[:,:] = np.NAN 

    if (verbose):
        print(frame,', image size:',ys,' x ',xs)
        print(frame,', tlm size:',ytlms,' x ',xtlms)

    x=np.arange(ys)+1

    # filter/smooth the original image to get an estimate of what the scattered light should look like
    # this can be a gaussian or lorentzian:
    if (profile == 'gaussian'):
        V = im.copy()
        V[im!=im]=0
        VV = nd.filters.gaussian_filter(V,sigma)

        W = 0*im.copy()+1
        W[im!=im] = 0
        WW = nd.filters.gaussian_filter(W,sigma)

        im_conv = (VV/WW) * scale 
    
    # alternative using lorentzian filter:
    elif (profile == 'lorentzian'):
        kern = lorentz2D_kernel(401,gamma)
        lkernel = convolution.kernels.CustomKernel(kern)
        im_conv = convolution.convolve_fft(im, lkernel,allow_huge=True) * scale

    else:
        print('error, profile not recognized: ',profile)
    
        
    # loop over each gap, and identify pixels that are in it that can be used
    # for the fit:
    # Currently we just use the location of the gaps:
    #gaploc=[-3,-2,-1,0,63,126,189,252,315,378,441,504,567,630,693,756,819]
    gaploc=[0,63,126,189,252,315,378,441,504,567,630,693,756,819]
    ngaps = np.size(gaploc)
    gap5577loc = np.zeros(ngaps)

    # calculate the location of 5577 for each gap:
    for i in range(ngaps):
        gapfib = gaploc[i]
        if (gapfib < 1):
            gapfib=1
        mindif = 1.0e10
        iloc = -1
        for j in range(xs):
            if (abs(wavela[gapfib-1,j]-557.7) < mindif):
                mindif = abs(wavela[gapfib-1,j]-557.7)
                # add 50 to iloc as we have not used the shifts array to get the real offset:
                iloc = j+50
        gap5577loc[i] = iloc

    print('location of 5577 line: ',gap5577loc)
    
    #
    # shorter test examples:
    #gaploc=[63]
    #gaploc=[63,126,189]
    #
    # arrays to store data:
    gapflux=np.zeros((xs*40))
    gapsig=np.zeros((xs*40))
    gapflux_conv=np.zeros((xs*40))
    gapcol=np.zeros((xs*40))
    gapallx=np.zeros(xs*400)
    gapally=np.zeros(xs*400)
    gapallflux=np.zeros(xs*400)
    gapallflux_conv=np.zeros(xs*400)
    sl_diff=np.zeros(xs*400)


    gap_model_flux_spl=np.zeros((xs,ngaps))

    gap_y_pos = np.zeros((xs,ngaps))
    
    binflux=np.zeros(xs*400)
    binsig=np.zeros(xs*400)
    bincol=np.zeros(xs*400)
    binflux_conv=np.zeros(xs*400)
    binsig_conv=np.zeros(xs*400)
    bincol_conv=np.zeros(xs*400)

    slmean = np.zeros(xs)
    slmedian = np.zeros(xs)
    slrms = np.zeros(xs)
    slmean2 = np.zeros(xs)
    slmean3 = np.zeros(xs)
    slrms2 = np.zeros(xs)
    slbin = np.zeros(xs)
    slmean_conv = np.zeros(xs)
    slmean3_conv = np.zeros(xs)
    slrms_conv = np.zeros(xs)
    slbin_conv = np.zeros(xs)

    ygappos = np.zeros((xs,np.size(gaploc)))
    xgappos = np.zeros((xs,np.size(gaploc)))
    fgapspl = np.zeros((xs,np.size(gaploc)))
    fgapspl_conv = np.zeros((xs,np.size(gaploc)))

    peakgap = np.zeros(np.size(gaploc))
    
    # arrays for lorentzian fits:
    lor_peak = np.zeros(np.size(gaploc))
    lor_pos = np.zeros(np.size(gaploc))
    lor_gamma = np.zeros(np.size(gaploc))
    g_peak = np.zeros(np.size(gaploc))
    g_pos = np.zeros(np.size(gaploc))
    g_sigma = np.zeros(np.size(gaploc))
    
        
    # set up the colours for plotting:
    if (doplot):
        cols=iter(py.cm.rainbow(np.linspace(0,1,ngaps)))
    
    #
    # loop over gaps:
    noff = 0
    nloc = 0
    ngapa = 0
    for loc in gaploc:

        ngap = 0
        # loop over columns and find valid gap pixels:
        for ix in range(xs):
            # for the first element in the list (assuming it is an end gap):
            if (loc == 0):
#                tlm1 = 0.0
# Test to define the lower edge gap as less than 10 pixels from fibre.  Increase to 15 for test!!!!!
                tlm1 = max(tlm[loc,ix]-15.0,0.0)
                tlm2 = tlm[loc,ix]
                sig1 = 0.0
                sig2 = prf[loc,ix]
# extra edge gaps:
            elif (loc == -1):
                tlm1 = max(tlm[0,ix]-15.0,0.0)
                tlm2 = max(tlm[0,ix]-10.0,0.0)
                sig1 = 0.0
                sig2 = 0.0
            elif (loc == -2):
                tlm1 = max(tlm[0,ix]-20.0,0.0)
                tlm2 = max(tlm[0,ix]-15.0,0.0)
                sig1 = 0.0
                sig2 = 0.0
            elif (loc == -3):
                tlm1 = max(tlm[0,ix]-25.0,0.0)
                tlm2 = max(tlm[0,ix]-20.0,0.0)
                sig1 = 0.0
                sig2 = 0.0
            # for the last element in the list (assuming its an end gap):
            elif (loc == gaploc[-1]):
                tlm1 = tlm[loc-1,ix]
                tlm2 = ys
                sig1 = prf[loc-1,ix]
                sig2 = 0.0
            else:
                tlm1 = tlm[loc-1,ix]
                tlm2 = tlm[loc,ix]
                sig1 = prf[loc-1,ix]
                sig2 = prf[loc,ix]
            # also loop over y pixels in a suitable range:
            yysum = 0
            for iy in range(int(tlm1),int(tlm2)):
                yy = float(iy) + 0.5
                # identify pixels in the gap:
                if ((abs(tlm1 - yy) > (minsep*sig1)) & (abs(tlm2 - yy) > (minsep*sig2))):
                    if (not np.isnan(im[iy,ix])):
                        # calculate the shift in wavelength (in units of pixels):
                        # dx = (pol(yy)-minlam)/pixsize
                        dx = 0
                        # list for just this gap (for testing):
                        gapflux[ngap] = im[iy,ix]
                        gapsig[ngap] = np.sqrt(var[iy,ix])
                        gapflux_conv[ngap] = im_conv[iy,ix]
                        gapcol[ngap] = float(ix)+0.5+dx
                        # list for all gaps together:
                        gapallx[ngapa] = float(ix)+0.5+dx
                        gapally[ngapa] = yy
                        gapallflux[ngapa] = im[iy,ix]
                        gapallflux_conv[ngapa] = im_conv[iy,ix]
                        
                        ngap =ngap +1
                        ngapa =ngapa +1
                        
            # derive the location in y (row) of the gap trace:
            ygappos[ix,nloc] = (tlm1 + tlm2)/2.0

        #peakgap[nloc] = np.max(gapflux_conv[0:ngap])
        if (verbose):
            print('ngap,ngapa:',ngap,ngapa)
            print('average gap pixels per col:',float(ngap)/float(xs))

        # try averaging the values (with some clipping) in bins:
        nbin = int(xs/binsize)
        for ib in range(nbin):
            ibin1 = ib * binsize
            ibin2 = (ib+1) * binsize
            nb = 0
            for ig in range(ngap):
                if ((gapcol[ig] > ibin1) & (gapcol[ig] < ibin2)):
                  binflux[nb] = gapflux[ig]
                  binflux_conv[nb] = gapflux_conv[ig]
                  nb =nb + 1  

            # get the mean in the bin (and sigma):
            slmean[ib],slrms[ib],nclip = clipped_mean_rms(binflux[0:nb],3.0)
            slrms[ib] = slrms[ib]/np.sqrt(nclip)
            slmedian[ib] = np.nanmedian(binflux[0:nb])
            
            slmean_conv[ib],slrms_conv[ib],nclip = clipped_mean_rms(binflux_conv[0:nb],3.0)
            slrms_conv[ib] = slrms_conv[ib]/np.sqrt(nclip)
            
            slbin[ib] = (ibin1+ibin2)/2.0

        # filter out obvious outlying pixels.   The first way to do this is via their anomolous variance.
        # outlying pixls tend to have high variance:
        medrms = np.nanmedian(slrms[0:nbin])
        per95rms = np.nanpercentile(slrms[0:nbin],95.0)
        print('median, 95 percentile rms:',medrms,per95rms)
        #reject points with 2x the rms of the 95th percentile:
        slmean = np.where((slrms>2.0*per95rms),np.nan,slmean)
        print(slmean[0:nbin])
        print(np.count_nonzero(~np.isnan(slmean[0:nbin])))

        # get the median ratio between the gaps and the convolved data:
        conv_ratio = np.nanmedian(slmean[0:nbin]/slmean_conv[0:nbin])

        print('ratio of gap flux to convolved image: ',conv_ratio)
        
        # now fit a bspline to the flux along the gap.  generate values at each x pixel based on this fit.
        ngk = 8
        step = xs/ngk
        tgx = np.linspace(0.0+step,float(xs)-step,num=ngk)
        print('gap knots: ',tgx)
        idx = np.where(np.isfinite(slmean[0:nbin]))
        gap_spline = interp.splrep(slbin[idx],slmean[idx],task=-1,t=tgx,k=3)
        xgappos[:,nloc] = np.linspace(0.0,float(xs-1),num=xs)
        fgapspl[:,nloc] = interp.splev(xgappos[:,nloc],gap_spline)

        # do a second pass.  Check for individual pixels that are more than X sigma from
        # the first pass spline fit.
        # try averaging the values (with some clipping) in bins:
        for ib in range(nbin):
            ibin1 = ib * binsize
            ibin2 = (ib+1) * binsize
            nb = 0
            for ig in range(ngap):
                gc = gapcol[ig]
                #gcf = interp.splev(gc,gap_spline)
                gcf = fgapspl[int(gc),nloc]
                if ((gapcol[ig] > ibin1) & (gapcol[ig] < ibin2)):
                    if (abs(gcf-gapflux[ig])<5.0*gapsig[ig]):
                        binflux[nb] = gapflux[ig]
                        binflux_conv[nb] = gapflux_conv[ig]
                        nb =nb + 1
                    else:
                        print('point clipped: ',gcf,gapflux[ig],gapsig[ig])

            # get the mean in the bin (and sigma):
            slmean2[ib],slrms2[ib],nclip = clipped_mean_rms(binflux[0:nb],3.0)
            slrms2[ib] = slrms2[ib]/np.sqrt(nclip)
            slmedian[ib] = np.nanmedian(binflux[0:nb])
            
            slmean_conv[ib],slrms_conv[ib],nclip = clipped_mean_rms(binflux_conv[0:nb],3.0)
            slrms_conv[ib] = slrms_conv[ib]/np.sqrt(nclip)
            
            slbin[ib] = (ibin1+ibin2)/2.0
        
        # now fit a bspline to the flux along the gap AGAIN.  generate values at each x pixel based on this fit.
        # for red arm, this is okay:
        #ngk = 6
        #step = xs/ngk
        #tgx = np.linspace(0.0+step,float(xs)-step,num=ngk)
        #print('gap knots: ',tgx)
        # for blue arm:
        ngk = 5
        step = xs/ngk
        tgx = np.linspace(0.0+step,float(xs)-step,num=ngk)
        #tgx_t = np.linspace(0.0+step,float(xs)-step,num=ngk-2)
        #tgx = np.append(tgx_t,[gap5577loc[nloc]-50.0,gap5577loc[nloc]+50.0])
        print('gap knots: ',tgx)

        # make a list of points that does not include 5577 region:
        c1 = int(gap5577loc[nloc]-70)
        c2 = int(gap5577loc[nloc]+70)
        print(c1,c2)
        #slbin_cut = np.concatenate((slbin[0:c1],slbin[c2:nbin]))
        #slmean_cut = np.concatenate((slmean2[0:c1],slmean2[c2:nbin]))
        #idx = np.where(((np.isfinite(slmean2[0:nbin])) & (slbin[0:nbin] < c1) & (slbin[0:nbin] > c2)))
        idx = np.where((np.isfinite(slmean2[0:nbin]) & ((slbin[0:nbin] < c1) | (slbin[0:nbin] > c2))))
        print(idx)
        print(slbin[idx])
        gap_spline = interp.splrep(slbin[idx],slmean2[idx],task=-1,t=tgx,k=3)
        xgappos[:,nloc] = np.linspace(0.0,float(xs-1),num=xs)
        fgapspl[:,nloc] = interp.splev(xgappos[:,nloc],gap_spline)
        gap_spline_conv = interp.splrep(slbin[idx],slmean_conv[idx],task=-1,t=tgx,k=3)
        fgapspl_conv[:,nloc] = interp.splev(xgappos[:,nloc],gap_spline_conv)

        
        # fit a gaussian around the scattered light for 5577, as this is a reasonably model
        # and can take into account the increasing flux towards the edges more closely.
        #
        # first subtract the spline fit from the data:
        # this is wrong!!!!
        #slmean3 = slmean2 - fgapspl[:,nloc]
        #slmean3_conv = slmean_conv - fgapspl_conv[:,nloc]

        for i in range(nbin):
            slmean3[i] = slmean2[i] - interp.splev(slbin[i],gap_spline)
            slmean3_conv[i] = slmean_conv[i] - interp.splev(slbin[i],gap_spline_conv)
            print(slbin[i],slmean2[i],fgapspl[i,nloc],slmean3[i],interp.splev(slbin[i],gap_spline))

        #
        # define data around the line:
        c1 = int(gap5577loc[nloc]-150)
        c2 = int(gap5577loc[nloc]+150)
        idx = np.where((slbin[0:nbin] > c1) & (slbin[0:nbin] < c2) & np.isfinite(slmean3[0:nbin]))
        #
        # do gaussian fit:
        pg2 = (10.0,gap5577loc[nloc],10.0)
        (poptg2,cov2)=sp.optimize.curve_fit(gaussian,slbin[idx],slmean3[idx],p0=pg2,sigma=slrms2[idx])
        print('gaussian fit:',poptg2)
        #
        # do lorentzian fit:
        #pl2 = (10.0,gap5577loc[nloc],10.0)
        pl2 = (10.0,gap5577loc[nloc],28.50)
#        (poptl2,cov2)=sp.optimize.curve_fit(lorentzian,slbin[idx],slmean3[idx],p0=pl2,sigma=slrms2[idx],bounds=([0.0,1500.0,28.49],[1.0e4,2000.0,28.51]))
        (poptl2,cov2)=sp.optimize.curve_fit(lorentzian,slbin[idx],slmean3[idx],p0=pl2,sigma=slrms2[idx],bounds=([0.0,1500.0,5.0],[1.0e4,2000.0,100.0]))
        print('lorentzian fit:',poptl2)
        #
        # generate the model:
        gmod = gaussian(xgappos[:,nloc],*poptg2)
        lmod = lorentzian(xgappos[:,nloc],*poptl2)
        lor_peak[nloc] = poptl2[0]
        lor_pos[nloc] = poptl2[1]
        lor_gamma[nloc] = abs(poptl2[2])        
        g_peak[nloc] = poptg2[0]
        g_pos[nloc] = poptg2[1]
        g_sigma[nloc] = abs(poptg2[2])        

        # fit the lorentzian profile to the convolved data in the gaps as well:
        (poptl2c,cov2)=sp.optimize.curve_fit(lorentzian,slbin[idx],slmean3_conv[idx],p0=pl2,bounds=([0.0,1500.0,28.49],[1.0e4,2000.0,28.51]))
        gmod_conv = gaussian(xgappos[:,nloc],*poptg2)
        lmod_conv = lorentzian(xgappos[:,nloc],*poptl2c)
        #lor_peak[nloc] = poptl2[0]
        #lor_pos[nloc] = poptl2[1]
        #lor_gamma[nloc] = abs(poptl2[2])        
        peakgap[nloc] = poptl2c[0]
        print('lorentzian gap fit results for convolved data:',poptl2c)
        
        
        if (doplot):
            c = next(cols)
            py.figure(3)
            py.errorbar(slbin[0:nbin],slmean[0:nbin]+noff,slrms[0:nbin],fmt='o',label=str(loc),color=c)
            py.plot(slbin[0:nbin],slmean2[0:nbin]+noff,'^',color=c)
            py.plot(slbin[0:nbin],slmean_conv[0:nbin]+noff,'--',color=c)
            #py.plot(slbin[0:nbin],slmedian[0:nbin]+noff,'--',color=c)
            py.plot(xgappos[:,nloc],fgapspl[:,nloc]+noff,'-',color=c)
            py.plot(xgappos[:,nloc],fgapspl[:,nloc]+noff+lmod,':',color=c)
            py.xlim(xmin=0.0,xmax=2100.0)

            #py.axvline(gap5577loc[nloc],color='b',linestyle='--')
            for i in range(ngk):
                py.axvline(tgx[i],color='g',linestyle=':')

            # plot just the fit around the 5577 line:
            py.figure(4)
            py.errorbar(slbin[idx],slmean3[idx]+noff,slrms2[idx],fmt='o',label=str(loc),color=c)
            py.axhline(noff,color=c,linestyle=':')
            py.plot(xgappos[:,nloc],noff+gmod,':',color=c)
            py.plot(xgappos[:,nloc],noff+lmod,'-',color=c)
            py.plot(xgappos[:,nloc],noff+lmod_conv,'--',color=c)
            label = "fit pars: {0:5.2f} {1:6.1f} {2:6.4f}".format(poptl2[0],poptl2[1],poptl2[2])
            print('test: ',label)
            py.text(1720.0,noff+1,label,size=10)
            py.xlim(xmin=1700.0,xmax=2000.0)


            #py.axvline(gap5577loc[nloc],color='b',linestyle='--')
            for i in range(ngk):
                py.axvline(tgx[i],color='g',linestyle=':')

            
                
            
            py.figure(8)
            py.errorbar(slbin[0:nbin],slmean[0:nbin],slrms[0:nbin],fmt='o',label=str(loc),color=c)
            py.xlim(xmin=0.0,xmax=2100.0)
            

            noff = noff+doff
        
        # now that the plots have been done, add the 5577 component to the main smooth SL:
        #fgapspl[:,nloc] = fgapspl[:,nloc] + lmod
        fgapspl[:,nloc] = fgapspl[:,nloc]

        nloc = nloc+1

    # get the median gamma value:
    med_gamma = np.nanmedian(lor_gamma)
    print('median gamma factor:',med_gamma)

    # convolve the image with the median lorentzian gamma.

    # as a function of row number find the peak flux in the convolved scattered light model.
    print(np.shape(im_conv))
    peak_row = np.max(im_conv[:,1750:1900],axis=1)
    rowax = np.linspace(1.0,float(ys),ys)
    print('test 1',np.shape(peak_row))
    print(np.shape(rowax))
    
    # derive the peak flux from the convolved model in the gaps


    # scale the peak flux vs row to the convolbved data by fitting a smooth function
    # a quadratic should be okay.
    pratio = lor_peak /peakgap
    ppos = np.zeros(nloc)
    for i in range(nloc):
        ppos[i] = ygappos[int(lor_pos[i]),i]
        print(ppos[i])

    peakratiofit = np.polyfit(ppos, pratio, 3)
    ppoly = np.poly1d(peakratiofit)
    print('polyfit:',peakratiofit)
    # also fit a function to the peak location:
    peaklocfit = np.polyfit(ppos, lor_pos, 3)
    lpoly = np.poly1d(peaklocfit)

    # now generate a model for the 
    sl_mod_line_only = np.zeros((ys,xs))
    sl_mod_extra = np.zeros((ys,xs))
    print(ys,xs)
    xaxx = np.linspace(0.0,float(xs-1),xs)
    peaks = peak_row*ppoly(rowax)
    cents = lpoly(rowax)
    # max range for simple gradient:
    # this gradient seems like a reasonable level to get the blue arm data right:
    dr = 0.15
    for i in range(ys):
        lpar = np.array([peaks[i],cents[i],28.5])
#        print(i,lpar)
        sl_mod_line_only[i,:] = lorentzian(xaxx,*lpar)
        # extra constant gradient up the CCD:
        sl_mod_extra[i,:] = (xaxx-xaxx)-(dr*2)*float(i)/float(ys-1)+dr
    
    # plot all the points that are used with their locations:
    if (doplot):
        py.figure(2)
        py.subplot(1,3,3)
        py.plot(gapallx[0:ngapa],gapally[0:ngapa],',')

        # plot the lortentzian fits:
        py.figure(15)
        nlocax = np.linspace(0.0,float(nloc-1),nloc)
        py.subplot(2,3,1)
        py.plot(nlocax,lor_peak,label='lorentzian')
        py.plot(nlocax,g_peak,label='gaussian')
        py.legend(prop={'size':8})
        py.plot(nlocax,peakgap,'o')
        py.title('peak')
        py.subplot(2,3,2)
        py.plot(nlocax,lor_pos,label='lorentzian')
        py.plot(nlocax,g_pos,label='gaussian')
        py.legend(prop={'size':8})
        py.title('position')
        py.subplot(2,3,3)
        py.plot(nlocax,lor_gamma,label='gamma')
        py.plot(nlocax,g_sigma,label='sigma')
        py.legend(prop={'size':8})
        py.axhline(med_gamma,color='r',linestyle=':')
        py.title('gamma')
        py.subplot(2,3,4)
        py.plot(rowax,peak_row)
        py.title('peak')
        py.subplot(2,3,5)
        py.plot(ppos,pratio)
        py.plot(rowax,ppoly(rowax))
        py.title('peak/peakgap')
        py.subplot(2,3,6)
        py.plot(ppos,lor_peak)
        py.plot(rowax,peak_row*ppoly(rowax))
        py.title('peak')
    

    # estimate the median ratio of all the pixels in gaps vs the simple convolved model:
    medscale = np.nanmedian(gapallflux[0:ngapa]/gapallflux_conv[0:ngapa])
    print('median scale:',medscale)

    hdu_out = pf.PrimaryHDU(im_conv*medscale)
    slfile=frame+'slfn_conv.fits'
    hdu_out.writeto(slfile,overwrite=True)
    #
    # fit an appropriate functional form to the points derived from the smoothed
    # spline function.
    #
    # first plot the points.  Currently just an example:
    xaxis = np.linspace(0.0,float(ys-1),4098)
    for j in range(xs):
        col=j
        print('fitting col: ',j,col%20)
        xslice = ygappos[col,:]
        yslice = fgapspl[col,:]
        if (doplot & (col%20 == 0)):
#        if (doplot & (col > 1800)):
            py.figure(10)
            py.clf()
            c1 = int(col-nsum/2)
            c2 = int(col+nsum/2)
            fp = np.nanmean(im[0:ys,c1:c2],axis=1)
            py.plot(xaxis,fp)
#            py.plot(xaxis,im_conv[0:ys,col]*medscale)
            py.plot(xaxis,im_conv[0:ys,col])
            py.plot(xslice,yslice,'o',label='gap spline values')

        global sl_sigma1
        global sl_sigma2
        global erf_start1
        global erf_start2
        # first set, matches 2018 data but, might be being fooled by extra light in first bundle.
        #sl_sigma1 = 100.0
        #sl_sigma2 = 400.0
        #erf_start1 = xslice[0]+50.0
        #erf_start2 = xslice[-1]-300.0
        #
        sl_sigma1 = 200.0
        sl_sigma2 = 200.0
        erf_start1 = xslice[0]+300.0
        erf_start2 = xslice[-1]-300.0
        porder = 2
        #standard:
        #p0 = [1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
        p0 = [1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
        popt,pcov = optimize.curve_fit(sl_poly_erf,xslice,yslice,p0=p0)

        # try alternative of fitting at bspline:
        nslk = 6
        step = ys/nslk
        #tsl = np.linspace(0.0+step,float(ys)-step,num=nslk)
        #tsl = np.array([485.0,1120.0,1730.0,2340.0,2970.0,3580.0])
        # good for red arm:
        #tsl = (xslice[5:16]+xslice[4:15])/2.0
        # knot if 3 extra gaps at end (4 in total):
        # tsl = np.array([(xslice[4]+xslice[5])/2,(xslice[6]+xslice[7])/2,(xslice[8]+xslice[9])/2,(xslice[10]+xslice[11])/2,(xslice[12]+xslice[13])/2,(xslice[14]+xslice[15])/2])
        # knots with only one end gap:
        #tsl = np.array([(xslice[1]+xslice[2])/2,(xslice[3]+xslice[4])/2,(xslice[5]+xslice[6])/2,(xslice[7]+xslice[8])/2,(xslice[9]+xslice[10])/2,(xslice[11]+xslice[12])/2])
        # knots with only one end gap and less mid knots for low S/N blue:
        tsl = np.array([(xslice[1]+xslice[2])/2,(xslice[4]+xslice[5])/2,(xslice[8]+xslice[9])/2,(xslice[11]+xslice[12])/2])
        #tsl = np.array([(xslice[4]+xslice[5])/2.0,xslice[6],xslice[8],xslice[10],xslice[12],(xslice[14]+xslice[15])/2.0])
        print('SL knots: ',tsl)
        sl_spline = interp.splrep(xslice,yslice,task=-1,t=tsl,k=2)
        fit_sl_bspl= interp.splev(xaxis,sl_spline)
        sl_image_bs[:,j]=fit_sl_bspl
 
        
        mslice=np.zeros(ys)
        for i in range(ys):
            mslice[i] = sl_poly_erf(xaxis[i],*popt)
            sl_image_fn[i,j]=mslice[i]
        
        if (doplot & (col%20 == 0)):
#        if (doplot & (col > 1800)):
            py.axvline(erf_start1,color='r')
            py.axvline(erf_start2,color='r')
        
            mslice2=np.zeros(ys)
            sl_sigma = 1.0
            erf_start1 = xaxis[0]
            erf_start2 = xaxis[-1]
            for i in range(ys):
                mslice2[i] = sl_poly_erf(xaxis[i],*popt)

            py.plot(xaxis,mslice)
            py.plot(xaxis,mslice2)
            py.plot(xaxis,fit_sl_bspl)
            py.ylim(ymin=-2.0,ymax=10.0)

            py.draw()
            yn = input('continue?')
            
    imdiff = im-sl_image_fn


    
    sl_final = sl_image_bs + sl_mod_line_only + sl_mod_extra
    
    # default polynomial fit:
    #hdu_out = pf.PrimaryHDU(sl_image_fn)
    #slfile=frame+'slfn.fits'
    #hdu_out.writeto(slfile,overwrite=True)

    # bspline fit:
    hdu_out = pf.PrimaryHDU(sl_final)
    slfile=frame+'slfn.fits'
    hdu_out.writeto(slfile,overwrite=True)
    
    hdu_out = pf.PrimaryHDU(sl_mod_line_only)
    slfile=frame+'slfn_line.fits'
    hdu_out.writeto(slfile,overwrite=True)

    hdu_out = pf.PrimaryHDU(imdiff)
    slfile=frame+'slfn_imdiff.fits'
    hdu_out.writeto(slfile,overwrite=True)

    exit()
    
    ###################################################################################
    # generate a 2D bspline:
    #
    # first need to define the knot locations in the x (wavelength) direction.  This will depend
    # on the type of frame.  For object frames, the order should be relatively low, as the S/N
    # in the gaps is lower.  For dome flats or twilights, the knots can be closer as the S/N
    # is better:
    
    if (obstype == 'OBJECT'):
        nxknot = 12
        tx_tmp = np.linspace(0.0,float(xs),nxknot)

        # if a blue object frame, add more knots around 5577.  At the moment, the location
        # if these in pixels is just approx (based on me looking at some frames), but we can
        # do this more precisely if we use the full wavelength solution:
        if (spectid == 'BL'):
            tx = np.sort(np.append(tx_tmp,[1840.0,1850.0,1860.0,1870.0,1880.0,1890.0]))
        else:
            tx = tx_tmp

    if (obstype == 'ARC'):
        nxknot = 12
        tx_tmp = np.linspace(0.0,float(xs),nxknot)
        
    if ((obstype == 'SKY') or (obstype == 'FLAT')): 
        nxknot = 48
        tx = np.linspace(0.0,float(xs),nxknot)
        
    # work out the spacing of knots in the y direction. This does not depend so much on the
    # data as it is driven by the number of gaps.  however, we might want this to be lower for
    # object frames that have very low levels of scattered light, as the noise can still drive
    # some features
    #
    # Set the knots at the approx location of the gaps:
    nyknot = 14
    ty = np.zeros(nyknot)
    for i in range(nloc):
        ty[i] = np.nanmean(ygappos[:,i])

    # add in extra knots at the end:
    ty = np.sort(np.append(ty,[0.0,float(ys)])) 

    # for blue object frames, reduce the number of knots as the scattered light is low S/N:
    # this actually works for most red object frames, but there may be some where the
    # structure from the vacuum gauge problem is too structured.
    #if (obstype == 'OBJECT'):
    #    if ((spectid == 'BL') or (spectid == 'RD')):
    #        nyknot = 6
    #        ty = np.zeros(nyknot)
    #        for i in range(nyknot):
    #            ty[i] = (np.nanmean(ygappos[:,i*2+1])+np.nanmean(ygappos[:,i*2+2]))/2.0

    #        # add in extra knots at the end:
    #        ty = np.sort(np.append(ty,[0.0,float(ys),np.nanmean(ygappos[:,0]),np.nanmean(ygappos[:,-1])]))
            
    # alternate smaller number of knots:
    #ty = [0.0,float(ys)/3.0,float(ys)*2.0/3.0,float(ys)]
            
    if (verbose):
        print('x knots:',tx)
        print('y knots:',ty)
        
    # define weights, all set to one for now.  At very low count levels we don't want to use variance weighting
    # as this can bias the fits low just a little (~0.5 counts).
    wt=np.ones(ngapa)

    # look at the distribution of values and clip out any extreme outliers before we start (removes cosmics etc):
    (mean1,rms1,nc1) = clipped_mean_rms(gapallflux[0:ngapa],5.0,verbose=verbose)

    if (doplot):
        py.figure(4)
        py.subplot(2,1,1)
        rmin = np.percentile(gapallflux[0:ngapa],0.1)
        rmax = np.percentile(gapallflux[0:ngapa],99.9)
        py.hist(gapallflux[0:ngapa],bins=100,range=(rmin,rmax),histtype='step',color='k')
        py.title('distribution of background values')
        py.xlabel('flux')
        py.ylabel('Number')

    # set the weights of any point outside of 10 sigma of mean background to be small.
    # this is conservative, to clip out obvious bad pixels before we start (e.g.
    # bright cosmic rays):
    for i in range(ngapa):
        if (abs(gapallflux[i]-mean1) > rms1*10.0):
            wt[i] = 1.0e-40

    # generate a first 2D spline:
    sl_spline = interp.bisplrep(gapallx[0:ngapa],gapally[0:ngapa],gapallflux[0:ngapa],w=wt,task=-1,tx=tx,ty=ty,kx=kx,ky=ky)

    # calculate the RMS deviation of the data from the spline model:
    for i in range(ngapa):
        sl_diff[i] = gapallflux[i] - interp.bisplev(gapallx[i],gapally[i],sl_spline)

    (mean,rms,nc) = clipped_mean_rms(sl_diff[0:ngapa],5.0,verbose=verbose)
     
    # plot histogram:
    if (doplot):
        py.subplot(2,1,2)
        py.hist(sl_diff[0:ngapa],bins=100,range=(-20.0,20.0),histtype='step',color='k')
        py.title('distribution of background residual values')
        py.xlabel('flux')
        py.ylabel('Number')

    if (verbose):
        print('mean,rms,nc of residuals from scattered light fit:',mean,rms,nc)

    # set the weights of any point outside of 5 sigma to zero:
    for i in range(ngapa):
        if (abs(sl_diff[i]) > rms*5.0):
            wt[i] = 1.0e-40

    # recalculate the spline with adjusted weights for clipped points:
    sl_spline = interp.bisplrep(gapallx[0:ngapa],gapally[0:ngapa],gapallflux[0:ngapa],w=wt,task=-1,tx=tx,ty=ty,kx=kx,ky=ky)

    # do a second test, fitting a 1d spline to all the data at once.  This is equivalent to
    # a constant scattered light model (for a given wavelength):
    # sort the input data first:
    fxx = gapallx[0:ngapa]
    fyy = gapallflux[0:ngapa]
    idx = fxx.argsort()

    print(fxx[idx])
    print(fyy[idx])
    if (doplot):
        py.figure(7)
        py.plot(fxx,fyy,',')
    #
    # average in bins before the 1d fit:
    nbin = int((xs+int(maxshift))/binsize)
    for ib in range(nbin):
        ibin1 = ib * binsize
        ibin2 = (ib+1) * binsize
        nb = 0
        for ig in range(ngapa):
            if ((gapallx[ig] > ibin1) & (gapallx[ig] < ibin2) & (wt[ig] == 1.0)):
                binflux[nb] = gapallflux[ig]
                nb =nb + 1  

        # get the mean in the bin (and sigma):
        slmean[ib],slrms[ib],nclip = clipped_mean_rms(binflux[0:nb],5.0)
        # std error on the mean:
        slrms[ib] = slrms[ib]/np.sqrt(nclip)
        slbin[ib] = (ibin1+ibin2)/2.0
        print(ib,ibin1,ibin2,slmean[ib],slrms[ib],nclip,slbin[ib])

    if (doplot):
        py.figure(7)
        py.errorbar(slbin[0:nbin],slmean[0:nbin],slrms[0:nbin],fmt='o')
        py.xlim(xmin=0.0,xmax=float(xs+int(maxshift)))

    # now fit a 1D spline:
    #sl_spline_1d = interp.splrep(slbin[0:nbin],slmean[0:nbin],w=slrms[0:nbin],k=3,t=tx)

    sl_poly_1d = polyfitr(slbin[0:nbin],slmean[0:nbin],8,5.0)
    sl_pol = np.poly1d(sl_poly_1d)
    
    #txx = np.linspace(0.0,float(xs+int(maxshift)),10)
    #print txx
    #sl_spline_1d = interp.splrep(slbin[0:nbin],slmean[0:nbin],t=txx)

    xax = np.arange(xs+int(maxshift))
    ysplmodel = sl_pol(xax)

    if (doplot):
        py.plot(xax,ysplmodel,'-')

    # generate a scattered light image from the 1D fit:
    sl_model_shift_1d = np.tile(ysplmodel,[ys,1])
    print('shape',np.shape(sl_model_shift_1d))
        
    # generate an image that is an expanded version allowing for shifts:
    sl_model_shift_im = np.zeros((ys,xs+int(maxshift)))
    yy = np.arange(ys)+0.5
    xxsh = np.arange(xs+int(maxshift))+0.5

    sl_model_shift_im = np.transpose(interp.bisplev(xxsh,yy,sl_spline))

    if (doplot):
        py.figure(5)
        py.subplot(1,4,1)
        py.imshow(sl_model_shift_im,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        py.colorbar()

    # now transform back the model (take the shift out):
    for i in range(ys):
        ishift = int(round(dslice[i]))
        sl_image[i,:] = sl_model_shift_im[i,0+ishift:xs+ishift]
        sl_image_1d[i,:] = sl_model_shift_1d[i,0+ishift:xs+ishift]

    if (doplot):
        py.subplot(1,4,2)
        py.imshow(sl_image,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        py.colorbar()
        py.subplot(1,4,3)
        py.imshow(sl_model_shift_1d,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        py.colorbar()
        py.subplot(1,4,4)
        py.imshow(sl_image_1d,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        py.colorbar()
      
    # write the final SL image to a file:
    #hdu_out = pf.PrimaryHDU(sl_image)
    hdu_out = pf.PrimaryHDU(sl_image_1d)
    slfile=frame+'sl.fits'
    hdu_out.writeto(slfile,overwrite=True)

    # Plot the model back onto the traces along the gaps to see if it
    # is a good match
    if (doplot):
        cols2=iter(py.cm.rainbow(np.linspace(0,1,14)))
        noff=0
        xaxis = np.arange(2048)
        for i in range(nloc):
            for j in range(xs):
                xx = xaxis[j]
                yy = ygappos[j,i]
                gap_model_flux_spl[j,i] = interp.bisplev(xx,yy,sl_spline)
    
            c=next(cols2)
            py.figure(3)
            py.plot(xaxis,gap_model_flux_spl[:,i]+noff,'--',label=str(loc),color=c)
            noff=noff+doff

    #Loop through columns and plot the scattered light model to see
    # if it is a good match:

        py.figure(6)
        yp = np.arange(ys)
        mp = np.zeros(ys)
        xp1 = np.ones(ys)
        for ix in range(0,xs,100):
            fp = np.nanmean(im[0:ys,ix:ix+nsum],axis=1)
            mp = np.nanmean(sl_image[0:ys,ix:ix+nsum],axis=1)
            mp1d = np.nanmean(sl_image_1d[0:ys,ix:ix+nsum],axis=1)
            xp = xp1 * float(ix)

            py.plot(yp,fp)
            py.plot(yp,mp,label='sl_image')
            py.plot(yp,mp1d,label='sl_image_1d')
            py.title('Col '+str(ix))
            py.xlabel('pixel')
            py.ylabel('Counts')
            py.xlim(xmin=0.0,xmax=float(ys))
            py.ylim(ymin=0.0,ymax=np.nanpercentile(fp,60.0))
            py.legend(prop={'size':8})
            py.draw()
            tnum = input('continue? (y/n):')
        

            py.clf()
        
        
   
################################################################################

def fit_sl(frame,tlmfile,col,plotres=False,verbose=True,xplotmin=0,xplotmax=4096,dopdf=False,nsum=1,nstep=1,doall=True,sp=3):
    """Function to test scattered light fitting to gaps in SAMI data.  Based on plot_optex.
    frame name without extension, e.g. 06mar20055 the tlm file to use and
    a column number to plot.
    Usage:
    sami_2dfdr_reduction_tests.fit_sl('06mar20066','06mar20066tlm.fits',1500,doall=False)
    """

    #spline order:
    #sp = 1 # this is not smooth
    #sp = 3 # cubic spline - sets up upphysical oscillations unless knots are carefully selected.
    
    
    
    imfile = frame+'im.fits'
    mimfile = frame+'_outdir/'+frame+'mim.fits'
    mslfile = frame+'_outdir/'+frame+'msl.fits'
    #    mslfile = 'background.fits'
    resfile = frame+'_outdir/'+frame+'res.fits'
    #    tlmfile = frame+'tlm.fits'
    #only get TLM file if a flat, otherwise need to use something else
    #tlmfile = frame+'tlm.fits'
    # hardcode for now
    #tlmfile = '06mar20066tlm.fits'

    #imfile = frame+'im.fits'
    #mimfile = frame+'mim.fits'
    #mslfile = frame+'msl.fits'
    #    mslfile = 'background.fits'
    #resfile = frame+'res.fits'
    #tlmfile = frame+'tlm.fits'

    
    hdulist_im = pf.open(imfile)
    hdulist_mim = pf.open(mimfile)
    hdulist_msl = pf.open(mslfile)
    hdulist_res = pf.open(resfile)
    hdulist_tlm = pf.open(tlmfile)

    # get image:
    im = hdulist_im[0].data
    mim = hdulist_mim[0].data
    msl = hdulist_msl[0].data
    res = hdulist_res[0].data
    tlm = hdulist_tlm[0].data
    prf = hdulist_tlm['SIGMAPRF'].data
    
    # get variance:
    variance = hdulist_im['VARIANCE'].data

    # Note y is first axis in python array, x is second
    (ys,xs) = im.shape

    # define an output image for the SL model:
    sl_image = np.zeros((ys,xs))
    # define to be NaN unless we fill it with something:
    sl_image[:,:] = np.NAN 

    # define an image to store the pixels we use for fitting:
    sl_pixels = np.zeros((ys,xs))

    if (verbose):
        print(frame,', image size:',xs,' x ',ys)

    x=np.arange(ys)+1

    if (doall):
        ncol = xs
        print('doing SL fit in ',ncol,' cols')
    else:
        ncol = 1
        print('doing SL fit in a single column')

    for ic in range(0,ncol,nstep):
        print('fitting SL in col ',ic)
         
        if (ncol > 1):
            colnum = ic
        else:
            colnum = col
            
        # get data for particular col:
        if (nsum == 1): 
            flux = im[:,colnum]
            model_flux = mim[:,colnum] 
            model_sl = msl[:,colnum]
            var = variance[:,colnum]
            sig = np.sqrt(variance)
            sigma = sig[:,colnum]
            tlmcol = tlm[:,colnum]
            prfcol = prf[:,colnum]

        # or sum over columns to increase S/N: 
        else:
            n1 = max(0,colnum-nsum/2)
            n2 = min(xs,colnum+nsum/2)
            flux = np.nanmean(im[:,n1:n2],axis=1)
            model_flux = np.nanmean(mim[:,n1:n2],axis=1) 
            model_sl = np.nanmean(msl[:,n1:n2],axis=1)
            # variance of summed flux is just sum of variances,
            # V_mean = V_sum/n_sum^2
            # below we calc the mean of the variance (we don't sum)
            # so we need to divide by another factor of nsum to
            # get the right value:
            var = np.nanmean(variance[:,n1:n2],axis=1)/nsum
            sigma = np.sqrt(var)
            tlmcol = np.nanmean(tlm[:,n1:n2],axis=1)
            prfcol = np.nanmean(prf[:,n1:n2],axis=1)


        
        nfib = np.size(tlmcol)
        if (verbose):
            print('number of tramlines in TLM:',nfib)


        # Now identify pixels that are sufficiently far from a tramline and create a new
        # array with just those pixels:
        #
        # number of sigma from the TLM to select pixels:
        min_sep_sl = 4.0
        npix = np.size(x)
        xsl = np.zeros(npix)
        ysl = np.zeros(npix)
        ssl = np.zeros(npix)
        ntlm1 = np.ones(npix,dtype=np.int16) * -1
        ntlm2 = np.ones(npix,dtype=np.int16) * -1
        nsl = 0

        # for each pixel, find the closest two TLMs:
        # first identify all the pixels that are less than the TLM position:
        i=0
        for j in range(nfib):
            while (x[i] < tlmcol[j]):
                ntlm2[i] = j
                i = i +1
            
        # now go through in the opposite direction:
        i=npix-1
        for j in range(nfib-1,-1,-1):
            while (x[i] > tlmcol[j]):
                ntlm1[i] = j
                i = i - 1

        for i in range(npix):
            # identify the nearest TLM above and below the pixel:
            ltlm = tlmcol[ntlm1[i]]
            utlm = tlmcol[ntlm2[i]]
            ltlmsig =  prfcol[ntlm1[i]]
            utlmsig =  prfcol[ntlm2[i]]
            if ((abs(ltlm - x[i]) > (min_sep_sl*ltlmsig)) & (abs(utlm - x[i]) > (min_sep_sl*ltlmsig))):

                xsl[nsl] = x[i]
                ysl[nsl] = flux[i]
                ssl[nsl] = sigma[i]
                nsl = nsl+1
                sl_pixels[i,ic] = flux[i]
                if (ncol == 1):
                    print(i,x[i])


        # now that we have points that are well away from the fibre profiles, we
        # can attempt to fit them.  The question is, what do we fit?  Probably the
        # best approach will be a bspline.  In this case we are best to define the knots
        #carefully.  These should be at the start/end and in the fibre gaps:
        nk = 0
        xk=np.zeros(npix)
        # list of TLm positions that are followed by a gap:
        #gaploc=[126,252,378,504,630,756]
        #gaploc=[63,189,315,441,567,693]
        #gaploc=[63,126,189,252,315,378,441,504,567,630,693,756]
        # try not using the first and last gap as a knot:
        gaploc=[126,189,252,315,378,441,504,567,630,693]
        # try not using 2nd last and 2nd first:
        #gaploc=[63,189,252,315,378,441,504,567,630,756]
        #knots every 10 pixels at the start:
        # find the last pixel of the edge:
        epix1 = 0
        for i in range(npix):
            if (x[i] < (tlmcol[0] - min_sep_sl*prfcol[0])):
                epix1 = i

        print('epix1=',epix1)
        
        for i in range(5,npix,4):
            if (x[i] < (tlmcol[0] - min_sep_sl*prfcol[0])+3):
                xk[nk] = x[i]
                if (ncol == 1):
                    print('knot:',nk,xk[nk])
                nk = nk +1
        # add knots at the gaps:
        for loc in gaploc:
            # place a knot at the point in the model of gaps:
            xk[nk] = (tlmcol[loc-1]+tlmcol[loc])/2.0
            # place a knot in the middle of the fibre block as an alternative:
            #xk[nk] = (tlmcol[loc-1]+tlmcol[loc-63])/2.0
            if (ncol == 1):
                print('knot:',nk,xk[nk],'  (interior)')
            nk = nk +1
        #knots every 10 pixels at the end:
        for i in range(0,npix,4):
            if (x[i] > (tlmcol[-1] + min_sep_sl*prfcol[-1])-3):
                xk[nk] = x[i]
                if (ncol == 1):
                    print('knot:',nk,xk[nk])
                nk = nk +1
        
        # do the bspline fitting.
        splmodel = interp.splrep(xsl[0:nsl],ysl[0:nsl],w=1/ssl[0:nsl],k=sp,task=-1,t=xk[0:nk])

        ysplmodel = interp.splev(x,splmodel)

        # put model into image:
        sl_image[:,ic] = ysplmodel 
            
        # if we want a pdf make it:
        if (dopdf):
            pdf = PdfPages('plot_optex.pdf')

        
        # do plot of data and 2dfdr fit:
        if (ncol == 1):
            py.figure(1)
            py.plot(x,flux, 'b-',label=frame+' Data')
            py.xlabel('pixel')
            py.ylabel('counts')
            #py.errorbar(x, flux, sigma)
            py.plot(x,model_flux+model_sl, 'r-',label=frame+' model profiles')
            py.plot(x,model_sl, 'g-',label=frame+' model scattered light')
            py.xlim(xmin = xplotmin, xmax = xplotmax)

            # plot points that are good for scattered light:
            py.plot(xsl[0:nsl],ysl[0:nsl], 'o',color='green',label=frame+' scattered light points')
    
            # plot spline scattered light model
            py.plot(x,ysplmodel,'-',color='magenta',label=frame+' spline SL model')

    #print idx

    if (doall):
        nid = 0
        mfilt = 5
        print('Replacing NaN values with local median...')
        for idx,val in np.ndenumerate(sl_image):
            if np.isnan(val):
                x1 = max(0,idx[1]-mfilt)
                x2 = min(ncol-1,idx[1]+mfilt)
                y1 = max(0,idx[0]-mfilt)
                y2 = min(npix-1,idx[0]+mfilt)
                #print x1,x2,y1,y2
                #print 'subimage:',sl_image[y1:y2,x1:x2]
                med = np.nanmedian(sl_image[y1:y2,x1:x2])
                sl_image[idx] = np.nanmedian(sl_image[y1:y2,x1:x2])
                #print 'test',idx,nid,sl_image[idx],x1,x2,y1,y2
                nid = nid+1
        
        # Now smooth the output:    
        sl_image_sm = signal.medfilt(sl_image,kernel_size=(1,21))

        # write the final SL image to a FITS file
        pf.writeto('sl_model.fits',sl_image_sm,clobber=True)

    # plot the output image:
    if (ncol > 1):
        py.figure(1)
        py.subplot(1,3,1)
        py.imshow(sl_image_sm,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        py.colorbar()

        py.subplot(1,3,2)
        py.imshow(sl_image,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        py.colorbar()

        py.subplot(1,3,3)
        py.imshow(sl_pixels,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        py.colorbar()


        
        if (dopdf):        
            py.savefig(pdf, format='pdf')        
            pdf.close()
            
