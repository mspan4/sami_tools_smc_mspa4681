
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
import matplotlib.ticker as ticker
from matplotlib import rc
#from matplotlib._png import read_png

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, AnchoredOffsetbox

import astropy.convolution as convolution
from astropy.convolution.kernels import CustomKernel

# local things from this package:
from .sami_2dfdr_read import find_fibre_table
from .sami_stats_utils import median_smooth, gaussian_filter_nan, clipped_mean_rms, gaussian, median_filter_nan, clipped_rms
from .sami_2dfdr_general import plot_bundlelims


################################################################################
# compare many throughput measurements from different files.
#
# usage:
# sami_2dfdr_reduction_tests.check_thput_many('06mar100??red.fits')
#
def check_thput_many(inlist,obstype='OBJECT',min_exp=900,ha_min=-90.0,ha_max=90.0,dopdf=True):
    """Function to compared the throughput measurements from many different frames.
    Usage:
    sami_2dfdr_reduction_tests.check_thput_many('06mar100??red.fits')
    """
    # plot?
    doplot=True
        
    # field radius in microns:
    field_radius = (1800/15.2) * 1000

    
    # "glob" the filelist to expand wildcards:
    files = glob.glob(inlist)

    # max number of bundles + sky fibs:
    mbun=100
    
    # initialize arrays

    # loop over files
    # get mean and rms per fibre
    # plot


    # initialize:
    #cont=np.zeros(0)
    #line=np.zeros(0)

    # arrays to store fibre specific values:
    #allcont=np.zeros((26,1000))
    #allline=np.zeros((26,1000))
     
    # loop over each 
    nfiles = 0
    for filename in files:

        # open fits file and check its an object with min exposure time:
        hdulist = pf.open(filename)
        primary_header=hdulist['PRIMARY'].header
        exposure = primary_header['EXPOSED']
        otype = primary_header['OBSTYPE']
        hastart = primary_header['HASTART']
        haend = primary_header['HAEND']
        ha = (hastart+haend)/2.0
        print(otype)
            
#        print filename,otype,exposure

        # only select objects frames with at least min-exp exposure time:
        if ((otype == obstype) and (exposure > min_exp) and (ha > ha_min) and (ha < ha_max)):

            # get througput extension:
            thput = hdulist['THPUT'].data

            # get binary table info:
            fib_tab_hdu=find_fibre_table(hdulist)
            table_data = hdulist[fib_tab_hdu].data
            types=table_data.field('TYPE')
            fid_mra = table_data.field('FIB_MRA')
            fid_mdec = table_data.field('FIB_MDEC')
            fibpos_x = table_data.field('FIBPOS_X')
            fibpos_y = table_data.field('FIBPOS_Y')
            
            # get size:
            xsize = thput.size

            # initialize arrays on first pass:
            if (nfiles == 0):
                thput_all=np.zeros((xsize,1000))
                thput_all_corrected=np.zeros((xsize,1000))
                thput_mean=np.zeros(xsize)
                thput_rms=np.zeros(xsize)
                thput_med=np.zeros(xsize)

                ra_all=np.zeros((xsize,1000))
                dec_all=np.zeros((xsize,1000))
                x_all=np.zeros((xsize,1000))
                y_all=np.zeros((xsize,1000))
                types_all=np.empty((xsize,1000), dtype='U1')

                thput_bundle=np.zeros((mbun,1000))
                thputrms_bundle=np.zeros((mbun,1000))
                thputn_bundle=np.zeros((mbun,1000))
                ra_bundle=np.zeros((mbun,1000))
                ra_bundle=np.zeros((mbun,1000))
                dec_bundle=np.zeros((mbun,1000))
                x_bundle=np.zeros((mbun,1000))
                y_bundle=np.zeros((mbun,1000))
                r_bundle=np.zeros((mbun,1000))
                types_bundle=np.empty((mbun,1000), dtype='U1')
                
            thput_all[:,nfiles] = thput
            ra_all[:,nfiles] = fid_mra
            dec_all[:,nfiles] = fid_mdec
            x_all[:,nfiles] = fibpos_x
            y_all[:,nfiles] = fibpos_y
            types_all[:,nfiles] = types

            # get average values for each bundle or sky fibre.  First work out which
            # bundle we are looking at:
            for ifib in range(xsize):
                ibundle = int(ifib/63)
                isky = int(ifib/63) +1 + int((ifib+1)/63) - 1

                if (types[ifib] == 'P'):
                    thput_bundle[ibundle,nfiles] = thput_bundle[ibundle,nfiles] + thput[ifib]
                    x_bundle[ibundle,nfiles] = x_bundle[ibundle,nfiles] + fibpos_x[ifib]
                    y_bundle[ibundle,nfiles] = y_bundle[ibundle,nfiles] + fibpos_y[ifib]
                    thputn_bundle[ibundle,nfiles] = thputn_bundle[ibundle,nfiles] + 1

                if (types[ifib] == 'S'):
                    thput_bundle[isky+13,nfiles] = thput_bundle[isky+13,nfiles] + thput[ifib]
                    x_bundle[isky+13,nfiles] = x_bundle[isky+13,nfiles] + fibpos_x[ifib]
                    y_bundle[isky+13,nfiles] = y_bundle[isky+13,nfiles] + fibpos_y[ifib]
                    r_bundle[isky+13,nfiles] = np.sqrt(x_bundle[isky+13,nfiles]**2 + y_bundle[isky+13,nfiles]**2)
                    

            for ibun in range(13):
                thput_bundle[ibun,nfiles] = thput_bundle[ibun,nfiles]/thputn_bundle[ibun,nfiles]
                x_bundle[ibun,nfiles] = x_bundle[ibun,nfiles]/thputn_bundle[ibun,nfiles]
                y_bundle[ibun,nfiles] = y_bundle[ibun,nfiles]/thputn_bundle[ibun,nfiles]
                r_bundle[ibun,nfiles] = np.sqrt(x_bundle[ibun,nfiles]**2 + y_bundle[ibun,nfiles]**2)

#            for ibun in xrange(39):
#                print ibun,thput_bundle[ibun,nfiles],thputn_bundle[ibun,nfiles],x_bundle[ibun,nfiles],y_bundle[ibun,nfiles]
            nfiles = nfiles + 1
            print(nfiles,filename,obstype,exposure,ha)

        hdulist.close()

        
    # get mean throughput
    thput_mean = np.average(thput_all[:,0:nfiles],1)
    thput_med = np.median(thput_all[:,0:nfiles],1)
    for i in range(xsize):
        thput_rms[i] = np.sqrt(np.sum((thput_all[i,0:nfiles]-thput_mean[i])**2)/nfiles)
        #        thput_rms[i] = np.sqrt(((thput_all[i,0:nfiles]-thput_mean[i])**2)/nfiles)
        #print i,thput_all[i,0:nfiles],thput_mean[i],thput_rms[i]


    # get the median RMS throughput:
    med_rms = np.median(thput_rms)
    print("median RMS of fibre throughputs",med_rms)
    
    # start with plotting the throughputs
    py.figure(1)
    py.subplot(2,1,1)
    py.axhline(1.0,color='k',linestyle='--')

    # define fibre number axis:
    xfib = range(xsize)
    py.plot(xfib,thput_mean,'.',label='mean througputs')
    py.plot(xfib,thput_med,'.',label='median througputs')
    py.xlabel('Fibres')
    py.ylabel('Relative throughput')
    py.legend(prop={'size':10})

    # plot rms
    py.subplot(2,1,2)
    py.axhline(1.0,color='k',linestyle='--')

    # define fibre number axis:
    xfib = range(xsize)
    py.plot(xfib,thput_rms,'.',label='rms througputs')
    py.xlabel('Fibres')
    py.ylabel('RMS of relative throughput')
    py.legend(prop={'size':10})

    # subtract median and look for systematic changes:
    for i in range(nfiles):
        thput_all_corrected[:,i] = thput_all[:,i] - thput_med
    

    xfile = range(nfiles)
    py.figure(2)
    py.axhline(0.0,color='k',linestyle='--')
    for i in range(xsize):
        py.plot(xfile,thput_all_corrected[i,0:nfiles],'-')



        
    # plot the spatial distribution of throughputs on the field:
    py.figure(3)
    for ifiles in range(nfiles):
        py.scatter(x_bundle[0:13,ifiles],y_bundle[0:13,ifiles],c=thput_bundle[0:13,ifiles],edgecolors='none',s=80,marker='o',vmin=0.7,vmax=1.2)
        for i in range(13):
            py.text(x_bundle[i,ifiles],y_bundle[i,ifiles],str(i+1))
#        py.scatter(x_bundle[13:39,ifiles],y_bundle[13:39,ifiles],c=thput_bundle[13:39,ifiles],edgecolors='none',s=40,marker='^',vmin=0.7,vmax=1.2)

    # field radius:
    circle1=py.Circle((0,0),field_radius,color='k',fill=False)
    py.gcf().gca().add_artist(circle1)
    # radius of possible ghost, although this is unlikely to be strong enough to impact throughput:
    circle2=py.Circle((0,0),30000,color='k',fill=False)
    py.gcf().gca().add_artist(circle2)
    py.axes().set_aspect('equal', 'datalim')
    py.colorbar(label='Relative throughput')
    py.xlim(xmin=-125000,xmax=125000)
    py.ylim(ymin=-125000,ymax=125000)
    py.xlabel('X position (microns)')
    py.ylabel('Y position (microns)')
    py.title('Bundle average throughput')

    py.figure(4)
    for ifiles in range(nfiles):
#        py.scatter(x_bundle[0:13,ifiles],y_bundle[0:13,ifiles],c=thput_bundle[0:13,ifiles],edgecolors='none',s=80,marker='o',vmin=0.7,vmax=1.2)
        py.scatter(x_bundle[13:39,ifiles],y_bundle[13:39,ifiles],c=thput_bundle[13:39,ifiles],edgecolors='none',s=40,marker='^',vmin=0.7,vmax=1.2)
        for i in range(13,39):
            py.text(x_bundle[i,ifiles],y_bundle[i,ifiles],str(i-12))

    # field radius:
    circle1=py.Circle((0,0),field_radius,color='k',fill=False)
    py.gcf().gca().add_artist(circle1)
    # radius of possible ghost, although this is unlikely to be strong enough to impact throughput:
    circle2=py.Circle((0,0),30000,color='k',fill=False)
    py.gcf().gca().add_artist(circle2)
    py.axes().set_aspect('equal', 'datalim')
    py.colorbar(label='Relative throughput')
    py.xlim(xmin=-125000,xmax=125000)
    py.ylim(ymin=-125000,ymax=125000)
    py.xlabel('X position (microns)')
    py.ylabel('Y position (microns)')
    py.title('Sky fibre throughput')

    # plot radial throughput variation:
    py.figure(5)
    # set up bins to take median/average:
    nbins = 10
    binsize = 120000/nbins
    thput_binned = np.zeros((nbins,10000))
    thput_binned_n =  np.zeros(nbins,dtype=np.int32)
    thput_binned_s = np.zeros((nbins,10000))
    thput_binned_s_n =  np.zeros(nbins,dtype=np.int32)
    for ifiles in range(nfiles):
        py.scatter(r_bundle[0:13,ifiles],thput_bundle[0:13,ifiles],facecolors='none',edgecolors='r',s=10,marker='o')
        for ibun in range(13):
            ibin = int(r_bundle[ibun,ifiles]/binsize)
            thput_binned[ibin,thput_binned_n[ibin]] = thput_bundle[ibun,ifiles]
            thput_binned_n[ibin] = thput_binned_n[ibin] + 1
        
        py.scatter(r_bundle[13:39,ifiles],thput_bundle[13:39,ifiles],facecolors='none', edgecolors='b',s=10,marker='^')
        for ibun in range(13,39):
            ibin = int(r_bundle[ibun,ifiles]/binsize)
            thput_binned_s[ibin,thput_binned_s_n[ibin]] = thput_bundle[ibun,ifiles]
            thput_binned_s_n[ibin] = thput_binned_s_n[ibin] + 1

    py.xlim(xmin=0.0,xmax=125000)
    py.xlabel('radial position (microns)')
    py.ylabel('Relative throughput')

    # get median/mean/rms throughput:
    mean_thput_binned = np.zeros(nbins)
    median_thput_binned = np.zeros(nbins)
    rms_thput_binned = np.zeros(nbins)
    mean_thput_binned_s = np.zeros(nbins)
    median_thput_binned_s = np.zeros(nbins)
    rms_thput_binned_s = np.zeros(nbins)
    r_binned = np.zeros(nbins)
    f_binned = np.zeros(nbins)
    xfit = np.zeros(nbins)
    yfit = np.zeros(nbins)
    
    print('\n')
    print('            Bundles           |      Sky fibres            | bundle-sky')
    print('nbin ndata mean median    rms | ndata  mean  median    rms |   mean')
    nfit = 0
    for ibin in range(nbins):
        # bundles (note points offset slightly from bin centre for clarity):
        mean_thput_binned[ibin] = np.nanmean(thput_binned[ibin,0:thput_binned_n[ibin]])
        median_thput_binned[ibin] = np.nanmedian(thput_binned[ibin,0:thput_binned_n[ibin]])
        rms_thput_binned[ibin] =  np.nanstd(thput_binned[ibin,0:thput_binned_n[ibin]])/np.sqrt(thput_binned_n[ibin])
        py.errorbar((ibin+0.48)*binsize,mean_thput_binned[ibin],rms_thput_binned[ibin],color='m',marker='o',lw=3,markersize=8)

        # sky:
        mean_thput_binned_s[ibin] = np.nanmean(thput_binned_s[ibin,0:thput_binned_s_n[ibin]])
        median_thput_binned_s[ibin] = np.nanmedian(thput_binned_s[ibin,0:thput_binned_s_n[ibin]])
        rms_thput_binned_s[ibin] =  np.nanstd(thput_binned_s[ibin,0:thput_binned_s_n[ibin]])/np.sqrt(thput_binned_s_n[ibin])
        print('{0:2d} {1:4d} {2:6.4f}  {3:6.4f} {4:6.4f} | {5:4d} {6:6.4f}  {7:6.4f} {8:6.4f} |  {9:6.4f}'.format(ibin,int(thput_binned_n[ibin]),mean_thput_binned[ibin],median_thput_binned[ibin],rms_thput_binned[ibin],int(thput_binned_s_n[ibin]),mean_thput_binned_s[ibin],median_thput_binned_s[ibin],rms_thput_binned_s[ibin], mean_thput_binned[ibin]-mean_thput_binned_s[ibin]))
        py.errorbar((ibin+0.52)*binsize,mean_thput_binned_s[ibin],rms_thput_binned_s[ibin],color='g',marker='^',lw=3,markersize=8)
        r_binned[ibin] = (ibin+0.5)*binsize
        if (thput_binned_n[ibin] > 0):
            xfit[nfit] = r_binned[ibin]
            yfit[nfit] = mean_thput_binned[ibin]
            nfit=nfit+1
            
        
    # fit a curve to the binned values:
    pfit = np.poly1d(np.polyfit(xfit[0:nfit],yfit[0:nfit],2))
    for ibin in range(nbins):
        f_binned[ibin] = pfit(r_binned[ibin])
        print(r_binned[ibin],f_binned[ibin])

    py.plot(r_binned,f_binned,'-',color='m')

    # plot the spatial distribution of throughputs on the field after removing the radial component:
    py.figure(6)
    for ifiles in range(nfiles):
        py.scatter(x_bundle[0:13,ifiles],y_bundle[0:13,ifiles],c=thput_bundle[0:13,ifiles]-pfit(r_bundle[0:13,ifiles]),edgecolors='none',s=80,marker='o',vmin=-0.15,vmax=0.15)
#        py.scatter(x_bundle[13:39,ifiles],y_bundle[13:39,ifiles],c=thput_bundle[13:39,ifiles],edgecolors='none',s=40,marker='^',vmin=0.7,vmax=1.2)

    # field radius:
    circle1=py.Circle((0,0),field_radius,color='k',fill=False)
    py.gcf().gca().add_artist(circle1)
    # radius of possible ghost, although this is unlikely to be strong enough to impact throughput:
    circle2=py.Circle((0,0),30000,color='k',fill=False)
    py.gcf().gca().add_artist(circle2)
    py.axes().set_aspect('equal', 'datalim')
    py.colorbar(label='Relative throughput')
    py.xlim(xmin=-125000,xmax=125000)
    py.ylim(ymin=-125000,ymax=125000)
    py.xlabel('X position (microns)')
    py.ylabel('Y position (microns)')
    py.title('Bundle average (radial trend removed)')

            
        
#    circle1=py.Circle((0,0),field_radius,color='k',fill=False)
#    py.gcf().gca().add_artist(circle1)
    
##############################################################################
# function defining a linear plane:
def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z



################################################################################
# check PCA sky subtraction residuals and how they correlate with spectrum
# flux
#
def check_pcasky_res(infile_pca,infile_nopca,doplot=True):

    # open files:
    hdulist = pf.open(infile_pca)
    im_pca = hdulist[0].data
    var = hdulist['VARIANCE'].data

    (ys,xs) = im_pca.shape
    print("input array sizes:",ys," x ",xs)

    # get wavelength solution:
    primary_header=hdulist['PRIMARY'].header
    crval1=primary_header['CRVAL1']
    cdelt1=primary_header['CDELT1']
    crpix1=primary_header['CRPIX1']
    naxis1=primary_header['NAXIS1']
    x=np.arange(naxis1)+1
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    lam=L0+x*cdelt1

    # get PCA extension:
    im_skymodpca = hdulist['PCASKYMOD'].data
    hdulist.close()
    
    # get data with no PCA correction:
    hdulist = pf.open(infile_nopca)
    im_nopca = hdulist[0].data
    hdulist.close()

    
    # default range for PCA is 5527A to 5627A.
    #define an index in this range, but ignore the 10 pixels nearest the line,
    # from 5567 to 5587.
    #idx = np.where(((lam > 5527.0) & (lam<5567.0)) | ((lam > 5587.0) & (lam < 5627.0)))
    # more limited band width:
    idx = np.where(((lam > 5547.0) & (lam<5567.0)) | ((lam > 5587.0) & (lam < 5607.0)))


    pcamod_band = np.zeros(ys)
    specflux_band = np.zeros(ys)
    
    # calc mean flux in the band:
    for i in range(ys):
        pcamod_band[i] = np.nanmean(im_skymodpca[i,idx])
        specflux_band[i] = np.nanmean(im_pca[i,idx])

    print(np.shape(pcamod_band))
    
    # median filter data:
    pcamod_band_sm = median_smooth(pcamod_band,50)
    
    # define fibre axis:
    fibax = np.linspace(1.0,float(ys),ys)

    #plot:
    if (doplot):
        fig1 = py.figure(1)
        ax1_1 = fig1.add_subplot(2,1,1)
        ax1_1.plot(fibax,pcamod_band)
        ax1_1.plot(fibax,pcamod_band_sm)
        
        ax1_2 = fig1.add_subplot(2,1,2)
        ax1_2.plot(fibax,specflux_band)

        fig2 = py.figure(2)
        ax2 = fig2.add_subplot(1,1,1)
        ax2.plot(specflux_band,pcamod_band,'.')
        
    # process the original model to make it more robust to
    # bright fibres:
    # 1) median filter each col, ignoring bright fibres
    # 2) gaussian smoother model.

    im_skymodcp = np.copy(im_skymodpca)

    # filter out fibres with high flux:
    for i in range(ys):
        if (pcamod_band[i] > (pcamod_band_sm[i]+4.0)):
            im_skymodcp[i,:] = np.nan
        
    i1 = 1785
    i2 = 1885
    model_sm = np.zeros((ys,xs))
    for i in range(xs):
        if ((i > i1) & (i< i2)): 
            model_sm[:,i] = median_smooth(im_skymodcp[:,i],101)

    model_sm_rm5577 = np.copy(model_sm)

    for i in range(xs):
        if ((lam[i]>5572.0) and (lam[i]<5582.0)):
                model_sm_rm5577[:,i] = np.nan

    # gaussian filter:
    model_sm_rm5577_gf = gaussian_filter_nan(model_sm_rm5577,(5,5))

    # generate the total model:
    model_sm_total = np.copy(model_sm_rm5577_gf)
    for i in range(xs):
        if ((lam[i]>5572.0) and (lam[i]<5582.0)):
                model_sm_total[:,i] = im_skymodpca[:,i]
    
    
    if (doplot):
        vmin=-10.0
        vmax=10.0
        fig3 = py.figure(3)
        ax3_1 = fig3.add_subplot(1,6,1)
        ax3_1.imshow(im_skymodpca[:,i1:i2],origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax)
        
        ax3_2 = fig3.add_subplot(1,6,2)
        ax3_2.imshow(model_sm[:,i1:i2],origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax)

        ax3_3 = fig3.add_subplot(1,6,3)
        ax3_3.imshow(im_skymodcp[:,i1:i2],origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax)
    
        ax3_4 = fig3.add_subplot(1,6,4)
        ax3_4.imshow(model_sm_rm5577[:,i1:i2],origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax)
        
        ax3_5 = fig3.add_subplot(1,6,5)
        ax3_5.imshow(model_sm_rm5577_gf[:,i1:i2],origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax)
        
        ax3_6 = fig3.add_subplot(1,6,6)
        ax3_6.imshow(model_sm_total[:,i1:i2],origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax)


    # subtract the new model from the data:
    im_new = im_nopca - model_sm_total
    
    if (doplot):

        vmin=-30.0
        vmax=30.0
        fig4 = py.figure(4)
        
        ax4_1 = fig4.add_subplot(1,4,1)
        ax4_1.imshow(im_nopca[:,i1:i2],origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax)
        
        ax4_2 = fig4.add_subplot(1,4,2)
        ax4_2.imshow(im_pca[:,i1:i2],origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax)
        
        ax4_3 = fig4.add_subplot(1,4,3)
        ax4_3.imshow(im_new[:,i1:i2],origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax)

##############################################################################
# function to return median sky subtraction residuals for many frames
# to get a reasonable distribution function over the 
#
# usage:
# sami_dr_smc.sami_2dfdr_skysub.check_sky_many('06mar100??red.fits',dopdf=True)
#
# run on full blue sample:
# sami_dr_smc.sami_2dfdr_skysub.check_sky_many('/import/opus1/nscott/SAMI_Survey/20??_??_??-20??_??_??/reduced/*/*/*/main/ccd_1/??????????red.fits',maxscale=0.07,dopdf=True)  
#
# run on full red sample:
# sami_dr_smc.sami_2dfdr_skysub.check_sky_many('/import/opus1/nscott/SAMI_Survey/20??_??_??-20??_??_??/reduced/*/*/*/main/ccd_1/??????????red.fits',maxscale=0.07,lmin=6300.0,lmax=7400.0,dopdf=True)  
#
#

def check_sky_many(inlist,dopdf=False,col='k',label=' ',maxscale=0.15,nlam=20,do_subtest=0,allfib=False):

    doplot = True

    # set up formating:
    py.rc('text', usetex=True)
    py.rcParams.update({'font.size': 14})
    py.rcParams.update({'lines.linewidth': 1})
    py.rcParams.update({'figure.autolayout': True})
    # this to get sans-serif latex maths:
    py.rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
        ]  
  

    
    # define min exposure time:
    min_exp = 900
#    py.rc('text', usetex=True)
#    py.rc('font', family='serif')
    py.rcParams['font.size'] = 16

    # if we want a pdf make it:
    if (dopdf):
        pdf = PdfPages('sky_sub_abs.pdf')
    
    # "glob" the filelist to expand wildcards:
    files = glob.glob(inlist)

    
    # read in the spectrograph ID from the first file 9assuming they are all the same:
    testfile = files[0]
    hdr = pf.getheader(testfile, 0)
    spec = hdr['SPECTID']         
    inst = hdr['INSTRUME']         

    print('Data for spectrograph: ',spec,inst)
    # setting wavelength range based on instrument:
    if (inst == 'SPECTOR'):
        if (spec == 'BL'):
            lmin=3700.0
            lmax=5700.0
        if (spec == 'RD'):
            lmin=5800.0
            lmax=7750.0
    if (inst == 'AAOMEGA-HECTOR'):
        if (spec == 'BL'):
            lmin=3700.0
            lmax=5700.0
        if (spec == 'RD'):
            lmin=6350.0
            lmax=7350.0

            
    
    
    # set size, depending on whether we use all fibres or not.  These numbers are the max
    # number of fibres used.
    #if (allfib):
        #fsize = 819
        #fsize = 900
    #else:
        #fsize = 26
        #fsize = 50
    fsize = 900
        
    # initialize:
    cont=np.zeros(0)
    line=np.zeros(0)
    contlam=np.zeros((fsize,nlam))

    # find the number of files to use to define array size:
    maxfiles = np.size(files)
    print('number of files in list:',maxfiles)
    
    # arrays to store fibre specific values:
    allcont=np.full((fsize,maxfiles),np.nan)
    allline=np.full((fsize,maxfiles),np.nan)
    alladjfrac=np.full((fsize,maxfiles),np.nan)
    allcontlam=np.full((fsize,nlam,maxfiles),np.nan)
    allcontlam_abs=np.full((fsize,nlam,maxfiles),np.nan)
    allfibnum=np.zeros((fsize,maxfiles),dtype=np.int16)
    nfibre=np.zeros(fsize)
    
    # arrays to store statistics for each frame
    medframe_cont = np.zeros(maxfiles)
    medframe_line = np.zeros(maxfiles)
    medframe_contlam = np.zeros(maxfiles)
    p90frame_cont = np.zeros(maxfiles)
    p90frame_line = np.zeros(maxfiles)
    p90frame_contlam = np.zeros(maxfiles)
    frame_fitsig = np.zeros(maxfiles)
    frame_fitcent = np.zeros(maxfiles)

    # open a text file to output stats on a file by file basis:
    output_file = open('skysub_stats.txt','w+')
    output_file.write('# Frame_N Frame_name cont_med line_med cont_p90 line_p90\n')
            
    
    # loop over each 
    nfiles = 0
    for filename in files:

        print(filename)
        # open fits file and check its an object with min exposure time:
        try:
            hdulist = pf.open(filename)

        except IOError:
            print('File not found, skipping ',filename)
            continue
                
        primary_header=hdulist['PRIMARY'].header
        exposure = primary_header['EXPOSED']
        obstype = primary_header['OBSTYPE']

        # get the reduction_args table info (particularly extr_operation):
        try:
            red_args = hdulist['REDUCTION_ARGS'].data
            thput_filename = 'none'
            for vals in red_args:
                if (vals[0] == 'EXTR_OPERATION'): 
                    extract = vals[1]
                if (vals[0] == 'THPUT_FILENAME'): 
                    thput_filename = vals[1]
        except KeyError:
            print('REDUCTION_ARGS extension not found, skipping ',filename)
            continue

        # check for different throughput files:
        print(thput_filename[5:7])
        # if 20, then using twilight:
        # if 29, then dome flat:
        # if _?, then skylines?
        #if (thput_filename[5:7] != '20'):
        #if (thput_filename[5:7] != '29'):
        #if (thput_filename[5:6] != '_'):
        #    print 'wrong throught type, skipping',thput_filename[5:7]
        #    continue
        
        hdulist.close()

        # only select objects frames with at least min-exp exposure time:
        if (obstype == 'OBJECT' and exposure > min_exp and ((extract == 'OPTEX') or (extract == 'SMCOPTEX'))):        
#        if (obstype == 'OBJECT' and exposure > min_exp and extract == 'OPTEX'):        
            #if (obstype == 'OBJECT' and exposure > min_exp and extract == 'GAUSS'):        
        #if (obstype == 'OBJECT' and exposure > min_exp and extract == 'GAUSS'):        

            # now call the check_sky_subtraction code:
            (fibs,contres,lineres,fraccontlam,lamcent,fitsig,fitcent,adjfrac)=check_sky_subtraction(filename,plotall=False,plot=False,returnall=True,verbose=False,nlam=nlam,lmin=lmin,lmax=lmax,do_subtest=do_subtest,allfib=allfib,use_median=False)
            # check that the frame has the right dimensions:                                                                                                               
            if (np.size(contres) > fsize):
                print('output is wrong dimension:',np.shape(contres),' ...skipping')
                continue

            # Store statistics for each frame
            medframe_cont[nfiles] = np.median(np.abs(contres))
            medframe_line[nfiles] = np.median(np.abs(lineres))
            medframe_contlam[nfiles] = np.median(np.abs(fraccontlam))
            p90frame_cont[nfiles] = np.percentile(np.abs(contres),90.0)
            p90frame_line[nfiles] = np.percentile(np.abs(lineres),90.0)
            p90frame_contlam[nfiles] = np.percentile(np.abs(fraccontlam),90.0)
            frame_fitsig[nfiles] = fitsig
            frame_fitcent[nfiles] = fitcent

            output_file.write('{0:5d} {1:s} {2:7.4f} {3:7.4f} {4:7.4f} {5:7.4f}\n'.format(nfiles,filename,medframe_cont[nfiles],medframe_line[nfiles],p90frame_cont[nfiles],p90frame_line[nfiles]))
            
            
            cont = np.append(cont,contres,0)
            line = np.append(line,lineres,0)

            if (nfiles ==0):
                contlam = fraccontlam
            else:
                contlam = np.append(contlam,fraccontlam,axis=0)

            print("array size =",np.shape(contlam))

            # put data into array by fibre number to allow us to calc stats per fibre:
            nn = np.size(fibs)
            for i in range(nn):
                ifib = fibs[i]
                allcont[ifib,nfiles]=contres[i]
                allline[ifib,nfiles]=lineres[i]
                alladjfrac[ifib,nfiles]=adjfrac[i]
                allcontlam_abs[ifib,:,nfiles] = np.abs(fraccontlam[i,:])
                allcontlam[ifib,:,nfiles] = fraccontlam[i,:]
                nfibre[ifib] = nfibre[ifib] + 1
                
            
            #cont + contres
            #line = line + lineres
            nfiles = nfiles + 1
            print(nfiles,filename,obstype,exposure,extract)

    print('nfiles = ',nfiles)

    # close output text file:
    output_file.close()
    
    # get the median residuals as an image in the fibre/lambda plane:
    medcontlam_im_abs = np.nanmedian(allcontlam_abs[:,:,0:nfiles],axis=2)
    medcontlam_im = np.nanmedian(allcontlam[:,:,0:nfiles],axis=2)
    print('size of median image=',np.shape(medcontlam_im_abs))

    # generate the median residuals in an image that only contains fibres with data:
    ngoodfib = np.count_nonzero(nfibre)
    medcontlam_im_good = np.zeros((ngoodfib,nlam))
    print('Number of fibres with data:',ngoodfib)
    nfill = 0
    for i in range(fsize):
        if (nfibre[i] > 0):
            print(i)
            medcontlam_im_good[nfill,:] = medcontlam_im[i,:]
            nfill=nfill+1
    
    # get the median residuals
    medcont = np.median(np.abs(cont))
    medline = np.median(np.abs(line))
    #print cont
    print('stats across absolute residuals per fibre:')
    print(cont)
    print('Absolute continuum residual percentiles (10,50,90,95,99):',np.nanpercentile(np.abs(cont),10.0),medcont,np.nanpercentile(np.abs(cont),90.0),np.percentile(np.abs(cont),95.0),np.nanpercentile(np.abs(cont),99.0))
    print('Absolute line residual percentiles (10,50,90,95,99):',np.nanpercentile(np.abs(line),10.0),medline,np.nanpercentile(np.abs(line),90.0),np.nanpercentile(np.abs(cont),95.0),np.nanpercentile(np.abs(cont),99.0))

    print('stats across absolute residuals per fibre and wavelength bin (cont only):')
    print('Absolute continuum residual percentiles (10,50,90,95,99):',np.nanpercentile(allcontlam_abs[:,:,0:nfiles],10.0),np.nanmedian(allcontlam_abs[:,:,0:nfiles]),np.percentile(allcontlam_abs[:,:,0:nfiles],90.0),np.nanpercentile(allcontlam_abs[:,:,0:nfiles],95.0),np.nanpercentile(allcontlam_abs[:,:,0:nfiles],99.0))
    
    
    # get the median for each fibre:
    fibnum=np.zeros(fsize)
    fibcont=np.zeros(fsize)
    fibline=np.zeros(fsize)
    fibcont_sign=np.zeros(fsize)
    fibline_sign=np.zeros(fsize)
    fibcont10=np.zeros(fsize)
    fibline10=np.zeros(fsize)
    fibcont90=np.zeros(fsize)
    fibline90=np.zeros(fsize)
    
    medsky_cont=np.zeros((nlam))
    sky_10pc_cont=np.zeros((nlam))
    sky_90pc_cont=np.zeros((nlam))

    ng = 0
    for i in range(fsize):
        if (nfibre[i] > 0):
            fibnum[ng]=ng+1
            fibcont[ng] = np.median(np.abs(allcont[i,0:nfiles]))
            fibline[ng] = np.median(np.abs(allline[i,0:nfiles]))
            fibcont_sign[ng] = np.median(allcont[i,0:nfiles])
            fibline_sign[ng] = np.median(allline[i,0:nfiles])
            fibcont10[ng] = np.percentile(np.abs(allcont[i,0:nfiles]),10.0)
            fibline10[ng] = np.percentile(np.abs(allline[i,0:nfiles]),10.0)
            fibcont90[ng] = np.percentile(np.abs(allcont[i,0:nfiles]),90.0)
            fibline90[ng] = np.percentile(np.abs(allline[i,0:nfiles]),90.0)
            ng=ng+1
        #print i,fibnum[i],fibcont[i],fibline[i]

    # generate the wavelength dependent version:
    for j in range(nlam):
        medsky_cont[j]=np.median(abs(contlam[:,j]))
        sky_10pc_cont[j]=np.percentile(abs(contlam[:,j]),10.0)
        sky_90pc_cont[j]=np.percentile(abs(contlam[:,j]),90.0)
        
        print('10%, median, 90% absolute continuum residuals:',sky_10pc_cont[j],medsky_cont[j],sky_90pc_cont[j],lamcent[j])


    # rework plots and only keep the useful ones.  No longer use absolute values:
    if (doplot):

        if (dopdf):
            pdf = PdfPages('sky_sub.pdf')

        # next generate the sample plot, but show the residuals with the sign, to see
        # any systematic trends there.
        py.figure(5)            
        # plot residuals of individual observations per fibre:
        #for i in range(fsize):
            #xi=np.ones(nfiles)*(i+1)
            #yi = allcont[i,0:nfiles]
            #if (not dopdf):
#                print i,nfiles,np.size(xi),np.size(yi)
                #py.plot(xi,yi,'x',color='b')
                
        lab = label+' continuum residual'
        py.plot(fibnum[0:ng],fibcont_sign[0:ng],'.',label=lab,color=col)
        lab = label+' line residual'
        py.plot(fibnum[0:ng],fibline_sign[0:ng],'x',label=lab,color=col)
        print(fibnum[0:ng],fibcont_sign[0:ng],fibline_sign[0:ng])
        py.axhline(0.0,color='k',linestyle='--')
        py.xlabel('Fibre')
        py.ylabel('fractional sky residual')
        py.xlim(xmin=0,xmax=ng+2)
        py.ylim(ymin=-1.0*maxscale,ymax=maxscale)
        
        py.legend(prop={'size':10},numpoints=1)
        
        if (dopdf):
            py.savefig(pdf, format='pdf')        
            pdf.close()
            #py.close()

        

        # generate second plot with wavelength dependence:
        if (dopdf):
            pdf = PdfPages('sky_sub_lam.pdf')


        py.figure(2)
        py.plot(lamcent,medsky_cont,'-',color=col)
        py.plot(lamcent,sky_90pc_cont,':',color=col)
        py.plot(lamcent,sky_10pc_cont,':',color=col)
        py.xlabel('Wavelength [\AA]')
        py.ylabel('fractional sky residual')
        py.xlim(xmin=lmin,xmax=lmax)
        py.ylim(ymin=0.0,ymax=0.1)
        
        if (dopdf):
            py.savefig(pdf, format='pdf')        
            pdf.close()
            #py.close()
            
        # generate third plot with wavelength dependence and fibre:
        if (dopdf):
            pdf = PdfPages('sky_sub_lam_fib.pdf')

        # set aspect, so that it looks the same whatever the wavelength range:
        asp = 100*(lmax-lmin)/(5700.0-3700.0)

        fig4 = py.figure(4)
        ax4 = fig4.add_subplot(1,1,1)
        cax4 = ax4.imshow(medcontlam_im_good,vmin=-1.0*maxscale,vmax=maxscale,origin='lower',interpolation='nearest',extent=[lmin,lmax,0,ngoodfib],cmap=py.cm.RdYlBu,aspect=asp)
        ax4.set(xlabel='Wavelength [\AA]',ylabel='Fibre number')
        ax4.xaxis.set_major_locator(ticker.LinearLocator(5))
        #ax4.get_yticklabels()[0].set_visible(False)
#py.xlim(xmin=3700.0,xmax=5700.0)
        #py.ylim(ymin=0.0,ymax=0.1)
        ticks = np.linspace(-1.0*maxscale,maxscale,5)
        cbar4 = fig4.colorbar(cax4,label='fractional sky residual',ticks=ticks)
        #py.colorbar(label='fractional sky residual',ticks=ticks)
        
        if (dopdf):
            py.savefig(pdf, format='pdf')        
            pdf.close()
            #py.close()

        #
        # plot the distribution of values from different frames:
        print('Per frame median abs cont residual:')
        (mean,rms,nc) = clipped_mean_rms(medframe_cont[0:nfiles],5.0)
        print('mean, rms, n, median =',mean,rms,nc,np.median(medframe_cont[0:nfiles]))
        print('Per frame median abs line residual:')
        (mean,rms,nc) = clipped_mean_rms(medframe_line[0:nfiles],5.0)
        print('mean, rms, n, median =',mean,rms,nc,np.median(medframe_line[0:nfiles]))
        print('Per frame median abs cont residual (lambda bins):')
        (mean,rms,nc) = clipped_mean_rms(medframe_contlam[0:nfiles],5.0)
        print('mean, rms, n, median =',mean,rms,nc,np.median(medframe_contlam[0:nfiles]))
        # get the stats for 90th percentile
        print(' ')
        print('stats for 90th percentile:')
        print('Per frame median abs cont residual:')
        (mean,rms,nc) = clipped_mean_rms(p90frame_cont[0:nfiles],5.0)
        print('mean, rms, n, median =',mean,rms,nc,np.median(p90frame_cont[0:nfiles]))
        print('Per frame median abs line residual:')
        (mean,rms,nc) = clipped_mean_rms(p90frame_line[0:nfiles],5.0)
        print('mean, rms, n, median =',mean,rms,nc,np.median(p90frame_line[0:nfiles]))
        print('Per frame median abs cont residual (lambda bins):')
        (mean,rms,nc) = clipped_mean_rms(p90frame_contlam[0:nfiles],5.0)
        print('mean, rms, n, median =',mean,rms,nc,np.median(p90frame_contlam[0:nfiles]))

        # get median fitted sigma:
        print(' ')
        print('per frame fit of sigma:')
        (mean,rms,nc) = clipped_mean_rms(frame_fitsig[0:nfiles],5.0)
        medsig = np.median(frame_fitsig[0:nfiles])
        print('mean, rms, n, median =',mean,rms,nc,medsig)
        print('per frame fit of offset:')
        (mean,rms,nc) = clipped_mean_rms(frame_fitcent[0:nfiles],5.0)
        medcent = np.median(frame_fitcent[0:nfiles])
        print('mean, rms, n, median =',mean,rms,nc,medcent)


        
        if (doplot):
            if (dopdf):
                pdf = PdfPages('sky_frame_medians.pdf')

            py.figure(6)
            py.subplot(2,1,1)
            py.hist(medframe_cont[0:nfiles],bins=25,range=(-0.1,0.1),color='r',histtype='step',label='cont')
            py.hist(medframe_line[0:nfiles],bins=25,range=(-0.1,0.1),color='b',histtype='step',label='line')
            py.hist(medframe_contlam[0:nfiles],bins=25,range=(-0.1,0.1),color='g',histtype='step',label='cont (lam bins)')
            py.legend(prop={'size':10})
            py.xlabel('Median absolute fractional sky residual')

            py.subplot(2,1,2)
            py.hist(p90frame_cont[0:nfiles],bins=25,range=(-0.1,0.1),color='r',histtype='step',label='cont')
            py.hist(p90frame_line[0:nfiles],bins=25,range=(-0.1,0.1),color='b',histtype='step',label='line')
            py.hist(p90frame_contlam[0:nfiles],bins=25,range=(-0.1,0.1),color='g',histtype='step',label='cont (lam bins)')
            py.xlabel('90th percentile absolute fractional sky residual')

            if (dopdf):
                py.savefig(pdf, format='pdf')        
                pdf.close()

            py.figure(10)
            #py.hist(cont,bins=80,range=(-0.1,0.1),color='r',histtype='step',label='cont')
            py.hist(cont,bins=80,range=(-0.1,0.1),histtype='step',label='cont')
            py.xlabel('fractional sky residual')
            py.ylabel('Number of fibres')
            medcont = np.nanmedian(cont)
            py.axvline(medcont,color='b',linestyle=':')


                
            # plot median values per frame as a function of frame number:
            if (dopdf):
                pdf = PdfPages('sky_vs_frames.pdf')

            py.figure(7)
            xax = np.arange(nfiles)
            py.plot(xax,medframe_cont[0:nfiles],'-',color='b',label='median continuum')
            py.plot(xax,p90frame_cont[0:nfiles],':',color='b',label='90% continuum')
            py.plot(xax,medframe_line[0:nfiles],'-',color='r',label='median line')
            py.plot(xax,p90frame_line[0:nfiles],':',color='r',label='90% line')
            py.legend(prop={'size':10},numpoints=1)
            py.xlabel('Frame number')
            py.ylabel('Fractional sky residual')
            
            if (dopdf):
                py.savefig(pdf, format='pdf')        
                pdf.close()

                
            # plot fitted offsets and sigma values per frame as a function of frame number:
            if (dopdf):
                pdf = PdfPages('fit_vs_frames.pdf')

            py.figure(8)
            py.subplot(2,1,1)
            xax = np.arange(nfiles)
            py.plot(xax,frame_fitsig[0:nfiles],'-',color='b',label='fitted sigma')
            py.ylim(ymin=0.8,ymax=1.2)
            py.axhline(1.0,color='k',linestyle='-')
            py.axhline(medsig,color='b',linestyle=':')
            py.xlabel('Frame number')
            py.ylabel('fitted sigma')
            
            py.subplot(2,1,2)
            py.plot(xax,frame_fitcent[0:nfiles],'-',color='b',label='fitted offset from zero')
            py.ylim(ymin=-0.1,ymax=0.1)
            py.axhline(0.0,color='k',linestyle='-')
            py.axhline(medcent,color='b',linestyle=':')
            py.xlabel('Frame number')
            py.ylabel('Fitted offset')
            
            if (dopdf):
                py.savefig(pdf, format='pdf')        
                pdf.close()


            # plot adjacent fibre flux vs. sky subtraction residual.
            py.figure(9)
            print(np.shape(alladjfrac[:,0:nfiles]))
            print(np.shape(allcont[:,0:nfiles]))
            py.plot(np.ravel(alladjfrac[:,0:nfiles]),np.ravel(allcont[:,0:nfiles]),'o')
            py.xlabel('(adjacent fibre flux)/(sky fibre flux)')
            py.ylabel('Fractional sky residual')

            # calculate binned estimate of residual:
            binsize = 0.1
            binmax = 4.0
            nbin = int(binmax/binsize)

            ffmean=np.zeros(nbin)
            ffmed=np.zeros(nbin)
            ffrms=np.zeros(nbin)
            ffbin=np.zeros(nbin)
            binfrac=np.zeros(100000)
            
            for ib in range(nbin):
                ibin1 = ib * binsize
                ibin2 = (ib+1) * binsize
                nb = 0
                for ii in range(fsize):
                    for jj in range(nfiles):
                        if ((alladjfrac[ii,jj] > ibin1) & (alladjfrac[ii,jj] < ibin2)):
                            binfrac[nb] = allcont[ii,jj]
                            nb =nb + 1  

                # get the mean in the bin (and sigma):
                ffmean[ib],ffrms[ib],nclip = clipped_mean_rms(binfrac[0:nb],5.0)
                ffmed[ib] = np.nanmedian(binfrac[0:nb])
                # std error on the mean:
                ffrms[ib] = ffrms[ib]/np.sqrt(nclip)
                ffbin[ib] = (ibin1+ibin2)/2.0
        

            py.errorbar(ffbin[0:nbin],ffmean[0:nbin],ffrms[0:nbin],fmt='o',color='r')
            py.plot(ffbin[0:nbin],ffmed[0:nbin],'o',color='g')
            
            py.axhline(0.0,color='k',linestyle='-')
            

##############################################################################
# script to test sky subtraction accuracy.  Now modified to run on a single
# frame, not red and blue at the same time, as these are actually two 
# separate function and could be merged into one
#
def check_sky_subtraction(infile,plotall=False,plot=True,returnall=False,verbose=True,allfib=False,do_legend=False,nlam=1,lmin=3700,lmax=5700,do_subtest=0,thputfile=None,use_median=False):
    """Function to check sky subtraction accuracy in SAMI and Hector data.
    This can be run multiple times to test several frames and compare.
    If return all is set to True, then two vectors of residuals are
    returned to the calling function (continuum and sky).
    Usage:
    sami_tools_smc.dr_tools.sami_2dfdr_skysub.check_sky_subtraction('06mar10055red.fits',plotall=False,plot=True,returnall=False,nlam=1)
    or
    (contres_vec,lineres_vec)=sami_2dfdr_reduction_tests.check_sky_subtraction('06mar10055red.fits',plotall=False,plot=True,returnall=True)

    the optional flags are:
    plotall    - generate all the plots.
    plot       - make plots
    returnall  - return results as output of function
    verbose    - output to the screen
    allfib     - do all fibres, not just skies
    nlam       - number of wavelength bins to independently calculate continuum sky residual
    do_subtest - test different ways of combing sky spectra and subtracting
    use_median - if true calculate the median residuals.  If false, use the mean.

    """
    
    #hwidth in pixels for region to check skylines:
    hwidth = 10

    # max number of true sky fibres:
    sfmax = 50
    # always max of 26 sky fibres with SAMI:
    #sfmax = 26
    
    # do we ignore some fibres, and if so, which ones:
    ignore = False
    ignore_list = np.array([379,454,492])

    # subtract 1 from ignore list as python list is zero indexed:
    if (ignore):
        print('fibres we will ignore:',ignore_list)
        ignore_list = ignore_list - 1

    # get the throughputs if needed:
    if (thputfile == None):
        print('No throughput file given')
        dothput=False
    else:
        dothput=True 


    if (dothput):
        hdulist1 = pf.open(thputfile)
        # get thput data:
        thput = hdulist1['THPUT'].data
        thput_med = median_smooth(thput,21)
        hdulist1.close() 
    
    # open files:
    hdulist = pf.open(infile)

    # min/max wavelength range:
    # set to ignore the 5577 sky line (now a variable in function call)
    #lmin = 3700
    #lmax = 5400
    
    # get data and variance
    im = hdulist[0].data
    var = hdulist['VARIANCE'].data

    # get array sizes:
    (ys,xs) = im.shape
    #print "input array sizes:",ys," x ",xs

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

    # find min/max pixel for lmin/lmax:
    imin=0
    imax=naxis1
    ld=1.0e10
    for i in range(naxis1):
        diff = abs(lam[i]-lmin)
        if (diff < ld):
            imin = i
            ld = diff
            
    ld=1.0e10
    for i in range(naxis1):
        diff = abs(lam[i]-lmax)
        if (diff < ld):
            imax = i
            ld = diff

    print('min/max lambda wavelength and pixels = ',lmin,lmax,imin,imax)
    irange = imax-imin
            
    # get binary table info:
    fib_tab_hdu=find_fibre_table(hdulist)
    table_data = hdulist[fib_tab_hdu].data
    types=table_data.field('TYPE')
    probename=table_data.field('PROBENAME')
    slitlet=table_data.field('SLITLET')
    
    # next add back in the sky spectrum to the data and then
    im_sky = im + sky

    # define arrays that will be used to store fibres used as sky fibres:
    sf_spec = np.zeros((sfmax,xs))
    sf_spec_plussky = np.zeros((sfmax,xs))
    sf_var = np.zeros((sfmax,xs))

    # fill arrays that contain only sky fibres:
    # define a sky only array:
    nsf = 0
    for i in range(ys):
        if ((types[i] == 'S')):
            # skip fibres if we want to ignore them (e.g. because contaminated):
            if (ignore):
                ig = 0
                for j in range(np.size(ignore_list)):
                    if (i == ignore_list[j]):
                        print("ignoring fibre ",i+1)
                        ig = 1
                if (ig == 1):
                    continue
                
            sf_spec[nsf,:] = im[i,:] 
            sf_spec_plussky[nsf,:] = im_sky[i,:] 
            sf_var[nsf,:] = var[i,:]

            nsf = nsf + 1
    
    # define arrays that can hold sky fibres or all the fibres:
    if (allfib):
        sky_only = np.zeros((ys,xs))
        sky_only_sky = np.zeros((ys,xs))
        sig_sky_only = np.zeros((ys,xs))
        var_sky_only = np.zeros((ys,xs))
    else:
        sky_only = np.zeros((sfmax,xs))
        sky_only_sky = np.zeros((sfmax,xs))
        sig_sky_only = np.zeros((sfmax,xs))
        var_sky_only = np.zeros((sfmax,xs))
        
    # get sigma image by dividing by sqrt(var):
    sig_im = im /np.sqrt(var)

    # define a sky only array:
    ns = 0
    for i in range(ys):
        if ((types[i] == 'S') or (allfib and ((types[i] == 'S') or (types[i] == 'P')))):
            # skip fibres if we want to ignore them (e.g. because contaminated):
            if (ignore):
                ig = 0
                for j in range(np.size(ignore_list)):
                    if (i == ignore_list[j]):
                        print("ignoring fibre ",i+1)
                        ig = 1
                if (ig == 1):
                    continue
                
            sky_only[ns,:] = im[i,:] 
            sky_only_sky[ns,:] = im_sky[i,:] 
            sig_sky_only[ns,:] = sig_im[i,:]
            var_sky_only[ns,:] = var[i,:]

            # define a median smoothed spectrum and subtract to estimate
            # the variance separately
            #medflux = median_smooth(im[i,:], 50, givenmad=False)
            #rms1,nclip = clipped_rms(im[i,:]-medflux,5.0)
            #medvar = np.nanmedian(np.sqrt(var[i,:]))
            #medvar = np.median(np.sqrt(var[i,:]))
            # find the ratio of scatter around median subtracted flux and sqrt(variance):
            #err_ratio = (im[i,:] - medflux)/np.sqrt(var[i,:])
            ns = ns + 1


    # now we can get an average sky in different ways.  For this we need to use the arrays
    # that are only true sky fibres. 
    if (do_subtest > 0): 
        sky_mean_nowt = np.nanmean(sf_spec_plussky[0:nsf,:],axis=0)
        sky_median_nowt = np.nanmedian(sf_spec_plussky[0:nsf,:],axis=0)
        sky_std_nowt = np.nanstd(sf_spec_plussky[0:nsf,:],axis=0)
        # this is a new function, so not in pythogn 2.7...
        #sky_mad_nowt = stats.median_absolute_deviation(sky_only_sky,axis=0,nan_policy='omit')
        # subtract unweighted median to get difference:
        sky_only_medsub =  sf_spec_plussky[0:nsf,:]- sky_median_nowt
        # get median of absolute difference:
        sky_mad_nowt = np.nanmedian(np.absolute(sky_only_medsub[0:nsf,:]),axis=0)*1.4826

        # flag points that are >5xMAD from the data:
        sky_only_sky_flagged = np.where((np.abs(sf_spec_plussky[0:nsf,:] - sky_median_nowt) > 5.0*sky_mad_nowt),np.nan,sf_spec_plussky[0:nsf,:])

        # define MAD clipped version of sky:
        sky_mean_nowt_madclipped = np.nanmean(sky_only_sky_flagged[0:nsf,:],axis=0)
        

        # only make these plots if doing sky-sub tests:
        if (plot):
            py.figure(10)
            py.subplot(3,1,1)
            py.plot(lam,sky,label='original 2dfdr median sky')
            py.plot(lam,sky_mean_nowt,label='unweighted mean sky')
            py.plot(lam,sky_median_nowt,label='unweighted median sky')
            py.legend(prop={'size':10})
            py.subplot(3,1,2)
            py.plot(lam,sky_mean_nowt-sky,label='mean-original')
            py.plot(lam,sky_mean_nowt-sky+2*sky_std_nowt,color='r',linestyle=':',label='2 sigma range')
            py.plot(lam,sky_mean_nowt-sky-2*sky_std_nowt,color='r',linestyle=':')
            py.axhline(0.0,color='k',linestyle='--')
            py.legend(prop={'size':10})
            py.subplot(3,1,3)
            py.plot(lam,(sky_mean_nowt-sky)/sky_mean_nowt,label='(mean-original)/mean')
            py.axhline(0.0,color='k',linestyle='--')
            py.legend(prop={'size':10})

    # also plot all the individual spectra on top of each other to get a sense of how they compare:
            py.figure(11)
            for i in range(nsf):
                py.plot(lam,sf_spec_plussky[i,:])
                py.plot(lam,sky_mean_nowt+5*sky_mad_nowt,color='r',linestyle=':')
                py.plot(lam,sky_mean_nowt-5*sky_mad_nowt,color='r',linestyle=':')
                
    # the same but flagged:
            py.figure(12)
            for i in range(nsf):
                py.plot(lam,sky_only_sky_flagged[i,:])
                py.plot(lam,sky_mean_nowt+5*sky_mad_nowt,color='r',linestyle=':')
                py.plot(lam,sky_mean_nowt-5*sky_mad_nowt,color='r',linestyle=':')
    
        # if we are using the different sky subtraction to test, then ajust all the arrays
        # used below to use the new sky sub:
        if (do_subtest == 1):
            im = im_sky - sky_mean_nowt
            sky_only = sky_only + sky - sky_mean_nowt
            sig_im = im /np.sqrt(var)
        elif(do_subtest == 2):
            im = im_sky - sky_median_nowt
            sky_only = sky_only + sky - sky_median_nowt
            sig_im = im /np.sqrt(var)
        elif(do_subtest == 3):
            im = im_sky - sky_mean_nowt_madclipped
            sky_only = sky_only + sky - sky_mean_nowt_madclipped
            sig_im = im /np.sqrt(var)

    # redefine some arrays once we have made the above changes:
    nss=0
    for i in range(ys):
        if ((types[i] == 'S') or (allfib and ((types[i] == 'S') or (types[i] == 'P')))):
#        if ((types[i] == 'S') or (allfib)):                
            sky_only[nss,:] = im[i,:] 
            sky_only_sky[nss,:] = im_sky[i,:] 
            sig_sky_only[nss,:] = sig_im[i,:]
            var_sky_only[nss,:] = var[i,:]
            nss=nss+1

            
    # next measure width of distribtuion at each wavelength in the
    # sigma image.  Because we have divided the residual spectrum by
    # sqrt(variance) the sigma should be 1. Here we calculate the
    # sigma in at a fixed wavelength, to see if it changes with
    # wavelength:
    sigma = np.nanstd(sig_sky_only,axis=0)
    # get the median, mean etc of sigma (should be 1):
    sig_mean = np.nanmean(sigma)
    sig_median = np.nanmedian(sigma)
    #sig_median = np.median(sigma)
    print('unclipped scaled sigma (should be 1 as scaled by sqrt(var)):')
    print('mean scaled sigma:',sig_mean)
    print('median scaled sigma:',sig_median)

    # redo the sigma measurement, but this time clipping:
    for i in range(np.size(sigma)):
        sigma[i],nclip = clipped_rms(sig_sky_only[0:ns,i],5.0)
        
    # get the median, mean etc of sigma (should be 1):
    sig_mean = np.nanmean(sigma)
    sig_median = np.nanmedian(sigma)
    #sig_median = np.median(sigma)
    print('clipped sigmas:')
    print('mean scaled sigma:',sig_mean)
    print('median scaled sigma:',sig_median)
    if (plot):
        # next plot the sigma spectrum, that should be close to 1:
        py.figure(3)
        py.axhline(1,color='k',linestyle='--')
        py.axhline(sig_median,color='r',linestyle='--')
        py.plot(lam,sigma)
        #py.text(lam[0]-50,yoff,lab, fontsize=10,color='r')
        py.ylim(ymin=0.0,ymax=2.0)
        py.xlim(xmin=lam[0]-100,xmax=lam[xs-1])
        py.xlabel('Wavelength (Angstoms)')
        py.ylabel('sigma for flux/sqrt(variance)')
        py.title('scaled sigma vs wavelength')
        
    # set up bins for noise histogram:
    nbin=200
    bmin=-5.0
    bmax=5.0
    bsize = (bmax-bmin)/float(nbin)
    print('bsize=',bsize)
        
    if (plot):
        # next, check to see if we have consistent variances, plot the histogram of all spectral
        # pixels:
        py.figure(4)
        #py.subplot(1,2,1)
        # plot the histogram, but also get the data from it:
        (nn,bins,patches) = py.hist(np.ravel(sig_sky_only),bins=nbin,range=(bmin,bmax),histtype='step')
        # plot a gaussian
        x = np.linspace(-5.0, 5.0, 1000)
        py.plot(x,stats.norm.pdf(x, 0, sig_mean)*xs*ns*bsize,label='measured normal dist')
        py.plot(x,stats.norm.pdf(x, 0.0,1.0)*xs*ns*bsize,color='r',label='sigma=1 normal dist')
    #    py.plot(x,stats.norm.pdf(x-median, 0.0,sig_clipped_zero)*xs*ys*0.1,color='c',label='measured normal dist (zeroed)')
        py.axvline(0.0,color='r',linestyle='--')
        py.xlim(xmin=-5.0,xmax=5.0)
        #py.subplot(1,2,2)
        #py.plot(np.ravel(sky_only),np.ravel(var_sky_only),'ko', markersize=2)

    # calculate the binned sigma distribution without plotting, so that we can do calculations even
    # when running in non-plot mode on many frames at once:
    (nn,bins) = np.histogram(np.ravel(sig_sky_only), bins=nbin, range=(bmin,bmax))
    
    # fit a Gaussian to the distribution:
    err = np.zeros(np.size(nn))
    xx = np.zeros(np.size(nn))
    for i in range(np.size(nn)):
        xx[i] = (bins[i]+bins[i+1])/2.0
        err[i] = max(1.0,np.sqrt(nn[i]))

    # The sigma can be affected by the spike at zero difference.  This is caused by
    # using the median sky spectrum.  Instead we need to fit the gaussian distribution
    # directly while cutting out the centre.
    # remove central spike (needed for median sky spectrum)
    indx = np.where(np.abs(xx)<0.02)
    xxfit = np.delete(xx,indx)
    nnfit = np.delete(nn,indx)
    errfit = np.delete(err,indx)
    pars=[200.0,0.0,1.0]
    (popt2,cov2)=sp.optimize.curve_fit(gaussian, xxfit,nnfit,p0=pars,sigma=errfit)
    print('fitted sigma:',popt2[2],'+-',cov2[2,2])
    print('fitted centre:',popt2[1],'+-',cov2[1,1])
    fitsig = popt2[2]
    fitcent = popt2[1]

    if (plot):
        py.plot(xx,gaussian(xx,*popt2),':',label='Fitted normal dist')
        py.axvline(popt2[1],color='r',linestyle=':')
        py.legend(prop={'size':10})
        


    # get min/max for scale.  First get a masked array to remove NaNs:
    spec_test = im[0,:]
    spec_test = spec_test[~np.isnan(spec_test)]
    zmin = np.percentile(spec_test,2.0)
    zmax = np.percentile(spec_test,98.0)

    # define sky lines:
    skylines_all=np.array([5577.346680,6300.308594,6363.782715,6533.049805,6553.625977,6923.192383,7316.289551,7340.900879,7358.680176])
    # remove sky lines that are outside the wavelength range:
    skyidx = np.where(((skylines_all>lmin) & (skylines_all<lmax)))
    skylines=skylines_all[skyidx]

    print('Wavelength of skylines to use:')
    print(skylines)
    
    # first overplot all the individual sky fibres to see how the residuals vary:
    if (plotall):
        yoff = 0
        for i in range(ys):
            ifig = 1
            if (i == 400):
                yoff = 0
                
            if (i >= 400):
                ifig = 2

            py.figure(ifig)

            if (types[i] == 'S'):
                # plot the sky-subtracted sky spectrum
                py.axhline(yoff,color='k',linestyle='--')
                py.plot(lam,im[i,:]+yoff)
                lab = str(i+1)
                py.text(lam[0]-50,yoff,lab, fontsize=10,color='r')
                yoff = yoff+40
                

                        # plot the first sky fibre in SAMI and scale by continuum level:
                if (i+1 == 757):
                    py.figure(5)
                    medcont = np.nanmedian(im_sky[i,:])
                    medmedcont = np.nanmedian(sky)
                    print(medcont,medmedcont,medcont/medmedcont)
                    py.plot(lam,im_sky[i,:],label='single fibre')
                    py.plot(lam,im_sky[i,:]*medmedcont/medcont,label='cont scaled single fibre')
                    py.plot(lam,sky,label='median sky')


                
    # set limits:
        py.figure(1)
        py.ylim(ymin=zmin,ymax=zmax+yoff)
        py.xlim(xmin=lam[0]-100,xmax=lam[xs-1])
        py.figure(2)
        py.ylim(ymin=zmin,ymax=zmax+yoff)
        py.xlim(xmin=lam[0]-100,xmax=lam[xs-1])
        py.figure(5)
    #    py.ylim(ymin=zmin,ymax=zmax+yoff)
        py.xlim(xmin=lam[0],xmax=lam[xs-1])
        py.legend(prop={'size':10})

    # calculate some basic statistics for each sky spectrum.
    # what do we want: 
    # 1) summed residual flux as a fraction of total sky
    # 2) summed residual flux as fraction for strong emission lines

    fraccontlam=np.zeros((ys,nlam))
    fluxcontlam=np.zeros((ys,nlam))
    lamcent=np.zeros((ys,nlam))
    lam1=np.zeros((ys,nlam))
    lam2=np.zeros((ys,nlam))

    medsky_cont=np.zeros((nlam))
    sky_10pc_cont=np.zeros((nlam))
    sky_90pc_cont=np.zeros((nlam))

    
    ns = 0 
    fracs=np.zeros(ys)
    fluxs=np.zeros(ys)
    adjfrac=np.zeros(ys)
    adjfrac_av=np.zeros(ys)
    fibs=np.zeros(ys,dtype=np.int16)
    line_fracs=np.zeros(ys)

    fracs_lam=np.zeros((ys,nlam))
    line_fracs_lam=np.zeros((ys,nlam))


    # calculate the residual sky flux.  To do this we calculate the median sky flux in each
    # fibre and the median residual, then take the ratio.  One issue is whether this is an
    # appropriate way to make the measurement.
    if (verbose):
        if (use_median):
            print('Median residuals per fibre:')
        else:
            print('Mean residuals per fibre:')
        print('   |       Cont flux       |      Line flux        ')
        print('fib| f(sky)   f(res)   frac| f(sky)   f(res)  frac ')
    for i in range(ys):
        if (types[i] == 'S' or allfib):
            # skip fibres if we want to ignore them (e.g. because contaminated):
            if (ignore):
                ig = 0
                for j in range(np.size(ignore_list)):
                    if (i == ignore_list[j]):
                        print("ignoring fibre ",i+1)
                        ig = 1
                if (ig == 1):
                    continue
                
            
            spec_test = im[i,:]
            spec_test = spec_test[~np.isnan(spec_test)]
            # take both the average and median.  Note that the median can be impacted by
            # the fact we subtract the median sky, so some pixels in the sky subtraction will
            # be exactly zero.  The average will be better in this case.
            sub_med = np.median(spec_test)
            sub_av = np.mean(spec_test)

            spec_test = im_sky[i,:]
            spec_test = spec_test[~np.isnan(spec_test)]
            sky_med = np.median(spec_test)
            sky_av = np.mean(spec_test)
            
            frac = sub_med/sky_med
            frac_av = sub_av/sky_av
            #print i,types[i],frac,sky_av,sub_av,frac_av

            # find the object fibre that is closest to this sky fibre and estimate the
            # flux in that fibre (i.e. how bright is the source).  Include the flux
            # in from the sky, even though this does get subtracted off.
            afound = False
            if (i > 0):
                if (types[i-1] == 'P'):
                    spec_adjacent = im_sky[i-1,:]
                    afound = True

            if (i<ys-1):
                if (types[i+1] == 'P'):
                    spec_adjacent = im_sky[i+1,:]
                    afound = True

            if (afound):
                spec_adjacent = spec_adjacent[~np.isnan(spec_adjacent)]
                adj_med = np.median(spec_adjacent)
                adj_av = np.mean(spec_adjacent)
            else:
                adj_med = np.nan
                adj_av = np.nan

            # estimate the adjacent fibre flux fraction relative to the median flux
            # in the sky fibre (not sky subtracted):
            adjfrac[ns] = adj_med/sky_med
            adjfrac_av[ns] = adj_av/sky_av
            if (use_median):
                fluxs[ns]=sub_med
                fracs[ns]=frac
            else:
                fluxs[ns]=sub_av
                fracs[ns]=frac_av
            
            fibs[ns]=i+1

            # next calculate the continuum sky residual in different wlavelength intervals
            for j in range(nlam):
                j1 = int(np.rint(imin + j * irange/nlam))
                j2 = int(np.rint(imin + (j + 1) * irange/nlam -1))
                #print 'pixel range:',j1,j2
                spec_test = im[i,j1:j2]
                spec_test = spec_test[~np.isnan(spec_test)]
                sub_medlam = np.median(spec_test)
                sub_avlam = np.mean(spec_test)

                spec_test = im_sky[i,j1:j2]
                spec_test = spec_test[~np.isnan(spec_test)]
                sky_medlam = np.median(spec_test)
                sky_avlam = np.mean(spec_test)

                if (use_median):
                    fraclam = sub_medlam/sky_medlam
                    fluxcontlam[ns,j]=sub_medlam
                else:
                    fraclam = sub_avlam/sky_avlam
                    fluxcontlam[ns,j]=sub_avlam
                    
                fraccontlam[ns,j]=fraclam
                lamcent[ns,j]=(lam[j1]+lam[j2])/2.0
                lam1[ns,j]=lam[j1]
                lam2[ns,j]=lam[j2]
                #print 'fibre ',i,'  wavelength bin ',j,j1,j2,fraclam,lamcent[ns,j]

            
            line_res=0
            line_flux=0
            line_res_s=0
            line_flux_s=0
            ibad=0

            nlines_used=0
            
            for line in skylines:
                # only use lines in range:
                if (line > lam[0] and line < lam[xs-1]):

                    #print 'testing...',line,lam[0],lam[xs-1]
                    ll = lam
                    ff = im
                    ss = im_sky

                    nlines_used=nlines_used+1
                
                    # get the index of the pixel nearest the sky line:

                    iloc = min(range(len(ll)), key=lambda i: abs(ll[i]-line))

                    # get the data around the sky line
                    xx = ll[iloc-hwidth:iloc+hwidth+1]
                    yy = ff[i,iloc-hwidth:iloc+hwidth+1]

                    # get median filtered continuum near the line:
                    #cont = nd.filters.median_filter(ff[i,:],size=51)
                    #cont_sky = nd.filters.median_filter(ss[i,:],size=51)
                    #cont = median_smooth(ff[i,:],51)
                    #cont_sky = median_smooth(ss[i,:],51)
                    cont = median_filter_nan(ff[i,:],51)
                    cont_sky = median_filter_nan(ss[i,:],51)
                    cc = cont[iloc-hwidth:iloc+hwidth+1]
                    cc_sky = cont_sky[iloc-hwidth:iloc+hwidth+1]
                
                    #sig = np.sqrt(yy)
                    # sum the flux over the line
                    line_res = np.sum(ff[i,iloc-hwidth:iloc+hwidth+1]-cont[iloc-hwidth:iloc+hwidth+1])
                    line_flux = np.sum(ss[i,iloc-hwidth:iloc+hwidth+1]-cont_sky[iloc-hwidth:iloc+hwidth+1])

                    # get the residual line flux 
                    if (np.isnan(line_res) or np.isnan(line_flux)):
                        ibad=ibad+1
                    else:
                        line_res_s = line_res_s + line_res
                        line_flux_s = line_flux_s + line_flux


                        #print 'test:',line_res_s,line_flux_s,ibad
            if (line_flux_s > 0):
                line_fracs[ns] = line_res_s/line_flux_s
            else:
                line_fracs[ns] = 0.0
                
            if (verbose):
                if (use_median):
                    print('{0:3d} {1:8.4f} {2:8.4f} {3:7.4f} {4:8.2f} {5:7.2f} {6:6.3f} {7:1s} {8:8s} {9:2d}'.format(i+1,sky_med,sub_med,frac,line_flux_s,line_res_s,line_fracs[ns],types[i],probename[i],slitlet[i]))
                else:
                    print('{0:3d} {1:8.4f} {2:8.4f} {3:7.4f} {4:8.2f} {5:7.2f} {6:6.3f} {7:1s} {8:8s} {9:2d}'.format(i+1,sky_av,sub_av,frac_av,line_flux_s,line_res_s,line_fracs[ns],types[i],probename[i],slitlet[i]))
                            
            #print i+1,sky_med_r,sub_med_r,frac_r,line_flux_r,line_res_r,line_fracs_r[ns]

            ns=ns+1
            #            print 'number of lines used:',nlines_used

    # get the median fractional sky residuals:
    medsky_cont1=np.nanmedian(abs(fracs[0:ns]))
    medsky_line1=np.nanmedian(abs(line_fracs[0:ns]))
    avsky_cont1=np.nanmean(abs(fracs[0:ns]))
    avsky_line1=np.nanmean(abs(line_fracs[0:ns]))
    
    if (verbose):
        print('for median residual and flux in each sky fibre:')
        print('median absolute continuum residuals:',medsky_cont1,' (fractional median flux)')
        print('median absolute line residuals:',medsky_line1,' (fractional median flux)')
        print('for mean residual and flux in each sky fibre:')
        print('mean absolute continuum residuals:',avsky_cont1,' (fractional mean flux)')
        print('mean absolute line residuals:',avsky_line1,' (fractional mean flux)')

    if (nlam > 1):
        for j in range(nlam):
            medsky_cont[j]=np.nanmedian(abs(fraccontlam[0:ns,j]))
            sky_10pc_cont[j]=np.nanpercentile(abs(fraccontlam[0:ns,j]),10.0)
            sky_90pc_cont[j]=np.nanpercentile(abs(fraccontlam[0:ns,j]),90.0)
    
            if (verbose):
                print('10%, median, 90% absolute continuum residuals:',sky_10pc_cont[j],medsky_cont[j],sky_90pc_cont[j],lamcent[0,j],lam1[0,j],lam2[0,j])

        # Calculate the median continuum residual once it has been binned by wavelength as
        # well as fibre:
        print('Median fractional continuum residual in wavelength bins:',np.nanmedian(np.abs(fraccontlam[0:ns,0:nlam])))
                
    if (plot):
        py.figure(8)
        py.plot(lamcent[0,:],medsky_cont,'-',color='k')
        py.plot(lamcent[0,:],sky_90pc_cont,':',color='k')
        py.plot(lamcent[0,:],sky_10pc_cont,':',color='k')
        py.xlabel('Wavelength (\AA)')
        py.ylabel('fractional sky residual')
        py.xlim(xmin=3700.0,xmax=5700.0)
        py.ylim(ymin=0.0,ymax=0.1)


                
    if (plot):
        py.figure(6)
        lab = infile+' line residual'
        py.plot(fibs[0:ns],line_fracs[0:ns],'-',label=lab,color='b')
        sidx = np.where((types == 'S'))
        lab = infile+' line residual (sky fibs)'
        py.plot(fibs[sidx],line_fracs[sidx],'.',color='m',markersize=12,label=lab)
            #else:
                #py.plot(fibs[i],line_fracs[i],'x',color='b')
                
        
        lab = infile+' cont residual'
        #        py.plot(fibs[0:ns],fracs[0:ns],'-',color='k',label=lab)
        py.plot(fibs[0:ns],fracs[0:ns],'-',label=lab,color='r')
        lab = infile+' cont residual (sky fibs)'
        py.plot(fibs[sidx],fracs[sidx],'.',color='g',markersize=12,label=lab)
            #else:
                #py.plot(fibs[i],fracs[i],'x',color='r')

                
        py.axhline(0.0,color='k',linestyle='--')
        py.xlabel('Fibre')
        py.ylabel('fractional sky residual')
        py.title('Fractional sky residuals')
        plot_bundlelims()
        if (do_legend):
            py.legend(prop={'size':10})

        # plot the residual vs. flux in adjacent fibre:
        py.figure(9)
        py.plot(adjfrac[0:ns],fracs[0:ns],'o')
        py.xlabel('(adjacent fibre flux)/(sky fibre flux)')
        py.ylabel('Fractional sky residual')

        if (dothput):
        # plot the residual vs. flux in adjacent fibre:
            py.figure(13)
            py.plot(thput[0:ns],fracs[0:ns],'o')
            py.plot(thput_med[0:ns],fracs[0:ns],'o')
            py.xlabel('throughput')
            py.ylabel('Fractional sky residual')
            
    if (plot):

        py.figure(7)            
        for j in range(nlam):
            lab1 = " $\lambda$=%6.1f" % lamcent[0,j]

            lab = infile+lab1
        #        py.plot(fibs[0:ns],fracs[0:ns],'-',color='k',label=lab)
            py.plot(fibs[0:ns],fraccontlam[0:ns,j],'-',label=lab)
            #for i in range(ys):
                #if (types[i] == 'S'):
                #    py.plot(fibs[i],fraccontlam[i,j],'x',color='g')
                #else:
                #    py.plot(fibs[i],fraccontlam[i,j],'.',color='r')

        py.axhline(0.0,color='k',linestyle='--')
        py.xlabel('Fibre')
        py.ylabel('fractional sky residual')
        py.title('Fractional sky residuals')
        plot_bundlelims()

        if (do_legend):
            py.legend(prop={'size':10})

        # plot flux of residual (not fractions):
        py.figure(12)            
        for j in range(nlam):
            lab1 = " $\lambda$=%6.1f" % lamcent[0,j]

            lab = infile+lab1
        #        py.plot(fibs[0:ns],fracs[0:ns],'-',color='k',label=lab)
            py.plot(fibs[0:ns],fluxcontlam[0:ns,j],'-',label=lab)
            #for i in range(ys):
                #if (types[i] == 'S'):
                #    py.plot(fibs[i],fraccontlam[i,j],'x',color='g')
                #else:
                #    py.plot(fibs[i],fraccontlam[i,j],'.',color='r')

        py.axhline(0.0,color='k',linestyle='--')
        py.xlabel('Fibre')
        py.ylabel('Flux sky residual (counts)')
        py.title('Flux sky residuals')
        plot_bundlelims()

        if (do_legend):
            py.legend(prop={'size':10})

        # do the same again, but smooth vector in fibre direction to get
        # a less noise estimate:
        py.figure(11)            
        for j in range(nlam):
            lab1 = " $\lambda$=%6.1f" % lamcent[0,j]

            lab = infile+lab1
        #        py.plot(fibs[0:ns],fracs[0:ns],'-',color='k',label=lab)

            smcont = median_smooth(fraccontlam[0:ns,j],21)
            py.plot(fibs[0:ns],smcont,'-',label=lab)
            #for i in range(ys):
                #if (types[i] == 'S'):
                #    py.plot(fibs[i],fraccontlam[i,j],'x',color='g')
                #else:
                #    py.plot(fibs[i],fraccontlam[i,j],'.',color='r')

        py.axhline(0.0,color='k',linestyle='--')
        py.xlabel('Fibre')
        py.ylabel('fractional sky residual')
        py.title('Smoothed Fractional sky residuals')
        plot_bundlelims()

        if (do_legend):
            py.legend(prop={'size':10})

        # do the same again, but smooth vector in fibre direction to get
        # a less noise estimate:
        py.figure(14)            
        for j in range(nlam):
            lab1 = " $\lambda$=%6.1f" % lamcent[0,j]

            lab = infile+lab1
        #        py.plot(fibs[0:ns],fracs[0:ns],'-',color='k',label=lab)

            smcont = median_smooth(fluxcontlam[0:ns,j],21)
            py.plot(fibs[0:ns],smcont,'-',label=lab)
            #for i in range(ys):
                #if (types[i] == 'S'):
                #    py.plot(fibs[i],fraccontlam[i,j],'x',color='g')
                #else:
                #    py.plot(fibs[i],fraccontlam[i,j],'.',color='r')

        py.axhline(0.0,color='k',linestyle='--')
        py.xlabel('Fibre')
        py.ylabel('Flux sky residual (counts)')
        py.title('Smoothed flux sky residuals')
        plot_bundlelims()

        if (do_legend):
            py.legend(prop={'size':10})

            
    if (returnall):
        return fibs[0:ns],fracs[0:ns],line_fracs[0:ns],fraccontlam[0:ns,0:nlam],lamcent[0,:],fitsig,fitcent,adjfrac[0:ns]
    else:    
        return medsky_cont1,medsky_line1
