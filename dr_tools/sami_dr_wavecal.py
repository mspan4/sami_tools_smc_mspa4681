"""This package contains a number of routines to test SAMI and other data reduced 
using the 2dFdr pipeline.  These were written with SAMI data specifically in mind, 
but in most cases can be used for data from HERMES, 2dF/AAOmega, SPIRAL/AAOmega etc.
This includes some functions taken from other SAMI packages (e.g. sami.utils.other.find_fibre_table
so that there are no major dependenceies other than standard python packages.
"""

import matplotlib
import pylab as py
import numpy as np
import pandas as pd
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
import datetime
#import spectres

import scipy.stats as stats
import scipy.interpolate as interp
import scipy.signal as signal
from scipy.interpolate import griddata
from scipy.signal import find_peaks
from scipy import special
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
#from matplotlib._png import read_png
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter, percentile_filter
from matplotlib.lines import Line2D

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, AnchoredOffsetbox

import astropy.convolution as convolution
from astropy.convolution.kernels import CustomKernel

from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

# local things from this package:
from .sami_2dfdr_read import find_fibre_table
from sami_tools_smc.dr_tools.hector_data_checks import get_lam_axis
#import .sami_stats_utils as sami_stats_utils
#import .sami_2dfdr_general as sami_2dfdr_general

##############################################################################
# read an AAOmega frame and calculate lam cent from grating angle etc.
#
def calc_lamcent(infile,dang=0.0):

    # assume 15 micron pixels:
    pix = 0.015
    
    # open and get header:
    hdulist = pf.open(infile)
    hdr = hdulist[0].header

    # get relevant keywords:
    spectid = hdr['SPECTID']
    gratlpmm = hdr['GRATLPMM']
    gratangl = hdr['GRATANGL']+dang/2
    camangl = hdr['CAMANGL']+dang

    print('Grating angle (deg): ',gratangl)
    print('Camera angle (deg): ',camangl)
    
    # get focal length of camera:
    if (spectid == 'RD'):
        flcam = 245.0
    elif (spectid == 'BL'):
        flcam = 245.4

    # get angular pixel size:
    apix = pix/flcam
        
    # calc angles:
    alpha = gratangl * np.pi/180.0 
    beta = (camangl-gratangl) * np.pi/180.0
    print('alpha, beta (radians) :',alpha, beta)
    
    # calc cen wavelength (assumes order = 1)
    cenwave = 1.0e7 * (np.sin(alpha) + np.sin(beta))/gratlpmm
    disp = 1.0e7 * np.cos(beta) * apix / gratlpmm
    print('Central wavelength (angstroms): ',cenwave)
    print('Dispersion (angstroms): ',disp)

    
    

##############################################################################
# find peaks in an arc frame.  Assumes an extracted, but not calibrated
# spectrum input (i.e. ex.fits file)
#
def arc_peakfind(infile,reffib=400,prom=30.0,shift=-10.0):

    # read input file:
    # read in spectrum
    hdulist = pf.open(infile)

    # get image:
    im = hdulist[0].data
    (nfib,nx) = im.shape
    print("image1 x axis",nx)
    print("image1 y axis (nfib)",nfib)

    # get WAVELA arrray (starting model from 2dfdr).
    # note we convert to angstroms (from nm):
    wavela = hdulist['WAVELA'].data * 10.0

    # read arclist:
    wavelist,waveinten = np.loadtxt('/Users/scroom/code/2dfdr_src/2dfdr_hector7/2dfdr/drcontrol_ac/arc_files/helium+cuar+fear.arc',usecols=(0,1), unpack=True,comments=['*','#'])
    print(wavelist)
    print(waveinten)
    nwave = np.size(wavelist)
    
    # do peak finding on reference spectrum:
    refspec = im[reffib,:]
    reflam = wavela[reffib,:]
    peaks, peak_properties = find_peaks(refspec,height=0.0,prominence=prom)

    print(peaks)
    print(peak_properties)
    # plot examples
    fig1 = py.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(refspec,label='spec 1')
    ax1.plot(peaks,refspec[peaks],'x',color='r',label='peaks')
    ax1.legend(prop={'size':10})
    
    # now plot peaks on nominal wavelength scale (from WAVELA):
    fig2 = py.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(reflam+shift,refspec,label='spec 1')
    ax2.plot(reflam[peaks]+shift,refspec[peaks],'x',color='r',label='peaks')
    ax2.legend(prop={'size':10})

    for i in range(nwave):
        if ((wavelist[i] > reflam[0]) & (wavelist[i] < reflam[-1])):
            ax2.axvline(wavelist[i],linestyle=':',color='g')
            ymin,ymax = ax2.get_ylim()
            label = str(wavelist[i])
            ax2.text(wavelist[i],ymax-(ymax-ymin)*0.5,label,rotation=90.0)
    
 


    
    # get wavelength axis:
    #primary_header=hdulist['PRIMARY'].header
    #crval=primary_header['CRVAL1']
    #cdelt=primary_header['CDELT1']
    #crpix=primary_header['CRPIX1']
    #naxis=primary_header['NAXIS1']

    #x=np.arange(naxis)+1
    #L0=crval-crpix*cdelt #Lc-pix*dL
    #lam=L0+x*cdelt


    
##############################################################################
# do some simple tests of arc line matching using dynamic time warping
#
def dtw_arcline_match(infile):

    # read input file:
    # read in spectrum
    hdulist = pf.open(infile)

    # get image:
    im = hdulist[0].data
    (nfib,nx) = im.shape
    print("image1 x axis",nx)
    print("image1 y axis (nfib)",nfib)

    # get wavelength axis:
    primary_header=hdulist['PRIMARY'].header
    crval=primary_header['CRVAL1']
    cdelt=primary_header['CDELT1']
    crpix=primary_header['CRPIX1']
    naxis=primary_header['NAXIS1']

    x=np.arange(naxis)+1
    L0=crval-crpix*cdelt #Lc-pix*dL
    lam=L0+x*cdelt

    # get two spectra
    i1 = 400
    i2 = 10
    sigma_smooth=1.0
    # replace nan's with zero and do a small gaussian filter:
    spec1 = gaussian_filter(np.nan_to_num(im[i1,:]),sigma=sigma_smooth,mode='reflect')
    spec2 = gaussian_filter(np.nan_to_num(im[i2,:]),sigma=sigma_smooth,mode='reflect')
    # Only keep the peaks and remove background and set to zero:
    med1 = median_filter(spec1,size=51)
    med2 = median_filter(spec2,size=51)
    per1 = percentile_filter(spec1,percentile=90.0,size=51)
    per2 = percentile_filter(spec2,percentile=90.0,size=51)
    # replace points 
    spec1a = np.where((spec1<per1),med1,spec1)
    spec2a = np.where((spec2<per2),med2,spec2)
    
    

    # do a simple plot of the two spectra:
    fig1 = py.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(lam,spec1,label='spec 1')
    ax1.plot(lam,spec2,label='spec 2')
    ax1.legend(prop={'size':10})
    
    fig1 = py.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(lam,spec1,label='spec 1')
    ax1.plot(lam,spec2,label='spec 2')
    ax1.plot(lam,med1,label='med 1')
    ax1.plot(lam,med2,label='med 2')
    ax1.plot(lam,per1,label='per 1')
    ax1.plot(lam,per2,label='per 2')
    ax1.legend(prop={'size':10})

    fig4 = py.figure()
    ax4 = fig4.add_subplot(1,1,1)
    ax4.plot(lam,spec1a,label='spec 1')
    ax4.plot(lam,spec2a,label='spec 2')

    # run dtw:
    distance, paths = dtw.warping_paths(spec1a,spec2a,window=200,penalty=10)
    #print(distance)
    best_path = dtw.best_path(paths)
    dtwvis.plot_warpingpaths(spec1a, spec2a, paths, best_path)
    #print(type(best_path))
    best_path_arr = np.array(best_path)
    #print(best_path_arr)
    
    path1 = best_path_arr[:,0]
    path2 = best_path_arr[:,1]

    #get the mean difference in the paths:
    diff = path1-path2
    print('path difference near the start:')
    print(diff[0:100])
    print('path difference near the end:')
    print(diff[-100:])
    # find the median difference near the ends:
    meddiff1 = np.nanmedian(diff[0:200])
    meddiff2 = np.nanmedian(diff[-200:])
    print('median differences (near start and end):',meddiff1,meddiff2)
    npath = np.size(path1)
    print('size of path array:',npath)
    i1 = abs(int(meddiff1))
    i2 = npath-abs(int(meddiff2))
    #print(i1,i2)

    
    # fit a smooth low order poly to the bestpath:
    fig2 = py.figure()
    ax2 = fig2.add_subplot(2,1,1)
    #print(path1[0:100])
    #print(path2[0:100])
    #print(path1[-100:])
    #print(path2[-100:])
    ax2.plot(path1,path2)
    ax2.plot(path1[i1:i2],path2[i1:i2])
    z = np.polyfit(path1[i1:i2], path2[i1:i2], 6)
    p = np.poly1d(z)
    yfit = p(path1)
    ax2.plot(path1,yfit)
    ax2.plot([-1000,5000],[-1000,5000])
    xmin=np.nanmin(path1)-100
    xmax=np.nanmax(path1)+100
    ax2.set(xlim=[xmin,xmax],ylim=[xmin,xmax])
    print(path1[i1:i2], path2[i1:i2])
    
    ax22 = fig2.add_subplot(2,1,2)
    ax22.plot(path1,path2-yfit)
    ax22.axhline(0.0,linestyle='-',color='k')
    ax22.set(title='residual from fit')

    # now use the fit to map between the two arcs:
    lam2 = L0+p(x)*cdelt
    print(lam)
    print(lam2)
    
    fig3=py.figure()
    ax3 = fig3.add_subplot(1,1,1)
    ax3.plot(lam2,spec1,label='spec 1')
    ax3.plot(lam,spec2,label='spec 2')
    
    
    return
    
    


##############################################################################
# format arc list to make a list that will work with 2dfdr.  Take the output
# from check_arclist and do a few minor fixes.
#
def format_arclist(infile):

    # read file:
    df = pd.read_csv(infile,sep='\s+')

    # remove rows with flag > 1 (these are all too weak, or blends):
    df.drop(df[df.flag > 1].index, inplace=True)

    # reset indices after removing some data:
    df.reset_index(drop=True, inplace=True)

    # reorder cols:
    dfout = df[['lam','peak','name','flag']]
    
    nrow = len(dfout.index)
    print(dfout)

    # write a header:
    current_time = datetime.datetime.now()
    user = os.getlogin()
    print(current_time)
    print(user)

    header = '* arclist generated by '+user+' at '+str(current_time)+'\n* based on list checked by sami_dr_wavecal.check_arclist()\n* lam peak name flag\n'
    fp = open('arclist_out.dat', 'w')
    fp.write(header)
    fp.close()

    # now write the data, but use mode='a' to append:
    dfout.to_csv('arclist_out.dat', sep=' ',index=False,mode='a',header=False)  
    

##############################################################################
# check arc lines:
#
# standard (all CCD3) run:
# sami_tools_smc.dr_tools.sami_dr_wavecal.check_arclist('26sep30025red.fits','26sep30026red.fits','26sep30027red.fits','26sep30028red.fits')
#
# useage including line file (not default):
# sami_tools_smc.dr_tools.sami_dr_wavecal.check_arclist('26sep10025red.fits','26sep10026red.fits','26sep10027red.fits','../ccd_3/26sep30025red.fits',linefile='/Users/scroom/code/2dfdr_src/2dfdr_hector7/2dfdr/drcontrol_ac/arc_files/helium+cuar+fear_spector.arc')
#
# if autopdf is true, then generate a multi-page pdf of each line.
    
def check_arclist(infile1,infile2,infile3,infile4,fibre=200,lstart=3650.0,lend=5950.0,linefile=None,autopdf=False,scale4=False):
    
    # read in spectrum
    hdulist1 = pf.open(infile1)
    im1 = hdulist1[0].data
    flux1 = im1[fibre,:]
    lam1 = get_lam_axis(hdulist1)
    file1 = os.path.basename(infile1)
    hdr1 = hdulist1[0].header 
    lamp1 = hdr1['LAMPNAME']
    label1 = file1+' '+lamp1
    nlam = np.size(lam1)
        
    hdulist2 = pf.open(infile2)
    im2 = hdulist2[0].data
    flux2 = im2[fibre,:]
    lam2 = get_lam_axis(hdulist2)
    file2 = os.path.basename(infile2)
    hdr2 = hdulist2[0].header 
    lamp2 = hdr2['LAMPNAME']
    label2 = file2+' '+lamp2

    hdulist3 = pf.open(infile3)
    im3 = hdulist3[0].data
    flux3 = im3[fibre,:]
    lam3 = get_lam_axis(hdulist3)
    file3 = os.path.basename(infile3)
    hdr3 = hdulist3[0].header 
    lamp3 = hdr3['LAMPNAME']
    label3 = file3+' '+lamp3

    hdulist4 = pf.open(infile4)
    im4 = hdulist4[0].data
    flux4 = im4[fibre,:]
    lam4 = get_lam_axis(hdulist4)
    file4 = os.path.basename(infile4)
    hdr4 = hdulist4[0].header 
    lamp4 = hdr4['LAMPNAME']
    label4 = file4+' '+lamp4
   

    # now read reference files.  This are hard-coded as they point to different
    # specific sets of data:
    refpath = '/Users/scroom/data/hector/wavecal/noao_linelists/'
    ref1 = 'scidoc2214' # FeAr
    ref2 = 'scidoc2215' # HeNeAr
    ref3 = 'scidoc2216' # CuAr


    if (linefile):
        print ('reading lines from linefile...')
        refdf_t = pd.read_csv(linefile, sep="\s+", header=None, names=["lam", "name"],comment='*',usecols=[0,2])
    else:
        # read line lists:
        file1 = refpath+ref1+'.txt'
        print('Reading ',file1)
        ref1df = pd.read_csv(file1, sep="\s+", header=None, names=["lam", "name"],comment='#')
        file2 = refpath+ref2+'.txt'
        print('Reading ',file2)
        ref2df = pd.read_csv(file2, sep="\s+", header=None, names=["lam", "name"],comment='#')
        file3 = refpath+ref3+'.txt'
        print('Reading ',file1)
        ref3df = pd.read_csv(file3, sep="\s+", header=None, names=["lam", "name"],comment='#')
    
        refdf_t = pd.concat([ref1df,ref2df,ref3df], ignore_index=True)
    
        
    # drop nans:
    refdf_t.dropna(inplace=True)

    # drop duplicates:
    refdf_t.drop_duplicates(inplace=True)

    # drop outside of useful wavelength range:
    refdf = refdf_t[(refdf_t.lam > lstart) & (refdf_t.lam < lend)]

    # add useful columns:
    refdf['peak'] = 0.0    # approx peak value to be filled in later
    refdf['flag'] = ""     # flag to say whether this line should be used.
    refdf['colour'] = ""   # colour to plot line

    # set all flag values to -1:
    refdf.loc[:,'flag'] = -1

    # colour mapping:
    cols = {"ArI": "r", "ArII": "r", "FeI": "g", "HeI": "b", "CuI": "m", "CuII": "m", "NeI": "y", "NeII": "y", "KrI": 'y'}

    refdf['colour'] = refdf['name'].map(cols)

    # sort in wavelength:
    refdf.sort_values(by=['lam'],inplace=True)
    
    # reset indices after removing some data:
    refdf.reset_index(drop=True, inplace=True)

    nref = len(refdf.index)

    print(refdf)

    # now read in reference spectra:
    rfits1 = refpath+ref1+'.fits'
    rfits2 = refpath+ref2+'.fits'
    rfits3 = refpath+ref3+'.fits'
    hdulist1 = pf.open(rfits1)
    rflux1 = hdulist1[0].data
    rlam1 = get_lam_axis(hdulist1)
    hdulist2 = pf.open(rfits2)
    rflux2 = hdulist2[0].data
    rlam2 = get_lam_axis(hdulist2)
    hdulist3 = pf.open(rfits3)
    rflux3 = hdulist3[0].data
    rlam3 = get_lam_axis(hdulist3)

    # scale reference spectra to approx the same peaks.  Set this to below 5000A for now:
    # scale compared to first ref spec: 
    rmax1 = np.nanmax(rflux1[np.where(rlam1<5000)])
    rmax2 = np.nanmax(rflux2[np.where(rlam2<5000)])
    rmax3 = np.nanmax(rflux3[np.where(rlam3<5000)])
    rflux2 = rflux2*rmax1/rmax2
    rflux3 = rflux3*rmax1/rmax3
        
    # plot the full data in one, as a quick look:
    fig1=py.figure()
    ax1 = fig1.add_subplot(1,1,1)    
    ax1.plot(rlam1,rflux1,label='Reference spec 1: FeAr')
    ax1.plot(rlam2,rflux2,label='Reference spec 2: HeNeAr')
    ax1.plot(rlam3,rflux3,label='Reference spec 3: CuAr')

    ax1.set(xlim=[lstart,lend],ylim=[0.0,rmax1*1.5])

    # plot the lines:
    for i in range(nref):
        lam_p = refdf.at[i,'lam']
        col_p = refdf.at[i,'colour']

        
        if ((lam_p > lstart) & (lam_p < lend)):
            
            ax1.axvline(lam_p,linestyle='-',color=col_p)
            #ymin,ymax = ax1.get_ylim()
            #label = str(wave[i])+' '+labels[i]
            #ax1.text(wave[i],ymax-(ymax-ymin)*0.5,label,rotation=90.0)

    # plot the Hector arcs:
    ax1.plot(lam1,flux1,label='CCD1')
    ax1.plot(lam2,flux2,label='CCD3')

    ax1.legend()
        
        
    # set up plotting:
    if (autopdf):
        pdf = PdfPages('arclines_zoom_all.pdf')
    
    # start main loop that plots each line
    fig2 = py.figure()
    ax2 = fig2.add_subplot(2,1,1)
    ax3 = fig2.add_subplot(2,1,2)
    dlam = 20.0
    i = -1
    while (i<nref-1):
    #for i in range(nref):
        i = i + 1
        ax2.cla()
        ax3.cla()
        lam_p = refdf.at[i,'lam']
        col_p = refdf.at[i,'colour']
        name_p = refdf.at[i,'name']
# reset value of flag:
        refdf.at[i,'flag'] = -1
        
        title = str(lam_p)+' '+name_p
        print(refdf.loc[i,:])
        l1 = lam_p-dlam
        l2 = lam_p+dlam
        print('Lam range to plot:',l1,l2)
        rmax1 = np.nanmax(rflux1[np.where((rlam1>l1) & (rlam1<l2))])
        rmax2 = np.nanmax(rflux2[np.where((rlam2>l1) & (rlam2<l2))])
        rmax3 = np.nanmax(rflux3[np.where((rlam3>l1) & (rlam3<l2))])
        ax2.plot(rlam1,rflux1/rmax1,label='Reference spec 1: FeAr')
        ax2.plot(rlam2,rflux2/rmax2,label='Reference spec 2: HeNeAr')
        ax2.plot(rlam3,rflux3/rmax3,label='Reference spec 3: CuAr')
        ax2.set(xlim=[l1,l2],ylim=[-0.1,1.1])

        # get indices for plotting in the subwindow:
        idx1 = np.where((lam1>l1) & (lam1<l2))
        idx2 = np.where((lam2>l1) & (lam2<l2))
        idx3 = np.where((lam3>l1) & (lam3<l2))
        idx4 = np.where((lam4>l1) & (lam4<l2))
        nf1 = np.count_nonzero(~np.isnan(flux1[idx1]))
        nf2 = np.count_nonzero(~np.isnan(flux2[idx2]))
        nf3 = np.count_nonzero(~np.isnan(flux3[idx3]))
        nf4 = np.count_nonzero(~np.isnan(flux4[idx4]))
        print(nf1,nf2,nf3,nf4)
        if (nf1 > 1):
            fmax1 = np.nanmax(flux1[idx1])
            print('fmax1:',fmax1)
            ax3.plot(lam1[idx1],flux1[idx1],label=label1)
        if (nf2 > 1):
            fmax2 = np.nanmax(flux2[idx2])
            print('fmax2:',fmax2)
            ax3.plot(lam2[idx2],flux2[idx2],label=label2)
        if (nf3 > 1):
            fmax3 = np.nanmax(flux3[idx3])
            print('fmax3:',fmax3)
            ax3.plot(lam3[idx3],flux3[idx3],label=label3)
        if (nf4 > 1):
            fmax4 = np.nanmax(flux4[idx4])
            print('fmax4:',fmax4)
            # scale only last one to the first.  This is only
            # when comparing CCD1 and CCD3 (scale down CCD3):
            if ((nf1>1) & scale4) :
                ax3.plot(lam4[idx4],flux4[idx4]*fmax1/fmax4,label=label4)
            else:
                ax3.plot(lam4[idx4],flux4[idx4],label=label4)

        ax3.set(xlim=[l1,l2])

        # find the index of the pixel closest to the line:
        lidx = (np.abs(lam1 - lam_p)).argmin()
        lpeak = flux1[lidx]
        print(lpeak,lidx)
        if ((lidx > 1) & (lidx < nlam) & (lpeak > 0.0)):
            refdf.at[i,'peak'] = np.round(lpeak)
        else:
            refdf.at[i,'peak'] = 0.0
            
        print('peak flux:',refdf.at[i,'peak'])

        if (autopdf):
            ax2.legend(loc='upper left',fontsize=6)
        else:
            ax2.legend(loc='upper left')

        # Add the legend manually to the Axis, as we will ad another later and
        # it will be overwritten if we don't do this:
        if (autopdf):
            first_legend = ax3.legend(loc='upper right',fontsize=6)
        else:
            first_legend = ax3.legend(loc='upper right')
        ax3.add_artist(first_legend)
        
        # plot the lines:
        for j in range(nref):
            lam_pp = refdf.at[j,'lam']
            col_pp = refdf.at[j,'colour']
            if ((lam_pp > l1) & (lam_pp < l2)):
                ax2.axvline(lam_pp,linestyle='-',color=col_pp)
                ax3.axvline(lam_pp,linestyle='-',color=col_pp)

        # mark actual line being checked
        ax2.axvline(lam_p,linestyle=':',color='k')
        ax3.axvline(lam_p,linestyle=':',color='k')
        ax2.axvspan(lam_p-1,lam_p+1, alpha=0.2, color='k')
        ax3.axvspan(lam_p-1,lam_p+1, alpha=0.2, color='k')

        
        labels = cols.keys()
        colours = cols.values()
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colours]
        if (autopdf):
            ax3.legend(lines, labels,loc='upper left',fontsize=6)
        else:
            ax3.legend(lines, labels,loc='upper left')
        ax3.set(xlabel='Wavelength (Angstroms)')
        ax2.set(title=title)
        
        
        # make sure things are plotted
        py.draw()

        # if this is autopdf, don't do interacive bit, just send to pdf:
        if (autopdf):
            pdf.savefig()
            continue
        
        py.pause(0.0001)
        #fig2.show()
        #fig2.draw()
        # pause and wait for input:
        goodkeys = ['g','m','b','w','p','q']
        while True:
            print('input: g - good; m - marginal; b - blend (bad); w - weak (bad); p - previous; q - quit')   
            yn = input('Input: ')
            print(yn)
            if ((yn == 'g') | (yn == 'm') | (yn == 'b') | (yn == 'w') | (yn == 'p') | (yn == 'q')):
                print('test ',yn)
                print('Done, moving to next line')
                break
            else:
                print('Incorrect input, try again...')
                continue
                
        # process feedback
        print('Processing input...')
        if (yn == 'g'):
            refdf.at[i,'flag'] = 0
            print('Line flagged as good')
        elif (yn == 'm'):
            refdf.at[i,'flag'] = 1
            print('Line flagged as marginal')
        elif (yn == 'b'):
            refdf.at[i,'flag'] = 2
            print('Line flagged as blend')
        elif (yn == 'w'):
            refdf.at[i,'flag'] = 3
            print('Line flagged as weak')
        elif (yn == 'p'):
            i = i - 2
            print('jump back to previous line...')
        elif (yn == 'q'):
            print('save and quit...')
            break

    # finished, so save the data:
    print(refdf)
    # don't keep the colour column:
    refdf.drop(columns=['colour'],inplace=True)

    # save the file (if no autopdf):
    if (autopdf):
        pdf.close()
    else:
        refdf.to_csv('arclines_new.dat', sep=' ',index=False)  
    



##############################################################################
# plot lines from a linelist on a reduced spectrum
#
def plot_linelist(infile,listfile,fibno=200,infile2=None,refspec=None):

    # read in spectrum
    hdulist = pf.open(infile)

    # get image:
    im = hdulist[0].data
    (ys,xs) = im.shape
    print("image1 x axis",xs)
    print("image1 y axis",ys)

    # get wavelength axis:
    primary_header=hdulist['PRIMARY'].header
    crval=primary_header['CRVAL1']
    cdelt=primary_header['CDELT1']
    crpix=primary_header['CRPIX1']
    naxis=primary_header['NAXIS1']

    x=np.arange(naxis)+1
    L0=crval-crpix*cdelt #Lc-pix*dL
    lam=L0+x*cdelt

    # if needed get the second infile:
    if (infile2):
        hdulist2 = pf.open(infile2)

        # get image:
        im2 = hdulist2[0].data
        (ys,xs) = im2.shape
        print("image2 x axis",xs)
        print("image2 y axis",ys)

        # get wavelength axis:
        primary_header2=hdulist2['PRIMARY'].header
        crval=primary_header2['CRVAL1']
        cdelt=primary_header2['CDELT1']
        crpix=primary_header2['CRPIX1']
        naxis=primary_header2['NAXIS1']

        print(crval,cdelt,crpix,naxis)
        x2=np.arange(naxis)+1
        L0=crval-crpix*cdelt #Lc-pix*dL
        lam2=L0+x2*cdelt
        
        print(lam2)

    # read in a reference spectrum:
    if (refspec):
        hdulist3 = pf.open(refspec)

        # get image:
        im3 = hdulist3[0].data
        print('Refspec shape:',im3.shape)

        # get wavelength axis:
        primary_header3=hdulist3['PRIMARY'].header
        crval=primary_header3['CRVAL1']
        cdelt=primary_header3['CDELT1']
        crpix=primary_header3['CRPIX1']
        naxis=primary_header3['NAXIS1']

        print(crval,cdelt,crpix,naxis)
        x3=np.arange(naxis)+1
        L0=crval-crpix*cdelt #Lc-pix*dL
        lam3=L0+x3*cdelt
        
        print(lam3)

        
    # get data from linelist:
    wave = np.loadtxt(listfile,usecols=(0), unpack=True,comments=['*','#'])
    print(wave)
    labels = np.loadtxt(listfile,usecols=(1), unpack=True,comments=['*','#'],dtype='U4')
    print(labels)

    nwave=len(wave)
    
    flux = im[fibno,:]

    fig1=py.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(lam,flux)
    if (infile2):
        flux2 = im2[fibno,:]
        ax1.plot(lam2,flux2)
        
    if (refspec):
        flux3 = im3
        ax1.plot(lam3,flux3)

    for i in range(nwave):
        if ((wave[i] > lam[0]) & (wave[i] < lam[-1])):
            ax1.axvline(wave[i],linestyle='-',color='g')
            ymin,ymax = ax1.get_ylim()
            label = str(wave[i])+' '+labels[i]
            ax1.text(wave[i],ymax-(ymax-ymin)*0.5,label,rotation=90.0)
    
    

##############################################################################
# estimate centroid S/N for a Gaussian fit.
# based on https://physics.stackexchange.com/questions/329765/error-of-centroid-position-with-noise
# and https://opg.optica.org/ao/abstract.cfm?uri=ao-48-36-6913
#
def plot_centroid_err(readnoise=3.61,gain=1.88,sigma=1.0):

    # set range of flux values.  This is the total counts in the line
    # in photons (i.e. obeys Poisson errors)
    logflux = np.linspace(0.0,5.0,101)
    flux = 10**logflux
    fluxadu = flux/gain
    logfluxadu = np.log10(fluxadu)

    # set background to readnoise squared - this essentially suggests that the background
    # is just bias variations, not actual counts, so an underestimate.
    B = readnoise**2

    cent_err = sigma*np.sqrt(1+4*np.sqrt(np.pi)*sigma*B/flux)/np.sqrt(flux)

    print(flux,cent_err)
    
    fig1 = py.figure()
    ax1 = fig1.add_subplot(1,2,1)
    ax1.plot(fluxadu,cent_err)
    ax1.set(xlim=[0,5000],ylim=[0,0.1],xlabel='line flux (ADU)',ylabel='centroid error (pix)')
    
    ax2 = fig1.add_subplot(1,2,2)
    ax2.plot(logfluxadu,np.log10(cent_err))
    ax2.set(xlim=[0,5],ylim=[-2.5,1.0],xlabel='log(line flux) (ADU)',ylabel='log(centroid error) (pix)')

    label = 'gain = {0:4.2f}, RN = {1:4.2f}, sigma={2:5.2f}'.format(gain,readnoise,sigma)
    fig1.suptitle(label)
    

##############################################################################
# simple plotting and comparison of measured shifts read from text files
#
def plot_shifts(inlist):

    # glob files:
    files = glob.glob(inlist)

    # find the number of files to use to define array size:
    maxfiles = np.size(files)
    print('number of files in list:',maxfiles)
    
    # loop over each 
    nfiles = 0
    py.figure(1)
    for filename in files:

        print(filename)
        fibnum, shift = np.loadtxt(filename,unpack=True)

        py.plot(fibnum,shift,label=filename)

        
        

        

##############################################################################
# script to compare 5577 locations of two different frames:
#
def comp_5577(infile1,infile2,doplot=True):

    line_cent1,type1 = centroid_5577(infile1,doplot=False,return_results=True)
    line_cent2,type2 = centroid_5577(infile2,doplot=False,return_results=True)

    delta_cent = line_cent1-line_cent2
    lam5577=5577.346680

    nfib = np.size(line_cent1)
    
    fibnum=np.linspace(1,nfib,nfib)

    diff = line_cent1-line_cent2
    
    if (doplot): 
        py.figure(1)
        py.plot(fibnum,line_cent1-lam5577,label=infile1)
        py.plot(fibnum,line_cent2-lam5577,label=infile2)
        for i in range(nfib):
            if ((type1[i] == 'S')):
                py.plot(fibnum[i],line_cent1[i]-lam5577,'o',color='r',label='sky fibres')
                py.plot(fibnum[i],line_cent2[i]-lam5577,'o',color='r',label='sky fibres')
        py.xlabel('fibre number')
        py.ylabel('Wavelength offset (Angstroms)')

        py.figure(2)
        py.plot(fibnum,diff,label='difference')
        
   
##############################################################################
# script to compare many 5577 line centroids
#
def comp_5577_many(inlist,min_exp=900.0,doplot=True,shift=True,dopdf=True):

    lam5577=5577.346680
    
    # glob files:
    files = glob.glob(inlist)

    # find the number of files to use to define array size:
    maxfiles = np.size(files)
    print('number of files in list:',maxfiles)

    # arrays for storage:
    line_cent_all = np.zeros((maxfiles,819))
    line_cent_all_medsub = np.zeros((maxfiles,819))
    line_cent_all_medsub_medfilt = np.zeros((maxfiles,819))
    line_cent_all_medsub_medfilt_fit = np.zeros((maxfiles,819))
    line_cent_all_fullcorr = np.zeros((maxfiles,819))
    line_cent_all_ls_corr = np.zeros((maxfiles,819))
    
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

        hdulist.close()

        # only select objects frames with at least min-exp exposure time:
        if (obstype == 'OBJECT' and exposure > min_exp):
            line_cent1,type1 = centroid_5577(filename,doplot=False,return_results=True)

            # is shift is true then offset by the mediam to just look at relative differences:
            if (shift):
                med = np.nanmedian(line_cent1)
            else:
                med = lam5577
                
            line_cent_all[nfiles,:] = line_cent1-med
            
            nfiles=nfiles+1


    #
    print('number of files used:', nfiles)
    # calculate mean, median, stddev etc:
    mean_cent = np.nanmean(line_cent_all[0:nfiles,:],axis=0)
    median_cent = np.nanmedian(line_cent_all[0:nfiles,:],axis=0)
    stdev_cent = np.nanstd(line_cent_all[0:nfiles,:],axis=0)

    nfib = np.size(mean_cent)
    fibnum=np.linspace(1,nfib,nfib)

    # define set of data with the median subtracted:
    for i in range(nfiles):
        line_cent_all_medsub[i,:] = line_cent_all[i,:] - median_cent
        line_cent_all_medsub_medfilt[i,:] = sami_stats_utils.median_smooth(line_cent_all_medsub[i,:],11)
        # fit a low order polynomial to large-scale shift:
        fitpar = np.polyfit(fibnum,line_cent_all_medsub_medfilt[i,:],2)
        pfit = np.poly1d(fitpar)
        line_cent_all_medsub_medfilt_fit[i,:] = pfit(fibnum)

        # fineal corrected version with small scale median subtracted as well
        # as large scale shift:
        line_cent_all_fullcorr[i,:] = line_cent_all[i,:] - median_cent - line_cent_all_medsub_medfilt_fit[i,:]
        # now a vesion that only applies the large-scale shift:
        line_cent_all_ls_corr[i,:] = line_cent_all[i,:] - line_cent_all_medsub_medfilt_fit[i,:]

    # calculate std dev for fully corrected data:
    stdev_cent_fullcorr = np.nanstd(line_cent_all_fullcorr[0:nfiles,:],axis=0)

    # calculate std dev for the version just corrected for large-scale trends:
    stdev_cent_ls_corr = np.nanstd(line_cent_all_ls_corr[0:nfiles,:],axis=0)
    median_cent_ls_corr = np.nanmedian(line_cent_all_ls_corr[0:nfiles,:],axis=0)
    
    mad_cent_ls_corr = sami_stats_utils.med_abs_dev(line_cent_all_ls_corr[0:nfiles,:])
    
    # calc distribution of std dev after correction:
    print('5th, 10th, 50th, 90th 95th percentile of fully corrected std dev:',np.nanpercentile(stdev_cent_fullcorr,5.0),np.nanpercentile(stdev_cent_fullcorr,10.0),np.nanpercentile(stdev_cent_fullcorr,50.0),np.nanpercentile(stdev_cent_fullcorr,90.0),np.nanpercentile(stdev_cent_fullcorr,95.0))
    
    if (doplot):

        if (dopdf):
            pdf = PdfPages('comp5577.pdf')

        py.figure(1)
        py.plot(fibnum,mean_cent,label='mean',color='r')
        py.plot(fibnum,median_cent,label='median',color='b')
        py.plot(fibnum,mean_cent+stdev_cent,':',color='r')
        py.plot(fibnum,mean_cent-stdev_cent,':',color='r')
        for i in range(nfib):
            if ((type1[i] == 'S')):
                py.plot(fibnum[i],mean_cent[i],'o',color='r')
                py.plot(fibnum[i],median_cent[i],'o',color='b')
        py.xlabel('fibre number')
        py.ylabel('Wavelength offset (Angstroms)')
        py.legend(prop={'size':10})
        
        # plot bundle boundaries:
        sami_2dfdr_general.plot_bundlelims()

        if (dopdf):
            py.savefig(pdf, format='pdf')        
            pdf.close()


        
        py.figure(2)
        py.plot(fibnum,stdev_cent,label='std dev',color='r')
        py.plot(fibnum,stdev_cent/np.sqrt(nfiles),label='std err',color='b')
        py.legend(prop={'size':10})
        py.xlabel('fibre number')
        py.ylabel('std dev / std err (Angstroms)')
        # plot bundle boundaries:
        sami_2dfdr_general.plot_bundlelims()


        py.figure(3)
        py.title('All fibre shifts, uncorrected')
        for i in range(nfiles):
            py.plot(fibnum,line_cent_all[i,:],label=filename)

        # plot bundle boundaries:
        sami_2dfdr_general.plot_bundlelims()
            
        py.figure(4)
        py.title('All fibre shifts - local fibre median')
        for i in range(nfiles):
            py.plot(fibnum,line_cent_all[i,:]-median_cent,label=filename)

        # plot bundle boundaries:
        sami_2dfdr_general.plot_bundlelims()
            
        py.figure(5)
        py.title('All fibre shifts - local fibre median and median filtered')
        for i in range(nfiles):
            py.plot(fibnum,line_cent_all_medsub_medfilt[i,:],label=filename)
            py.plot(fibnum,line_cent_all_medsub_medfilt_fit[i,:],label=filename)

        # plot bundle boundaries:
        sami_2dfdr_general.plot_bundlelims()
    
        py.figure(6)
        for i in range(nfiles):
            py.plot(fibnum,line_cent_all_fullcorr[i,:],label=filename)

        # plot bundle boundaries:
        sami_2dfdr_general.plot_bundlelims()
        
        py.figure(7)
        py.plot(fibnum,stdev_cent_fullcorr,label=filename)

        # plot bundle boundaries:
        sami_2dfdr_general.plot_bundlelims()


        py.figure(8)
        py.plot(fibnum,median_cent_ls_corr,label='median',color='b')
        py.plot(fibnum,median_cent_ls_corr+stdev_cent_ls_corr,':',color='r')
        py.plot(fibnum,median_cent_ls_corr-stdev_cent_ls_corr,':',color='r')
        # plot bundle boundaries:
        sami_2dfdr_general.plot_bundlelims()


        py.figure(9)
        py.plot(fibnum,stdev_cent_ls_corr,color='r')
        py.plot(fibnum,mad_cent_ls_corr,color='b')
        # plot bundle boundaries:
        sami_2dfdr_general.plot_bundlelims()

        # write an output file of the median local shifts:
        np.savetxt('shift5577_median.txt',np.column_stack([fibnum,median_cent]))
        
    
##############################################################################


    

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

    # get median line centre:
    med_line_cent = np.nanmedian(line_cent[0:nfib])-lam5577

    # get mean and rms:
    (mean,rms,nc) = sami_stats_utils.clipped_mean_rms(line_cent[0:nfib],5.0,verbose=False)
    mean = mean - lam5577
    print('median measured line cent:',med_line_cent)
    print('mean and rms line cent:',mean,rms)
    print('number of fibres used for rms:',nc)

    # calculate what the velocity scatter is in km/s:
    v_mean = (mean/lam5577) * 2.997e5
    v_rms = (rms/lam5577) * 2.997e5

    print('mean and rms velocity offset (km/s):',v_mean,v_rms) 
        
        
    if (doplot): 
        fig1 = py.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.plot(fibnum,line_cent-lam5577,label='all fibres')
        for i in range(nfib):
            if ((types[i] == 'S')):
                ax1.plot(fibnum[i],line_cent[i]-lam5577,'o',color='r',label='sky fibres')
        ax1.set(xlabel='fibre number',ylabel='Wavelength offset (Angstroms)')
        ax1.axhline(mean,linestyle='-',color='g')
        ax1.axhline(mean+rms,linestyle=':',color='g')
        ax1.axhline(mean-rms,linestyle=':',color='g')
        ax1.text(0.05, 0.92,infile,
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes)
        label1 = 'mean={0:7.4f},  rms = {1:7.4f} Ang.'.format(mean,rms)
        label2 = 'mean={0:7.4f},  rms = {1:7.4f} km/s.'.format(v_mean,v_rms)
        ax1.text(0.05, 0.12,label1,
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes)
        ax1.text(0.05, 0.05,label2,
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes)

    if (return_results):
        return line_cent,types
    else:
        return
            
    
