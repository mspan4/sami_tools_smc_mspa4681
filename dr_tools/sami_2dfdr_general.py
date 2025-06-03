
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
import scipy.fftpack as fftpack
from scipy.interpolate import griddata
from scipy import special
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
#from matplotlib._png import read_png


from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, AnchoredOffsetbox

import astropy.convolution as convolution
from astropy.convolution.kernels import CustomKernel

from .sami_2dfdr_read import find_fibre_table

################################################################################

def update_badpix(infile):
    """Function to update bad pixel mask image for AAOega CCDs that go into 2dfdr"""

    # first copy the file:
    outfile = infile.replace('.fits','_new.fits')
    shutil.copyfile(infile,outfile)

    # now open the new file for editing:
    hdulist = pf.open(outfile,mode='update')
    im = hdulist[0].data
    
    # Plot the image:
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(1,2,1)
    cax1 = ax1.imshow(im,origin='lower',interpolation='nearest')

    x1 = []
    x2 = []
    y1 = []
    y2 = []
    # define coords to replace.  At the moment hard coded for fixes
    # for a particular file.  Note that these are inclusive pixel ranges to flag as
    # bad, wher the first pixel is at (1,1), NOT (0,0):
    if (infile == 'E2V3.fits'):
        x1 = [23,76,143,271]
        x2 = [75,142,270,286]
        y1 = [2973,2975,2976,2978]
        y2 = [2976,2976,2978,2981]


    nfix = len(x1)
    print('Number of regions to fix:',nfix)

    for i in range(nfix):
        print('Fixing region ',i)

        for ix in range(x1[i],x2[i]+1):
            for iy in range(y1[i],y2[i]+1):
                print(ix,iy)
                # note that we need to subtract 1 for the zero referencesing of python arrays:
                im[iy-1,ix-1] = 1


    ax2 = fig1.add_subplot(1,2,2)
    cax2 = ax2.imshow(im,origin='lower',interpolation='nearest')

                
    # update and close:
    hdulist.flush()
    hdulist.close()
    


################################################################################

def check_frames_sci(inlist,append=True):
    """Function to plot (on a nice scale) a reduced sami frame (could be red or
    sci.fits files) and allow user to check it.  Write a file at the end of 
    all the checked flags."""

    # glob the list:
    files = glob.glob(inlist)
    maxfiles = np.size(files)

    #
    dpath=5
    
    # get the run - assume there is just one run being looked at:
    path_list = files[0].split(os.sep)
    run = path_list[0+dpath]
    date = path_list[2+dpath]
    field = path_list[4+dpath]
    outfile = 'check_frames_'+run+'.txt'


    print('number of files to check: ',maxfiles)
    print('will write results to '+outfile)

    # read the outfile and get the frames already done:
    try:
        f=open(outfile,"r")
        lines=f.readlines()
        prevfile=[]
        prevgood=[]
        for x in lines:
            cols = x.split(' ')
            prevfile.append(cols[0])
            prevgood.append(int(cols[1].rstrip()))
            f.close()
        print(prevfile)
        print(prevgood)
        nprev = np.size(prevgood)
        print(nprev)
    except FileNotFoundError:
        print('output file not found...')
        nprev = 0
        
        
    #
    good = np.ones(maxfiles,dtype=bool)

    
    fig1 = py.figure(1)

    filename = np.empty(maxfiles,dtype='U256')
    
    nfiles = 0
    i = 0
    while i < maxfiles:

        if (i<0):
            print('skipped back to the start')
            i = 0

        print(i,nfiles,files[i])
        filename[nfiles] = files[i]

        
        # check if the file is ready done:
        skip = False
        for j in range(nprev):
            if (filename[nfiles] == prevfile[j]):
                print('already done... skipping ',filename[nfiles])
                skip = True

        if (skip == True):
            i = i +1
            continue
                
        path_list = filename[nfiles].split(os.sep)
        run = path_list[0]
        date = path_list[2]
        field = path_list[4]
        
        print(i,filename[nfiles])
        fname = os.path.basename(filename[nfiles])

        # get image:
        hdulist = pf.open(filename[nfiles])
        im = hdulist[0].data

        # get wavelength axis (assume the same for both files):
        primary_header=hdulist['PRIMARY'].header
        crval1=primary_header['CRVAL1']
        cdelt1=primary_header['CDELT1']
        crpix1=primary_header['CRPIX1']
        naxis1=primary_header['NAXIS1']
        x=np.arange(naxis1)+1
        L0=crval1-crpix1*cdelt1 #Lc-pix*dL

        texp = primary_header['EXPOSED']
        
        # get scalings for image:
        med = np.nanmedian(im)
        p5 = np.nanpercentile(im,5.0)
        p95 = np.nanpercentile(im,95.0)

        vmin=p5
        vmax=p95
        
        
        iter = True
        qq = False
        while (iter):
            # make the plot:
            fig1.clf()
            ax1 = fig1.add_subplot(111)
            #ax2 = ax1.twiny()
            

            cmap = py.cm.gray
            cmap.set_bad('red',1.)
            cax1=ax1.imshow(im,origin='lower',interpolation='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
            cbar1=fig1.colorbar(cax1,fraction=0.046, pad=0.04)
            ax1.set(title=str(i)+': '+fname+'    Texp = '+str(texp))

            py.draw()
            # do the interactive bit:
            user = input("Frame {0:4d}: Y - data okay; N - data not okay; P - increase scale; M - decrease scale; Q - quit?".format(i))
            print(user)
            if (user.upper() == 'Y'):
                good[nfiles] = True
                iter = False
            elif (user.upper() == 'B'):
                i = i - 2
                nfiles = nfiles - 2
                iter = False
            elif (user.upper() == 'N'):
                good[nfiles] = False
                iter = False
            elif (user.upper() == 'P'):
                # increase range of scales:
                vmin = med - 2.0*(abs(med-vmin))
                vmax = med + 2.0*(abs(med-vmax))
            elif (user.upper() == 'M'):
                # increase range of scales:
                vmin = med - 0.5*(abs(med-vmin))
                vmax = med + 0.5*(abs(med-vmax))
            elif (user.upper() == 'Q'):
                qq = True
                iter = False
            else:
                good[nfiles] = False

        if (qq == True):
            print('Quitting...')
            break
        
        i = i +1
        nfiles = nfiles +1
        hdulist.close()

    #output results:
    print ('writing output to '+outfile)
    
    if (append):
        ftab = open(outfile,mode='a+')
    else:
        ftab = open(outfile,mode='w+')


    for i in range(nfiles):
        ftab.write('{0:s} {1:1d}\n'.format(filename[i],int(good[i])))

    ftab.close()
        

################################################################################
# script to plot course cuts for tlm creep
#
def plot_tlmcreep_course(creepfile,vmin=-0.05,vmax=0.05):

    iy, im, tlm = np.loadtxt(creepfile, usecols=(0, 1,2), unpack=True)

    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(111)
    medim = np.nanmedian(im)
    medtlm = np.nanmedian(tlm)
    ax1.plot(iy,im)
    ax1.plot(iy,tlm*medim/medtlm)
    
    nim = np.size(im)
    print('image size:',nim)
    # robustly identify the first peak:
    per05=np.nanpercentile(im,5.0)
    per95=np.nanpercentile(im,95.0)
    
    ax1.axhline(per05)
    ax1.axhline(per95)

    fig3 = py.figure(3)
    ax3 = fig3.add_subplot(111)
    gradim = np.gradient(im)
    gradtlm = np.gradient(tlm)
    ax3.plot(iy,gradim)
    ax3.plot(iy,gradtlm*medim/medtlm)
    
    
    
    xcorr = signal.correlate(im,tlm)
    fig2 = py.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.plot(xcorr)
    xcorr_peak =np.max(xcorr)
    index_peak = np.argmax(xcorr)
    print('xcorr peak at:',index_peak)
    

################################################################################
# script to read and plot TLM creep shifts
#
def plot_tlmcreep(creepfile,vmin=-0.05,vmax=0.05):

    ix, iy, x1, x2, y1 , y2, shift = np.loadtxt(creepfile, usecols=(0, 1,2, 3, 4, 5, 6), unpack=True)

    # calc centres of each bin:
    xc = (x1+x2)/2
    yc = (y1+y2)/2

    # plot the distribution of shifts across the field:
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(111)
    cax1 = ax1.scatter(xc,yc,c=shift,marker='o',cmap=py.cm.RdYlBu,vmin=vmin,vmax=vmax)
    cbar1 = fig1.colorbar(cax1)
    ax1.set_xlabel('x pixels')
    ax1.set_ylabel('y pixels')
    ax1.set_title('TMP shifts vs CCD location')
    
    # now generate histograms:
    fig2 = py.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.hist(shift,bins=100,range=(-0.1,0.1),histtype='step',label=creepfile)
    ax2.set_xlabel('TLM shift')
    ax2.set_ylabel('Number')
    ax2.legend(prop={'size':8})

    # calculate some statistics:
    medshift = np.nanmedian(shift)
    print('median shift:',medshift)
    ax2.axvline(medshift,linestyle=':')

    # go along each axis and look for a gradient by averaging shifts:
    
    xc_unique = np.unique(xc)
    nxb = np.size(xc_unique)
    shift_avx = np.zeros(nxb)
    shift_sigx = np.zeros(nxb)
    ixb = 0
    for xcval in xc_unique:
        idx = np.where((xcval==xc))
        shift_avx[ixb] = np.nanmean(shift[idx])
        shift_sigx[ixb] = np.nanstd(shift[idx])/np.sqrt(np.size(shift[idx]))
        ixb = ixb + 1
        
    fig3 = py.figure(3)
    ax3 = fig3.add_subplot(111)
    ax3.errorbar(xc_unique,shift_avx,shift_sigx,fmt='o')
    ax3.set_xlabel('x pixels')
    ax3.set_ylabel('average shift')
    ax3.set_title('average shift as fn of x')
        
    yc_unique = np.unique(yc)
    nyb = np.size(yc_unique)
    shift_avy = np.zeros(nyb)
    shift_sigy = np.zeros(nyb)
    iyb = 0
    for ycval in yc_unique:
        idx = np.where((ycval==yc))
        shift_avy[iyb] = np.nanmean(shift[idx])
        shift_sigy[iyb] = np.nanstd(shift[idx])/np.sqrt(np.size(shift[idx]))
        iyb = iyb + 1

    fig4 = py.figure(4)
    ax4 = fig4.add_subplot(111)
    ax4.errorbar(yc_unique,shift_avy,shift_sigy,fmt='o')
    ax4.set_xlabel('y pixels')
    ax4.set_ylabel('average shift')
    ax4.set_title('average shift as fn of y')
        
    
################################################################################

def compare_spec(infile1,infile2,fibnum1=1,fibnum2=None,scale=False,interactive=False,lammin=1.0,lammax=1.0e10):
    """Function to simply plot the same spectrum of a given fibre from 2 different 
    frames to make a comparison."""
    
    #  first open the files:
    hdulist1 = pf.open(infile1)
    hdulist2 = pf.open(infile2)

    # get image:
    im1 = hdulist1[0].data
    im2 = hdulist2[0].data

    if (fibnum2==None):
        fibnum2=fibnum1
    
    # get variance:
    variance1 = hdulist1['VARIANCE'].data
    variance2 = hdulist2['VARIANCE'].data

    # Note y is first axis in python array, x is second
    (ys1,xs1) = im1.shape
    (ys2,xs2) = im2.shape
    print("image1 x axis",xs1)
    print("image1 y axis",ys1)
    print("image2 x axis",xs2)
    print("image2 y axis",ys2)

    # get wavelength axis:
    primary_header=hdulist1['PRIMARY'].header
    crval1=primary_header['CRVAL1']
    cdelt1=primary_header['CDELT1']
    crpix1=primary_header['CRPIX1']
    naxis1=primary_header['NAXIS1']

    x1=np.arange(naxis1)+1
        
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL
        
    lam1_all=L0+x1*cdelt1

    primary_header=hdulist2['PRIMARY'].header
    crval1=primary_header['CRVAL1']
    cdelt1=primary_header['CDELT1']
    crpix1=primary_header['CRPIX1']
    naxis1=primary_header['NAXIS1']

    x2=np.arange(naxis1)+1
        
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL
        
    lam2_all=L0+x2*cdelt1

    # Just use ~500 pixels with no strong lines:
    #x1 = 1200
    #x2 = 1700
    x11=1
    x12=len(x1)
    x21=1
    x22=len(x2)

    # define wavelength lims:
    l11 = max(lam1_all[0],lammin)
    l12 = min(lam1_all[xs1-1],lammax)
    l21 = max(lam2_all[0],lammin)
    l22 = min(lam2_all[xs2-1],lammax)

    print(x11,x12,l11,l12)
    print(x21,x22,l21,l22)

    repeat = True
    while (repeat):
        py.figure(1)
        py.clf()
        if (interactive):
            repeat = True
        else:
            repeat = False
    
        print('displaying fibre:',fibnum1,' from frame 1')
        print('displaying fibre:',fibnum2,' from frame 2')
        # set up plotting arrays:
        pfibnum1=fibnum1-1
        pfibnum2=fibnum2-1
        lam1 = lam1_all[x11:x12]
        lam2 = lam2_all[x21:x22]
        if (scale):
            flux1 = im1[pfibnum1,x11:x12]/np.nanmedian(im1[pfibnum1,x11:x12])
            flux2 = im2[pfibnum2,x21:x22]/np.nanmedian(im2[pfibnum2,x21:x22])
        else:
            flux1 = im1[pfibnum1,x11:x12]
            flux2 = im2[pfibnum2,x21:x22]
            
        var1 = variance1[pfibnum1,x11:x12]
        var2 = variance2[pfibnum2,x21:x22]
        sig1 = np.sqrt(var1)
        sig2 = np.sqrt(var2)
        print('sizes:',np.size(flux1),np.size(flux2))
        negidx = np.where(var1<0)
        print('pixels where variance is negative:')
        print(negidx)
        print(lam1[negidx])
        
        
    # plot:
        py.subplot(311)
        py.plot(lam1,flux1, 'b-',label=infile1)
        py.plot(lam2,flux2, 'r-',label=infile2)
        py.xlabel('Wavelength $\AA$')
        py.ylabel('Flux')
        py.title('fibres '+str(pfibnum1+1)+' and '+str(pfibnum2+1))
        py.xlim(l11,l12)
        
        py.legend(prop={'size':8})
        py.draw()
        
        # get the right scale for the variance:
        var12 = np.concatenate((sig1,sig2))
        y1 = 0
        y2 = sp.percentile(var12[np.logical_not(np.isnan(var12))],99.0)
    
        # plot variance:
        py.subplot(312)
        py.plot(lam1,sig1, 'b-')
        py.plot(lam2,sig2, 'r-')
        py.xlabel('Wavelength $\AA$')
        py.ylabel('sigma (from var array)')
        py.xlim(l11,l12)
        py.draw()
        #py.ylim(y1,y2)

        # plot S/N:
        py.subplot(313)
        py.plot(lam1,flux1/sig1, 'b-')
        py.plot(lam2,flux2/sig2, 'r-')
        py.xlabel('Wavelength $\AA$')
        py.ylabel('S/N')
        py.xlim(l11,l12)
        py.draw()
        #py.ylim(y1,y2)

        # now plot ratios:
        py.figure(2)
        py.subplot(211)
        if (xs1 == xs2):
            py.plot(lam1,flux1/flux2, 'b-',label=infile1)
        py.xlabel('Wavelength $\AA$')
        py.ylabel('Flux1/Flux2')
        py.title('Ratios for fibres '+str(pfibnum1+1)+' and '+str(pfibnum2+1))
        py.xlim(l11,l12)
        py.draw()
        py.subplot(212)
        if (xs1 == xs2):
            py.plot(lam1,sig1/sig2, 'b-')
        py.xlabel('Wavelength $\AA$')
        py.ylabel('sigma1/sigma2 (from var array)')
        py.xlim(l11,l12)
        py.draw()

        if (interactive):
            user = input("Continue or (Q)uit:")
            if (user == 'Q'):
                repeat = False
            if (user.isdigit()):
                print('skipping to fibre ',user)
                fibnum1 = int(user) - 2
                fibnum1 = fibnum1 + 1
                fibnum2 = fibnum1
            if (fibnum1 > ys1):
                repeat = False
            fibnum1 = fibnum1 + 1
            fibnum2 = fibnum1
        
    
    # finish and close file:
    hdulist1.close()
    hdulist2.close()

################################################################################

def plot_optex(frame,tlmfile,colnum,plotres=False,verbose=True,xplotmin=0,xplotmax=4096,dopdf=False,markfibres=False,nsum=1,logflux=False,scale=1.0):
    """Function to plot optimal extraction fits from 2dfdr.  Requires a
    frame name without extension, e.g. 06mar20055 the tlm file to use and
    a column number to plot.
    Usage:
    sami_2dfdr_reduction_tests.plot_optex('06mar20055','06mar20066tlm.fits',1500,nsum=1)
    optional arguments: plotres - do you want to plot residuals?
                        verbose - lots of output?

    Example useage when making a pdf:
    sami_2dfdr_reduction_tests.plot_optex('06mar20066','06mar20066tlm.fits',1500,xplotmin=628,xplotmax=700,dopdf=True)
    """

    # do you want to mark the centres of each fibre?
    #markfibres = True
    
    imfile = frame+'im.fits'
    mimfile = frame+'_outdir/'+frame+'mim.fits'
    mslfile = frame+'_outdir/'+frame+'msl.fits'
    #mslfile = frame+'_outdir/OPTEXBG.fits'
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
    #prf = hdulist_tlm['SIGMAPRF'].data
    
    # get variance:
    variance = hdulist_im['VARIANCE'].data

    # Note y is first axis in python array, x is second
    (ys,xs) = im.shape

    if (verbose):
        print(frame,', image size:',xs,' x ',ys)

    x=np.arange(ys)+1


    # get data for particular col:
    if (nsum == 1): 
        flux = im[:,colnum]
        model_flux = mim[:,colnum] 
        model_sl = msl[:,colnum]
        var = variance[:,colnum]
        sig = np.sqrt(variance)
        sigma = sig[:,colnum]
        tlmcol = tlm[:,colnum]
        #prfcol = prf[:,colnum]

    # or sum over columns to increase S/N: 
    else:
        n1 = int(np.rint(colnum-nsum/2))
        n2 = int(np.rint(colnum+nsum/2))
        print(n1,n2)
        flux = np.average(im[:,n1:n2],axis=1)*scale
        model_flux = np.average(mim[:,n1:n2],axis=1)*scale 
        model_sl = np.average(msl[:,n1:n2],axis=1)*scale
        var = np.average(variance[:,n1:n2],axis=1)/nsum
        sig = np.sqrt(variance/nsum)*scale
        sigma = np.average(sig[:,n1:n2],axis=1)
        tlmcol = np.average(tlm[:,n1:n2],axis=1)
        #prfcol = prf[:,colnum]


        
    nfib = np.size(tlmcol)
    if (verbose):
        print('number of tramlines in TLM:',nfib)

    # if we want a pdf make it:
    if (dopdf):
        pdf = PdfPages('plot_optex.pdf')

        
    # do plot of data and 2dfdr fit:
    py.figure(1)
    py.xlabel('pixel')
    py.ylabel('counts')
    #plot_bundlelims()
    if (logflux):
        py.plot(x,np.log10(flux), 'b-',label=frame+' Data')
        py.plot(x,np.log10(model_flux+model_sl), 'r-',label=frame+' model profiles')
        py.plot(x,np.log10(model_sl), 'g-',label=frame+' model scattered light')
    else:
        py.plot(x,flux, 'b-',label=frame+' Data')
        py.errorbar(x, flux, sigma)
        py.plot(x,model_flux+model_sl, 'r-',label=frame+' model profiles')
        py.plot(x,model_sl, 'g-',label=frame+' model scattered light')
        
    py.xlim(xmin = xplotmin, xmax = xplotmax)

    xmin,xmax,ymin,ymax = py.axis()

    # mark centres of fibres:
    if (markfibres):
        for i in range(np.size(tlmcol)):
            if (tlmcol[i] > xmin and tlmcol[i] < xmax):
                py.axvline(tlmcol[i]+0.5,ymin=0,ymax=0.9,color='r',linestyle=':')
                py.text(tlmcol[i]+0.5,ymin+(ymax-ymin)*0.95,i+1,horizontalalignment='center')    
    
            #py.legend(prop={'size':8})

    if (dopdf):        
        py.savefig(pdf, format='pdf')        
        pdf.close()
            
    # plot the residuals:
    if (plotres):    
        py.figure(2)
        py.plot(x,flux-model_flux-model_sl, 'b-',label=frame+' Data')
        py.xlabel('pixel')
        py.ylabel('residual')
        py.errorbar(x, flux-model_flux-model_sl, sigma)
        py.axhline(0.0,color='k',linestyle='-',linewidth=0.5)                
        py.legend(prop={'size':8})
            # mark centres of fibres:
        if (markfibres):
            for i in range(np.size(tlmcol)):
                if (tlmcol[i] > xmin and tlmcol[i] < xmax):
                    py.axvline(tlmcol[i]+0.5,ymin=0,ymax=0.9,color='r',linestyle=':')
                    py.text(tlmcol[i]+0.5,ymin+(ymax-ymin)*0.95,i+1,horizontalalignment='center')    


###########################################################################
# Function to plot the measured TLMSHIFT in reduced data frames as a fuction
# of time, checking that they look okay.
#
# usage:
# sami_2dfdr_reduction_tests.plot_tlmshift('06mar100??red.fits',dopdf=True)
#

def plot_tlmshift(inlist,dopdf=False):

    # if we want a pdf make it:
    if (dopdf):
        pdf = PdfPages('tlmshift.pdf')

    
    # "glob" the filelist to expand wildcards:
    files = glob.glob(inlist)
    maxfiles = np.size(files)
    
    # initialize:
    utmjd=np.zeros(maxfiles)
    tlmshift=np.zeros(maxfiles)
    exposure=np.zeros(maxfiles)
    obstype=np.empty(maxfiles,dtype='S10')
    
    print('number of files in list:',maxfiles)
    
    # loop over each 
    nfiles = 0
    for filename in files:

        # open fits file and check its an object with min exposure time:
        try:
            hdulist = pf.open(filename)

        except IOError:
            print('File not found, skipping ',filename)
            continue
                
        primary_header=hdulist['PRIMARY'].header

        # get the keywords we need:
        try:
            tlmshift[nfiles] = primary_header['TLMSHIFT']
        except KeyError:
            print('TLMSHIFT keyword not found, skipping ',filename)
            continue
        try:
            utmjd[nfiles] = primary_header['UTMJD']
        except KeyError:
            print('UTMJD keyword not found, skipping ',filename)
            continue

        exposure[nfiles] = primary_header['EXPOSED']
        obstype[nfiles] = primary_header['OBSTYPE']

        print(filename,nfiles,tlmshift[nfiles],utmjd[nfiles],exposure[nfiles],obstype[nfiles].decode('utf-8'))
        hdulist.close()
        nfiles = nfiles + 1

    # now plot, but use different colours for different file types:
    py.figure(1)
    nfit = 0
    xfit = np.zeros(nfiles)
    yfit = np.zeros(nfiles)
    for i in range(nfiles):
        col = 'k'
        if (obstype[i].decode('utf-8') == 'OBJECT'):
            col = 'red'
            if (exposure[i] < 600.0):
                col = 'green'
        elif (obstype[i].decode('utf-8') == 'SKY'):
            col = 'blue'
        elif (obstype[i].decode('utf-8') == 'FLAT'):
            col = 'cyan'
            
        py.plot(utmjd[i],tlmshift[i],'o',color=col)
        if (obstype[i].decode('utf-8') != 'ARC'):
            xfit[nfit] = utmjd[i]
            yfit[nfit] = tlmshift[i]
            nfit = nfit +1

    #
    xmin=np.min(utmjd)
    xmax=np.max(utmjd)
    z1 = np.polyfit(xfit[0:nfit],yfit[0:nfit],1)
    z2 = np.polyfit(xfit[0:nfit],yfit[0:nfit],2)
    print('1st order fit parameters:',z1)
    print('2nd order fit parameters:',z2)
    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)
    xp = np.linspace(xmin,xmax, 100)
    yp1 = p1(xp)
    yp2 = p2(xp)
    py.plot(xp,yp1)
    py.plot(xp,yp2)
    #py.axvline(0.0,color='red',linestyle=':')
    py.axhline(0.0,color='red',linestyle=':')
    py.xlabel('Time from reference exposure (sec)')
    py.ylabel('TLM offset (pixels)')

        
    py.xlabel('UT Mean Julian Date (days)')
    py.ylabel('TLM shift (pixels)')

#############################################################################

def compare_thput(infile1,infile2,maxdiff=0.25):
    """Function to compared the throughput measurements from two different frames.
    Usage:
    sami_2dfdr_reduction_tests.compare_thput('06mar10063red.fits','06mar10064red.fits')
    maxdiff - the max fractional difference in throughput before being flagged
    
    """

    
    # open file:
    hdulist1 = pf.open(infile1)
    hdulist2 = pf.open(infile2)

    # get data:
    data1 = hdulist1[0].data
    data2 = hdulist2[0].data
    
    # get throughput extension:
    thput1 = hdulist1['THPUT'].data
    thput2 = hdulist2['THPUT'].data

    # get binary table info:
    try: 
        fib_tab_hdu=find_fibre_table(hdulist1)
        table_data = hdulist1[fib_tab_hdu].data
        types=table_data.field('TYPE')
    except KeyError:
        print('No fibre table found')
        types = np.empty(np.size(thput1), dtype='string')
        types.fill('P')
    
    
    # get array sizes:
    xs = thput1.size
    print("input array size:",xs)
    print("1st python index is ",xs)

    
    # define fibre number axis:
    xfib = range(xs)

    # get ratio of throughputs:
    ratio = thput1/thput2

    # median filter ratios to get smoothed version:
    ratio_medf=nd.filters.median_filter(ratio,size=51)

    # calculate the scatter in the ratios locally (compared to local median):
    ratio_rms = np.sqrt(np.sum((ratio-ratio_medf)**2)/xs)

    print('rms about median ratio:',ratio_rms)
    
    # calculate some statistics:
    thput1_med = np.median(thput1)
    thput1_p1 = np.percentile(thput1,86.4)
    thput1_m1 = np.percentile(thput1,13.6)
    thput1_p2 = np.percentile(thput1,97.5)
    thput1_m2 = np.percentile(thput1,2.5)

    thput2_med = np.median(thput2)
    thput2_p1 = np.percentile(thput2,86.4)
    thput2_m1 = np.percentile(thput2,13.6)
    thput2_p2 = np.percentile(thput2,97.5)
    thput2_m2 = np.percentile(thput2,2.5)

    ratio_med = np.median(ratio)
    ratio_p1 = np.percentile(ratio,86.4)
    ratio_m1 = np.percentile(ratio,13.6)
    ratio_p2 = np.percentile(ratio,97.5)
    ratio_m2 = np.percentile(ratio,2.5)

    print("                                 percentiles")
    print(" Measurement         median 13.6  86.4  2.5  97.5")
    print('{0:20s} {1:5.3f} {2:5.3f} {3:5.3f} {4:5.3f} {5:5.3f}'.format(infile1,thput1_med,thput1_m1,thput1_p1,thput1_m2,thput1_p2))
    print('{0:20s} {1:5.3f} {2:5.3f} {3:5.3f} {4:5.3f} {5:5.3f}'.format(infile2,thput2_med,thput2_m1,thput2_p1,thput2_m2,thput2_p2))
    print('{0:20s} {1:5.3f} {2:5.3f} {3:5.3f} {4:5.3f} {5:5.3f}'.format('Ratio (1/2)',ratio_med,ratio_m1,ratio_p1,ratio_m2,ratio_p2))

    
    # start with plotting the individual throughputs
    py.figure(1)
    py.subplot(2,1,1)
    py.axhline(1.0,color='k',linestyle='--')

    # plot all throughputs
    py.plot(xfib,thput1,'.',color='r',label=infile1)
    py.plot(xfib,thput2,'.',color='b',label=infile2)
    plot_bundlelims()
    # plot sky fibres sepeartely with different symbols:
    ns=0
    for i in range(xs):
        if (types[i] == 'S'):
            if (ns == 0):
                lab = infile1+' skies'
                py.plot(xfib[i],thput1[i],'o',color='r')
                lab = infile2+' skies'
                py.plot(xfib[i],thput2[i],'o',color='b')
            else:
                py.plot(xfib[i],thput1[i],'o',color='r')
                py.plot(xfib[i],thput2[i],'o',color='b')                
        ns=ns+1
        
    py.xlabel('Fibres')
    py.ylabel('Rel throughput')
    py.title('Throughput comparison (skys are large symbols)')
    #py.legend(loc='lower right',prop={'size':10})
    py.legend(loc='upper right',prop={'size':10})

    # plot ratio of throughputs:
    py.subplot(2,1,2)
    py.axhline(1.0,color='k',linestyle='--')
    py.plot(xfib,ratio,'.',label='Ratio')
    py.plot(xfib,ratio_medf,linestyle='-',label='median ratio')
    plot_bundlelims()
    # plot sky ratios:
    ns=0
    for i in range(xs):
        if (types[i] == 'S'):
            if (ns == 0):
                py.plot(xfib[i],ratio[i],'o',color='r',label='Sky ratio')
            else:
                py.plot(xfib[i],ratio[i],'o',color='r')
            ns=ns+1
                
    py.xlabel('Fibres')
    py.ylabel('thput1/thput2')
    py.legend(prop={'size':10})

    # now look to see if there is a relationship between throughput and the continuum
    # flux in the fibre.  This could be the case for the skyline methods.
    data1_med = np.nanmedian(data1,axis=1)
    data2_med = np.nanmedian(data2,axis=1)
    
    # output bad throughputs, i.e. throughputs with large differnces in ratio:
    for i in range(xs):
        if (abs(ratio[i]-1.0) > maxdiff):
            print("discrepant throughput in fibre ",i+1,thput1[i],thput2[i],ratio[i],data2_med[i])
            py.plot(i,ratio[i],'x',color='r')



    py.figure(2)
    py.subplot(2,1,1)
    py.plot(data1_med,thput1,'.')
    py.subplot(2,1,2)
    py.plot(data2_med,ratio,'.')
    py.xlabel('median flux')
    py.ylabel('thput1/thput2')

    # bin the ratio data by the continuum flux:
    nbins=50
    binsize = 150000.0/nbins
    fbin = np.arange(nbins,dtype='float')*binsize+binsize/2.0
    thbin = np.zeros(nbins)
    nthbin = np.zeros(nbins)
    for i in range(xs):
        print(data2_med[i])
        ibin = int(data2_med[i]/binsize)
        thbin[ibin] = thbin[ibin] + ratio[i]
        nthbin[ibin] = nthbin[ibin] + 1

    for i in range(nbins):
        thbin[i] = thbin[i]/nthbin[i]
        print(fbin[i],thbin[i],nthbin[i])
    
    rho,prob = stats.spearmanr(data2_med,ratio)

    print(rho,prob)
        
    py.plot(fbin,thbin,'o')
    print(np.shape(data1_med))

################################################################################
# plot bundle lims on a plot with fib num on the x axis.  Assume first fibre is
# number 1.       
#
def plot_bundlelims(color='g',linestyle=':'):

    for i in range(14):
        
        lim = 63*i+0.5
        py.axvline(lim,color=color,linestyle=linestyle)
    
################################################################################
# compare two tlm files:
#
def compare_tlm(frame1,frame2):
    """
    
    """

    hdu1 = pf.open(frame1)
    hdu2 = pf.open(frame2)

    # read the TLM
    tlm1 = hdu1[0].data
    tlm2 = hdu2[0].data
    #
    # do the plots:
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(2,1,1)
    cax1 = ax1.imshow(tlm1,origin='lower',interpolation='nearest')
    cbar1 = fig1.colorbar(cax1)
    ax2 = fig1.add_subplot(2,1,2)
    cax2 = ax2.imshow(tlm2,origin='lower',interpolation='nearest')
    cbar2 = fig1.colorbar(cax2)

    # plot difference:
    tlm_diff = tlm2-tlm1
    fig2 = py.figure(2)
    ax3 = fig2.add_subplot(1,1,1)
    cax3 = ax3.imshow(tlm_diff,origin='lower',interpolation='nearest')
    cbar3 = fig2.colorbar(cax3)
    

    

#    py.title(exten_name+' for '+frame1)


    
################################################################################
# compare sigma profiles from different tlm files:
#
def compare_sigmaprf(frame1,frame2,frame3):
    """
    Frame 3 is the frame with a fibre table
    """

    exten_name = 'SIGMAPRF'
    
    hdu1 = pf.open(frame1)
    hdu2 = pf.open(frame2)
    hdu3 = pf.open(frame3)

    # read the TLM
    tlm1_tmp = hdu1[0].data
    tlm2_tmp = hdu2[0].data
    # write the TLM to the array, but do not use the raw TLM,
    # instead, get the difference between adjacent TLMs and
    # use this.  The relative shifts should not change!
    # use numpy.roll to do the shift
    tlm1 = tlm1_tmp - np.roll(tlm1_tmp,1,axis=0)
    tlm2 = tlm2_tmp - np.roll(tlm2_tmp,1,axis=0)
    
    sig1 = hdu1[exten_name].data
    sig2 = hdu2[exten_name].data

    diff = sig1 - sig2
    
    # get binary table info:
    fib_tab_hdu=find_fibre_table(hdu3)
    table_data = hdu3[fib_tab_hdu].data
    types=table_data.field('TYPE')
    
    v1 = 1.0
    v2 = 1.3
    
    py.figure(1)
    py.subplot(3,1,1)
    py.title(exten_name+' for '+frame1)
    py.imshow(sig1,vmin=v1,vmax=v2,origin='lower',interpolation='nearest')
    py.colorbar()

    py.subplot(3,1,2)
    py.title(exten_name+' for '+frame2)
    py.imshow(sig2,vmin=v1,vmax=v2,origin='lower',interpolation='nearest')
    py.colorbar()
    
    py.subplot(3,1,3)
    py.title(frame1+' - '+frame2)
    py.imshow(diff,origin='lower',interpolation='nearest')
    py.colorbar()

    py.figure(2)
    py.subplot(2,1,1)

    print('shape:',np.shape(sig1))
    
    sig1_av = np.nanmean(sig1,axis=1)
    sig2_av = np.nanmean(sig2,axis=1)

    xs = sig1_av.size
    print("input array size:",xs)
    
    # define fibre number axis:
    xfib = range(xs)

    # get ratio of throughputs:
    ratio = sig1_av/sig2_av

    #py.axhline(1.0,color='k',linestyle='--')

    # plot all throughputs
    py.plot(xfib,sig1_av,'.',color='r',label=frame1)
    py.plot(xfib,sig2_av,'.',color='b',label=frame2)

    # plot sky fibres sepeartely with different symbols:
    ns=0
    for i in xrange(xs):
        if (types[i] == 'S'):
            py.plot(xfib[i],sig1_av[i],'o',color='r',markersize=10)
            py.plot(xfib[i],sig2_av[i],'o',color='b',markersize=10)                
        ns=ns+1
        
    py.xlabel('Fibres')
    py.ylabel('Fibre profile sigma')
    py.title('Sigma comparison (skys are large symbols)')
    py.legend(loc='lower right',prop={'size':10})
    #py.legend(loc='upper right',prop={'size':10})

    # plot ratio of throughputs:
    py.subplot(2,1,2)
    py.axhline(1.0,color='k',linestyle='--')
    py.plot(xfib,ratio,'.',label='Ratio')
    #py.plot(xfib,ratio_medf,linestyle='-',label='median ratio')

    # plot sky ratios:
    ns=0
    for i in xrange(xs):
        if (types[i] == 'S'):
            py.plot(xfib[i],ratio[i],'o',color='r')
            ns=ns+1
                
    py.xlabel('Fibres')
    py.ylabel('sigma1/sigma2')
    py.legend(prop={'size':10})

    # plot the tlms:
    tlm1_av = np.nanmean(tlm1,axis=1)
    tlm2_av = np.nanmean(tlm2,axis=1)
    
    py.figure(3)
    py.subplot(2,1,1)

    print('shape:',np.shape(tlm1))

    xs = tlm1_av.size
    print("input array size:",xs)
    
    # define fibre number axis:
    xfib2 = range(xs)

    # get ratio of throughputs:
    ratio = tlm1_av/tlm2_av

    #py.axhline(1.0,color='k',linestyle='--')

    # plot all throughputs
    py.plot(xfib2,tlm1_av,'.',color='r',label=frame1)
    py.plot(xfib2,tlm2_av,'.',color='b',label=frame2)

    # plot sky fibres sepeartely with different symbols:
    ns=0
    for i in xrange(xs):
        if (types[i] == 'S'):
            py.plot(xfib2[i],tlm1_av[i],'o',color='r',markersize=10)
            py.plot(xfib2[i],tlm2_av[i],'o',color='b',markersize=10)                
        ns=ns+1
        
    py.xlabel('Fibres')
    py.ylabel('Fibre tlm difference')
    py.ylim(ymin=3.0,ymax=20.0)
    #py.title('Sigma comparison (skys are large symbols)')
    py.legend(loc='lower right',prop={'size':10})
    #py.legend(loc='upper right',prop={'size':10})
    
    

            
    
        
