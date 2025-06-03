##########################################################################
# functions to process Hector simulation data and do various things to it

import pylab as py
import numpy as np
import astropy.io.fits as fits
import random
import glob
import csv
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from symfit import parameters, variables, Fit, Poly, Model
from symfit.core.objectives import LogLikelihood
from symfit.core.minimizers import NelderMead,BFGS

from scipy.stats import linregress
from shutil import copyfile
from os.path import basename
from .sami_2dfdr_read import find_fibre_table
from .sami_stats_utils import clipped_mean_rms
from image_registration import chi2_shift
from image_registration.fft_tools import shift
#from dtw import *

##############################################################
# fix hector error in ccd4 that has a bias col in the
# middle of the detector!!!
# only in early (prior to ~feb 2024 data).  Code taken from Sree's
# fix in the main hector manager code:
#
def fix_biascol(infile):

    hdulist = fits.open(infile, 'update')
    hdr = hdulist[0].header
    # check to see if the header keyword for the update is present (don't want
    # to do it twice): 
    try:
        biascol_modified = hdr['BIASCOL']
    except KeyError:
        biascol_modified = False

    if (biascol_modified):
        print('bias column already modified, not doing it again')
    else:
        image = hdulist[0].data
        bias_col = image[:,2048].copy()
        image[:,2048:-1] = image[:, 2049:]
        image[:, -1] = bias_col
        hdr['BIASCOL'] = (True, 'BIAS column at x=2048 is removed')
        hdulist.flush()
        print('update done.')

    hdulist.close()


##############################################################
# test code for dynamic time warping

def dtw_test():
    
    ## A noisy sine wave as query
    idx = np.linspace(0,6.28,num=100)
    query = np.sin(idx) + np.random.uniform(size=100)/10.0

    ## A cosine is for template; sin and cos are offset by 25 samples
    #template = np.cos(idx)

    ## Find the best match with the canonical recursion formula
    #alignment = dtw(query, template, keep_internals=True)

    ## Display the warping curve, i.e. the alignment curve
    #alignment.plot(type="threeway")

    ## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
    #dtw(query, template, keep_internals=True, 
    #    step_pattern=rabinerJuangStepPattern(6, "c"))\
    #    .plot(type="twoway",offset=-2)

    ## See the recursion relation, as formula and diagram
    #print(rabinerJuangStepPattern(6,"c"))
    #rabinerJuangStepPattern(6,"c").plot()

## And much more!

###########################################################################
# Analyise shifts in wavelength in twilights frames:
#
def twilight_waveshifts(infile,midspec=None,bandwidth=100):

    # open file and read data:
    hdulist = fits.open(infile)
    im = hdulist[0].data
    var = hdulist['VARIANCE'].data
    stdev = np.sqrt(var)
    
    lam = get_lam_axis(hdulist)
    (ys,xs) = im.shape
    print("input array sizes:",ys," x ",xs)
    nfib = ys

    # get pix size in Angstoms:
    primary_header=hdulist['PRIMARY'].header
    cdelt1=primary_header['CDELT1']
    
    fib_tab_hdu=find_fibre_table(hdulist)
    table_data = hdulist[fib_tab_hdu].data
    type=table_data.field('TYPE')

    if midspec is None:
        midspec = int(ys/2)
        print('central fibre:',midspec,type[midspec])

    # calculate median flux of each fibre:
    medfib = np.nanmedian(im,axis=1)
    print(np.size(medfib))

    imnorm = np.zeros((ys,xs))
    stdevnorm = np.zeros((ys,xs))
    for i in range(nfib):
        imnorm[i,:] = im[i,:]/medfib[i]
        stdevnorm[i,:] = stdev[i,:]/medfib[i]

    print('fibre to plot:',midspec)
        
    # display the image:
    fig2 =plt.figure(2)
    ax2 = fig2.add_subplot(1,1,1)
    vmin = np.nanpercentile(imnorm,1.0)
    vmax = np.nanpercentile(imnorm,99.0)
    ax2.imshow(imnorm,vmin=vmin,vmax=vmax,origin='lower',interpolation='nearest')
    
    # do a simple plot of spectrum to make sure everything is working:
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1,1,1)
    #plot the 5 spectra around midspec:
    for i in range(midspec-2,midspec+2):
        ax1.plot(lam,imnorm[i,:])
        
    # loop through spectra and estimate shifts.  Do this is short wavelength
    # intervals:
    nbins = int(xs/bandwidth)

    ymin, ymax = ax1.get_ylim()
    # plot vertical lines to mark out the bands we are going to fit:
    for i in range(nbins+1):
        ax1.axvline(lam[i*bandwidth],color='k',linestyle=':')
        ax1.text(lam[int(i*bandwidth+bandwidth/2)],ymin+0.05*(ymax-ymin),str(i),horizontalalignment='center')
    
    print('will sample each spectrum in ',nbins,' bins')
    # set up array to store results:
    shifts = np.zeros((nfib,nbins))
    shifts_corr = np.zeros((nfib,nbins))
    for j in range(nbins):
        print('running bin ',j)
        x1 = j*bandwidth
        x2 = (j+1)*bandwidth
        pad = 0
        r1 = x1-pad
        r2 = x2+pad
        ref = imnorm[midspec,r1:r2].reshape(r2-r1,1)
        for i in range(nfib):
            if ((type[i] == 'P') or (type[i] == 'S')):

                # now do cross-correlation to find the shift:
                spec = imnorm[i,x1:x2].reshape(x2-x1,1)
                err = stdevnorm[i,x1:x2].reshape(x2-x1,1)
                # check number of good pixels and only do measurement
                # if there is a reasonable number:
                ngood = np.count_nonzero(~np.isnan(spec))
                if (ngood>bandwidth/2):
                    xoff,yoff,exoff,eyoff = chi2_shift(ref,spec,err=err,zeromean=True,boundary='nearest')
                else:
                    yoff = np.nan
                    
                shifts[i,j] = yoff

    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(1,1,1)
    vmin = np.nanpercentile(shifts[:,1:nbins-1],5.0)
    vmax = np.nanpercentile(shifts[:,1:nbins-1],95.0)    
    cax4 = ax4.imshow(shifts,origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax,aspect=0.05*100/bandwidth)
    cbar4 = fig4.colorbar(cax4,ax=ax4)
    ax4.set(xlabel = 'Wavelength axis',ylabel='Fibre')
    
    # plot the histogram of shifts values and calc stats:
    shifts1d = shifts.flatten()
    fig5 = plt.figure(5)
    ax5 = fig5.add_subplot(1,1,1)
    ax5.hist(shifts1d,bins = 100,range=[-1.0,1.0],histtype='step')

    # do some stats:
    shift_med = np.nanmedian(shifts1d)
    shift_05p = np.nanpercentile(shifts1d,5.0)
    shift_84p = np.nanpercentile(shifts1d,84.14)
    shift_16p = np.nanpercentile(shifts1d,15.87)
    shift_95p = np.nanpercentile(shifts1d,95.0)

    ax5.axvline(0.0,color='k',linestyle='-')
    ax5.axvline(shift_med,color='r',linestyle=':')
    ax5.axvline(shift_16p,color='g',linestyle=':')
    ax5.axvline(shift_84p,color='g',linestyle=':')
    ax5.axvline(shift_05p,color='b',linestyle=':')
    ax5.axvline(shift_95p,color='b',linestyle=':')
    print('Stats for original shift array:')
    print('median shift:',shift_med)
    print('1 sigma range (percentile):',shift_16p,shift_84p)
    print('2 sigma range (percentile):',shift_05p,shift_95p)
    onesigma = (shift_84p-shift_16p)/2.0
    print('1 sigma (from percentile):',onesigma)
    ax5.set(xlabel='pixel shift',ylabel='Number')
    shift_stdev = np.nanstd(shifts1d)
    print('regular standard deviation:',shift_stdev)
    
    
    # replot the shifts, but this time subtract the median shift in each bin, as this
    # can be a little systematic due to particularly strong features in the spectra:
    medshiftbin = np.nanmedian(shifts,axis=0)
    for j in range(nbins):
        for i in range(nfib):
            shifts_corr[i,j] = shifts[i,j]-medshiftbin[j]

    fig6 = plt.figure(6)
    ax6 = fig6.add_subplot(1,1,1)
    vmin = np.nanpercentile(shifts_corr,2.0)
    vmax = np.nanpercentile(shifts_corr,98.0)    
    cax6 = ax6.imshow(shifts_corr,origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax,aspect=0.05*100/bandwidth)
    cbar6 = fig6.colorbar(cax6,ax=ax6)
    ax6.set(xlabel = 'Wavelength axis',ylabel='Fibre')
    
    # plot the histogram of shifts values and calc stats:
    shifts_corr1d = shifts_corr.flatten()
    fig7 = plt.figure(7)
    ax7 = fig7.add_subplot(1,1,1)
    ax7.hist(shifts_corr1d,bins = 100,range=[-1.0,1.0],histtype='step')

    # do some stats:
    shift_med = np.nanmedian(shifts_corr1d)
    shift_05p = np.nanpercentile(shifts_corr1d,5.0)
    shift_84p = np.nanpercentile(shifts_corr1d,84.14)
    shift_16p = np.nanpercentile(shifts_corr1d,15.87)
    shift_95p = np.nanpercentile(shifts_corr1d,95.0)

    ax7.axvline(0.0,color='k',linestyle='-')
    ax7.axvline(shift_med,color='r',linestyle=':')
    ax7.axvline(shift_16p,color='g',linestyle=':')
    ax7.axvline(shift_84p,color='g',linestyle=':')
    ax7.axvline(shift_05p,color='b',linestyle=':')
    ax7.axvline(shift_95p,color='b',linestyle=':')
    print('\n Stats for corrected shift array:')
    print('median shift:',shift_med)
    print('1 sigma range (percentile):',shift_16p,shift_84p)
    print('2 sigma range (percentile):',shift_05p,shift_95p)
    onesigma = (shift_84p-shift_16p)/2.0
    print('1 sigma (from percentile):',onesigma)
    ax7.set(xlabel='pixel shift',ylabel='Number')
    shift_stdev = np.nanstd(shifts_corr1d)
    print('regular standard deviation:',shift_stdev)
    n5sigma = (np.abs(shifts_corr1d)>5.0*onesigma).sum()
    print('Number > 5 sigma:',n5sigma,n5sigma/(nbins*nfib))

    # plot spectra (with and without shifts for a few examples):
    fig3 = plt.figure(3)
    ax31 = fig3.add_subplot(2,1,1)
    ax32 = fig3.add_subplot(2,1,2)
    # plot the 5 spectra around midspec:
    # plot ref spec:
    #label='ref fib '+str(midspec)
    #ax3.plot(lam[r1:r2],imnorm[i,r1:r2],label=label)
    for i in range(midspec-2,midspec+3):
        label='fib '+str(i)
        ax31.plot(lam[x1:x2],imnorm[i,x1:x2],label=label)
        j = 2
        x1 = j*bandwidth
        x2 = (j+1)*bandwidth
        pad = 0
        r1 = x1-pad
        r2 = x2+pad
        # now do cross-correlation to find the shift:
        ref = imnorm[midspec,r1:r2].reshape(r2-r1,1)
        spec = imnorm[i,x1:x2].reshape(x2-x1,1)
        err = stdevnorm[i,x1:x2].reshape(x2-x1,1)
        xoff,yoff,exoff,eyoff = chi2_shift(ref,spec,err=err,zeromean=True,boundary='nearest')
        # if offset is large, redo accounting for this shift ro take care fo edge effects:
        #if (abs(yoff)>2):
        #    print('doing second pass due to large offset...')
        #    xx1 = x1-yoff
        #    xx2 = x2-yoff
        #    spec = imnorm[i,xx1:xx2].reshape(xx2-xx1,1)
        #    err = stdevnorm[i,xx1:xx2].reshape(xx2-xx1,1)
        #    xoff,yoff,exoff,eyoff = chi2_shift(ref,spec,err=err,zeromean=True,boundary='nearest')
            

        #print(midspec,i,xoff,yoff,exoff,eyoff)
        
        ax32.plot(lam[x1:x2]-yoff*cdelt1,imnorm[i,x1:x2],label=label)

        
    plt.legend(prop={'size':10})

    
###########################################################################
# plot TLM slice and peaks found to test TLM matching:
#
# usage:
# sami_tools_smc.dr_tools.hector_data_checks.plot_tlmpeaks('ycut.dat','peaks.dat','/Users/scroom/code/2dfdr_src/2dfdr_hector3/2dfdr/drcontrol_ac/dat_files/FIBPOS_AAOmega_Hector_580V.dat')

def plot_tlmpeaks(ycutfile,peakfile,origposfile,typefile,delta=0):

    # read ycut file:
    iycut,ycut = np.loadtxt(ycutfile,unpack=True)
    nycut = len(ycut)
    
    # read peak file:
    ipeak,peak = np.loadtxt(peakfile,unpack=True)
    # add the 0.5 TLM offset
    peak = peak + 0.5
    npeak = len(peak)
    
    # read types file:
    types = np.loadtxt(typefile,usecols=[1],dtype='U1')
    ntype = len(types)
    print(types)

    
    # read orignal (default) peak file:
    iorig,orig = np.loadtxt(origposfile,unpack=True)
    # add the 0.5 TLM offset
    orig = orig + 0.5
    norig = len(orig)

    print ('Number of pixels in cut: ',nycut)
    print ('Number of peaks in original list: ',norig)
    print ('Number of peaks found: ',npeak)

    # set up plotting:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)

    # plot cut:
    ax1.plot(iycut,ycut,'k')
    # plot original peaks:
    for i in range(norig):
        ax1.axvline(orig[i],color='b',linestyle='--',ymax=0.9)
    # plot peaks found:
    for i in range(npeak):
        ax1.axvline(peak[i],color='r',linestyle=':',ymax=0.9)

    
    # second plot to find best match:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)

    good = np.where((types == 'S') | (types == 'P')) 
    
    ax2.plot(orig[0:npeak],peak[0:npeak],'o',color='r')
    ax2.set(xlabel = 'default fib location',ylabel = 'peak location')
    for i in range(npeak):
        if ((types[i] == 'S') | (types[i] == 'P')):
            ax2.plot(orig[i],peak[i],'o',color='b')

    # mark block edges:
    for i in range(13):
        i1 = i*63
        i2 = (i+1)*63-1
        print(i1,i2)
        ax2.axvline(orig[i1],color='g',linestyle=':',ymax=0.9)
        ax2.axvline(orig[i2],color='g',linestyle=':',ymax=0.9)

    # do a fit:
    a,b = np.polyfit(orig[0:npeak],peak[0:npeak],1)

    print(a,b)

    # plot the line:
    ax2.plot(orig[0:npeak],a*orig[0:npeak]+b,'-',color='g')

    # what are the possible actions?
    # 1) add a blank fibre at either end of a block
    # 2) add a blank fibre at an arbitrary location.
    # 3) assum to start with, only P & S fibres are visible?

    # define an array that gives index of peaks to master positions.
    peakindx=-1*np.ones(norig,dtype=np.int16)
    
    ip = 0
    for i in range(norig):
        if ((types[i] == 'P') | (types[i] == 'S')):
            peakindx[i] = ip
            ip = ip + 1

    print(peakindx)

    # define x and ypoints to be fit/plotted:
    nf = 0
    x=np.zeros(norig)
    y=np.zeros(norig)
    for i in range(norig):
        if (peakindx[i] > -1):
            x[nf] = orig[i]
            y[nf] = peak[peakindx[i]]
            nf = nf+1
            
    ax2.plot(x[0:nf],y[0:nf],'x',color='g')
    
    # do a fit:
    a2,b2 = np.polyfit(x[0:nf],y[0:nf],1)

    #Now do something....
    # what are the cases?
    # 1) too many peaks found.  Need to assign a peak to an orig location
    #    that has an N/U etc fibre.
    # 2) too few peaks and need to find which are missing.
    # 3) the right number but they are mixed....?
    # 4) right number but not mixed - all good.

    for i in range(nf):
            delta = y[i] - (a2*x[i]+b2)
            print(delta)
            

    
    
    # to begin with assume that peaks are only assigned to S and P fibres and
    # do them in order.  Then make corrections to this...

    # plot the line:
    ax2.plot(orig,a2*orig+b2,'-',color='c')
    

    
    

###########################################################################
# update FIBPOS files, particularly to set sensible values for missing
# fibres
#
def fix_fibpos(tlmfile,imfile,fibposfile,blocksize=63):

    # first read the tlm file:
    hdulist = fits.open(tlmfile)
    tlm = hdulist[0].data
    (nfib,nx) = np.shape(tlm)
    print('size of TLM file: (nfib,nx)=',nfib,nx)
    hdulist.close()
    
    # now read the im file:
    hdulist = fits.open(imfile)
    im = hdulist[0].data
    (nyi,nxi) = np.shape(im)
    print('size of im file: (nyi,nxi)=',nyi,nxi)
    hdulist.close()

    print('Blocksize = ',blocksize)
    
    # take a slice of the TLM data:
    # note +0.5 is because there is a 0.5 pixel offset betwen the tlm and pixels
    cent = int(nx/2+1)
    tlm_slice = tlm[:,cent]+0.5
    im_slice = im[:,cent]

    # get difference:
    diff = tlm_slice[1:]-tlm_slice[0:-1]

    med_diff = np.nanmedian(diff)
    print('Median gap between TLMs: ',med_diff)
    
    yax = np.linspace(1.0,nyi,nyi)

    # plot the slice:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(yax,im_slice,'-',color='b')
    (ymin,ymax) = ax1.get_ylim()
    yrange = ymax-ymin
    yloc = ymax-0.05*yrange
    print(ymin,ymax)
    # plot the TLMs:
    for  i in range(nfib):
        ifib = i+1
        ax1.axvline(tlm_slice[i],color='r',linestyle='--',ymax=0.9)
        ax1.text(tlm_slice[i],yloc,str(ifib), horizontalalignment='center',verticalalignment='center')


    # read the FIBPOS file:
    outfile = fibposfile.replace('.dat','_new.dat')
    file1 = open(fibposfile, 'r')
    file2 = open(outfile,'w')

    lines = file1.readlines()

    fib = np.zeros(nfib)
    pos = np.zeros(nfib)
    
    n = 0
    for index, line in enumerate(lines):
        # skip comments:
        if '#' in line:
            file2.write(line)
            continue
        else:
            # split the line up:
            tmp = line.split()
            fib[n] = int(tmp[0])
            pos[n] = float(tmp[1])

            n = n + 1

    # now loop through all the FIBPOS data:
    for i in range(n):
        
        # check fibres within blocks to make sure they all 
        # have consistent spacing:
        # first look at fibres at the end of a block:
        if (fib[i] % blocksize == 0):
            diff = pos[i]-pos[i-1]
            print(fib[i],pos[i],diff)
            if (abs(diff-med_diff) > 2):
                pos[i] = pos[i-1] + med_diff
                print('Change pos to ',pos[i])

        # do the same for start of block:
        if (fib[i] % blocksize == 1):
            diff = pos[i+1]-pos[i]
            print(fib[i],pos[i],diff)
            if (abs(diff-med_diff) > 2):
                pos[i] = pos[i+1] - med_diff
                print('Change pos to ',pos[i])

                
        ax1.axvline(pos[i],color='g',linestyle=':',ymax=0.95)
        # write to output file:
        line = ' {0:4d} {1:8.3f}\n'.format(int(fib[i]),pos[i])
        file2.write(line)
        
            
    file1.close()
    file2.close()

        
    return





###########################################################################
# get wavelength axis array:
#
def get_lam_axis(hdulist):

        # get wavelength axis keywords:
        primary_header=hdulist['PRIMARY'].header
        crval1=primary_header['CRVAL1']
        cdelt1=primary_header['CDELT1']
        crpix1=primary_header['CRPIX1']
        naxis1=primary_header['NAXIS1']

        # define axis:
        x=np.arange(naxis1)+1
        L0=crval1-crpix1*cdelt1 #Lc-pix*dL
        lam=L0+x*cdelt1

        return lam
        

###########################################################################
# Routine to estimate the gain from a flat frame, or set of flat frames.
# based on an old version for SAMI (converted from python 2.7)
#
def measure_gain(flatlist,plot=True,verbose=True,pdf=False,scaleim=True,y1=2000,y2=2100):

    '''
    Script to calculate the gain of a CCD for SAMI (or other similar - Hector...) data
    using a pair of flat fields (need to be defocussed or detector flats).
    
    usage:
    sami_tools_smc.dr_tools.hector_data_checks.measure_gain('12mar200??im.fits',scaleim=True)

    scaleim allows for the second image to be scaled slightly to match the median flux, so 
    accounting for variations in illumination, but if this is a few percent, it generally 
    makes no difference to the resultant gain values.


    '''
    

    # "glob" the filelist to expand wildcards:
    files = sorted(glob.glob(flatlist))

    # find the number of files to use to define array size:
    maxfiles = np.size(files)
    if (verbose):
        print('number of files in list:',maxfiles)

    #define arrays to hold data from each file:
    gain_measured = np.zeros(maxfiles)
    gain_fromfile = np.zeros(maxfiles)
    
    # loop over each file, but as we are using pairs, only
    # go to the last but one:
    ngood=0
    for nfiles in range(maxfiles-1):

        # get the file names:
        file1 = files[nfiles]
        file2 = files[nfiles+1]
        
        # open fits file and check its an object with min exposure time:
        try:
            hdulist1 = fits.open(file1)
        except IOError:
            print('File not found, skipping ',file1)
            continue
        try:
            hdulist2 = fits.open(file2)
        except IOError:
            print('File not found, skipping ',file2)
            continue
                
        primary_header1=hdulist1['PRIMARY'].header
        exposure1 = primary_header1['EXPOSED']
        obstype1 = primary_header1['OBSTYPE']
        gain1 = primary_header1['RO_GAIN']
        ron1 = primary_header1['RO_NOISE']
        
        primary_header2=hdulist2['PRIMARY'].header
        exposure2 = primary_header2['EXPOSED']
        obstype2 = primary_header2['OBSTYPE']
        gain2 = primary_header2['RO_GAIN']
        ron2 = primary_header2['RO_NOISE']

        if (verbose):
            print(file1,'  OBSTYPE=',obstype1)
            print(file2,'  OBSTYPE=',obstype2)
            
        # get the data:
        im1 = hdulist1[0].data
        im2 = hdulist2[0].data
        hdulist1.close()
        hdulist2.close()

        # get size:
        (ys1,xs1) = im1.shape
        (ys2,xs2) = im2.shape
        if (verbose):
            print("input array sizes:",ys1," x ",xs1)
            print("input array sizes:",ys2," x ",xs2)
            
        # get the median flux in the image, to compare the illumination:
        medflux1 = np.nanmedian(im1)
        medflux2 = np.nanmedian(im2)
        fluxratio= medflux1/medflux2

        if (verbose):
            print('median flux (frome 1):',medflux1)
            print('median flux (frome 2):',medflux2)
            print('flux ratio:',fluxratio)
        

        # only select the right obstype:
        if (obstype1 == 'FLAT'):

            # get the difference and average:
            if (scaleim):
                diff = im1-im2*fluxratio
                av = (im1+im2*fluxratio)/2.0
            else:
                diff = im1-im2
                av = (im1+im2)/2.0


            # calc ranges:
            vmin=0.0
            vmax=np.nanpercentile(im1,99.0)
            dmin=np.nanpercentile(diff,1.0)
            dmax=np.nanpercentile(diff,99.0)
            
            # plot the data first:
            if (plot):
                plt.figure(1)
                plt.subplot(1,4,1)
                plt.imshow(im1,vmin=vmin,vmax=vmax,origin='lower',interpolation='nearest')
                plt.title('image 1')
                plt.colorbar()
                plt.subplot(1,4,2)
                plt.imshow(im2,vmin=vmin,vmax=vmax,origin='lower',interpolation='nearest')
                plt.title('image 2')
                plt.colorbar()
                plt.subplot(1,4,3)
                plt.imshow(diff,vmin=dmin,vmax=dmax,origin='lower',interpolation='nearest')
                plt.title('difference')
                plt.colorbar()
                plt.subplot(1,4,4)
                plt.imshow(av,vmin=vmin,vmax=vmax,origin='lower',interpolation='nearest')
                plt.title('average')
                plt.colorbar()
            

            var = np.zeros(1000)
            flux = np.zeros(1000)
            # loop along the band and calculate stats in 100 pixel blocks:
            nbin=0
            xsize =100
            for ix in range(0,2000,xsize):
                x1 = ix+xsize
                x2 = x1 + xsize
                (mean_av,rms,nc) = clipped_mean_rms(np.ravel(av[y1:y2,x1:x2]),5.0)
                (meand,rmsd,ncd) = clipped_mean_rms(np.ravel(diff[y1:y2,x1:x2]),5.0)
                var[nbin] = rmsd*rmsd/2.0
                flux[nbin] = mean_av
                nbin=nbin+1

            # plot flux vs variance:
            plt.figure(2)
            plt.plot(flux[0:nbin],var[0:nbin],'o')
            plt.xlim(xmin=0.0,xmax=np.max(flux[0:nbin])*1.05)
            plt.ylim(ymin=0.0,ymax=np.max(var[0:nbin])*1.05)
            plt.xlabel('Counts (ADU)')
            plt.ylabel('Variance')

            slope, intercept, r_value, p_value, std_err = linregress(flux[0:nbin],var[0:nbin])

            print('best fit slope and intercept:',slope, intercept)
            xx = np.arange(0,100000)
            yy = xx*slope+intercept
            plt.plot(xx,yy)
            # try plotting an example with the nominal gain:
            yy2 = xx/gain1 + intercept
            plt.plot(xx,yy2)
            print('FITS header values:')
            print('Gain:',gain1,' Readout amplifier (inverse) gain (e-/ADU) from header')
            print('Read noise:',ron1,' Readout noise (electrons)')
            print('Measured gain (e-/ADU):',1.0/slope)
            print('measured ron (e-):',np.sqrt(intercept)/slope)

            gain_measured[ngood] = 1/slope
            gain_fromfile[ngood] = gain1
            ngood = ngood+1


    # finally calculate mean/median values and make some sumary plots:
    (mean_gain,rms_gain,nc) = clipped_mean_rms(np.ravel(gain_measured[0:ngood]),5.0,verbose=True)
    # get median gain:
    med_gain = np.nanmedian(np.ravel(gain_measured[0:ngood]))

    if (verbose):
        print('mean,rms measured gain (e-/ADU:',mean_gain,rms_gain,nc)
        print('Median measured gain (e-/ADU):',med_gain)
        
    if (plot):
        plt.figure(3)
        xax = np.arange(ngood)
        plt.plot(xax,gain_measured[0:ngood],'-',color='b',label='Measured gain')
        plt.axhline(mean_gain,color='red',linestyle='--',label='Mean measured gain')
        plt.axhline(med_gain,color='green',linestyle='--',label='Median measured gain')
        plt.axhline(gain1,color='k',linestyle=':',label='Nominal gain')
        plt.xlabel('Frame number')
        plt.ylabel('Gain (e-/ADU)')
        plt.legend(prop={'size':10})

        
    
    return


##########################################################################
# plot twilight sky spectrum (or other) for comparison of wavelength
# cutoff

# sami_tools_smc.dr_tools.hector_data_checks.compare_dichroic('ccd_1/02dec10021red.fits')
#
def compare_dichroic(s1file,fibno=200,av=True,med=True,percentile=True):

    

   # plotting font setup etc:
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='sans-serif')
    #plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'lines.linewidth': 1})

    py.rc('text', usetex=True)
    py.rcParams.update({'font.size': 14})
    #py.rcParams.update({'lines.linewidth': 1})
    py.rcParams.update({'figure.autolayout': True})
    # this to get sans-serif latex maths:
    py.rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmathfonts}',  # load up the sansmath so that math -> helvet
       #r'\sansmathfonts'               # <- tricky! -- gotta actually tell tex to use!
        ]  

    # generate the other file names:
    s2file = s1file.replace('100','200') 
    s3file = s1file.replace('100','300') 
    s4file = s1file.replace('100','400') 
    
    
    # read in files:
    hdulist1 = fits.open(s1file)
    ims1 = hdulist1[0].data
    ny1,nx1 = np.shape(ims1)
    lam1 = get_lam_axis(hdulist1)
    header1 = hdulist1[0].header
    gain1 = header1['RO_GAIN']
    cdelt1 = header1['CDELT1']
    exp1 = header1['EXPOSED']
    date1 = header1['UTDATE']
    print(s1file,' ',gain1,cdelt1,exp1,date1)
    
    hdulist2 = fits.open(s2file)
    ims2 = hdulist2[0].data
    ny2,nx2 = np.shape(ims2)
    lam2 = get_lam_axis(hdulist2)
    header2 = hdulist2[0].header
    gain2 = header2['RO_GAIN']
    cdelt2 = header2['CDELT1']
    exp2 = header2['EXPOSED']
    date2 = header2['UTDATE']
    print(s2file,' ',gain2,cdelt2,exp2,date2)

    # get fibre types.  Assume these are the same for CCD1 and 2:
    fib_tab_hdu1=find_fibre_table(hdulist1)
    table_data1 = hdulist1[fib_tab_hdu1].data
    types12=table_data1.field('TYPE')

    
    hdulist3 = fits.open(s3file)
    ims3 = hdulist3[0].data
    ny3,nx3 = np.shape(ims3)
    lam3 = get_lam_axis(hdulist3)
    header3 = hdulist3[0].header
    gain3 = header3['RO_GAIN']
    cdelt3 = header3['CDELT1']
    exp3 = header3['EXPOSED']
    date3 = header3['UTDATE']
    print(s3file,' ',gain3,cdelt3,exp3,date3)
    
    hdulist4 = fits.open(s4file)
    ims4 = hdulist4[0].data
    ny4,nx4 = np.shape(ims4)
    lam4 = get_lam_axis(hdulist4)
    header4 = hdulist4[0].header
    gain4 = header4['RO_GAIN']
    cdelt4 = header4['CDELT1']
    exp4 = header4['EXPOSED']
    date4 = header4['UTDATE']
    print(s4file,' ',gain4,cdelt4,exp4,date4)

    # get fibre types.  Assume these are the same for CCD3 and 4:
    fib_tab_hdu3=find_fibre_table(hdulist3)
    table_data3 = hdulist3[fib_tab_hdu3].data
    types34=table_data3.field('TYPE')
    
    # get the spectrum from the chosen fibre:
    if (av):
        # if doing the median/mean, first set all unused fibres to NaN so they
        # do not bias the measurements:
        nfib12 = np.size(types12)
        print('Number of fibres in CCD1/2:',nfib12)
        print(np.shape(ims1))
        for i in range(nfib12):
            if ((types12[i] != 'P') & (types12[i] !='S')):
                ims1[i,:] = np.nan
                ims2[i,:] = np.nan
                
        nfib34 = np.size(types34)
        print('Number of fibres in CCD3/4:',nfib34)
        print(np.shape(ims3))
        for i in range(nfib34):
            if ((types34[i] != 'P') & (types34[i] !='S')):
                ims3[i,:] = np.nan
                ims4[i,:] = np.nan

        
        print('Calculating mean spectrum over fibres...')
        if (med): 
            spec1 = np.nanmedian(ims1,axis=0)*gain1/(cdelt1*exp1)
            spec2 = np.nanmedian(ims2,axis=0)*gain2/(cdelt2*exp2)
            spec3 = np.nanmedian(ims3,axis=0)*gain3/(cdelt3*exp3)
            spec4 = np.nanmedian(ims4,axis=0)*gain4/(cdelt4*exp4)
            #spec1 = np.nanpercentile(ims1,90.0,axis=0)*gain1/(cdelt1*exp1)
            #spec2 = np.nanpercentile(ims2,90.0,axis=0)*gain2/(cdelt2*exp2)
            #spec3 = np.nanpercentile(ims3,90.0,axis=0)*gain3/(cdelt3*exp3)
            #spec4 = np.nanpercentile(ims4,90.0,axis=0)*gain4/(cdelt4*exp4)
            if (percentile):
                spec1p = np.nanpercentile(ims1,75.0,axis=0)*gain1/(cdelt1*exp1)
                spec2p = np.nanpercentile(ims2,75.0,axis=0)*gain2/(cdelt2*exp2)
                spec3p = np.nanpercentile(ims3,75.0,axis=0)*gain3/(cdelt3*exp3)
                spec4p = np.nanpercentile(ims4,75.0,axis=0)*gain4/(cdelt4*exp4)
                spec1m = np.nanpercentile(ims1,25.0,axis=0)*gain1/(cdelt1*exp1)
                spec2m = np.nanpercentile(ims2,25.0,axis=0)*gain2/(cdelt2*exp2)
                spec3m = np.nanpercentile(ims3,25.0,axis=0)*gain3/(cdelt3*exp3)
                spec4m = np.nanpercentile(ims4,25.0,axis=0)*gain4/(cdelt4*exp4)
        else:
            spec1 = np.nanmean(ims1,axis=0)*gain1/(cdelt1*exp1)
            spec2 = np.nanmean(ims2,axis=0)*gain2/(cdelt2*exp2)
            spec3 = np.nanmean(ims3,axis=0)*gain3/(cdelt3*exp3)
            spec4 = np.nanmean(ims4,axis=0)*gain4/(cdelt4*exp4)

            
    else:
        spec1 = ims1[fibno,:]*gain1/(cdelt1*exp1)
        spec2 = ims2[fibno,:]*gain2/(cdelt2*exp2)
        spec3 = ims3[fibno,:]*gain3/(cdelt3*exp3)
        spec4 = ims4[fibno,:]*gain4/(cdelt4*exp4)


    # do plots:
    fig1 = plt.figure()

    ax1 = fig1.add_subplot(1,1,1)

    ax1.plot(lam1,spec1,color='b',label='AAOmega blue')
    ax1.plot(lam2,spec2,color='r',label='AAOmega red')
    
    ax1.plot(lam3,spec3,color='c',label='Spector blue')
    ax1.plot(lam4,spec4,color='m',label='Spector red')

    if (percentile):
        ax1.plot(lam1,spec1p,color='b',alpha=0.5)
        ax1.plot(lam2,spec2p,color='r',alpha=0.5)
    
        ax1.plot(lam3,spec3p,color='c',alpha=0.5)
        ax1.plot(lam4,spec4p,color='m',alpha=0.5)
        
        ax1.plot(lam1,spec1m,color='b',alpha=0.5)
        ax1.plot(lam2,spec2m,color='r',alpha=0.5)
    
        ax1.plot(lam3,spec3m,color='c',alpha=0.5)
        ax1.plot(lam4,spec4m,color='m',alpha=0.5)

    
    ax1.axhline(0.0,color='k',linestyle=':')

    ax1.legend(prop={'size':12})

    label = basename(s1file)+'  '+basename(s2file)+'  '+basename(s3file)+'  '+basename(s4file)
    #ax1.set(xlabel='Wavelength (Angstroms)',ylabel='counts',title=label)
    ax1.set(xlabel='Wavelength (Angstroms)',ylabel='counts')

    py.savefig('hector_response_twilight.png', format='png',dpi=300)      
    
    # plot scaled versions:
    fig2 = plt.figure()

    ax2 = fig2.add_subplot(1,1,1)

    idx1 = np.where((lam1>4900) & (lam1<5100))
    med1 = np.nanmedian(spec1[idx1])
    idx2 = np.where((lam2>6600) & (lam2<6700))
    med2 = np.nanmedian(spec2[idx2])
    idx3 = np.where((lam3>4900) & (lam3<5100))
    med3 = np.nanmedian(spec3[idx3])
    idx4 = np.where((lam4>6600) & (lam4<6700))
    med4 = np.nanmedian(spec4[idx4])
    ax2.plot(lam1,spec1/med1,color='b',label='AAOmega blue')
    ax2.plot(lam2,spec2/med2,color='r',label='AAOmega red')

    # output some stats:
    print('Relative througput spector/AAOmega:')
    print('4900-5100A:',med3/med1)
    print('6600-6700A:',med4/med2)

    
    ax2.plot(lam3,1.2*spec3/med3,color='c',label='Spector blue')
    ax2.plot(lam4,spec4/med4,color='m',label='Spector red')

    ax2.axhline(0.0,color='k',linestyle=':')

    ax2.legend(prop={'size':12})
    ax2.set(xlabel='Wavelength (Angstroms)',ylabel='normalized counts',title=label)
    

##########################################################################
# get x,y,lam data from a combination of the tlm and the arcfits.dat file
# this is generated so that better optical models can be generated.
#
def get_xylam(tlmfile,arcfitsfile):

    # first read the tlm file:
    hdulist = fits.open(tlmfile)
    tlm = hdulist[0].data
    (ny,nx) = np.shape(tlm)
    print('size of TLM file: (ny,nx)=',ny,nx)

    # open output file:
    outfile = 'xylam.dat'
    fout = open(outfile,'w')
    writer = csv.writer(fout)
    
    fin = open(arcfitsfile, 'r')
    # read arcfits.dat file:
    while True:
        # get the header lines:
        header1 = fin.readline()
        header2 = fin.readline()
        header3 = fin.readline()
        if not header1:
            break
        fibno = int(header1.split()[2])
        nl = header2.split()[3]
        print('Fibre ',fibno)
        print('Number of lines ',nl)
        
        # read the next nl lines:
        lines = []
        for i in range(int(nl)):
            lines.append(fin.readline().strip())

        #I LNECHAN INTEN LNEWID CHANS WAVES FIT DEV, ORIGPIX, ORIGWAVE
        i, lnechan,inten,lnewid,chans,waves,fit,dev,origpix,origwave = np.genfromtxt(lines,unpack=True)   

        for i in range(int(nl)):
            # origpix is the x value.  
            xval = origpix[i]
            # origwave is the wavelenfgth value:
            lamval = origwave[i]
            # get the y value ftom the tlm file for this fibre:
            ix = round(xval)
            yval = tlm[fibno-1,ix-1]

            writer.writerow([xval,yval,lamval,fibno])

    fout.close()
    
    return

##########################################################################
# plot reference spectrum and lines from 2dfdr (REF.DAT REFSPEC.DAT)
#
def plot_refspec(specfile,linefile,shift=0):

    specdata = np.loadtxt(specfile)
    # Split the data into two arrays
    lam = specdata[:,0]
    flux = specdata[:,1]
    npix = np.size(lam)
    print('number of pixels:',npix)

    linedata = np.loadtxt(linefile)
    # Split the data into two arrays
    linepix = linedata[:,0]
    lineinten = linedata[:,1]
    linetruelam = linedata[:,2]
    nline = np.size(linepix)
    print('number of lines:',nline)
    
    
    # note the line positions are the positions in pixels
    # of the scrunched arc (based on the nominal wavelength
    # solution from the optics model).  We need to map from
    # linepix to linelam
    pixaxis = np.arange(npix)+0.5
    linelam = np.interp(linepix,pixaxis,lam)
    
    
    # plot
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(lam,flux,color='b')
    ax1.set(xlim=[lam[0],lam[-1]])
    for i in range(nline):
        if (linetruelam[i] > 0):
            color = 'g'
        else:
            color = 'r'
                
        ax1.axvline(linelam[i],color=color,linestyle=':')

    
    

##########################################################################
# plot results of 2dfdr arc fitting.  specfile should be the ex.fits file
# if used.
#
def plot_arcfits(infile,specfile=None,fib=-1):

    # set up plots:
    fig1 = plt.figure()
    fig2 = plt.figure()

    #
    if (specfile):
        hdulist = fits.open(specfile)
        im = hdulist[0].data
        ny,nx = np.shape(im)
        print('image size:',ny,nx)
        flux = im[fib-1,:]
        pax = np.arange(0,nx)+1
 
        #ax4.plot(pax,flux)

        
    
    with open(infile, 'r') as f:
        while True:
            # get the header lines:
            header1 = f.readline()
            header2 = f.readline()
            header3 = f.readline()
            print(header1)
            print(header2)
            print(header3)
            fibno = header1.split()[2]
            nl = header2.split()[3]
            print('Fibre ',fibno)
            print('Number of lines ',nl)
        
        # read the next nl lines:
            lines = []
            for i in range(int(nl)):
                lines.append(f.readline().strip())
            #I LNECHAN INTEN LNEWID CHANS WAVES FIT DEV
            i, lnechan,inten,lnewid,chans,waves,fit,dev,origpix,origwave = np.genfromtxt(lines,unpack=True)   

            # if not the right fibre, skip:
            if ((fib > 0) and (fib > int(fibno))):
                print('fibre not chosen:',fib,fibno)
                continue

            for i in range(int(nl)):
                print(lnechan[i],inten[i],lnewid[i],chans[i],waves[i],fit[i],dev[i],origpix[i],origwave[i])
                
            

            ifibno = int(fibno)
                
            linten = np.log10(inten)
            imin = np.nanmin(linten)
            imax = np.nanmax(linten)

            # get points with width > 5
            iwide = np.where(lnewid > 5)
            
            # calculate RMS:
            rms = np.sqrt(np.sum(dev**2)/float(nl))
            print('RMS on fit:',rms)

            fig1.clf()
            fig2.clf()
            label = 'fibre: {0:3d}'.format(int(fibno))
            ax1 = fig1.add_subplot(4,1,1)
            ax2 = fig1.add_subplot(4,1,2)
            ax3 = fig1.add_subplot(4,1,3)
            ax4 = fig1.add_subplot(4,1,4)

            ax5 = fig2.add_subplot(1,1,1)
            
            if (specfile):
                ax4.set(xlabel='Pixel',ylabel='Counts',title='Uncalibrated spectrum')
                ax5.set(xlabel='Pixel',ylabel='Counts',title='Uncalibrated spectrum: '+specfile)
                for i in range(int(nl)):
                    ax4.plot(pax,im[ifibno-1,:],color='b')
                    ax4.axvline(origpix[i],color='r',linestyle=':')
                    # get ymin/max:
                    (ymin,ymax) = ax4.get_ylim()
                    lamlabel = "{:7.2f}".format(origwave[i])
                    ax4.text(origpix[i],ymax-0.3*(ymax-ymin),lamlabel,rotation=90.0,fontsize=10.0)
                    
                    ax5.plot(pax,im[ifibno-1,:],color='b')
                    ax5.axvline(origpix[i],color='r',linestyle=':')
                    # get ymin/max:
                    (ymin,ymax) = ax5.get_ylim()
                    ax5.text(origpix[i],ymax-0.3*(ymax-ymin),lamlabel,rotation=90.0,fontsize=10.0)
            
            cax1 = ax1.scatter(waves,chans,c=linten,marker='o',cmap=plt.cm.RdYlBu,vmin=imin,vmax=imax)
            ax1.plot(waves,fit,'-')
            cbar1 = fig1.colorbar(cax1,ax=ax1)
            ax1.set(title=label)

            cax2 = ax2.scatter(waves,dev,c=linten,marker='o',cmap=plt.cm.RdYlBu,vmin=imin,vmax=imax)
            ax2.scatter(waves[iwide],dev[iwide],c=linten[iwide],marker='o',edgecolor='k',cmap=plt.cm.RdYlBu,vmin=imin,vmax=imax)
            cbar2 = fig1.colorbar(cax2,ax=ax2)
            ax2.axhline(0.0,color='k',linestyle='--')
            ax2.set(xlabel='True Wavelength - pixel 1 (Angstrom)',ylabel='True-fit (Angstroms)')
            label = 'RMS={0:5.4f}'.format(rms)
            ax2.text(0.8, 0.9,label,horizontalalignment='center',verticalalignment='center', transform=ax2.transAxes)

            cax3 = ax3.scatter(waves,waves-chans,c=linten,marker='o',cmap=plt.cm.RdYlBu,vmin=imin,vmax=imax)
            ax3.scatter(waves[iwide],waves[iwide]-chans[iwide],c=linten[iwide],marker='o',edgecolor='k',cmap=plt.cm.RdYlBu,vmin=imin,vmax=imax)
            ax3.plot(waves,fit-chans,'-')
            cbar3 = fig1.colorbar(cax3,ax=ax3)
            #ax3.axhline(0.0,color='k',linestyle='--')
            ax3.set(xlabel='True Wavelength - pixel 1 (Angstrom)',ylabel='True-measured')
            #label = 'RMS={0:5.4f}'.format(rms)
            #ax3.text(0.8, 0.9,label,horizontalalignment='center',verticalalignment='center', transform=ax2.transAxes)
            
            plt.draw()
            plt.pause(0.0001)
            #fig1.show()
            #fig1.draw()
            # pause and wait for input:
            yn = input('Continue?')
            
            
    return


##########################################################################
# compare vertical cuts:

def compare_vert_cut(frame1,frame2,col=1500,offset=65,scale=0.15):
    """
    plot a vertical cut of two frames
    """

    hdu1 = fits.open(frame1)
    hdu2 = fits.open(frame2)

    # read the data
    im1 = hdu1[0].data
    im2 = hdu2[0].data

    # take vertical cut:
    col1 = im1[:,col] 
    col2 = im2[:,col] 

    nx = np.size(col1)
    x = np.arange(nx)

    # make plot:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    
    ax1.plot(x,col1)
    ax1.plot(x+offset,col2*scale)

    title = frame1+' '+frame2+', col '+str(col)
    title = title.replace('_', '\_')
    ax1.set(title=title)
    
    return

##########################################################################
#  Read in binary tables and check what the fibres are flagged as
#
def check_fib_types(infile):

    # read in original image:
    hdulist = fits.open(infile)

    # find the fibre table:
    try: 
        fib_tab_hdu=find_fibre_table(hdulist)
        table_data = hdulist[fib_tab_hdu].data
        types=table_data.field('TYPE')

    except KeyError:
        print('No fibre table found')

    ntype = np.size(types)
        
    # 
    print('P fibres:')
    nnp=0
    for i in range(ntype):
        if (types[i] == 'P'):
            #print(i+1,types[i])
            nnp=nnp+1

    print('Number of Ps: ',nnp)

    print('S fibres:')
    ns=0
    for i in range(ntype):
        if (types[i] == 'S'):
            #print(i+1,types[i])
            ns=ns+1

    print('Number of Ss: ',ns)

    # update the fibre types:
    print('N fibres:')
    nn=0
    for i in range(ntype):
        if (types[i] == 'N'):
            print(i+1,types[i])
            nn=nn+1

    print('Number of Ns: ',nn)

    # update the fibre types:
    print('U fibres:')
    nu=0
    for i in range(ntype):
        if (types[i] == 'U'):
            print(i+1,types[i])
            nu=nu+1

    print('Number of Ns: ',nu)

    # update the fibre types:
    print('D fibres:')
    nd=0
    for i in range(ntype):
        if (types[i] == 'D'):
            print(i+1,types[i])
            nd=nd+1

    print('Number of Ds: ',nd)

    print('total of all types:',nnp+ns+nu+nn+nd)
    print('total of S+P:',nnp+ns)
    
    
    # close file:
    hdulist.close()


    return

####################################################################
# flip fibres table:

def flip_fib_table(infile,inplace=False):

    print('NOTE: only flipping TYPE and NAME columns')
    # copy to output file:
    if (not inplace):
        outfile = infile.replace('.fits','flip.fits')
        copyfile(infile,outfile)
        # open output file and make changes:
        hdulist = fits.open(outfile,mode='update')
        print('writing flipped data to: ',outfile) 
    else:
        hdulist = fits.open(infile,mode='update')
        print('Doing flip in place') 

    # find the fibre table:
    try: 
        fib_tab_hdu=find_fibre_table(hdulist)
        table_data = hdulist[fib_tab_hdu].data
        types=table_data.field('TYPE')
        names=table_data.field('NAME')
        spaxid=table_data.field('SPAX_ID')

        # make flipped copies:
        ttypes = np.flip(types)
        tnames = np.flip(names)
        tspaxid = np.flip(spaxid)
        
        # copy back in:
        table_data['TYPE'][:] = ttypes
        table_data['NAME'][:] = tnames
        table_data['SPAX_ID'][:] = tspaxid
        
    except KeyError:
        print('No fibre table found')


    # close file:
    hdulist.close()


    return

####################################################################
# flip fibres table:

def flip_image(infiles,inplace=False):

    files = sorted(glob.glob(infiles))

    # loop over files:
    for infile in files:
    
        # copy to output file:
        if (not inplace):
            outfile = infile.replace('.fits','flip.fits')
            copyfile(infile,outfile)
            # open output file and make changes:
            hdulist = fits.open(outfile,mode='update')
            print('writing flipped data to: ',outfile) 
        else:
            hdulist = fits.open(infile,mode='update')
            print('Doing flip in place') 

        # read in image:
        im = hdulist[0].data

        # flip:
        tim = np.flip(im,axis=0)
        
        # copy back in:
        hdulist[0].data = tim

        # close file:
        hdulist.close()

        

    return

####################################################################
# flip fibres table:

def change_fib_type(infiles,fib,newtype):

    files = sorted(glob.glob(infiles))

    # loop over files:
    for infile in files:
    
        hdulist = fits.open(infile,mode='update')

        try: 
            fib_tab_hdu=find_fibre_table(hdulist)
            table_data = hdulist[fib_tab_hdu].data
            types=table_data.field('TYPE')
            names=table_data.field('NAME')

            print(names[fib-1],types[fib-1],' -> ',newtype)

            # copy back in:
            table_data['TYPE'][fib-1] = newtype
        
        except KeyError:
            print('No fibre table found')

        hdulist.close()
        

    return


####################################################################
# change ndf frame class:

def change_frame_class(infiles,newclass):

    files = sorted(glob.glob(infiles))

    # loop over files:
    for infile in files:
    
        hdulist = fits.open(infile,mode='update')

        try:
            oldclass = hdulist[0].header['NDFCLASS']
            print(oldclass,' -> ', newclass)
            hdulist[0].header['NDFCLASS'] = newclass

            hdulist.close()
            
        except KeyError:
            print('No NDFCLASS heacder found')

        hdulist.close()
        

    return


####################################################################




