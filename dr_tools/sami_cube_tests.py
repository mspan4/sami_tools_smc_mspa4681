
import numpy as np
import glob
import astropy.io.fits as pf
import os
import pylab as py
import scipy as sp
import scipy.ndimage as nd
import scipy.ndimage.filters as filters
from matplotlib.backends.backend_pdf import PdfPages

import .sami_utils
import .sami_stats_utils

##############################################################################
# script to test the variance in a cube by comparing it to the measured
# scatter in the spectrum.  Assumes the spectrum is relatively smooth
#
def test_cube_var(cubefile,doplot=True,verbose=True,xc=25,yc=25,scalevar=1.0,usemax=True):

    hdulist = pf.open(cubefile)
    primary_header=hdulist['PRIMARY'].header
    crval3=primary_header['CRVAL3']
    cdelt3=primary_header['CDELT3']
    crpix3=primary_header['CRPIX3']
    naxis3=primary_header['NAXIS3']
    x=np.arange(naxis3)+1
    L0=crval3-crpix3*cdelt3 #Lc-pix*dL        
    lam=L0+x*cdelt3

    flux = hdulist[0].data
    variance = hdulist['VARIANCE'].data
    # get sigma, but also include the possibility of scaling the variance by
    # some amount:
    sigma = np.sqrt(variance/scalevar)
    # set any zero sigmas to nan:
    sigma[sigma==0.0] = np.nan
    hdulist.close()

    (zs,ys,xs) = flux.shape
    if (verbose):
        print('cube shape: ',zs,ys,xs)

    # get summed flux spectrum in cube:
    flux_sum = np.nansum(flux,axis=(1,2))
        
        
    #yc = int(ys/2)
    #xc = int(xs/2)
    # median filter the data in the spectral direction to get a smooth
    # version with no scatter:
    flux_med = sami_dr_smc.sami_stats_utils.median_filter_nan(flux,(51,1,1))
    
    # at a collapsed 2d image:
    im_med = np.nanmean(flux_med,axis=0)
    if (verbose):
        print('collapsed image size: ',np.shape(im_med))

    # find where the max in the image is:
    print('max image value:',np.nanmax(im_med))
    if (usemax):
        indexes = np.where(im_med == np.nanmax(im_med))
        yc = indexes[0]
        xc = indexes[1]
     
    # plot the central spectrum as an example:
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(2,2,1)
    ax1.plot(lam,flux[:,yc,xc],label='spectrum')
    ax1.plot(lam,flux_med[:,yc,xc],label='median filtered')
    ax1.plot(lam,sigma[:,yc,xc],label='sqrt(var)')
    if (usemax):
        title_string = 'brightest spaxel x={0:2d} y={1:2d}'.format(int(xc),int(yc))
    else:
        title_string = 'spaxel x={0:2d} y={1:2d}'.format(xc,yc)
        
    ax1.set(xlabel='wavelength',ylabel='flux',title=title_string)
    py.legend(prop={'size':8})
    
    # estimate the pull dist for all the pixels:
    pull = (flux-flux_med)/sigma

    # get the mean and rms of the pull for the central pixel:
    print(pull[:,yc,xc])
    (mean,rms,nc) = sami_dr_smc.sami_stats_utils.clipped_mean_rms(pull[:,yc,xc],5.0,verbose=True)
    if (verbose):
        print('mean and rms of pull:',mean,rms)
        print('number of pixels used:',nc)
     
    
    ax2 = fig1.add_subplot(2,2,2)
    ax2.hist(pull[:,yc,xc],bins=100,range=(-5.0,5.0),histtype='step',density=True,label='pull dist')
    ax2.set(xlabel='pull',ylabel='number',title=title_string+' pull',xlim=[-3.0,3.0])

    # plot gaussian:
    xx = np.linspace(-5.0,5.0,100)
    gg = sami_dr_smc.sami_stats_utils.gaussian(xx,1.0,mean,rms)/np.sqrt(2.0*np.pi*rms*rms)
    gg1 = sami_dr_smc.sami_stats_utils.gaussian(xx,1.0,mean,1.0)/np.sqrt(2.0*np.pi*rms*rms)
    ax2.plot(xx,gg,label='Gaussian sigma={0:6.3f}'.format(rms))
    ax2.plot(xx,gg1,label='Gaussian sigma=1')
    py.legend(prop={'size':8})

    # generate an image of the pull across the cube:
    im_pull = np.zeros((ys,xs))
    im_med_scatter = np.zeros((ys,xs))
    im_med_stdev = np.zeros((ys,xs))
    im_pull.fill(np.nan)
    im_med_scatter.fill(np.nan)
    im_med_stdev.fill(np.nan)
    for ix in range(xs):
        for iy in range(ys):
            if (np.isfinite(im_med[iy,ix])):
                (mean,rms,nc) = sami_dr_smc.sami_stats_utils.clipped_mean_rms(pull[:,iy,ix],5.0,verbose=False)
                if (np.isfinite(mean)):
                    im_pull[iy,ix] = rms

                # calculate the median scatter and median sqrt(var) to compare to median flux:
                im_med_scatter[iy,ix] = np.nanmedian(np.abs(flux[:,iy,ix]-flux_med[:,iy,ix]))
                im_med_stdev[iy,ix] = np.nanmedian(sigma[:,iy,ix])

                    

    ax3 = fig1.add_subplot(2,2,3)
    cax3 = ax3.imshow(im_pull,origin='lower',interpolation='nearest',cmap=py.cm.gray)
    cbar = fig1.colorbar(cax3)
    ax3.set(xlabel='x',ylabel='y')
    ax3.text(0.5, 0.9, 'rms of pull per spaxel',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax3.transAxes)
    ax3.plot(xc,yc,'o',color='r')

    # histogram of spaxel pulls:
    
    ax4 = fig1.add_subplot(2,2,4)
    ax4.hist(np.ravel(im_pull),bins=100,range=(0.0,2.0),histtype='step')
    ax4.set(xlabel='pull rms per spaxel',ylabel='number')


    # now make some plots that look at S/N based on both the variance and scatter:
    fig2 = py.figure(2)
    ax5 = fig2.add_subplot(2,3,1)
    cax5 = ax5.imshow(im_med,origin='lower',interpolation='nearest',cmap=py.cm.gray)
    cbar5 = fig2.colorbar(cax5)
    ax5.set(title='median flux')

    ax6 = fig2.add_subplot(2,3,2)
    cax6 = ax6.imshow(im_med_scatter,origin='lower',interpolation='nearest',cmap=py.cm.gray)
    cbar6 = fig2.colorbar(cax6)
    ax6.set(title='median scatter')
    
    ax7 = fig2.add_subplot(2,3,3)
    cax7 = ax7.imshow(im_med/im_med_scatter,origin='lower',interpolation='nearest',cmap=py.cm.gray)
    cbar7 = fig2.colorbar(cax7)
    ax7.set(title='median flux/median scatter')

    ax8 = fig2.add_subplot(2,3,4)
    cax8 = ax8.imshow(im_med/im_med_stdev,origin='lower',interpolation='nearest',cmap=py.cm.gray)
    cbar8 = fig2.colorbar(cax8)
    ax8.set(title='median flux/median stdev')

    ax9 = fig2.add_subplot(2,3,5)
    maxs = np.nanmax(im_med_stdev)
    ax9.hist(np.ravel(im_med_scatter),bins=50,range=(0.0,maxs),histtype='step')
    ax9.set(title='median scatter')

    ax10 = fig2.add_subplot(2,3,6)
    ax10.hist(np.ravel(im_med_stdev),bins=50,range=(0.0,maxs),histtype='step')
    ax10.set(title='median stdev')

    # output some useful values:
    #
    med_scatter = np.nanmedian(im_med_scatter)
    med_stdev = np.nanmedian(im_med_stdev)
    print('median value (across spaxels) of median scatter:',med_scatter) 
    print('median value (across spaxels) of median stdev:',med_stdev) 
    print('max value (across spaxels) of median scatter:',np.nanmax(im_med_scatter)) 
    print('max value (across spaxels) of median stdev:',np.nanmax(im_med_stdev))
    print('median pull value: ',np.nanmedian(im_pull))


    fig3 = py.figure(3)
    ax11 = fig3.add_subplot(2,2,1)
    maxs = np.nanmax(im_med_stdev)
    ax11.hist(np.ravel(im_med_scatter),bins=50,range=(0.0,maxs),histtype='step',label=cubefile)
    ax11.set(title='median scatter')
    py.legend()

    ax12 = fig3.add_subplot(2,2,2)
    ax12.hist(np.ravel(im_med_stdev),bins=50,range=(0.0,maxs),histtype='step')
    ax12.set(title='median stdev')

    ax13 = fig3.add_subplot(2,2,3)
    ax13.hist(np.ravel(im_pull),bins=100,range=(0.0,2.0),histtype='step')
    ax13.set(xlabel='pull rms per spaxel',ylabel='number')

    fig4 = py.figure(4)
    ax14 = fig4.add_subplot(111)
    ax14.plot(lam,flux_sum,label=cubefile)
    py.legend()
    
    
    

#############################################################################
# compare the data in two cubes.  Typically the same data, but reduced
# differently:
#
# examples:
# sami_2dfdr_reduction_tests.comp_cube('current/10000122_blue_7_Y13SAR1_P002_15T006.fits','cubes_Feb/10000122blue_50pix_05arcsec_cube.fits')
# 
#
def comp_cube(cubefile1,cubefile2):

    # read the first cube:
    hdulist = pf.open(cubefile1)
    primary_header=hdulist['PRIMARY'].header
    primary_header=hdulist['PRIMARY'].header
    crval3=primary_header['CRVAL3']
    cdelt3=primary_header['CDELT3']
    crpix3=primary_header['CRPIX3']
    naxis3=primary_header['NAXIS3']
    x=np.arange(naxis3)+1
    L0=crval3-crpix3*cdelt3 #Lc-pix*dL        
    lam1=L0+x*cdelt3

    cube1 = hdulist[0].data
    hdulist.close()

    # read section cube:
    hdulist = pf.open(cubefile2)
    primary_header=hdulist['PRIMARY'].header
    primary_header=hdulist['PRIMARY'].header
    crval3=primary_header['CRVAL3']
    cdelt3=primary_header['CDELT3']
    crpix3=primary_header['CRPIX3']
    naxis3=primary_header['NAXIS3']
    x=np.arange(naxis3)+1
    L0=crval3-crpix3*cdelt3 #Lc-pix*dL        
    lam2=L0+x*cdelt3

    cube2 = hdulist[0].data
    hdulist.close()

    (zs,ys,xs) = cube2.shape
    print('cube 2 shape: ',zs,ys,xs)
    
    ######################
    # next do comparisons:
    # 1) plot collapsed image (median)
    med_image1 = np.nanmedian(cube1,axis=0) 
    med_image2 = np.nanmedian(cube2,axis=0)
    # get the peak index:
    (yp1,xp1) = np.where(med_image1 == np.nanmax(med_image1))
    print(yp1,xp1)
    (yp2,xp2) = np.where(med_image2 == np.nanmax(med_image2))
    print(yp2,xp2)

    py.figure(1)
    py.subplot(2,2,1)
    py.imshow(med_image1,origin='lower',interpolation='nearest',cmap=py.cm.gray)
    py.colorbar()
    py.subplot(2,2,2)
    py.imshow(med_image2,origin='lower',interpolation='nearest',cmap=py.cm.gray)
    py.colorbar()

    # 2) plot cross-section for the median image through the peak:
    # get the peak:
    xax = np.arange(0,xs)
    # plot Y cut:
    py.subplot(2,2,3)
    py.plot(xax,np.ravel(med_image1[yp1-1,:]),color='b')
    py.plot(xax,np.ravel(med_image1[yp1,:]),color='b')
    py.plot(xax,np.ravel(med_image1[yp1+1,:]),color='b')
    py.plot(xax,np.ravel(med_image2[yp2-1,:]),color='r')
    py.plot(xax,np.ravel(med_image2[yp2,:]),color='r')
    py.plot(xax,np.ravel(med_image2[yp2+1,:]),color='r')
    py.xlim(xmin=10.0,xmax=40.0)
    # plot x cut:
    py.subplot(2,2,4)
    py.plot(xax,np.ravel(med_image1[:,xp1-1]),color='b')
    py.plot(xax,np.ravel(med_image1[:,xp1]),color='b')
    py.plot(xax,np.ravel(med_image1[:,xp1+1]),color='b')
    py.plot(xax,np.ravel(med_image2[:,xp2-1]),color='r')
    py.plot(xax,np.ravel(med_image2[:,xp2]),color='r')
    py.plot(xax,np.ravel(med_image2[:,xp2+1]),color='r')
    py.xlim(xmin=10.0,xmax=40.0)

    # plot spectra:
    py.figure(2)
    py.subplot(2,1,1)
    for ix in xrange(xp1-2,xp1+2):
        for iy in xrange(yp1-2,yp1+2):
            medval = np.nanmedian(cube1[:,iy,ix])
            py.plot(lam1,cube1[:,iy,ix]/medval)

    py.subplot(2,1,2)
    for ix in xrange(xp2-2,xp2+2):
        for iy in xrange(yp2-2,yp2+2):
            medval = np.nanmedian(cube2[:,iy,ix])
            py.plot(lam2,cube2[:,iy,ix]/medval)

        
    # plot RGB images of cube:
    # wavelength ranges for bands to use:
    lam_band1 = np.array([3800.0,4600.0,5400.0])
    #lam_band2 = np.array([3801.0,4601.0,5401.0])
    lam_band2 = np.array([4000.0,4800.0,5600.0])
    rgb_im1 = make_rgb_im(cube1,lam1,lam_band1,lam_band2)
    rgb_im2 = make_rgb_im(cube2,lam2,lam_band1,lam_band2)

    py.figure(3)
    py.subplot(1,2,1)
    py.imshow(rgb_im1,origin='lower',interpolation='nearest')
    py.title('RGB, sqrt(flux)')
    py.subplot(1,2,2)
    py.imshow(rgb_im2,origin='lower',interpolation='nearest')
    py.title('RGB, sqrt(flux)')

    # plot original images:
    py.figure(4)
    py.subplot(2,3,1)
    py.imshow(rgb_im1[:,:,0],origin='lower',interpolation='nearest')
    py.colorbar()
    py.subplot(2,3,2)
    py.imshow(rgb_im1[:,:,1],origin='lower',interpolation='nearest')
    py.colorbar()
    py.subplot(2,3,3)
    py.imshow(rgb_im1[:,:,2],origin='lower',interpolation='nearest')
    py.colorbar()

    py.subplot(2,3,4)
    py.imshow(rgb_im2[:,:,0],origin='lower',interpolation='nearest')
    py.colorbar()
    py.subplot(2,3,5)
    py.imshow(rgb_im2[:,:,1],origin='lower',interpolation='nearest')
    py.colorbar()
    py.subplot(2,3,6)
    py.imshow(rgb_im2[:,:,2],origin='lower',interpolation='nearest')
    py.colorbar()

    

##########################################################################
# routine to make a 3 colour image (x,y,3) sized array:

def make_rgb_im(cube,lam,lam_band1,lam_band2):

    # will always be getting 3 bands:
    nbands=3
    
    # first get the cube dimensions:
    (zs,ys,xs) = cube.shape

    # next define the output array:
    im_rgb = np.zeros((ys,xs,3))
    im_3 = np.zeros((ys,xs,3))
    
    # get the correct index values for the different bands.  First
    # define arrays that will hold index values:
    i1 = np.zeros(nbands)
    i2 = np.zeros(nbands)

    # Then loop over wavelength array to find index of bands:
    for j in xrange(nbands):
        for i in xrange(zs):
            if (lam[i] < lam_band1[j]):
                i1[j] = i

        for i in xrange(zs-1,0,-1):
            if (lam[i] > lam_band2[j]):
                i2[j] = i

        print('Band ',j,' wavelength range and pixel range:',lam_band1[j],lam_band2[j],i1[j],i2[j]) 

    # also generate a version of the images that are Gaussian smoothed to reduce noise and
    # also mask out bad pixels in the spectral direction:
    sigma=5.0
    V = cube.copy()
    V[cube!=cube]=0
    V[np.abs(cube)> 100] = 0
    VV = nd.filters.gaussian_filter(V,(sigma,0,0))

    W = 0*cube.copy()+1
    W[cube!=cube] = 0
    W[np.abs(cube)> 100] = 0
    WW = nd.filters.gaussian_filter(W,(sigma,0,0))

    cube_conv = VV/WW
        
    # look over the cube for each pixel and sum the flux within the bands:
    for ix in xrange(xs):
        for iy in xrange(ys):

            for ibin in xrange(nbands):
                im_3[iy,ix,ibin] = np.sum(cube_conv[i1[ibin]:i2[ibin],iy,ix])


    smin = 0.0
    smax = np.nanmax(im_3)

    im_rgb[:,:,0] = sqrt(im_3[:,:,2],scale_min=smin,scale_max=smax)
    im_rgb[:,:,1] = sqrt(im_3[:,:,1],scale_min=smin,scale_max=smax)
    im_rgb[:,:,2] = sqrt(im_3[:,:,0],scale_min=smin,scale_max=smax)
    
    return im_rgb
        

##############################################################################
# script to look at the scatter from spaxel to spaxel in a cube to test the
# impact of drizzling + DAR on the spectral shape.  Should br run on galaxies
# that we expect to have uniform colour and SED (e.g. old passive galaxies).
#
def spaxel_scatter(cube_list,snlim=15,dostar=False,dogal=True):

    """Function to calculate the spaxel-to-spaxel scatter in a SAMI cube.  Should
    be run on a galaxy that is expected to have uniform colour and SED (e.g. an
    old passive galaxy).
    Parameters: snlim - median S/N for spaxels to be included.
                dostar - do we run the test on SAMI stars?
                dogal  - do we run the test on SAMI galaxies?
                (the last two are based on their object names)

    """

    # width of Gaussian filter to use in the spectral direction:
    sigma=10.0
    
    # loop over list of cubes:
    files = glob.glob(cube_list)

    # arrays for holding spectra:
    speclist = np.zeros((2048,5000))
    speclist_bin2 = np.zeros((2048,5000))

    # arrays for holding summed flux:
    col_sum=np.zeros(5000)
    col_sum_bin2=np.zeros(5000)

    # set up a set of arrays that contain info for muliple bands, so that we
    # can use arbitrary sets of bands.  At the moment these are set up as
    # hard coded, and it probably makes sense to keep it like this for now.
    # we could have nbands, lam1 and lam2 as arguments.  This might help when
    # we want to look at the red arm as well.
    #nbands = 2
    #lam1 = (3800.0,5400.0)
    #lam2 = (4000.0,5600.0)
    nbands = 3
    lam1 = np.array([3800.0,4600.0,5400.0])
    lam2 = np.array([4000.0,4800.0,5600.0])
    #lamcent = np.zeros(nbands)
    lamcent = (lam1+lam2)/2.0
    
    bands_sum=np.zeros((5000,nbands))
    bands_sn=np.zeros((5000,nbands))
    bands_sum_var=np.zeros((5000,nbands))
    bands_sum_bin2=np.zeros((5000,nbands))


    # set up counters:
    nspec=0
    nspec_b=0
    ngal = 0

    # define galaxy/cube arrays:
    sigma_gal=np.zeros(np.size(files))
    nspec_gal=np.zeros(np.size(files))
    sigma_gal_bin2=np.zeros(np.size(files))
    nspec_gal_bin2=np.zeros(np.size(files))
    fileused = np.empty(np.size(files),dtype="S50")

    # loop over all cubes, but only using the cubes that
    # according to flags set (e.g. only stars, only gals etc) 
    for cubefile in files: 

        base = os.path.basename(cubefile)
        # check to see if this is a star or a galaxy:
        if (base[0:4] == '1000') or (base[0:3] == '888'):
            sgtype = 1 # star
        else:
            sgtype = 0 # galaxy

        if (sgtype == 1) and (not dostar):
            print('skipping star cube:',cubefile)
            continue

        if (sgtype == 0) and (not dogal):
            print('skipping galaxy cube:',cubefile)
            continue
            
        print('opening ',cubefile)
        #  first open the file   
        hdulist = pf.open(cubefile)
    
        # get cube:
        cube = hdulist[0].data
        # get variance:
        var = hdulist['VARIANCE'].data

        # get cube size:
        (zs, ys, xs) = np.shape(cube)
        print(('cube dimensions:',zs,ys,xs))

        # define array to hold band images.  This should only
        # need to be done once, but reseting this for every cube
        # means that the arrays are re-initialized, so no problems
        # with data left in them from previous cubes:
        bands_im = np.zeros((ys,xs,nbands))
        bands_var = np.zeros((ys,xs,nbands))
        bands_im_rgb = np.zeros((ys,xs,3))

        # get wavelength info:
        primary_header=hdulist['PRIMARY'].header
        crval3=primary_header['CRVAL3']
        cdelt3=primary_header['CDELT3']
        crpix3=primary_header['CRPIX3']
        naxis3=primary_header['NAXIS3']
        x=np.arange(naxis3)+1
        L0=crval3-crpix3*cdelt3 #Lc-pix*dL        
        lam=L0+x*cdelt3

        # get the correct index values for the different bands.  First
        # define arrays that will hold index values:
        i1 = np.zeros(nbands,dtype=np.int32)
        i2 = np.zeros(nbands,dtype=np.int32)

        # Then loop over wavelength array to find index of bands:
        for j in range(nbands):
            for i in range(zs):
                if (lam[i] < lam1[j]):
                    i1[j] = i

            for i in range(zs-1,0,-1):
                if (lam[i] > lam2[j]):
                    i2[j] = i

            print('Band ',j,' wavelength range and pixel range:',lam1[j],lam2[j],i1[j],i2[j]) 

        # Get the median and mean flux and variance in each spaxel:
        medflux_im = np.zeros((ys,xs))
        medvar_im = np.zeros((ys,xs))

        medflux_im = np.nanmedian(cube,axis=0)
        # for the mean flux, take the mean only in the central 1000 pixels of
        # the cube: 
        meanflux_im = np.nanmean(cube[500:1500,:,:],axis=0)
        medvar_im = np.nanmedian(var,axis=0)
        medsn_im = medflux_im/np.sqrt(medvar_im)

        # plot basic figures of flux, variance etc for the images:
        py.figure(1)
        py.subplot(2,2,1)
        py.imshow(medflux_im,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        py.title('Median flux')
        py.colorbar()
        py.subplot(2,2,2)
        py.imshow(np.sqrt(medvar_im),origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        py.title('Median error (sigma)')
        py.colorbar()
        py.subplot(2,2,3)
        py.imshow(medsn_im,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        py.title('Median S/N')
        py.colorbar()
        py.subplot(2,2,4)
        py.imshow(meanflux_im,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
        py.title('Mean flux')
        cur_ax = py.gca()
        ymin,ymax = cur_ax.get_ylim()
        xmin,xmax = cur_ax.get_xlim()
                
        # put small points at location of high S/N spaxels that are going to be used:
        for ix in range(xs):
            for iy in range(ys):
                # now only pick spectra that are above a given S/N and process these:
                if (medsn_im[iy,ix] > snlim):
                    py.plot(ix,iy,'o',color='k',markersize=2)

        # reset axis values, as they can be changed but the plot command just above.
        py.xlim(xmin=xmin,xmax=xmax)           
        py.ylim(ymin=ymin,ymax=ymax)           
        py.colorbar()

        # now generate a cube that is filtered in the spectral direction.  Currently this uses
        # a Gaussian filter, with a sigma of 10 (although can be set above).  The complexity of
        # this is that we need to get rid of NaNs, and this is what the extra steps do in the
        # filtering. 
        V = cube.copy()
        V[cube!=cube]=0
        VV = nd.filters.gaussian_filter(V,(sigma,0,0))

        W = 0*cube.copy()+1
        W[cube!=cube] = 0
        WW = nd.filters.gaussian_filter(W,(sigma,0,0))

        cube_conv = VV/WW
        
        # now generate a VARIANCE cube that is filtered in the spectral direction.
        V = var.copy()
        V[var!=var]=0
        VV = nd.filters.gaussian_filter(V,(sigma,0,0))

        W = 0*var.copy()+1
        W[var!=var] = 0
        WW = nd.filters.gaussian_filter(W,(sigma,0,0))

        cube_conv_var = VV/WW

        # generate a binned version of the spectrally smoothed cube.  This is binned 2x2:
        cube_conv_bin2 = sami_dr_smc.sami_utils.bin_ndarray(cube_conv, new_shape=(int(zs),int(ys/2),int(xs/2)), operation='mean')
        medflux_im_bin2 = np.nanmedian(cube_conv_bin2,axis=0)        
        
        # For each spaxel generate fluxes in the bands and plot the scaled spectrum for good S/N pixels.
        py.figure(2)
        py.subplot(2,1,1)
        py.title(cubefile)
        nstart=nspec
        for ix in range(xs):
            for iy in range(ys):

                # sum flux within bands for all spaxels:
                for ibin in range(nbands):
                    print(iy,ix,ibin,i1[ibin],i2[ibin],iy,ix)
                    bands_im[iy,ix,ibin] = np.sum(cube_conv[i1[ibin]:i2[ibin],iy,ix])
                    bands_var[iy,ix,ibin] = np.sum(cube_conv_var[i1[ibin]:i2[ibin],iy,ix])

                # now only pick spectra that are above a given S/N and process these:
                if (medsn_im[iy,ix] > snlim):

                    # normalize by the median flux, and then plot:
                    speclist[:,nspec] = cube_conv[:,iy,ix]/medflux_im[iy,ix]
                    py.plot(lam,speclist[:,nspec])

                    # sum the flux in specific bands and normalize by the median flux in that spaxel:
                    for ibin in range(nbands):
                        bands_sum[nspec,ibin] = np.sum(cube_conv[i1[ibin]:i2[ibin],iy,ix])/medflux_im[iy,ix]
                        bands_sum_var[nspec,ibin]=np.sum(cube_conv_var[i1[ibin]:i2[ibin],iy,ix])/medflux_im[iy,ix]/medflux_im[iy,ix]
                        bands_sn[nspec,ibin]= bands_sum[nspec,ibin]/np.sqrt(bands_sum_var[nspec,ibin])

                    # define a specific colour that will be used for statistical tests.  Base this on the colour
                    # with the broadest range of wavelength (assuming that the bands are ordered in wavelength): 
                    col_sum[nspec] = bands_sum[nspec,0]/bands_sum[nspec,nbands-1]
                    nspec = nspec +1


        # get average spectrum and rms:
        avspec = np.nanmean(speclist[:,nstart:nspec],axis=1)
        rmsspec = np.nanstd(speclist[:,nstart:nspec],axis=1)

        # divide each spectrum by the average for that galaxy to get the fractional residual:
        for i in range(nstart,nspec):
            speclist[:,i] = speclist[:,i]/avspec
            for ibin in range(nbands):
                bands_sum[i,ibin] = bands_sum[i,ibin]/np.sum(avspec[i1[ibin]:i2[ibin]])

            col_sum[i] = col_sum[i] - np.sum(avspec[i1[0]:i2[0]])/np.sum(avspec[i1[nbands-1]:i2[nbands-1]])
            

        # calculate the number of spectra and rms for just this galaxy:
        nspec_gal[ngal] = nspec-nstart
        sigma_gal[ngal] = np.nanstd(col_sum[nstart:nspec])
        print('RMS and number of spectra for this galaxy:',sigma_gal[ngal],nspec_gal[ngal])

        # plot mean and rms on the spectrum plots, but with an offset to allow us to see it
        # clearly:
        py.plot(lam,avspec+0.5,color='k',label='Average')
        py.plot(lam,avspec+0.5+rmsspec,linestyle=':',color='k',label='RMS')
        py.plot(lam,avspec+0.5-rmsspec,linestyle=':',color='k')
        py.ylim(0,2.5)
        py.xlim(lam[0],lam[zs-1])
        py.xlabel('Wavelength (Angstroms)')
        py.ylabel('Relative flux')
        # add the colour bands used to the spectral plot:
        for ibin in range(nbands):
            py.axvspan(lam1[ibin],lam2[ibin], alpha=0.2, color='red')

        
        py.legend(loc='upper left',prop={'size':10})

        # plot the RMS separately:
        py.subplot(2,1,2)
        py.plot(lam,rmsspec/avspec,label='Fractional RMS')
        py.ylim(ymin=0,ymax=0.5)
        py.xlim(lam[0],lam[zs-1])
        py.xlabel('Wavelength (Angstroms)')
        py.ylabel('RMS')
        py.legend(loc='upper left',prop={'size':10})
        # add the colour bands used to the spectral plot:
        for ibin in range(nbands):
            py.axvspan(lam1[ibin],lam2[ibin], alpha=0.2, color='red')

            
        # plot the spectra again, but this time, for different radii from the centre of the star:
        # First get the centre:
        xcent = 0.0
        ycent = 0.0
        ncent = 0.0
        for ix in range(xs):
            for iy in range(ys):
                if (np.isfinite(medflux_im[iy,ix])):
                    xcent = xcent + float(ix) * medflux_im[iy,ix]
                    ycent = ycent + float(iy) * medflux_im[iy,ix]
                    ncent = ncent + medflux_im[iy,ix]
        xcent = xcent/ncent
        ycent = ycent/ncent
        print('Centroid of star in median image:',xcent,ycent)
        py.figure(1)
        py.subplot(2,2,1)
        cur_ax = py.gca()
        ymin,ymax = cur_ax.get_ylim()
        xmin,xmax = cur_ax.get_xlim()
        py.plot(xcent,ycent,'x',color='k',lw=3)
        py.xlim(xmin=xmin,xmax=xmax)           
        py.ylim(ymin=ymin,ymax=ymax)

        # now we have the centre, actually plot the spectra:
        py.figure(3)
        isubplot_max = 0
        rbin=1.5
        for ix in range(xs):
            for iy in range(ys):

                # calculate the range:
                rad = np.sqrt((ix-xcent)**2 + (iy-ycent)**2)

                # use the radius to select the bin:
                isubplot = int(rad/rbin)+1
                if (medsn_im[iy,ix] > snlim):
                    py.subplot(2,3,isubplot)
                    # now only pick spectra that are above a given S/N and plot these:
                    py.plot(lam,cube_conv[:,iy,ix]/medflux_im[iy,ix])
                    if (isubplot > isubplot_max):
                        isubplot_max = isubplot

        for i in range(isubplot_max):
            py.subplot(2,3,i+1)
            py.plot(lam,avspec,color='k',label='Average',lw=3)
            py.plot(lam,avspec+rmsspec,linestyle=':',color='k',label='RMS')
            py.plot(lam,avspec-rmsspec,linestyle=':',color='k')
            py.ylim(0.5,1.5)
            py.xlim(lam[0],lam[zs-1])
            py.xlabel('Wavelength (Angstroms)')
            py.ylabel('Relative flux')
            py.title('Radius = '+str(float(i)*rbin)+' to '+str(float(i+1)*rbin))
          
        
            
        # plot individual band images.  Do this on a log scale so that
        # we can more clearly see any systematics etc:
        py.figure(4)
        for i in range(nbands):
            py.subplot(2,3,i+1)
            py.imshow(np.log10(bands_im[:,:,i]),origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu)
            py.title('band '+str(i+1)+', log(flux), lam='+str(int(lamcent[i])))
            py.colorbar()
        
        py.subplot(2,3,nbands+1)

        # plot an RGB image from the first 3 medium band images.  Note need to re-order array to that
        # it is RGB rather than BGR.  Also use sqrt(flux) so that the scaling is reasonable without
        # bad values if we have -ve flux using logs:
        smin = 0.0
        smax = np.nanmax(bands_im)
        bands_im_rgb[:,:,0] = sami_dr_smc.sami_utils.sqrt_scale(bands_im[:,:,2],scale_min=smin,scale_max=smax)
        bands_im_rgb[:,:,1] = sami_dr_smc.sami_utils.sqrt_scale(bands_im[:,:,1],scale_min=smin,scale_max=smax)
        bands_im_rgb[:,:,2] = sami_dr_smc.sami_utils.sqrt_scale(bands_im[:,:,0],scale_min=smin,scale_max=smax)
        py.imshow(bands_im_rgb,origin='lower',interpolation='nearest')
        py.title('RGB, sqrt(flux)')
        py.colorbar()

        # plot colour:
        py.subplot(2,3,nbands+2)
        vmed = np.nanmedian(bands_im[:,:,0]/bands_im[:,:,2])
        vhwidth=0.8
        py.imshow(bands_im[:,:,0]/bands_im[:,:,2],origin='lower',interpolation='nearest',vmin=vmed-vhwidth,vmax=vmed+vhwidth,cmap=py.cm.RdYlBu)
        py.title('band1/band3 colour')
        py.colorbar()
        
        py.subplot(2,3,nbands+3)
        vmed = np.nanmedian(bands_im[:,:,0]/bands_im[:,:,1])
        vhwidth=0.8
        py.imshow(bands_im[:,:,0]/bands_im[:,:,1],origin='lower',interpolation='nearest',vmin=vmed-vhwidth,vmax=vmed+vhwidth,cmap=py.cm.RdYlBu)
        py.title('band1/band2 colour')
        py.colorbar()
        

        # finally, plot a colour-colour diagram for the spaxels:
        py.figure(5)
        fbands_im0 = bands_im[:,:,0].flatten()
        fbands_im1 = bands_im[:,:,1].flatten()
        fbands_im2 = bands_im[:,:,2].flatten()
        fmedsn_im = medsn_im.flatten()
        # get flux ratios in terms of magnitudes:
        col1 = -2.5*np.log10(fbands_im0) + 2.5*np.log10(fbands_im1)
        col2 = -2.5*np.log10(fbands_im1) + 2.5*np.log10(fbands_im2)

        # specify index of good spaxels, and then plot them as separate colours:
        good_index = np.where((fmedsn_im > snlim))
        py.plot(col1,col2,'o',color='b')
        py.plot(col1[good_index],col2[good_index],'o',color='r')
        py.xlabel('band 0 - band 1 (mag)')
        py.ylabel('band 1 - band 2 (mag)')
        py.xlim(xmin=-0.5,xmax=3.0)
        py.ylim(ymin=-1.0,ymax=1.5)
        

        # repeat for the binned cube:
        py.figure(6)
        py.subplot(2,1,1)
        nstart=nspec_b
        for ix in range(int(xs/2)):
            for iy in range(int(ys/2)):
                # approx S/N cut, using the nearest spaxel in the original S/N image:
                if (medsn_im[iy*2,ix*2] > snlim):

                    # plot good binned spectra:
                    py.plot(lam,cube_conv_bin2[:,iy,ix]/medflux_im_bin2[iy,ix])
                    
                    speclist_bin2[:,nspec_b] = cube_conv_bin2[:,iy,ix]/medflux_im_bin2[iy,ix]

                    for ibin in range(nbands):
                        bands_sum_bin2[nspec_b,ibin] = np.sum(cube_conv_bin2[i1[ibin]:i2[ibin],iy,ix])/medflux_im_bin2[iy,ix]

                    # define colour to use for stats:
                    col_sum_bin2[nspec_b] = bands_sum_bin2[nspec_b,0]/bands_sum_bin2[nspec_b,nbands-1]

                    nspec_b = nspec_b +1


        # get average spectrum and rms:
        avspec = np.nanmean(speclist_bin2[:,nstart:nspec_b],axis=1)
        rmsspec = np.nanstd(speclist_bin2[:,nstart:nspec_b],axis=1)
        
        # divide each spectrum by the average for that galaxy to get the residual:
        for i in range(nstart,nspec_b):
            speclist_bin2[:,i] = speclist_bin2[:,i]/avspec

            for ibin in range(nbands):
                bands_sum_bin2[i,ibin] = bands_sum_bin2[i,ibin]/np.sum(avspec[i1[ibin]:i2[ibin]])

#            print col_sum_bin2[i],np.sum(avspec[ib1:ib2]),np.sum(avspec[ir1:ir2])
            col_sum_bin2[i] = col_sum_bin2[i] - np.sum(avspec[i1[0]:i2[0]])/np.sum(avspec[i1[1]:i2[1]])
            

        # calculate the number of spectra and rms for just this galaxy:
        nspec_gal_bin2[ngal] = nspec_b-nstart
        sigma_gal_bin2[ngal] = np.nanstd(col_sum_bin2[nstart:nspec_b])
        print('RMS and number of spectra for this galaxy:',sigma_gal_bin2[ngal],nspec_gal_bin2[ngal])

        print(ngal)
        fileused[ngal] = cubefile

        py.plot(lam,avspec+0.5,color='k',label='Average')
        py.plot(lam,avspec+0.5+rmsspec,linestyle=':',color='k',label='RMS')
        py.plot(lam,avspec+0.5-rmsspec,linestyle=':',color='k')
        py.ylim(0,2.5)
        py.xlim(lam[0],lam[zs-1])
        py.xlabel('Wavelength (Angstroms)')
        py.ylabel('Relative flux')
        py.legend(loc='upper left',prop={'size':10})

        py.subplot(2,1,2)
        py.plot(lam,rmsspec/avspec,label='Fractional RMS')
        py.ylim(ymin=0,ymax=0.5)
        py.xlim(lam[0],lam[zs-1])
        py.xlabel('Wavelength (Angstroms)')
        py.ylabel('RMS')
        py.legend(loc='upper left',prop={'size':10})

        print(' ')
        ngal = ngal+1
        hdulist.close()



    # Now plot each smoothed normalized spectrum for every galaxy to see the typical scatter:
    py.figure(7)
    py.subplot(2,1,1)
    for i in range(nspec):
        py.plot(lam,speclist[:,i])
        

    # finally, get the average and rms of the divided spectra:
    avspec = np.nanmean(speclist[:,0:nspec],axis=1)
    py.plot(lam,avspec,color='k',lw=3)
    rmsspec = np.nanstd(speclist[:,0:nspec],axis=1)
    py.plot(lam,avspec+rmsspec,color='k',lw=3)
    py.plot(lam,avspec-rmsspec,color='k',lw=3)
    #py.xlabel('Wavelength (Angstroms)')
    py.ylabel('flux/(average flux)')
    py.title('Fractional flux variation per spectrum')
    py.ylim(ymin=0.7,ymax=1.3)
    py.xlim(lam[0],lam[zs-1])

    py.subplot(2,1,2)
    for i in range(nspec_b):
        py.plot(lam,speclist_bin2[:,i])
        

    # finally, get the average and rms of the divided spectra:
    avspec = np.nanmean(speclist_bin2[:,0:nspec_b],axis=1)
    py.plot(lam,avspec,color='k',lw=3)
    rmsspec = np.nanstd(speclist_bin2[:,0:nspec_b],axis=1)
    py.plot(lam,avspec+rmsspec,color='k',lw=3)
    py.plot(lam,avspec-rmsspec,color='k',lw=3)
    py.xlabel('Wavelength (Angstroms)')
    py.ylabel('flux/(average flux)')
    py.title('Fractional flux variation per spectrum (2x2 binning)')
    py.ylim(ymin=0.7,ymax=1.3)
    py.xlim(lam[0],lam[zs-1])


    # plot a histogram of the summed band data:
    # what is the correct thing to plot???
    # an estimate of spectral shape is blue/red.  This will be
    # different for each galaxy, but dispersion in blue/red per galaxy
    # will give us scatter in spectral shape.  Then we want to shift the
    # red/blue ratio to the mean for each galaxy.
    # so we want the scatter in:   (blue(pix)/red(pix))-(blue(tot)/red(tot)) maybe!!!!
    py.figure(9)
    py.hist(col_sum[0:nspec],bins=40,range=(-0.2,0.2),histtype='step',color='r',normed=True,label='1x1 binning')
    py.hist(col_sum_bin2[0:nspec_b],bins=40,range=(-0.2,0.2),histtype='step',color='b',normed=True,label='2x2 binning')
    py.xlabel('(flux ratio) - (mean flux ratio of galaxy)')
    py.ylabel('Number')
    py.legend(prop={'size':10})
#    py.hist(blue_sum,bins=40,range=(0.0,2.0),histtype='step',color='b')

#    print col_sum_bin2[0:nspec_b]

# get the stdev of colour difference
    col_rms=np.nanstd(col_sum[0:nspec])
    print('rms scatter in colour:',col_rms)
    print(' 5th percentile',np.nanpercentile(col_sum[0:nspec],5.0))
    print('95th percentile',np.nanpercentile(col_sum[0:nspec],95.0))
    col_rms=np.nanstd(col_sum_bin2[0:nspec_b])
    print('rms scatter in colour (2x2 binned):',col_rms)

# plot the distribution of rms colour variations for each galaxy (so that we are not weighted towards
# a handful of galaxies with high numbers of good S/N spaxels.
    pdf = PdfPages('spaxel_rms.pdf')
    py.figure(10)
    py.hist(sigma_gal[0:ngal],bins=15,range=(0,0.15),histtype='step',color='k',linestyle='solid',label='RMS 0.5 arcsec')
    py.hist(sigma_gal_bin2[0:ngal],bins=15,range=(0,0.15),histtype='step',color='k',linestyle='dotted',label='RMS 1.0 arcsec')
    py.xlabel('Spaxel-to-spaxel colour RMS')
    py.ylabel('Number of galaxies')
    py.ylim(ymin=0.0,ymax=10.0)
    py.xlim(xmin=0.0,xmax=0.15)
    py.savefig(pdf, format='pdf')        
    pdf.close()

#    py.legend(prop={'size':10})
    print('individual RMS values:',sigma_gal[0:ngal])
    print('                Cube                    0.5 arcsec spaxels   1.0 arcsec spaxels')
    print('                Cube                   Sigma_blue/red Nspec Sigma_blue/red Nspec')
    for i in range(ngal):
        if (sigma_gal[i] < 0.1):
            print('{0:50s} {1:6.4f} {2:4d} {3:6.4f} {4:4d}'.format(fileused[i],sigma_gal[i],int(nspec_gal[i]),sigma_gal_bin2[i],int(nspec_gal_bin2[i])))
        else:
            print('{0:50s} {1:6.4f} {2:4d} {3:6.4f} {4:4d} High sigma!'.format(fileused[i],sigma_gal[i],int(nspec_gal[i]),sigma_gal_bin2[i],int(nspec_gal_bin2[i])))
    
    # calculate the median and other stats on the distribution of sigmas (0.5 arcsec pixels):
    print('statistics for 0.5 arcsec spaxels:')
    print('median RMS:',np.nanmedian(sigma_gal[0:ngal]))
    print('min/max RMS:',np.nanmin(sigma_gal[0:ngal]),np.nanmax(sigma_gal[0:ngal]))
    print('5th and 95th percentile RMS:',np.nanpercentile(sigma_gal[0:ngal],5.0),np.nanpercentile(sigma_gal[0:ngal],95.0))

    # calculate the median and other stats on the distribution of sigmas (1.0 arcsec pixels):
    print(' ')
    print('statistics for 1.0 arcsec spaxels:')
    print('median RMS:',np.nanmedian(sigma_gal_bin2[0:ngal]))
    print('min/max RMS:',np.nanmin(sigma_gal_bin2[0:ngal]),np.nanmax(sigma_gal_bin2[0:ngal]))
    print('5th and 95th percentile RMS:',np.nanpercentile(sigma_gal_bin2[0:ngal],5.0),np.nanpercentile(sigma_gal_bin2[0:ngal],95.0))

    print('number of galaxies used: ',ngal)
     
