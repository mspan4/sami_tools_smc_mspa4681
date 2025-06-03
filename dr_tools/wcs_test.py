"""
Functions for measuring and recording WCS information.

In particular, wcs_position_coords is supposed to determine the WCS for
a file based on cross-correlating a collapsed image from the datacube with
an external photometric image. However, this was never shown to work
properly (the results were clustering around particular values, for
unknown reasons), so it was put to one side and never finished. Instead,
the SAMI Galaxy Survey has been using the 'nominal' WCS, which assumes
that the catalogued object is in the centre of the data.
"""

import numpy as np
import scipy as sp
import scipy.signal as signal
import pylab as py
import astropy.io.ascii as ascii
from scipy.interpolate import griddata
import scipy.ndimage as nd
import astropy.io.fits as pf
import os
import math
import glob
import urllib.request, urllib.parse, urllib.error
import time
from matplotlib.backends.backend_pdf import PdfPages


import sami.samifitting as fitting
#from ..sdss import sdss

#########################
# fix WCS CRVAL1,CRVAL2 for objects with BAD_CLASS=5 that have offsets.
#
def fix_wcs_badclass5(catfile,inlist,sim=False):

    """Use the coordinates from the catalogue file to fix the WCS values
    for objects that have BAD_CLASS=5.  These have coords that are based
    of the IFU pointing, but they should have values based on the object 
    coordinates, as the IFU centre is based on the gaussing fit to the 
    centre of the object.

    catfile - input catalogue file with _IFU and _OBJ coordinates
    inlist - list of potential files to update
    sim - run in simulation mode (i.e. don't actually make the changes).

    """

    # glob the list:
    infiles = glob.glob(inlist)
    
    # read the catalogue file
    hdulist = pf.open(catfile)
    tab = hdulist[1].data

    # make a list of CATIDs with BAD_CLASS=5
    idx = np.where(tab['BAD_CLASS'] == 5)
    catid = tab['CATID'][idx]
    ra_obj = tab['RA_OBJ'][idx]
    dec_obj = tab['DEC_OBJ'][idx]
    ra_ifu = tab['RA_IFU'][idx]
    dec_ifu = tab['DEC_IFU'][idx]

    nobj = np.size(catid)
    print('number of objects to fix:',nobj)
    
    # loop through all the CATIDs:
    for i in range(nobj):

        catid_str = str(catid[i])
        # identify files in the list that are for this CATID:
        for infile in infiles:
            if catid_str in infile:
                print('updating ',infile)
                hdulist = pf.open(infile,mode='update')
                # loop through all HDUs in the file:
                nhdu = 0
                for hdu in hdulist:
                    hdr = hdu.header
                    if (nhdu != 0):
                        extname = hdr['EXTNAME']
                    else:
                        extname = 'Primary HDU'
                    # check to see if WCS info is in the header.  Check
                    # for CRVAL2, as this means the data must be 2D, e.g.
                    # can't be a 1D spectrum.
                    if 'CRVAL2' in hdr:
                        print('Checking WCS in extension: ',extname,', CRVAL1,CRVAL2 = ',hdr['CRVAL1'],hdr['CRVAL2'])
                        # as a test, calculate the difference between the header items and the
                        # ra/dec.  As we are testing against the IFU coords these should be very close
                        # zero:
                        ra_diff = (hdr['CRVAL1'] - ra_ifu[i])*3600.0
                        dec_diff = (hdr['CRVAL2'] - dec_ifu[i])*3600.0
                        # if the difference is small, then these must coordinates as we expect
                        # we can then update them:
                        if ((abs(ra_diff) < 0.2) & (abs(dec_diff) < 0.2)):
                            # only update if not in simulation mode:
                            if (not sim):
                                hdr['CRVAL1'] = ra_obj[i]
                                hdr['CRVAL2'] = dec_obj[i]
                                print('Updated WCS in extension ',extname) 
                                
                    nhdu = nhdu+1
                    
                # close the file
                hdulist.close()


                
#########################

def wcs_solve(myIFU, object_flux_cube, object_name, band, size_of_grid, output_pix_size_arcsec, plot=False, write=False, nominal=False, remove_thput_file=True):
    
    """Wrapper for wcs_position_coords, extracting coords from IFU.
        
        This function cross-correlates a g or r-band convolved SAMI cube with its
        respective SDSS g-band image and pins down the positional WCS for the
        central spaxel of the cube.
        """
    
    # Get Object RA + DEC from fibre table (this is the input catalogues RA+DEC in deg)
    object_RA = np.around(myIFU.obj_ra[myIFU.n == 1][0], decimals=6)
    object_DEC = np.around(myIFU.obj_dec[myIFU.n == 1][0], decimals=6)
    
    # Build wavelength axis.
    CRVAL3 = myIFU.crval1
    CDELT3 = myIFU.cdelt1
    Nwave  = np.shape(object_flux_cube)[0]
    
    # -- crval3 is middle of range and indexing starts at 0.
    # -- this wave-axis agrees with QFitsView interpretation.
    CRVAL3a = CRVAL3 - ((Nwave-1)/2)*CDELT3
    wave = CRVAL3a + CDELT3*np.arange(Nwave)
    
    object_flux_cube = np.transpose(object_flux_cube, (2,0,1))
    
    return wcs_position_coords(object_RA, object_DEC, wave, object_flux_cube, object_name, band, size_of_grid, output_pix_size_arcsec, plot=plot, write=write, nominal=nominal)


def wcs_position_coords(object_RA, object_DEC, wave, object_flux_cube, object_name, band, size_of_grid, output_pix_size_arcsec, plot=False, write=False, nominal=False, remove_thput_file=True,verbose=False,url_show=False):
    """Calculate a revised WCS position from a cross-correlation between a
       g-band SAMI cube and a g-band SDSS image at the nominal location of the cube.
       There are a few important points to remember in the FITS standard.  The first
       is that the centre of the first pixel in an image is (1,1) so the top right corner
       will be (1.5,1.5).  In the standard SAMI 50x50 arrays the middle pixels will have
       coords for their centres of (25,25) (25,26), (26, 25), (26, 26). As a result the
       middle of the array will be (25.5,25.5).

       In contrast, the centre of the first pixel in a typical python array is (0,0) as
       python arrays are zero indexed.
        """
    
    if nominal:
        img_crval1 = object_RA
        img_crval2 = object_DEC
        xcube = size_of_grid
        ycube = size_of_grid
        img_cdelt1 = -1.0 * output_pix_size_arcsec / 3600.0
        img_cdelt2 = output_pix_size_arcsec / 3600.0
    
    else:

        # Get SDSS g-band throughput curve
        if not os.path.isfile("sdss_"+str(band)+".dat"):
            urllib.request.urlretrieve("http://www.sdss.org/dr3/instruments/imager/filters/"+str(band)+".dat", "sdss_"+str(band)+".dat")
        
        # and convolve with the SDSS throughput
        sdss_filter = ascii.read("SDSS_"+str(band)+".dat", comment="#", names=["wave", "pt_secz=1.3", "ext_secz=1.3", "ext_secz=0.0", "extinction"])
        
        # re-grid g["wave"] -> wave
        thru_regrid = griddata(sdss_filter["wave"], sdss_filter["ext_secz=1.3"], wave, method="cubic", fill_value=0.0)
        
        # initialise a 2D simulated g' band flux array.
        len_axis = np.shape(object_flux_cube)[1]
        Nwave = len(wave)
        reconstruct = np.zeros((len_axis,len_axis))
        tester = np.zeros((len_axis,len_axis))
        data_bit = np.zeros((Nwave,len_axis,len_axis))
        
        # Sum convolved flux.  This could be made more robust by filtering the
        # cube data before concolving with the filter:
        for i in range(Nwave):
            data_bit[i] = object_flux_cube[i]*thru_regrid[i]

        # need to make this more robust to hot pixels etc:
        reconstruct = np.nansum(data_bit,axis=0) # not absolute right now
        reconstruct[np.isnan(reconstruct)] = 0. # replacing nan with 0.0
        reconstruct[reconstruct < 0] = 0.       # replacing negative fluxes with 0.0
        
        cube_image = reconstruct
        xcube = len(cube_image[0])
        ycube = len(cube_image[1])
        # This does some cropping or filtering, but we don't need to do this at the moment, so
        # ignore (not sure why this was put into the code):
        #cube_image_crop = cube_image[(len(cube_image[0])/2)-10:(len(cube_image[0])/2)+10,(len(cube_image[1])/2)-10:(len(cube_image[1])/2)+10]
        #cube_image_crop = sp.ndimage.zoom(cube_image_crop, 5, order=3)
        #cube_image_crop_norm = (cube_image_crop - np.min(cube_image_crop))/np.max(cube_image_crop - np.min(cube_image_crop))
        
        # Check if the user supplied a red RSS file, throw exception.
        if np.array_equal(cube_image, tester):
            raise SystemExit("All values are zero: please provide the cube corresponding to the requested spectral band of the image!")
        
        ##########

        # get the size of the cube so that we can use this to get the SDSS image:
        cube_size = np.around((size_of_grid*output_pix_size_arcsec)/3600, decimals=6)
        
        # Get SDSS Image
        if not os.path.isfile(str(object_name)+"_SDSS_"+str(band)+".fits"):

            if (verbose):
                print("Getting SDSS image for object ",object_name)
        
            sdss.getSDSSimage(object_name=object_name, RA=object_RA, DEC=object_DEC,
                         band=str(band), size=cube_size, number_of_pixels=size_of_grid,url_show=url_show)
        
            
        # Open SDSS image and extract data & header information
        image_file = pf.open(str(object_name)+"_SDSS_"+str(band)+".fits")
        image_data = image_file['Primary'].data        
        
        image_header = image_file['Primary'].header
        img_crval1 = float(image_header['CRVAL1']) #RA
        img_crval2 = float(image_header['CRVAL2']) #DEC
        img_crpix1 = float(image_header['CRPIX1']) #Reference x-pixel
        img_crpix2 = float(image_header['CRPIX2']) #Reference y-pixel
        img_cdelt1 = float(image_header['CDELT1']) #Delta RA
        img_cdelt2 = float(image_header['CDELT2']) #Delta DEC
        
        SDSS_image = image_data
        # no need to crop:
        #SDSS_image_crop = SDSS_image[(len(SDSS_image[0])/2)-10:(len(SDSS_image[0])/2)+10,(len(SDSS_image[1])/2)-10:(len(SDSS_image[1])/2)+10]
        #SDSS_image_crop_norm = (SDSS_image_crop - np.min(SDSS_image_crop))/np.max(SDSS_image_crop - np.min(SDSS_image_crop))

        # plot the SAMI and SDSS images, marking the centre pixel
        # of the images (typically 25.5):
        if (plot):
            py.figure(1)
            py.clf()
            py.subplot(2,2,1)
            py.scatter([img_crpix1],[img_crpix2],marker='x',facecolors='white', edgecolors='white',s=100,linewidths=1.5,label='image centre')
            py.imshow(cube_image,origin='lower',interpolation='nearest')
            py.title('SAMI, x=array cent')
            py.xlabel('RA (0.5\" pixels)')
            py.ylabel('Dec (0.5\" pixels)')
            py.subplot(2,2,2)
            py.scatter([img_crpix1],[img_crpix2],marker='x',facecolors='white', edgecolors='white',s=100,linewidths=1.5,label='image centre')
            py.imshow(np.log(SDSS_image),origin='lower',interpolation='nearest')
            py.title('SDSS, x=array cent')
            py.xlabel('RA (0.5\" pixels)')
            py.ylabel('Dec (0.5\" pixels)')
    
    ##########
    
    if (not nominal) and np.size(np.where(image_data == 0.0)) != 2*np.size(image_data):
        # Cross-correlate normalised SAMI-cube g-band image and SDSS g-band image
        WCS_flag = 'SDSS'
        # here we use the cropped images:
        #crosscorr_image = sp.signal.correlate2d(SDSS_image_crop_norm, cube_image_crop_norm)
        # here we use the full images:
        crosscorr_image = signal.correlate2d(SDSS_image, cube_image)

        # get shape of output cross-correlation:
        xcorrshape = np.shape(crosscorr_image)

        # first find the index of the max value in the cross-correlation:
        ymax,xmax = np.nonzero(crosscorr_image.max() == crosscorr_image)
        if (verbose):
            print('coordinate of max x-corr:',xmax,ymax)
            print('x-corr max value:',crosscorr_image.max()) 
            print('x-corr min value:',crosscorr_image.min())
        
        # extract a region of size (n_crop x n_crop) around the peak and fit a 2D quadratic to find
        # the best peak.  n_crop should be odd.
        n_crop = 5
        crosscorr_image_crop = crosscorr_image[int(ymax-n_crop/2):int(ymax+n_crop/2+1),int(xmax-n_crop/2):int(xmax+n_crop/2+1)]
        xc = np.linspace(xmax-n_crop/2,xmax+n_crop/2,n_crop)
        yc = np.linspace(ymax-n_crop/2,ymax+n_crop/2,n_crop)
        XC, YC = np.meshgrid(xc, yc, copy=False)
        qshape = np.shape(XC)
        XC = XC.flatten()
        YC = YC.flatten()

        # fit the local region around the centre using least squares (note that this is unweighted):
        A = np.array([XC*0+1, XC, YC, XC**2, XC*YC, YC**2]).T
        B = crosscorr_image_crop.flatten()
        cf, r, rank, s = np.linalg.lstsq(A, B)

        # find the model centre:
        mxcent = (cf[4]*cf[2] - 2*cf[5]*cf[1])/(4.0*cf[5]*cf[3] - cf[4]*cf[4])
        mycent = (cf[4]*cf[1] - 2*cf[3]*cf[2])/(4.0*cf[5]*cf[3] - cf[4]*cf[4])
        
        if (verbose):
            print('coefficients of fit: ',cf)
            print('model centre:',mxcent,mycent)
                        
        if (plot):
            #derive a model based on the best fit:
            model = cf[0] + cf[1]*XC + cf[2]*YC + cf[3]*XC*XC + cf[4]*XC*YC + cf[5]*YC*YC
            model_im = np.reshape(model,qshape)
            # place the model into the same sized array (filled with zeros):
            model_im_full = np.zeros(xcorrshape)
            model_im_full.fill(np.nan)
            model_im_full[(ymax-n_crop/2):(ymax+n_crop/2+1),(xmax-n_crop/2):(xmax+n_crop/2+1)]=model_im
            # plot cross-correlation image and mark the best fit centre:
            py.subplot(2,2,3)
            py.scatter([mxcent],[mycent],marker='x',facecolors='white', edgecolors='white',s=100,linewidths=1)
            py.imshow(crosscorr_image,origin='lower',interpolation='nearest')
            py.title('Cross-correlation')
            py.xlabel('RA (0.5\" pixels)')
            py.ylabel('Dec (0.5\" pixels)')
            
                        
                    
        # 2D Gauss Fit the cross-correlated image.  The fitter handles data in
        # 1D, so we use ravel to flatten the array.
        crosscorr_image_1d = np.ravel(crosscorr_image)
        #use for loops to recover indicies in x and y positions of flux values
        x_pos = []
        y_pos = []
        for i in range(np.shape(crosscorr_image)[0]):
            for j in range(np.shape(crosscorr_image)[1]):
                x_pos.append(i)
                y_pos.append(j)
        x_pos=np.array(x_pos)
        y_pos=np.array(y_pos)
        
        #define guess parameters for TwoDGaussFitter:
        amplitude = max(crosscorr_image_1d)
        mean_x = (np.shape(crosscorr_image)[0])/2
        mean_y = (np.shape(crosscorr_image)[1])/2
        sigma_x = 5.0
        sigma_y = 6.0
        rotation = 60.0
        offset = 4.0
        p0 = [amplitude, mean_x, mean_y, sigma_x, sigma_y, rotation, offset]
        
        # call SAMI TwoDGaussFitter
        GF2d = fitting.TwoDGaussFitter(p0, x_pos, y_pos, crosscorr_image_1d)
        # execute gauss fit using
        GF2d.fit()
        GF2d_xpos = GF2d.p[2]
        GF2d_ypos = GF2d.p[1]
        
        # reconstruct the fit.  The first array will be 1D, and then
        # we use reshape to recover the original 2D shape:
        GF2d_reconstruct=GF2d(x_pos, y_pos)
        GF2d_image = np.reshape(GF2d_reconstruct,xcorrshape)
        
        if (plot):
            py.subplot(2,2,3)
            py.plot([GF2d_xpos],[GF2d_ypos],'x',color='white')
            py.contour(GF2d_image,color='white')
            py.subplot(2,2,4)
            py.imshow(GF2d_image,origin='lower',interpolation='nearest')
            py.plot([GF2d_xpos],[GF2d_ypos],'x',color='white')
            py.contour(GF2d_image,color='white')

        # compare the centres derived using the gaussian and local quadratic
        # fits:
        diff = np.sqrt((mxcent-GF2d_xpos)**2 + (mycent-GF2d_ypos)**2)
        if (verbose):
            print("difference between Gaussian and local quadratic centres (pixels): ",diff)

        if (diff > 1.0):
            print("WARNING: gaussian and local quadratic centres disagree.  Diff=",diff)
            
        x_shape = len(crosscorr_image[0])
        y_shape = len(crosscorr_image[1])
# use the quadratic fit to the peak to find the peak:
        x_offset_pix = mxcent - x_shape/2
        y_offset_pix = mycent - y_shape/2
# use the gaussian fit to the x-corr peak to define the shift:
#        x_offset_pix = GF2d_xpos - x_shape/2
#        y_offset_pix = GF2d_ypos - y_shape/2
# scale by 5 due to zoom of SAMI image.  This is no-longer done:
#        x_offset_arcsec = -x_offset_pix * output_pix_size_arcsec/5
#        y_offset_arcsec = y_offset_pix * output_pix_size_arcsec/5
        x_offset_arcsec = -x_offset_pix * output_pix_size_arcsec
        y_offset_arcsec = y_offset_pix * output_pix_size_arcsec
        # for the final offset in degrees, also apply the cos(dec) term to get
        # the right shift in RA:
        x_offset_degree = (x_offset_arcsec/3600)/math.cos(math.radians(object_DEC))
        y_offset_degree = (y_offset_arcsec/3600)

        if (verbose):
            print('estimated offset in degrees (including cos(dec) term): ',x_offset_degree,y_offset_degree)
            print('estimated offset in arcsec: ',x_offset_arcsec,y_offset_arcsec)
            print('estimated offset in pixels: ',x_offset_pix,y_offset_pix)
        
        
    else:
        WCS_flag = 'Nominal'
        y_offset_degree = 0.0
        x_offset_degree = 0.0
    
    # Create dictionary of positional WCS.  Only make the changes for the CRVAl1 and CRVAL2 values, as we are not
    # measuring the other values or changing them:
    if isinstance(xcube/2, int):
#        WCS_pos={"CRVAL1":(img_crval1 + x_offset_degree), "CRVAL2":(img_crval2 + y_offset_degree), "CRPIX1":(xcube/2 + 0.5),
#            "CRPIX2":(ycube/2 + 0.5), "CDELT1":(img_cdelt1), "CDELT2":(img_cdelt2), "CTYPE1":"RA---TAN", "CTYPE2":"DEC--TAN",
#            "CUNIT1": 'deg', "CUNIT2": 'deg'}
        WCS_pos={"CRVAL1":(img_crval1 + x_offset_degree), "CRVAL2":(img_crval2 + y_offset_degree)}
    else:
#        WCS_pos={"CRVAL1":(img_crval1 + x_offset_degree), "CRVAL2":(img_crval2 + y_offset_degree), "CRPIX1":(xcube/2),
#            "CRPIX2":(ycube/2), "CDELT1":(img_cdelt1), "CDELT2":(img_cdelt2), "CTYPE1":"RA---TAN", "CTYPE2":"DEC--TAN",
#            "CUNIT1": 'deg', "CUNIT2": 'deg'}
        WCS_pos={"CRVAL1":(img_crval1 + x_offset_degree), "CRVAL2":(img_crval2 + y_offset_degree)}
    
    ##########
    
    # Remove temporary files
#    if remove_thput_file and os.path.exists("sdss_"+str(band)+".dat"):
#        os.remove("sdss_"+str(band)+".dat")
#    if os.path.exists(str(object_name)+"_SDSS_"+str(band)+".fits"):
#        os.remove(str(object_name)+"_SDSS_"+str(band)+".fits")
    
    return WCS_pos,WCS_flag,x_offset_arcsec,y_offset_arcsec

def wcs_position_coords_local(cubefile,object_RA, object_DEC, wave, object_flux_cube, object_var_cube, object_name, band, size_of_grid, output_pix_size_arcsec, plot=False, write=False, nominal=False, remove_thput_file=True,verbose=False,impath='/Users/scroom/data/sami/images/gama_cutouts/',fluxtest=True,sami_radius=12.0):
    """Calculate a revised WCS position from a cross-correlation between a
       g-band SAMI cube and a g-band SDSS image.

       This function is based on wcs_position_coords() but cross-matches to local images
       (usually SDSS) that need not be the same size or scale as the cubes.  This means
       that some extra steps are required, e.g. resampling and correcting for different
       WCS pixel CRPIX1, CRPIX2 values etc.

       There are a few important points to remember in the FITS standard.  The first
       is that the centre of the first pixel in an image is (1,1) so the top right corner
       will be (1.5,1.5).  In the standard SAMI 50x50 arrays the middle pixels will have
       coords for their centres of (25,25) (25,26), (26, 25), (26, 26). As a result the
       middle of the array will be (25.5,25.5).

       In contrast, the centre of the first pixel in a typical python array is (0,0) as
       python arrays are zero indexed.  We can also consider this to be the middle of the
       pixel.
        """

    # set some parameters for the plots:
    py.rcParams.update({'font.size': 10})
    # move the axis labels on the tick-marks further apart to avoid overlaps:
    py.rcParams.update({'xtick.major.pad': 6})
    py.rcParams.update({'ytick.major.pad': 6})
    
    # parameters for the code:
    # convolution to apply to the SDSS and SAMI image before cross-correlation and
    # flux integration.  Median SAMI seeing is 2.1", median for SDSS is 1.4".
    sami_seeing = 2.1 #FWHM
    sdss_seeing = 1.4 #FWHM
    conv_sdss = np.sqrt((sami_seeing)**2 - (sdss_seeing)**2)
    # we convolve SDSS image by 1.6", we will get to 2.1".  This assumes a Gaussian
    # PSF, and the value below if for FWHM (converted to sigma below):
    # conv_sdss = 1.6
    conv_sami = 0.0

    # SAMI radius (in 0.5 arcsec pixels) to integrate within to get flux
    # now passed as argument
    #sami_radius = 12.0  # 6 arcsec radius

    # half width in Angstroms to define the range to estimate S/N.  Will go
    # from g_lam-lwidth to g_lam-lwidth where g_lam is the pivot point for the
    # g-band filter:
    lwidth=100


    # read in mags and psf parameters from cube file:
    cube_hdu = pf.open(cubefile)
    cube_header = cube_hdu['Primary'].header
    try:
        magu = float(cube_header['MAGU'])
        magg = float(cube_header['MAGG'])
        magr = float(cube_header['MAGR'])
        magi = float(cube_header['MAGI'])
        magz = float(cube_header['MAGZ'])
        catmagu = float(cube_header['CATMAGU'])
        catmagg = float(cube_header['CATMAGG'])
        catmagr = float(cube_header['CATMAGR'])
        catmagi = float(cube_header['CATMAGI'])
        catmagz = float(cube_header['CATMAGZ'])
    except KeyError:
        magu = -99.9
        magg = -99.9
        magr = -99.9
        magi = -99.9
        magz = -99.9
        catmagu = -99.9
        catmagg = -99.9
        catmagr = -99.9
        catmagi = -99.9
        catmagz = -99.9

        
    #psfalpha = float(cube_header['PSFALPHA'])
    #psfbeta = float(cube_header['PSFBETA'])
    psffwhm = float(cube_header['PSFFWHM'])        
    #rescale = float(cube_header['RESCALE'])
    
    if nominal:
        img_crval1 = object_RA
        img_crval2 = object_DEC
        xcube = size_of_grid
        ycube = size_of_grid
        img_cdelt1 = -1.0 * output_pix_size_arcsec / 3600.0
        img_cdelt2 = output_pix_size_arcsec / 3600.0
    
    else:
        
        # Get SDSS g-band throughput curve
        if not os.path.isfile("SDSS_"+str(band)+".dat"):
            urllib.request.urlretrieve("http://www.sdss.org/dr3/instruments/imager/filters/"+str(band)+".dat", "SDSS_"+str(band)+".dat")
        
        # and read in the SDSS transmissions:
        sdss_filter = ascii.read("SDSS_"+str(band)+".dat", comment="#", names=["wave", "pt_secz=1.3", "ext_secz=1.3", "ext_secz=0.0", "extinction"])
        
        # re-grid g["wave"] -> wave
        # Which SDSS filter response curve should we use?
        # if we want to get the right flux above the atmosphere, then we should use the ext_secz=0 version.  However,
        # the SDSS observations use an effective filter curve that is a combination of the filter+atmosphere with a
        # typical airmass of 1.3.  The absoulte normalization does not matter (this is corrected for by the integration
        # over the filter band-pass of the AB zero0point flux).  It is just the shape that matters, and the airmass=1.3
        # version should be the right one:
        thru_regrid = griddata(sdss_filter["wave"], sdss_filter["ext_secz=1.3"], wave, method="cubic", fill_value=0.0)
        #thru_regrid = griddata(sdss_filter["wave"], sdss_filter["ext_secz=0.0"], wave, method="cubic", fill_value=0.0)
        
        # initialise a 2D simulated g' band flux array.
        len_axis = np.shape(object_flux_cube)[1]
        Nwave = len(wave)
        reconstruct = np.zeros((len_axis,len_axis))
        reconstruct_magsb = np.zeros((len_axis,len_axis))
        snimage =  np.zeros((len_axis,len_axis))
        tester = np.zeros((len_axis,len_axis))
        data_bit = np.zeros((Nwave,len_axis,len_axis))

        # next we need to convert the SAMI cube to a g-band image.  We will generate 2 images.  One that is in units of
        # magnitudes and the other that is in linear flux units.  For the magniutude version we will have to decide
        # whether to use normal mags or SDSS asinh mags.  Currently we will just use normal Pogson magnitudes.
        #
        # The code below is borrowed from the qc.fluxcal.measure_band() routine in the sami manager.  At some point
        # we should probably merge so that we only have one version of the code.
        #
        # Convert to SI units. Wavelength was in A, flux was in 1e-16 erg/s/cm^2/A
        wl_m = wave * 1e-10
        flux_wm3 = object_flux_cube * 1e-16 * 1e-7 * (1e2)**2 * 1e10
        # AB magnitudes are zero for flux of 3631 Jy.  For integrating per unit frequency, this is constant, but
        # as we are integrating over wavelength, we need the 1/wavelength^2 term:
        flux_zero = 3631.0 * 1.0e-26 * 2.99792458e8 / (wl_m**2)
        # Get the wavelength bin sizes - don't assume constant!
        delta_wl = np.hstack((wl_m[1] - wl_m[0],0.5 * (wl_m[2:] - wl_m[:-2]),wl_m[-1] - wl_m[-2]))
        #
        # finally, for each spaxel, integrate the spectrum and normalize by the zero-point flux integral.
        # only sum over points that are not NaN in the original spectrum.
        #
        # TO DO: Could also smooth (e.g. median filter) spectrum in the wavelength direction a little to
        # remove the impact of rogue CRs.  However, this could also cut out real emission lines.
        #
        # filter the cubes in the wavelength direction:
        

        # convolve with filter:
        filter_interpolated = thru_regrid
        for i in range(len_axis):
            for j in range(len_axis):
                flux_int = np.nansum(delta_wl * wl_m * filter_interpolated * flux_wm3[:,i,j])
                flux_int_zp = np.nansum(np.where((flux_wm3[:,i,j] != np.nan),delta_wl * wl_m * filter_interpolated * flux_zero,0))
                flux_band = flux_int/flux_int_zp
                reconstruct[i,j] = flux_band
                # get the surface brightness in magnitudes per sq arcsec:
                reconstruct_magsb[i,j] = np.where((flux_band>0),-2.5 * np.log10(flux_band/(output_pix_size_arcsec*output_pix_size_arcsec)),np.nan)

        # the reconstructed image is now in units of 3631Jy, so multiply by this to get into Jy.
        # can then multiply by 10^6 to get into microJy:
        reconstruct = reconstruct*3631.0*1.0e6
                
        # here we will replace NaNs with zeros, as otherwise this will break the sums and/or the cross-correlation.
        reconstruct[np.isnan(reconstruct)] = 0. # replacing nan with 0.0
        # could replace -ve values, but not obvious we need to do this.
        #reconstruct[reconstruct < 0] = 0.       # replacing negative fluxes with 0.0

        # define a range around the pivot point of the g-band
        # to calculate the S/N.
        # find index for +-100A around 4686 (centre of g-band):
        dmin = 1.0e10
        dmax = 1.0e10
        imin = 0
        imax = 0
        for i in range(Nwave):
            dlam = np.abs(wave[i]-(4686-lwidth)) 
            if (dlam < dmin):
                imin = i
                dmin = dlam

        for i in range(Nwave):
            dlam = np.abs(wave[i]-(4686+lwidth)) 
            if (dlam < dmax):
                imax = i
                dmax = dlam

        print('pixel range in wavelength to use for S/N:',imin,imax,wave[imin],wave[imax])

        # derive S/N map at the centre of the g-band, using the median.
        for i in range(len_axis):
            for j in range(len_axis):
                snimage[i,j] = np.nanmedian(object_flux_cube[imin:imax,i,j]/np.sqrt(object_var_cube[imin:imax,i,j]))

        # set up plotting.  At this point just define the axes, and clear of previous plots:
        if (plot):
            fig = py.figure(1)
            py.clf()
            ax1 = fig.add_subplot(2,3,1)
            ax2 = fig.add_subplot(2,3,2)
            ax3 = fig.add_subplot(2,3,3)
            ax4 = fig.add_subplot(2,3,4)
            ax5 = fig.add_subplot(2,3,5)
            ax6 = fig.add_subplot(2,3,6)


        # finally try gaussian smoothing of the image.  We may not actually need to do
        # this.
        sigma = (conv_sami/2.3548)/np.abs(output_pix_size_arcsec)  # sigma in arcsec
        cube_image =  nd.filters.gaussian_filter(reconstruct,sigma)

        # define shape of cube image:
        xcube = len(cube_image[0])
        ycube = len(cube_image[1])
        
        # Check if the user supplied a red RSS file, throw exception.
        if np.array_equal(cube_image, tester):
            raise SystemExit("All values are zero: please provide the cube corresponding to the requested spectral band of the image!")
        
        ################################################################
        # output_pix_size_arcsec: is the CDELT1, CDELT2, but converted to arcsec (from deg).
        # size_of_grid: the number of pixels in the SAMI cube image
        # object_RA, object_DEC: ra,dec of the centre of the SAMI cube
        #                        that should be pixel ref (25.5,25.5)
        #

        # Get the local image:
        imfile = impath+"G"+str(object_name)+"_SDSS_"+str(band)+".fits"
        if not os.path.isfile(imfile):

            # if the file cannot be found, exit and return null values.
            # this will need to be properly caught later...
            # just return nominal values
            print('cannot find file ',imfile)
            WCS_pos={"CRVAL1":(np.nan), "CRVAL2":(np.nan)}
            if (fluxtest):
                return WCS_pos,'none',0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
#               return WCS_pos,WCS_flag,x_offset_arcsec,y_offset_arcsec,sami_sum,sdss_sum,sb_sn5,sn220,sn225,sn230,sn235,sn240
            else:
                return WCS_pos,'none',0.0,0.0
#               return WCS_pos,WCS_flag,x_offset_arcsec,y_offset_arcsec

                    
            
        # Open SDSS image and extract data & header information
        image_file = pf.open(impath+"G"+str(object_name)+"_SDSS_"+str(band)+".fits")
        image_data = image_file['Primary'].data

        if (verbose):
            print('Reading SDSS image file:',image_file) 
        
        image_header = image_file['Primary'].header
        img_crval1 = float(image_header['CRVAL1']) #RA
        img_crval2 = float(image_header['CRVAL2']) #DEC
        img_crpix1 = float(image_header['CRPIX1']) #Reference x-pixel
        img_crpix2 = float(image_header['CRPIX2']) #Reference y-pixel
        # TODO: check if CDELT1/CDELT2 are present before this...
        try:
            img_cdelt1 = float(image_header['PC1_1']) #Delta RA
        #except KeyError:
        #    img_cdelt1 = float(image_header['CDELT1']) #Delta RA
        except KeyError:
            img_cdelt1 = float(image_header['CD1_1']) #Delta RA

        try:
            img_cdelt2 = float(image_header['PC2_2']) #Delta RA
        #except KeyError:
        #    img_cdelt2 = float(image_header['CDELT2']) #Delta DEC
        except KeyError:
            img_cdelt2 = float(image_header['CD2_2']) #Delta DEC

        # convert SDSS image to the same flux scale as SAMI.  The SAMI
        # image has been converted to microJy.  We are using GAMA reprocessed
        # SDSS images, and for these the zero point for 1ADU is 30mag in AB.
        # This allows us convert to flux units:
        zpmag_ab = 30.0   # zero point (from Driver et al. 2016)
        zp_fnu = 10.0**(-0.4*zpmag_ab)  
        image_data = image_data * zp_fnu

        # convert SDSS image to microJy:
        image_data = image_data * 3631.0 * 1.0e6
        
        SDSS_image = image_data
        ny,nx = np.shape(image_data)
        if (verbose):
            print("size of SDSS input image:",nx,ny)

        # get the ra/dec of the centre of the image by calculating it at
        # each pixel and then taking a mean.  This will then be correct whether
        # we have even or odd numbers of pixels in the image.
        ra_t = (img_crval1 + (1 + np.arange(nx) - img_crpix1) * img_cdelt1)
        dec_t = (img_crval2 + (1 + np.arange(ny) - img_crpix2) * img_cdelt2)
        SDSS_cent_ra = np.mean(ra_t)
        SDSS_cent_dec = np.mean(dec_t)
        if (verbose):
            print("original WCS for SDSS image",img_crval1,img_crval2,img_crpix1,img_crpix2,img_cdelt1,img_cdelt2)
            print("coordinates of centre of SDSS image:",SDSS_cent_ra,SDSS_cent_dec)

        # process the SDSS image.  We need to do a number of things
        # first we will convolve the image with a Gaussian kernel to
        # get closer to the correct seeing.
        # convolve with 1-1.5 arcsec seeing.  Need to convert FWHM to
        # sigma and then from arcsec to pixels:
        sigma = (conv_sdss/2.3548)/np.abs(img_cdelt1*3600)  # sigma in arcsec

        SDSS_image =  nd.filters.gaussian_filter(image_data,sigma)

        print('okay 1: ',img_cdelt1)
        
        
        # re-sample the arrays by the relative difference in the pixel size:
        # The resampling uses the zoom function.  Zoom maintains the centre of the
        # image at the centre of the new zoomed image:
        pix_ratio = output_pix_size_arcsec/np.abs(img_cdelt1*3600)
        print(pix_ratio,output_pix_size_arcsec)
        SDSS_zoom_t = sp.ndimage.zoom(SDSS_image, 1.0/pix_ratio, order=3)
        print('done zoom')
        (ysdsszoom,xsdsszoom) = np.shape(SDSS_zoom_t)
        # finally trim the SDSS image to retain only the central regions to make sure
        # that bright stars around the edge of the field do not
        ntrim_SDSS = 10
        SDSS_zoom = SDSS_zoom_t[(0 + ntrim_SDSS):(ysdsszoom-ntrim_SDSS),(0 + ntrim_SDSS):(xsdsszoom-ntrim_SDSS)]
        if (verbose):
            print('ratio of SAMI to image pixel size is:',pix_ratio)

        if (verbose):
            print("shape of SDSS_zoom:",np.shape(SDSS_zoom))

        # the SDSS images that we using here have been zoomed in, so have been resampled by
        # a factor 1.0/pix_ratio.  As the levels are
        # kept the same, this means that we have to scale the flux back to the correct level.
        # the GAMA cutouts have pixel scale of 0.339 arcsec, so when we scale to 0.5 arcsec
        # then we end up with less total flux.  The pix_ratio is then 0.5/0.339 = 1.475 
        # need to rescale the resampled image to conserve flux:
        SDSS_zoom = SDSS_zoom * pix_ratio * pix_ratio

        # plot the SAMI and SDSS images, marking the centre pixel
        # of the images (typically 25.5):
            # plot the location of the centre of the SDSS image, based on the WCS from the SAMI
            # cube
        pix1= (SDSS_cent_ra - object_RA)/(-1.0*output_pix_size_arcsec/3600.0) + 25.5 -1
        pix2= (SDSS_cent_dec - object_DEC)/(output_pix_size_arcsec/3600.0) + 25.5 -1
        if (plot):
            # get the vertical scale to plot.  Base the max flux on the peak in the central
            # part of the SAMI cube:
            nys,nxs = np.shape(cube_image)
            nx1 = int(nxs/4)
            nx2 = int(3*nxs/4)
            ny1 = int(nys/4)
            ny2 = int(3*nys/4)
            vmin=np.min(cube_image)
            vmax=np.max(cube_image[ny1:ny2,nx1:nx2]) 
            print('SAMI scale:',vmin,vmax)
            ax1.scatter([pix1],[pix2],marker='x',facecolors='white', edgecolors='white',s=100,linewidths=1.5,label='image centre')
            im1 = ax1.imshow(cube_image,origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax)
            ax1.set_title('SAMI g-band, x=array cent')
            ax1.set_xlabel('RA (0.5\" pixels)')
            ax1.set_ylabel('Dec (0.5\" pixels)')
            ax1.set_xlim(0-0.5,xcube-0.5)
            ax1.set_ylim(0-0.5,ycube-0.5)
            ax1.text(0.5,0.9,object_name, horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes,color='w')
            cb = fig.colorbar(im1, ax=ax1, shrink=0.8)
            cb.ax.set_title('uJy')

            # plot S/N image:
            vminsn=np.nanmin(snimage)
            vmaxsn=np.nanmax(snimage)
            print('SAMI scale for S/N:',vminsn,vmaxsn)
            im6 = ax6.imshow(snimage,origin='lower',interpolation='nearest',vmin=vminsn,vmax=vmaxsn)
            ax6.set_title('SAMI S/N map')
            ax6.set_xlabel('RA (0.5\" pixels)')
            ax6.set_ylabel('Dec (0.5\" pixels)')
            ax6.set_xlim(0-0.5,xcube-0.5)
            ax6.set_ylim(0-0.5,ycube-0.5)
            fig.colorbar(im6, ax=ax6, shrink=0.8)


        # make the plot of the SDSS image:
        #py.subplot(2,3,2)
        pix2,pix1 = np.shape(SDSS_zoom)
        # Calculate the pixel values of the centre of the image.  As an example, 
        #if size is 62, then range is 0 to 61 central pixels are at 30,31 and centre is 30.5
        pix1 = (pix1-1)/2.0
        pix2 = (pix2-1)/2.0
            
        if (plot):
            #vmin=np.min(SDSS_zoom)
            #vmax=np.max(SDSS_zoom)
            #print 'SDSS scale:',vmin,vmax            
            ax2.scatter([pix1],[pix2],marker='x',facecolors='white', edgecolors='white',s=100,linewidths=1.5,label='image centre')
            # use the same scale as the SAMI image:
            im2=ax2.imshow(SDSS_zoom,origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax)
            #py.imshow(np.log(SDSS_zoom),origin='lower',interpolation='nearest')
            ax2.set_title('SDSS g-band, x=array cent')
            ax2.set_xlabel('RA (0.5\" pixels)')
            ax2.set_ylabel('Dec (0.5\" pixels)')
            ax2.set_xlim(0-0.5,len(SDSS_zoom[0])-0.5)
            ax2.set_ylim(0-0.5,len(SDSS_zoom[1])-0.5)
            cb = fig.colorbar(im2, ax=ax2, shrink=0.8)
            cb.ax.set_title('uJy')
    
    ##########

    
    if (not nominal) and np.size(np.where(image_data == 0.0)) != 2*np.size(image_data):
        # Cross-correlate normalised SAMI-cube g-band image and SDSS g-band image
        WCS_flag = 'SDSS'
        # here we use the cropped images:
        #crosscorr_image = sp.signal.correlate2d(SDSS_image_crop_norm, cube_image_crop_norm)
        # here we use the full images:
        #crosscorr_image = sp.signal.correlate2d(SDSS_image, cube_image)
        # cross-correlate the zoomed, rescaled image:
        gaussfit=False
        repxcor=True
        if (plot):
            x_offset_pix, y_offset_pix = find_offset_xcor(SDSS_zoom,cube_image,verbose=verbose,plot=plot,axcf=ax3,gaussfit=gaussfit)
        else:
            x_offset_pix, y_offset_pix = find_offset_xcor(SDSS_zoom,cube_image,verbose=verbose,plot=plot,gaussfit=gaussfit)
        print('Done 1st pass x-cor')

        # if the offset is smallish, then re-do the cross correlation with a smaller image, particularly if
        if ((abs(x_offset_pix) < 5.0) & (abs(y_offset_pix) < 5.0) & repxcor):
            SDSS_zoom2 = cutout_cent(SDSS_zoom,10)
            cube_image2 = cutout_cent(cube_image,10)
            if (plot):
                x_offset_pix, y_offset_pix = find_offset_xcor(SDSS_zoom2,cube_image2,verbose=verbose,plot=plot,axcf=ax3,gaussfit=gaussfit)
            else:
                x_offset_pix, y_offset_pix = find_offset_xcor(SDSS_zoom2,cube_image2,verbose=verbose,plot=plot,gaussfit=gaussfit)
            print('Done 2nd pass x-cor')
        
        

        x_wcs_diff = object_RA - SDSS_cent_ra
        y_wcs_diff = object_DEC - SDSS_cent_dec
        print("difference in image WCS centres (arcsec)",x_wcs_diff*3600,y_wcs_diff*3600)



#
# The true WCS for the central pixel in the SAMI cube will be the WCS in the SDSS image shifted
# by the amount found in the cross-correlation.
# NOTE!!!  Need to check the sense of this offset, and think about whether we need to apply
# a cos(dec) term to the RA coordinate.  This depends on the specific implementation of the
# TAN projection in the FITS standard:
        crval1_new = SDSS_cent_ra - x_offset_pix * output_pix_size_arcsec/3600
        crval2_new = SDSS_cent_dec + y_offset_pix * output_pix_size_arcsec/3600

        print('x,y offsets from cross-correlation',x_offset_pix,y_offset_pix)
        print('old CRVALs for the cenrre of the SAMI cubes:',object_RA,object_DEC)
        print('new CRVALs for the centre of the SAMI cubes:',crval1_new,crval2_new)

        x_offset_degree =  (object_RA - crval1_new)
        y_offset_degree =   (object_DEC - crval2_new)

        x_offset_arcsec = x_offset_degree*3600
        y_offset_arcsec = y_offset_degree*3600

        # plot the location of the SDSS image centre on the SAMI cube with the new WCS info:
        if (plot):
            #py.figure(1)
            #py.subplot(2,3,1)
            # plot the location of the centre of the SDSS image, based on the WCS from the SAMI
            # cube
            print('pix1,pix2:',pix1,pix2)
            pix1a= (SDSS_cent_ra - crval1_new)/(-1*output_pix_size_arcsec/3600.0) + 25.5 -1
            pix2a= (SDSS_cent_dec - crval2_new)/(output_pix_size_arcsec/3600.0) + 25.5 -1
            ax1.scatter([pix1a],[pix2a],marker='x',facecolors='red', edgecolors='red',s=100,linewidths=1.5,label='image centre')
            #ax1.legend(loc='upper right',prop={'size':10})


        if (verbose):
            print('estimated offset in degrees (including cos(dec) term): ',x_offset_degree,y_offset_degree)
            print('estimated offset in arcsec: ',x_offset_arcsec,y_offset_arcsec)
            print('estimated offset in pixels: ',x_offset_pix,y_offset_pix)
        
        
    else:
        WCS_flag = 'Nominal'
        y_offset_degree = 0.0
        x_offset_degree = 0.0
    
    # Create dictionary of positional WCS.  Only make the changes for the CRVAl1 and CRVAL2 values, as we are not
    # measuring the other values or changing them:
    if isinstance(xcube/2, int):
#        WCS_pos={"CRVAL1":(img_crval1 + x_offset_degree), "CRVAL2":(img_crval2 + y_offset_degree), "CRPIX1":(xcube/2 + 0.5),
#            "CRPIX2":(ycube/2 + 0.5), "CDELT1":(img_cdelt1), "CDELT2":(img_cdelt2), "CTYPE1":"RA---TAN", "CTYPE2":"DEC--TAN",
#            "CUNIT1": 'deg', "CUNIT2": 'deg'}
        WCS_pos={"CRVAL1":(crval1_new), "CRVAL2":(crval2_new),"CTYPE1":"RA---TAN", "CTYPE2":"DEC--TAN","CUNIT1": 'deg', "CUNIT2": 'deg'}
    else:
#        WCS_pos={"CRVAL1":(img_crval1 + x_offset_degree), "CRVAL2":(img_crval2 + y_offset_degree), "CRPIX1":(xcube/2),
#            "CRPIX2":(ycube/2), "CDELT1":(img_cdelt1), "CDELT2":(img_cdelt2), "CTYPE1":"RA---TAN", "CTYPE2":"DEC--TAN",
#            "CUNIT1": 'deg', "CUNIT2": 'deg'}
        WCS_pos={"CRVAL1":(crval1_new), "CRVAL2":(crval2_new), "CTYPE1":"RA---TAN", "CTYPE2":"DEC--TAN","CUNIT1": 'deg', "CUNIT2": 'deg'}
    
    ##########
    
    # Remove temporary files
#    if remove_thput_file and os.path.exists("sdss_"+str(band)+".dat"):
#        os.remove("sdss_"+str(band)+".dat")
#    if os.path.exists(str(object_name)+"_SDSS_"+str(band)+".fits"):
#        os.remove(str(object_name)+"_SDSS_"+str(band)+".fits")

    # If the fluxtest flag is on, then we will sum up the flux in a large circular aperture and make a
    # comparison between the SAMI and SDSS images.  Note that the flux units (as converted above) are
    # in microJy.
    if (fluxtest):
        rad_pos_sami=np.zeros(xcube*ycube)
        rad_flux_sami=np.zeros(xcube*ycube)
        sn_arr_sami=np.zeros(xcube*ycube)
        rad_mag_sami=np.zeros(xcube*ycube)
        npt_sami = 0
        npt_sami_sn = 0
        sami_sum = 0.0
        sami_cent_x = 25.5 
        sami_cent_y = 25.5 
        for i in range(ycube):
            for j in range(xcube):
                sep = np.sqrt((sami_cent_x-j)**2 + (sami_cent_y-i)**2)
                if ((sep < sami_radius) & (~np.isnan(cube_image[i,j]))):
                    sami_sum = sami_sum + cube_image[i,j]
                    rad_pos_sami[npt_sami] = sep
                    rad_flux_sami[npt_sami] = cube_image[i,j]
                    npt_sami = npt_sami +1
                    if ((~np.isnan(reconstruct_magsb[i,j])) & (~np.isnan(snimage[i,j]))):
                        rad_mag_sami[npt_sami_sn] = reconstruct_magsb[i,j]
                        sn_arr_sami[npt_sami_sn] = snimage[i,j]
                        npt_sami_sn = npt_sami_sn +1

        print('number of pixels summed:',npt_sami)
        #print np.shape(rad_mag_sami[0:npt_sami])
        #print np.shape(sn_arr_sami[0:npt_sami])

        # fit a polynomial to the SN data vs mag:
        #print rad_mag_sami[0:npt_sami]
        #print sn_arr_sami[0:npt_sami]
        # need to do exception handing incase the fit fails:
        try:
            snfit = np.polyfit(rad_mag_sami[0:npt_sami_sn],sn_arr_sami[0:npt_sami_sn], 4)

        except ValueError:
            f_5 = -99.9
            print('could not fit surface brightness vs. S/N')
            snfit_good = False
            
        else:
            snfit_good=True
            p = np.poly1d(snfit)
            xp = np.linspace(20,26, 100)
            yp = p(xp)
            print('polynomial fit to S/N:',snfit)
            # find the value for S/N=5 (search in range 21 - 24):
            mindif = 1.0e10
            f_5 = 0.0
            for f in np.arange(21,25,0.01):
                sndif = np.abs(p(f)-5.0)
                if (sndif < mindif):
                    mindif = sndif
                    f_5 = f
            # get S/N for a range of surface brightnesses
            sn220 = p(22.0)
            sn225 = p(22.5)
            sn230 = p(23.0)
            sn235 = p(23.5)
            sn240 = p(24.0)
            
            print('surface brightness for S/N=5:',f_5)
            if (plot):
                lab = 'SB(S/N=5)={0:6.2f}'.format(f_5)
                ax4.text(0.9,0.8,lab, horizontalalignment='right',verticalalignment='center',transform=ax4.transAxes)
                            
        # plot the aperture on the image:
        if (plot):
            circle1=py.Circle((sami_cent_y,sami_cent_x),sami_radius,color='r',fill=False)
            ax1.add_artist(circle1)

        rad_pos_sdss=np.zeros(xcube*ycube)
        rad_flux_sdss=np.zeros(xcube*ycube)
        npt_sdss = 0
        sdss_sum = 0.0
        sdss_cent_x = pix1
        sdss_cent_y = pix2 
        sdss_radius = sami_radius
        sdss_ny,sdss_nx = np.shape(SDSS_zoom)
        for i in range(sdss_ny):
            for j in range(sdss_nx):
                sep = np.sqrt((sdss_cent_x-j)**2 + (sdss_cent_y-i)**2)
                if (sep < sdss_radius):
                    sdss_sum = sdss_sum + SDSS_zoom[i,j]
                    rad_pos_sdss[npt_sdss] = sep
                    rad_flux_sdss[npt_sdss] = SDSS_zoom[i,j]
                    npt_sdss = npt_sdss +1

        print('raw SDSS flux sum:',sdss_sum,' for object: ',object_name)
        print('number of pixels summed:',npt_sdss)

        # plot the profile:
        if (plot):
            ax5.plot(rad_pos_sami[0:npt_sami],rad_flux_sami[0:npt_sami],'+',color='r',label='SAMI')
            ax5.plot(rad_pos_sdss[0:npt_sdss],rad_flux_sdss[0:npt_sdss],'+',color='b',label='SDSS')
            ax5.legend(loc='upper right',prop={'size':10})
            ax5.axhline(y=0.0,color = 'k',linestyle=':')
            lab = 'SAMI flux = {0:6.3f}'.format(sami_sum)
            ax5.text(0.9,0.8,lab, horizontalalignment='right',verticalalignment='center',transform=ax5.transAxes)
            lab = 'SDSS flux = {0:6.3f}'.format(sdss_sum)
            ax5.text(0.9,0.75,lab, horizontalalignment='right',verticalalignment='center',transform=ax5.transAxes)
            lab = 'SAMI/SDSS = {0:6.3f}'.format(sami_sum/sdss_sum)
            ax5.text(0.9,0.7,lab, horizontalalignment='right',verticalalignment='center',transform=ax5.transAxes)
            ax5.set_xlabel('Radius (0.5" pixels)')
            ax5.set_ylabel('microJy')

            # plot S/N vs. flux for SAMI:
            ax4.plot(rad_mag_sami[0:npt_sami_sn],sn_arr_sami[0:npt_sami_sn],'+',color='r',label='SAMI')
            ax4.set_xlabel('surface brightness (mag/arcsec$^2$)')
            ax4.set_ylabel('S/N')
    
            if (snfit_good):
                ax4.plot(xp,yp,color='b',linestyle='-')
                
            ax4.set_ylim(ymin=np.min(sn_arr_sami[0:npt_sami_sn]),ymax=np.max(sn_arr_sami[0:npt_sami_sn]))
            ax4.set_xlim(xmin=np.min(rad_mag_sami[0:npt_sami_sn]),xmax=np.max(rad_mag_sami[0:npt_sami_sn]))
            ax4.axhline(5.0,ls=':',color='k')

        # test whether we get the right AB mag for the object.  Need to convert back to Jy:
        mag_ab = -2.5*np.log10(sdss_sum/(1.0e6*3631.0))
        print('recalculated SDSS AB mag:',mag_ab,' for object:',object_name)
        print('SDSS catalogue PSF AB mag:',catmagg)
        print('SAMI original star mag:',magg)

        
        if (plot):
            circle1=py.Circle((sdss_cent_y,sdss_cent_x),sdss_radius,color='r',fill=False)
            ax2.add_artist(circle1)


        print('summed flux (SAMI): ',sami_sum)
        print('summed flux (SDSS): ',sdss_sum)
        print('flux ratio = ',sami_sum/sdss_sum)

    if (plot):
        fig.canvas.draw()
        
    sb_sn5 = f_5
    if (fluxtest):
        return WCS_pos,WCS_flag,x_offset_arcsec,y_offset_arcsec,sami_sum,sdss_sum,sb_sn5,sn220,sn225,sn230,sn235,sn240
    else:
        return WCS_pos,WCS_flag,x_offset_arcsec,y_offset_arcsec

################################################################################
# simple cut out of centre of array:
#
def cutout_cent(im,n):

    ny,nx = np.shape(im)

    # calculate the amount to trim off the edge.  Make sure that we take the same
    # amount off each side, so that image is still centred at the same place.
    # this may mean the output image is not exactly n*n in size, but should only be 
    # off by 1 pixel.

    xcut = int((nx - n)/2)
    ycut = int((ny - n)/2)
    
    new_im = np.copy(im[(0+ycut):(ny-ycut),(0+xcut):(nx-xcut)])
    print('TEST',np.shape(new_im),xcut,ycut)
    

    return new_im

################################################################################
# actually get the offset from cross-correlation:
#
def find_offset_xcor(SDSS_zoom,cube_image,verbose=False,plot=True,axcf=None,gaussfit=False):

    """Actually run the cross-correlation"""

    crosscorr_image = signal.correlate2d(SDSS_zoom, cube_image)
    

    # get shape of output cross-correlation:
    xcorrshape = np.shape(crosscorr_image)
    (xcs,ycs) = np.shape(crosscorr_image)
    if (verbose):
        print('shape of x-corr image: ',xcorrshape)

    # now we need to find the best peak in the cross-correlation image.  This needs to 
    # locate the peak robustly, and get the right one, without being dragged off by
    # other bright objects that might be at the end of the image.  There are a few ways
    # to do this.  A hard limit on the range of the peaks is likely to be the best approach.
    # To do this we can define a kernel that is 1 within a certain radius and zero outside it:
    max_krad=40
    lim_kernel = np.zeros(xcorrshape)
    yk,xk = np.ogrid[-xcs/2:xcs/2,-ycs/2:ycs/2]
    kmask = xk**2 + yk**2 <= max_krad**2
    lim_kernel[kmask] = 1
    masked_xcorr = crosscorr_image * lim_kernel

    # first find the index of the max value in the cross-correlation:
    ymax,xmax = np.nonzero(masked_xcorr.max() == masked_xcorr)
    if (verbose):
        print('coordinate of max x-corr:',xmax,ymax)
        print('x-corr max value:',crosscorr_image.max()) 
        print('x-corr min value:',crosscorr_image.min())
        
    # extract a region of size (n_crop x n_crop) around the peak and fit a 2D quadratic to find
    # the best peak.  n_crop should be odd.
    n_crop = 5
    crosscorr_image_crop = crosscorr_image[int(ymax-n_crop/2+1):int(ymax+n_crop/2+1),int(xmax-n_crop/2+1):int(xmax+n_crop/2+1)]
    xc = np.linspace(xmax-n_crop/2,xmax+n_crop/2,n_crop)
    yc = np.linspace(ymax-n_crop/2,ymax+n_crop/2,n_crop)
    XC, YC = np.meshgrid(xc, yc, copy=False)
    qshape = np.shape(XC)
    XC = XC.flatten()
    YC = YC.flatten()

    # fit the local region around the centre using least squares (note that this is unweighted):
    A = np.array([XC*0+1, XC, YC, XC**2, XC*YC, YC**2]).T
    B = crosscorr_image_crop.flatten()
    cf, r, rank, s = np.linalg.lstsq(A, B)

    # find the model centre, using the first derivative of the quadratic model
    # in each direction:
    mxcent = (cf[4]*cf[2] - 2*cf[5]*cf[1])/(4.0*cf[5]*cf[3] - cf[4]*cf[4])
    mycent = (cf[4]*cf[1] - 2*cf[3]*cf[2])/(4.0*cf[5]*cf[3] - cf[4]*cf[4])
        
    if (verbose):
        print('coefficients of fit: ',cf)
        print('model centre:',mxcent,mycent)
                        
    if (plot):
        #derive a model based on the best fit:
        model = cf[0] + cf[1]*XC + cf[2]*YC + cf[3]*XC*XC + cf[4]*XC*YC + cf[5]*YC*YC
        model_im = np.reshape(model,qshape)
        # place the model into the same sized array (filled with zeros):
        model_im_full = np.zeros(xcorrshape)
        model_im_full.fill(np.nan)
        model_im_full[int(ymax-n_crop/2+1):int(ymax+n_crop/2+1),int(xmax-n_crop/2+1):int(xmax+n_crop/2+1)]=model_im
        # plot cross-correlation image and mark the best fit centre:
        axcf.scatter([mxcent],[mycent],marker='x',facecolors='white', edgecolors='white',s=100,linewidths=1)
        axcf.imshow(crosscorr_image*lim_kernel,origin='lower',interpolation='nearest')
        axcf.set_title('Cross-correlation')
        axcf.set_xlabel('RA (0.5\" pixels)')
        axcf.set_ylabel('Dec (0.5\" pixels)')
        axcf.set_xlim(0-0.5,len(crosscorr_image[0])-0.5)
        axcf.set_ylim(0-0.5,len(crosscorr_image[1])-0.5)
            
                    
    # 2D Gauss Fit the cross-correlated image.  The fitter handles data in
    # 1D, so we use ravel to flatten the array.
    crosscorr_image_1d = np.ravel(crosscorr_image)
    #use for loops to recover indicies in x and y positions of flux values
    x_pos = []
    y_pos = []
    for i in range(np.shape(crosscorr_image)[0]):
        for j in range(np.shape(crosscorr_image)[1]):
            x_pos.append(i)
            y_pos.append(j)
    x_pos=np.array(x_pos)
    y_pos=np.array(y_pos)
        
    #define guess parameters for TwoDGaussFitter:
    amplitude = max(crosscorr_image_1d)
    mean_x = (np.shape(crosscorr_image)[0])/2
    mean_y = (np.shape(crosscorr_image)[1])/2
    sigma_x = 5.0
    sigma_y = 6.0
    rotation = 60.0
    offset = 4.0
    p0 = [amplitude, mean_x, mean_y, sigma_x, sigma_y, rotation, offset]
        
    # call SAMI TwoDGaussFitter
    GF2d = fitting.TwoDGaussFitter(p0, x_pos, y_pos, crosscorr_image_1d)
    # execute gauss fit using
    GF2d.fit()
    GF2d_xpos = GF2d.p[2]
    GF2d_ypos = GF2d.p[1]
        
    # reconstruct the fit.  The first array will be 1D, and then
    # we use reshape to recover the original 2D shape:
    GF2d_reconstruct=GF2d(x_pos, y_pos)
    GF2d_image = np.reshape(GF2d_reconstruct,xcorrshape)
        
    if (plot):
        axcf.plot([GF2d_xpos],[GF2d_ypos],'x',color='white')
        axcf.contour(GF2d_image,color='white')
            #py.subplot(2,3,4)
            #ax4.imshow(GF2d_image,origin='lower',interpolation='nearest')
            #ax4.plot([GF2d_xpos],[GF2d_ypos],'x',color='white')
            #ax4.contour(GF2d_image,color='white')
            #ax4.set_xlim(0-0.5,len(GF2d_image[0])-0.5)
            #ax4.set_ylim(0-0.5,len(GF2d_image[1])-0.5)


    # compare the centres derived using the gaussian and local quadratic
    # fits:
    diff = np.sqrt((mxcent-GF2d_xpos)**2 + (mycent-GF2d_ypos)**2)
    if (verbose):
        print("difference between Gaussian and local quadratic centres (pixels): ",diff)

    if (diff > 1.0):
        print("WARNING: gaussian and local quadratic centres disagree.  Diff=",diff)


    # we have found the cross-correlation in pixel space in the image, but we have to transform
    # this to an offset in the reference pixel coordinate.
    #
    # This is the process (I think):
    # 1) find WCS of the middle of the SAMI image (we already have this)
    # 2) find WCS of the middle of the SDSS image (need to calculate this and account for the
    #    fact that we have re-sampled the SDSS image.
    # 3) measure the offset in the cross-correlation function from the centre of the x-corr.
    # 4) calculate the expected offset based on WCS (i.e. items 1 and 2 above).
    # 5) calculate the offset after removing the expected difference.
            
    x_shape = len(crosscorr_image[0])
    y_shape = len(crosscorr_image[1])
    if (gaussfit):
        # use the gaussian fit to the x-corr peak to define the shift:
        x_offset_pix = GF2d_xpos - (x_shape-1.0)/2
        y_offset_pix = GF2d_ypos - (y_shape-1.0)/2
    else:
        # use the quadratic fit to the peak to find the peak:
        x_offset_pix = mxcent - (x_shape-1.0)/2
        y_offset_pix = mycent - (y_shape-1.0)/2
#

    return x_offset_pix,y_offset_pix


################################################################################
#
def update_wcs_coords(filename, nominal=False, remove_thput_file=False,update=True,plot=False,verbose=False,url_show=False):
    """Recalculate the WCS data in a SAMI datacube."""
    
    # Pick out the relevant data from header of file:
    header = pf.getheader(filename)
    ra = (header['CRVAL1'] + (1 + np.arange(header['NAXIS1']) - header['CRPIX1']) * header['CDELT1'])
    dec = (header['CRVAL2'] + (1 + np.arange(header['NAXIS2']) - header['CRPIX2']) * header['CDELT2'])
    wave = (header['CRVAL3'] + (1 + np.arange(header['NAXIS3']) - header['CRPIX3']) * header['CDELT3'])
    object_RA = np.mean(ra)
    object_DEC = np.mean(dec)
    object_name = header['NAME']
    if header['GRATID'] == '580V':
        band = 'g'
    elif header['GRATID'] == '1000R':
        band = 'r'
    else:
        raise ValueError('Could not identify band. Exiting')

    # read in cube data:    
    object_flux_cube = pf.getdata(filename)
            
    # get the important size info for cubes:
    size_of_grid = np.shape(object_flux_cube)[1] #should be = 50
    output_pix_size_arcsec = abs(header['CDELT1']*3600.0) #should be = 0.5

    if (verbose):

        print("Original cube centre (ra,dec): ",object_RA,object_DEC)
        print("size_of_grid: ",size_of_grid,", output_pix_size_arcsec: ",output_pix_size_arcsec)

    # Calculate the WCS
    WCS_pos, WCS_flag, x_offset_arcsec, y_offset_arcsec = wcs_position_coords(object_RA, object_DEC, wave, object_flux_cube, object_name, band, size_of_grid, output_pix_size_arcsec, nominal=nominal, remove_thput_file=remove_thput_file,verbose=verbose,url_show=url_show,plot=plot)

    if (verbose):
        print(WCS_pos,WCS_flag)

    # Update the file (if flag set to true):
    if (update):
        hdulist = pf.open(filename, 'update', do_not_scale_image_data=True)
        header = hdulist[0].header
        for key, value in list(WCS_pos.items()):
            header[key] = value
            header['WCS_SRC'] = WCS_flag
        hdulist.close()

    return WCS_pos,x_offset_arcsec,y_offset_arcsec

###########################################################################################

def update_wcs_coords_local(filename, nominal=False, remove_thput_file=False,update=True,plot=False,verbose=False,impath='/Users/scroom/data/sami/images/gama_cutouts/',fluxtest=True,sami_radius=12.0):
    """Recalculate the WCS data in a SAMI datacube, but use a locally stored image to compare to"""
    
    # Pick out the relevant data from header of file:
    header = pf.getheader(filename)
    # calculate the RA/Dec at the centre of the cube by averaging over the pixel coords:
    ra = (header['CRVAL1'] + (1 + np.arange(header['NAXIS1']) - header['CRPIX1']) * header['CDELT1'])
    dec = (header['CRVAL2'] + (1 + np.arange(header['NAXIS2']) - header['CRPIX2']) * header['CDELT2'])
    wave = (header['CRVAL3'] + (1 + np.arange(header['NAXIS3']) - header['CRPIX3']) * header['CDELT3'])
    object_RA = np.mean(ra)
    object_DEC = np.mean(dec)
    # get other header info:
    object_name = header['NAME']
    if header['GRATID'] == '580V':
        band = 'g'
    elif header['GRATID'] == '1000R':
        band = 'r'
    else:
        raise ValueError('Could not identify band. Exiting')

    # read in cube data:    
    object_flux_cube = pf.getdata(filename)
    # get variance cube:
    object_var_cube = pf.getdata(filename,extname='VARIANCE')
    
            
    # get the important size info for cubes:
    size_of_grid = np.shape(object_flux_cube)[1] #should be = 50
    output_pix_size_arcsec = abs(header['CDELT1']*3600.0) #should be = 0.5

    if (verbose):
        print("Original cube centre (ra,dec): ",object_RA,object_DEC)
        print("size_of_grid: ",size_of_grid,", output_pix_size_arcsec: ",output_pix_size_arcsec)

    # Calculate the WCS
    if (fluxtest):
        WCS_pos, WCS_flag, x_offset_arcsec, y_offset_arcsec, sami_sum, SDSS_sum, sb_sn5, sn220, sn225, sn230, sn235, sn240 = wcs_position_coords_local(filename,object_RA, object_DEC, wave, object_flux_cube, object_var_cube, object_name, band, size_of_grid, output_pix_size_arcsec, nominal=nominal, remove_thput_file=remove_thput_file,verbose=verbose,plot=plot,impath=impath,fluxtest=fluxtest,sami_radius=sami_radius)
    else:
        WCS_pos, WCS_flag, x_offset_arcsec, y_offset_arcsec = wcs_position_coords_local(filename,object_RA, object_DEC, wave, object_flux_cube, object_var_cube, object_name, band, size_of_grid, output_pix_size_arcsec, nominal=nominal, remove_thput_file=remove_thput_file,verbose=verbose,plot=plot,impath=impath,fluxtest=fluxtest,sami_radius=sami_radius)

    # do not do the update of the WCS in the file here anymore,
    # do it in the higher level routine.


    if (fluxtest):
        return WCS_pos,WCS_flag,x_offset_arcsec,y_offset_arcsec,sami_sum,SDSS_sum, sb_sn5,sn220,sn225,sn230,sn235,sn240
    else:
        return WCS_pos,WCS_flag,x_offset_arcsec,y_offset_arcsec

##########################################################################################################

def update_wcs_coords_multi(in_cube_list,plot=False,plotall=False,verbose=False,update=False,fluxtest=True,impath='/Users/scroom/data/sami/images/gama_cutouts/',interactive=False,outpath='',sami_radius=12.0,listfile=None):
    """Function to update WCS centres for multiple SAMI cubes.  To do this it calls
       update_wcs_coords().  The WCS is compared to that in an SDSS image.  A collapsed
       cube and an SDSS image (that has previously been downloaded) are cross-correlated and the offset
       in the cross correlation is used to define an offset for the WCS.  This version
       allows many files to be updated at once.  The comparison is made between the SAMI
       blue cube and SDSS g-band images.  The red cube WCS is then updated to match the blue
       cube. 

       Usage is typically:
       > sami.general.wcs.update_wcs_coords_multi('*blue*.fits',plot=False,plotall=False,verbose=False,update=True)

       or if you want just want to do all the checks, plots etc:
       > wcs_test.update_wcs_coords_multi('*blue*.fits',plot=True,plotall=True,verbose=True,update=False,outpath='.')

       # interactive, useful for checking one-by-one:
       > wcs_test.update_wcs_coords_multi('*blue*.fits',plot=True,plotall=True,verbose=True,update=False,outpath='.'interactive=True)

        run on bill:
        > wcs_test.update_wcs_coords_multi('*blue*.fits',plot=False,plotall=False,verbose=False,update=True,impath='/import/bill1/sami_data/imaging/gama/SDSS_gama_cutouts/')

        run on bill for flux test:
        > wcs_test.update_wcs_coords_multi('*blue*.fits',plot=False,plotall=False,verbose=False,update=False,fluxtest=True,impath='/import/bill1/sami_data/imaging/gama/SDSS_gama_cutouts/')
        with plots:
        > wcs_test.update_wcs_coords_multi('*blue*.fits',plot=True,plotall=True,verbose=False,update=False,fluxtest=True,impath='/import/bill1/sami_data/imaging/gama/SDSS_gama_cutouts/')
        with plots+interactive:
        > wcs_test.update_wcs_coords_multi('*blue*.fits',plot=True,plotall=True,verbose=False,update=False,fluxtest=True,interactive=True,impath='/import/bill1/sami_data/imaging/gama/SDSS_gama_cutouts/')
       
       Note that just the blue cubes are passed to the script, the matching red cubes are
       then located and updated as well.

       There are a number of tests that need to be done.  In particular we need to flag objects that outliers in
       that there is a large shift between the original and new cross-correlation WCS.  We can run the code in
       auto mode and then get it to output the results to a file.  The results can then be passed to the code
       again and only the objects with large offsets can be looked at individually.
       
       """

    # better fonts:
    py.rcParams.update({'font.size': 6})
    
    # define start time to see how long this takes to run:
    start_time = time.time()

    # define parameters that control some of the functionality of the code:
    #
    # value above which to output object in a list of potential outliers:
    outlier_lim = 0.5
    # plot limits (apply in both RA and Dec, i.e. x and y):
    xmin = -2.0
    xmax= 2.0
    # number of sigma to clip to get clipped rms of distribution of position differences:
    nsig = 3.0
        

    print('looking for images in:',impath)

    # get the list of cube files to update:
    # in the default mode there is no list file and just use the globbed
    # list from the input
    if (listfile == None):
        files_tmp = glob.glob(in_cube_list)

        # get rid of any aperture spectra in the list:
        files = [i for i in files_tmp if 'apspec' not in i]

    else:
        # read the listfile:
        files = np.genfromtxt(listfile,dtype='str')[:,0]
        
    #print(files)


    # define arrays for each object:
    nfiles = np.size(files)
    print("starting to correct WCS for ",nfiles," cubes")
    CRVAL1_orig = np.zeros(nfiles)
    CRVAL2_orig = np.zeros(nfiles)
    CRVAL1_new = np.zeros(nfiles)
    CRVAL2_new = np.zeros(nfiles)
    sdss_sum = np.zeros(nfiles)
    sami_sum = np.zeros(nfiles)
    surb_sn5 = np.zeros(nfiles)
    sn220a = np.zeros(nfiles)
    sn225a = np.zeros(nfiles)
    sn230a = np.zeros(nfiles)
    sn235a = np.zeros(nfiles)
    sn240a = np.zeros(nfiles)
    flux_ratio = np.zeros(nfiles)
    catmagg = np.zeros(nfiles)
    magg = np.zeros(nfiles)
    rescale = np.ones(nfiles)
    samifwhm = np.zeros(nfiles)
    basename = np.empty(nfiles,dtype='S64')
    dirname = np.empty(nfiles,dtype='S128')
    
    # loop over each cube
    nfile = 0
    for file in files:

        # split file name into path and actual file:
        basename[nfile]=os.path.basename(file)
        dirname[nfile]=os.path.dirname(file)
        
        if (verbose):
            print("updating WCS for: ",file)

        #first read the header for the file to read in the original WCS CRVAL1 and CRVAL2:
        header = pf.getheader(file)
        CRVAL1_orig[nfile] = header['CRVAL1']
        CRVAL2_orig[nfile] = header['CRVAL2']
        # if this is a calibration star, get the mags etc:
        try: 
            magg[nfile] = float(header['MAGG'])
            catmagg[nfile] = float(header['CATMAGG'])
        except KeyError:
            magg[nfile] = np.nan
            catmagg[nfile] = np.nan

        psfalpha = float(header['PSFALPHA'])
        psfbeta = float(header['PSFBETA'])
        psffwhm = float(header['PSFFWHM'])
        samifwhm[nfile] = psffwhm
        #rescale[nfile] = float(header['RESCALE'])
        
        # next call the main routine that finds the new WCS.  Note that we do NOT update the WCS with this call, although we could
        # instead we will do the update in the main code below at the same time we update the red cube.
        # TODO: Need to have a error handling line here for cases when this fails (e.g. failed to get the SDSS image):
        if (fluxtest):
            WCS_pos,WCS_flag,x_offset_arcsec,y_offset_arcsec, SAMI_sum,SDSS_sum, sb_sn5,sn220,sn225,sn230,sn235,sn240 = update_wcs_coords_local(file,impath=impath,update=False,plot=plotall,verbose=verbose,fluxtest=fluxtest,sami_radius=sami_radius)
        else:
            WCS_pos,WCS_flag,x_offset_arcsec,y_offset_arcsec = update_wcs_coords_local(file,impath=impath,update=False,plot=plotall,verbose=verbose,fluxtest=fluxtest,sami_radius=sami_radius)

        # store the new values:
        CRVAL1_new[nfile] = WCS_pos["CRVAL1"]
        CRVAL2_new[nfile] = WCS_pos["CRVAL2"]
        sami_sum[nfile] = SAMI_sum
        sdss_sum[nfile] = SDSS_sum
        surb_sn5[nfile] = sb_sn5
        sn220a[nfile] = sn220
        sn225a[nfile] = sn225
        sn230a[nfile] = sn230
        sn235a[nfile] = sn235
        sn240a[nfile] = sn240

        if (verbose):
            print("old CRVAL1/2:",CRVAL1_orig[nfile],CRVAL2_orig[nfile])
            print("new CRVAL1/2:",CRVAL1_new[nfile],CRVAL2_new[nfile])
            print("difference (arcsec):",(CRVAL1_orig[nfile]-CRVAL1_new[nfile])*3600,(CRVAL2_orig[nfile]-CRVAL2_new[nfile])*3600)
        

        # if we go for interactive mode, then ask the user whether this object is okay or not.
        # if not okay, need to select some other options as to why.
        # TODO: add options if not okay - at the moment, it just pauses and does not do anything else
        if (interactive):
            # wait for user input:
            print("Is this okay (Y/N)?")
            yntest = input()


        
        # Update the file (if flag set to true):
        if (update):
            # first update the blue file that we made the measurement on:
            hdulist = pf.open(file, 'update', do_not_scale_image_data=True)
            header = hdulist[0].header
            for key, value in list(WCS_pos.items()):
                header[key] = value
                header['WCS_SRC'] = WCS_flag
            hdulist.close()

            # next update the red file to be the same:
            redfile = file.replace('blue','red')
            hdulist = pf.open(redfile, 'update', do_not_scale_image_data=True)
            header = hdulist[0].header
            for key, value in list(WCS_pos.items()):
                header[key] = value
                header['WCS_SRC'] = WCS_flag
            hdulist.close()
            

        print('processed file:',nfile)
        nfile = nfile + 1

    
    # now that we have calculated all the offsets, generate some statistics
    # and plots:
    #
    # calculate the mean offsets an RMS:
    ra_off = (CRVAL1_new - CRVAL1_orig)*3600.0
    dec_off = (CRVAL2_new - CRVAL2_orig)*3600.0
    #
    # get unclipped RMS:
    ra_off_rms_noclip = np.nanstd(ra_off)
    ra_off_mean_noclip = np.nanmean(ra_off)
    dec_off_rms_noclip = np.nanstd(dec_off)
    dec_off_mean_noclip = np.nanmean(dec_off)

    # clip the offsets to remove outliers, to get a robust RMS:
    ra_off_clipped = reject_outliers_iter(ra_off, m=nsig,verbose=False)
    dec_off_clipped = reject_outliers_iter(dec_off, m=nsig,verbose=False)

    ra_off_rms = np.nanstd(ra_off_clipped)
    ra_off_mean = np.nanmean(ra_off_clipped)
    dec_off_rms = np.nanstd(dec_off_clipped)
    dec_off_mean = np.nanmean(dec_off_clipped)

    # calc the flux from the catalogue magnitude:
    catfluxg = 1.0e6 * 3631.0 * 10.0**(-0.4*catmagg)
    # calc the flux from the sami mag from the pipeline:
    samifluxg = 1.0e6 * 3631.0 * 10.0**(-0.4*magg)
    # -2.5*np.log10(sdss_sum/(1.0e6*3631.0))
        
    # if doing flux test, get stats:
    if (fluxtest):
        flux_ratio = sami_sum/sdss_sum
        med_flux_ratio = np.median(flux_ratio)
        print('median flux ratio',med_flux_ratio)
        #normalize by median:
        flux_ratio_rms = np.sqrt(np.sum(((flux_ratio-med_flux_ratio)**2)/nfiles))
        print('rms flux ratio:',flux_ratio_rms)
        flux_ratio_rms = flux_ratio_rms/med_flux_ratio
        print('rms flux ratio (scaled):',flux_ratio_rms)
    
    # plot histograms:
    if (plot):
    # plot the scatter in measured offsets:
        py.figure(2)
        py.clf()
        py.subplot(2,2,1)
        py.scatter(ra_off,dec_off,marker='x',facecolors='black', edgecolors='black',s=100,linewidths=1.0)
        py.xlabel('RA offset (arcsec)')
        py.ylabel('Dec offset (arcsec)')
        py.xlim(xmin = xmin, xmax = xmax)
        py.ylim(ymin = xmin, ymax = xmax)
        py.axhline(y=0.0,color = 'k')
        py.axvline(x=0.0,color = 'k')
        # this doesn't work:
        py.Circle((0,0),0.5,color='b',fill=True)
        # plot histograms for each direction:
        py.subplot(2,2,3)
        n, bins, patches = py.hist(ra_off, bins=40, range=(xmin,xmax),histtype='step',color='k')
        ymin, ymax = py.ylim()
        yrange = (ymax-ymin)
        py.ylim(ymin=ymin,ymax=ymax+0.1*yrange)
        py.xlabel('RA offset (arcsec)')
        py.axvline(x=0.0,ls=':',color='k')
        py.subplot(2,2,2)
        n, bins, patches = py.hist(dec_off, bins=40, range=(xmin,xmax),histtype='step',orientation='horizontal',color='k')
        xmin, xmax = py.xlim()
        xxrange = (xmax-xmin)
        py.xlim(xmin=xmin,xmax=xmax+0.1*xxrange)
        py.xlabel('Dec offset (arcsec)')
        py.axhline(y=0.0,ls=':',color='k')
        # plot a gaussian with the correct RMS to match the histograms:

        # if doing the flux ratios, plot a histogram of them
        if (fluxtest & plot):
            py.subplot(2,2,4)
            n, bins, patches = py.hist(flux_ratio, bins=20,histtype='step',color='k')

            # also plot the flux ratio vs offset:
            py.figure(3)


            py.rcParams.update({'font.size': 8})
            py.subplot(3,2,1)
            py.scatter(samifluxg/catfluxg,sami_sum/sdss_sum,marker='x',facecolors='black', edgecolors='black',linewidths=1.0,label='rescaled')
            py.scatter(samifluxg/catfluxg,(sami_sum/rescale)/sdss_sum,marker='x',facecolors='red',label='no rescale')
            py.xlabel('SAMImag/SDSSmag')
            py.ylabel('SAMIim/SDSSmag')
            py.legend(loc='upper right',prop={'size':8})
            
            # plot the difference in mag for various different measurements:
            py.subplot(3,2,2)
            py.scatter(samifluxg/catfluxg,1/rescale,marker='x',facecolors='black', edgecolors='black',linewidths=1.0)
            py.xlabel('SAMImag/SDSSmag')
            py.ylabel('1/rescale')
            py.plot([0.9,1.1],[0.9,1.1],color='k',linestyle='-')
            
            py.subplot(3,2,3)
            py.scatter(samifluxg/catfluxg,(sami_sum/rescale)/sdss_sum,marker='x',facecolors='black')
            py.xlabel('SAMImag/SDSSmag')
            py.ylabel('SAMIim/SDSSmag (no rescale)')
            # plot diffs against each other:
            py.subplot(3,2,4)
            py.scatter(flux_ratio,catfluxg/sdss_sum,marker='x',facecolors='black')
            py.plot([0.9,1.2],[0.9,1.2],color='k',linestyle='-')
            py.xlabel('SAMIim/SDSSim flux')
            py.ylabel('SDSScat/SDSSim flux')

            # now plot a comparision of SAMI fluxes:
            py.subplot(3,2,5)
            py.scatter(flux_ratio,sami_sum/samifluxg,marker='x',facecolors='black')
            py.plot([0.9,1.2],[0.9,1.2],color='k',linestyle='-')
            py.xlabel('SAMIim/SDSSim flux')
            py.ylabel('SAMIim/SAMImag flux')
            
            py.subplot(3,2,6)
            py.scatter(flux_ratio,samifluxg/sdss_sum,marker='x',facecolors='black')
            py.plot([0.9,1.2],[0.9,1.2],color='k',linestyle='-')
            py.xlabel('SAMIim/SDSSim flux')
            py.ylabel('SAMImag/SDSSim flux')

            py.tight_layout()
            
    # output list of object that have outliers in the
    print("outliers:")
    
    for i in range(nfiles):
        if ((abs(ra_off[i]) > outlier_lim) or (abs(dec_off[i]) > outlier_lim)):
            print(files[i],CRVAL1_orig[i],CRVAL2_orig[i],CRVAL1_new[i],CRVAL2_new[i],ra_off[i],dec_off[i])

    print("total number of files tested:",nfiles)
    print("number of RAs clipped: ",np.size(ra_off)-np.size(ra_off_clipped))
    print("number of Decs clipped: ",np.size(dec_off)-np.size(dec_off_clipped))
    print("offset statistics (no clipping):")
    print("mean RA and Dec offsets: ",ra_off_mean_noclip,dec_off_mean_noclip)
    print("sigma of RA and Dec offsets: ",ra_off_rms_noclip,dec_off_rms_noclip)
    print("offset statistics clipping outliers:")
    print("mean RA and Dec offsets: ",ra_off_mean,dec_off_mean)
    print("sigma of RA and Dec offsets: ",ra_off_rms,dec_off_rms)

    # finally, create new FITS binary table that contains the old snd new WCS values so that these can
    # be placed into a data product.
    #
    # we need to start by making the columns:
    col1 = pf.Column(name='Cubename', format='64A', array=basename)
    col2 = pf.Column(name='path', format='128A', array=dirname)
    col3 = pf.Column(name='CRVAL1_orig', format='D', array=CRVAL1_orig,unit='degrees')
    col4 = pf.Column(name='CRVAL2_orig', format='D', array=CRVAL2_orig,unit='degrees')
    col5 = pf.Column(name='CRVAL1_new', format='D', array=CRVAL1_new,unit='degrees')
    col6 = pf.Column(name='CRVAL2_new', format='D', array=CRVAL2_new,unit='degrees')
    col7 = pf.Column(name='SAMI_flux_sum', format='D', array=sami_sum,unit='microJy')
    col8 = pf.Column(name='SDSS_flux_sum', format='D', array=sdss_sum,unit='microJy')
    col9 = pf.Column(name='SB_SN_5', format='D', array=surb_sn5,unit='mag arcsec-2')
    col10 = pf.Column(name='SN_220', format='D', array=sn220a,unit=' ')
    col11 = pf.Column(name='SN_225', format='D', array=sn225a,unit=' ')
    col12 = pf.Column(name='SN_230', format='D', array=sn230a,unit=' ')
    col13 = pf.Column(name='SN_235', format='D', array=sn235a,unit=' ')
    col14 = pf.Column(name='SN_240', format='D', array=sn240a,unit=' ')
    col15 = pf.Column(name='SDSS_psf_mag', format='D', array=catfluxg,unit=' ')
    col16 = pf.Column(name='SAMI_psf_mag', format='D', array=samifluxg,unit=' ')
    col17 = pf.Column(name='rescale', format='D', array=rescale,unit=' ')

    
    # next we make the column definitions:
    cols = pf.ColDefs([col1, col2, col3, col4, col5, col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17])

    #Now, create a new binary table HDU object:
    tbhdu = pf.BinTableHDU.from_columns(cols)

    # finally write the table HDU to a file:
    outfile = outpath+'WCS_crval_crosscorr.fits'
    print('Writing results to ',outfile)
    tbhdu.writeto(outfile,clobber=True)



    # output time taken:
    print(("Time taken %s sec" % (time.time() - start_time)))

    
                
    return

##################################################################################
# A function to analyse the output of the WCS testing code.  It reads in the 
# output FITS binary table from update_wcs_coords_multi(), makes some plots
# and calculates some statistics.  Most of these are taken from the plots
# and stats at the end of update_wcs_coords_multi()
#
# example:
# wcs_test.wcs_coords_test('WCS_crval_crosscorr_flux_v0.10_8pixrad.fits',fluxlim=100)
#
#
def wcs_coords_test(infile,nsigclip=3.0,dopdf=True,catfile='None',fluxlim=0.0,col='k',lstyle='solid',rem_rescale=False):

    
    xmin=-2.5
    xmax=2.5
    
    outlier_lim1 = 1.0
    outlier_lim2 = 3.0

    outlier_flux = 0.2
    
    # if we want a pdf make it:
    if (dopdf):
        pdf = PdfPages('wcs_coords_test.pdf')

    
    # open and read FITS file:
    hdulist = pf.open(infile)
    tbdata = hdulist[1].data
    print(hdulist[1].columns)

    # get columns:
    cubenames = tbdata['Cubename']
    paths = tbdata['path']
    CRVAL1_orig = tbdata['CRVAL1_orig'] 
    CRVAL2_orig = tbdata['CRVAL2_orig'] 
    CRVAL1_new = tbdata['CRVAL1_new'] 
    CRVAL2_new = tbdata['CRVAL2_new']
    SAMI_flux_sum = tbdata['SAMI_flux_sum']
    SDSS_flux_sum = tbdata['SDSS_flux_sum']
    sb_sn5 = tbdata['SB_SN_5']
    sn220 = tbdata['SN_220']
    sn225 = tbdata['SN_225']
    sn230 = tbdata['SN_230']
    sn235 = tbdata['SN_235']
    sn240 = tbdata['SN_240']
    rescale =  tbdata['rescale']
    
    nall = np.size(cubenames)
    print('number of results read:',nall)
    
    # If require read in a catalogue file that is the list of
    # objects or cubes that we want to test.    
    if (catfile != 'None'):
        good = np.zeros((nall),dtype=bool)
        ncat = 0
        catfitsname = []
        # go through the cat file to get the FITS file names to use:
        print('Cross-matching with catalogue file: ',catfile)
        f = open(catfile,'r')
        for line in f:
            vals = line.split()
            fitsfile = (vals[1].strip('.gz')).replace('red','blue')
            print(fitsfile)
            catfitsname.append(fitsfile)
            ncat = ncat +1
        f.close()
        print('number of catalogued files read:',ncat)
        # now go through all the data read in and see if they are on the list:
        for i in range(nall):
            for j in range(ncat):
                if (cubenames[i] == catfitsname[j]):
                    #print "match!",cubenames[i],catfitsname[j]
                    good[i] = True
            if (not good[i]):
                print("no match found for ",cubenames[i])
                    
    else:
        # if a file is not give, assume all are good:
        good = np.ones((nall),dtype=bool)
        
    # define the index list of good values:
    igood = np.where((good))

    # if fluxlim > 0, then apply a cut to only analyse brighter objects.  ALso check
    # for objects that don't have good values.  We can test for this using CRVAL1_new
    if (fluxlim > 0):
        igood = np.where((good & (SDSS_flux_sum > fluxlim) & (CRVAL1_new != 0) & np.isfinite(CRVAL1_new)))
    else:
        igood = np.where((good & (CRVAL1_new != 0) & np.isfinite(CRVAL1_new)))
    
        
    print('number of good objects to use:',np.size(cubenames[igood]))
    
        
    if (rem_rescale):
        ratio = (SAMI_flux_sum/SDSS_flux_sum)/(rescale)
    else:
        ratio = SAMI_flux_sum/SDSS_flux_sum
        
    nrow = np.size(CRVAL1_orig[igood])
    print('number of rows:',nrow)

    # calculate the mean offsets an RMS:
    ra_off = (CRVAL1_new - CRVAL1_orig)*3600.0
    dec_off = (CRVAL2_new - CRVAL2_orig)*3600.0

    # get radial magnitude of offset:
    r_off = np.sqrt(ra_off**2 + dec_off**2)
    
    #
    # get unclipped RMS:
    ra_off_rms_noclip = np.nanstd(ra_off[igood])
    ra_off_mean_noclip = np.nanmean(ra_off[igood])
    dec_off_rms_noclip = np.nanstd(dec_off[igood])
    dec_off_mean_noclip = np.nanmean(dec_off[igood])
    r_off_rms_noclip = np.nanstd(r_off[igood])
    r_off_mean_noclip = np.nanmean(r_off[igood])

    ratio_mean_noclip = np.nanmean(ratio[igood])
    ratio_rms_noclip = np.nanstd(ratio[igood])
    ratio_median_noclip = np.nanmedian(ratio[igood])
    ratio_158_noclip = np.nanpercentile(ratio[igood],15.8)
    ratio_841_noclip = np.nanpercentile(ratio[igood],84.1)
    ratio_95_noclip = np.nanpercentile(ratio[igood],95.0)
    ratio_05_noclip = np.nanpercentile(ratio[igood],5.0)

    sb_mean = np.nanmean(sb_sn5[igood])
    sb_rms = np.nanstd(sb_sn5[igood])
    sb_median = np.nanmedian(sb_sn5[igood])
    
    sn220_median = np.nanmedian(sn220[igood])
    sn225_median = np.nanmedian(sn225[igood])
    sn230_median = np.nanmedian(sn230[igood])
    sn235_median = np.nanmedian(sn235[igood])
    sn240_median = np.nanmedian(sn240[igood])

    # clip the offsets to remove outliers, to get a robust RMS:
    ra_off_clipped = reject_outliers_iter(ra_off[igood], m=nsigclip,verbose=False)
    dec_off_clipped = reject_outliers_iter(dec_off[igood], m=nsigclip,verbose=False)

    ra_off_rms = np.nanstd(ra_off_clipped)
    ra_off_mean = np.nanmean(ra_off_clipped)
    dec_off_rms = np.nanstd(dec_off_clipped)
    dec_off_mean = np.nanmean(dec_off_clipped)

    # define a radial distance that is relative to the corrected mean centre:
    r_off_corrected = np.sqrt((ra_off-ra_off_mean)**2 + (dec_off-dec_off_mean)**2)
    #r_off_corrected_rms = np.sqrt(np.nanmean(r_off_corrected**2))

    # clip the coordinates based on a N-sigma radial cut:
    nrej = 1
    niter = 0
    max_iter=10
    ndata1 = np.size(r_off_corrected[igood])
    data_tmp =  r_off_corrected[igood]
    while (nrej > 0):

        # get rms:
        rms = np.sqrt(np.nanmean(data_tmp**2))
        new = data_tmp[abs(data_tmp) < nsigclip * rms]
        ndata2 = np.size(new)
        nrej = ndata1 - ndata2
        niter = niter+1        
        print('iteration ',niter,', n_rej = ',nrej,ndata1,ndata2,rms)
            
        data_tmp = np.copy(new)
        ndata1 = ndata2

        if (niter > max_iter):
            print("Maximum iterations reached:",niter)                
            break 
        
    r_off_corrected_clipped = new

    #exit()
    
    r_off_clipped = reject_outliers_iter(r_off[igood], m=nsigclip,verbose=False)

    ratio_clipped = reject_outliers_iter(ratio[igood], m=nsigclip,verbose=False)
    

    # need to think more carefully about the rms of the radial offset, as this is not
    # uniformly distributed around the mean (as its the two offsets added in quadrature).
    # we actually want the rms around the mean position:

    #print r_off_corrected_rms
    #exit()
    
    r_off_rms = np.nanstd(r_off_clipped)
    r_off_mean = np.nanmean(r_off_clipped)

    ratio_mean = np.nanmean(ratio_clipped)
    ratio_rms = np.nanstd(ratio_clipped)
    ratio_median = np.nanmedian(ratio_clipped)


    # plot the scatter in measured offsets:
    #py.figure(1)
    #py.clf()
    py.rcParams.update({'font.size': 10})
    py.rcParams.update({'lines.linewidth': 2})
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(2,2,1)
    ax2 = fig1.add_subplot(2,2,2)
    ax3 = fig1.add_subplot(2,2,3)
    ax4 = fig1.add_subplot(2,2,4)
    ax1.text(0.9,0.9,'a)', horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)
    ax1.scatter(ra_off[igood],dec_off[igood],marker='x',facecolors='black', edgecolors='black',s=100,linewidths=1.0)
    ax1.set_xlabel('RA offset (arcsec)',labelpad=1)
    ax1.set_ylabel('Dec offset (arcsec)',labelpad=1)
    ax1.set_xlim(xmin = xmin, xmax = xmax)
    ax1.set_ylim(ymin = xmin, ymax = xmax)
    #ax1.set_aspect('equal')
    ax1.axhline(y=0.0,color = 'k')
    ax1.axvline(x=0.0,color = 'k')
    #ax1.set_aspect('equal')
        # this doesn't work:
    #py.Circle((0,0),0.5,color='b',fill=True)
        # plot histograms for each direction:

    # plot RA scatter:
    ax3.text(0.9,0.9,'c)', horizontalalignment='center',verticalalignment='center',transform=ax3.transAxes)
    n, bins, patches = ax3.hist(ra_off[igood], bins=50, range=(xmin,xmax),histtype='step',color='k')
    ymin, ymax = ax3.get_ylim()
    yrange = (ymax-ymin)
    ax3.set_xlim(xmin = xmin, xmax = xmax)
    ax3.set_ylim(ymin=ymin,ymax=ymax+0.1*yrange)
    ax3.set_ylabel('Number',labelpad=1)
    ax3.set_xlabel('RA offset (arcsec)',labelpad=1)
    ax3.axvline(x=0.0,ls=':',color='k')

    # plot dec scatter:
    ax2.text(0.9,0.9,'b)', horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes)
    n, bins, patches = ax2.hist(dec_off[igood], bins=50, range=(xmin,xmax),histtype='step',orientation='horizontal',color='k')
    xxmin, xxmax = ax2.get_xlim()
    xxrange = (xxmax-xxmin)
    ax2.set_ylim(ymin = xmin, ymax = xmax)
    ax2.set_xlim(xmin=xxmin,xmax=xxmax+0.1*xxrange)
    ax2.set_ylabel('Dec offset (arcsec)',labelpad=1)
    ax2.set_xlabel('Number',labelpad=1)
    ax2.axhline(y=0.0,ls=':',color='k')
    
    #plot histogram of radial offsets:
    ax4.text(0.9,0.9,'d)', horizontalalignment='center',verticalalignment='center',transform=ax4.transAxes)
    n, bins, patches = py.hist(r_off[igood], bins=50, range=(0,xmax),histtype='step',color='k')
    ax4.set_xlabel('Radial offset (arcsec)',labelpad=1)
    ax4.set_ylabel('Number',labelpad=1)
    # this doesn't work in a script on a mac:
    #py.tight_layout()


    
    # output list of object that have outliers in the coordinates:
    print("outliers:")
    f=open('coord_outliers.txt','w')
    
    print("Outliers for coordinates ratio (d(ra) or d(dec)> {0:5.2f})".format(outlier_lim1))
    for i in range(nall):
        if (((abs(ra_off[i]) > outlier_lim1) or (abs(dec_off[i]) > outlier_lim1)) and (good[i])):
            print(cubenames[i],CRVAL1_orig[i],CRVAL2_orig[i],CRVAL1_new[i],CRVAL2_new[i],ra_off[i],dec_off[i])
            outstr = '{0:s}/{1:s} {2:6.3f} {3:6.3f} {4:6.3f} {5:6.3f} {6:6.3f}\n'.format(paths[i],cubenames[i],SAMI_flux_sum[i],SDSS_flux_sum[i],ratio[i],ra_off[i],dec_off[i])
            f.write(outstr)

    print("")
    print("ExtremeOutliers for coordinates ratio (d(ra) or d(dec)> {0:5.2f})".format(outlier_lim2))
    for i in range(nall):
        if (((abs(ra_off[i]) > outlier_lim2) or (abs(dec_off[i]) > outlier_lim2)) and (good[i])):
            print(cubenames[i],CRVAL1_orig[i],CRVAL2_orig[i],CRVAL1_new[i],CRVAL2_new[i],ra_off[i],dec_off[i])
            
    print("")
    print("Exteme outliers for flux ratio (d(ratio) > {0:5.2f})".format(outlier_flux))
    print("Cube         CRVAL1_orig CRVAL2_orig CRVAL1_new CRVAL2_new RA_off dec_off SAMI_flux SDSS_flux SAMI/SDSS")

    f.close()

    # check putliers in flux:
    nextreme_flux=0
    f=open('flux_outliers.txt','w')
    
    for i in range(nall):
        if ((abs(ratio[i]-ratio_mean) > outlier_flux) and (good[i]) and (SDSS_flux_sum[i] > fluxlim)):
            print(cubenames[i],CRVAL1_orig[i],CRVAL2_orig[i],CRVAL1_new[i],CRVAL2_new[i],ra_off[i],dec_off[i],SAMI_flux_sum[i],SDSS_flux_sum[i],ratio[i])
            nextreme_flux = nextreme_flux + 1
            outstr = '{0:s}/{1:s} {2:6.3f} {3:6.3f} {4:6.3f} {5:6.3f} {6:6.3f}\n'.format(paths[i],cubenames[i],SAMI_flux_sum[i],SDSS_flux_sum[i],ratio[i],ra_off[i],dec_off[i])
            f.write(outstr)
            
    print("total number of extreme outliers in flux:",nextreme_flux)
    f.close()
    
            
    print("total number of files tested:",nrow)
    print("number of RAs clipped: ",np.size(ra_off[igood])-np.size(ra_off_clipped))
    print("number of Decs clipped: ",np.size(dec_off[igood])-np.size(dec_off_clipped))
    print("number of radial coords clipped: ",np.size(r_off[igood])-np.size(r_off_corrected_clipped))
    print("offset statistics (no clipping):")
    print("mean RA, Dec, radial offsets: ",ra_off_mean_noclip,dec_off_mean_noclip,r_off_mean_noclip)
    print("sigma of RA, Dec, radial offsets: ",ra_off_rms_noclip,dec_off_rms_noclip,r_off_rms_noclip)
    print("offset statistics clipping outliers:")
    print("mean RA, Dec, radial offsets: ",ra_off_mean,dec_off_mean,r_off_mean)
    print("sigma of RA, Dec, radial offsets: ",ra_off_rms,dec_off_rms,r_off_rms)
    print("error on mean of RA, Dec, radial offsets: ",ra_off_rms/np.sqrt(np.size(ra_off_clipped)),dec_off_rms/np.sqrt(np.size(dec_off_clipped)),r_off_rms/np.sqrt(np.size(r_off_clipped)))

    print("")
    print("9th, 50th 68th 90th percentile radial offsets",np.percentile(r_off[igood],5.0),np.percentile(r_off[igood],50.0),np.percentile(r_off[igood],68.0),np.percentile(r_off[igood],90.0))
    print("9th, 50th 68th 90th percentile radial offsets (clipped)",np.percentile(r_off_corrected_clipped,5.0),np.percentile(r_off_corrected_clipped,50.0),np.percentile(r_off_corrected_clipped,68.0),np.percentile(r_off_corrected_clipped,90.0))

    #r_off_sort = np.sort(r_off_corrected_clipped)
    #for i in xrange(np.size(r_off_sort)):
    #    print i,r_off_sort[i],float(i)/np.size(r_off_sort)
    
    print("")
    print("flux ratio stats (no clipping):",np.size(ratio[igood])," objects")
    print("mean SAMI/SDSS flux ratio (no clipping): ",ratio_mean_noclip,'+-',ratio_rms_noclip/np.sqrt(np.size(ratio[igood])))
    print("median SAMI/SDSS flux ratio (no clipping): ",ratio_median_noclip,'+-',1.253*ratio_rms_noclip/np.sqrt(np.size(ratio[igood])))    
    print("sigma SAMI/SDSS flux ratio (no clipping): ",ratio_rms_noclip)
    print("flux ratio stats (clipping):",np.size(ratio_clipped)," objects.  Number clipped: ",np.size(ratio[igood])-np.size(ratio_clipped))
    print("mean SAMI/SDSS flux ratio (clipping): ",ratio_mean,'+-',ratio_rms/np.sqrt(np.size(ratio_clipped)))
    print("median SAMI/SDSS flux ratio (clipping): ",ratio_median,'+-',1.253*ratio_rms/np.sqrt(np.size(ratio_clipped)))   
    print("sigma SAMI/SDSS flux ratio (clipping): ",ratio_rms)
    print("1 sigma percentile range SAMI/SDSS flux ratio (no clipping): ",ratio_158_noclip,ratio_841_noclip,', equivalent 1sigma error:',(ratio_841_noclip-ratio_158_noclip)/2.0)
    print("2 sigma percentile range SAMI/SDSS flux ratio (no clipping): ",ratio_05_noclip,ratio_95_noclip,', equivalent 2sigma error:',(ratio_95_noclip-ratio_05_noclip)/2.0)

    print("")
    print("mean surface brightness for S/N=5: ",sb_mean)
    print("median surface brightness for S/N=5: ",sb_median)
    print("rms surface brightness for S/N=5: ",sb_rms)

    print("")
    print("Median S/N at SB of 22.0 mag arcsec-2 (g-band):",sn220_median)
    print("Median S/N at SB of 22.5 mag arcsec-2 (g-band):",sn225_median)
    print("Median S/N at SB of 23.0 mag arcsec-2 (g-band):",sn230_median)
    print("Median S/N at SB of 23.5 mag arcsec-2 (g-band):",sn235_median)
    print("Median S/N at SB of 24.0 mag arcsec-2 (g-band):",sn240_median)
    

    if (dopdf):
        py.savefig(pdf, format='pdf')        
        pdf.close()


    # make the photometry comparison plots:
    if (dopdf):
        pdf = PdfPages('wcs_coords_test_flux.pdf')

    py.clf()
    py.rcParams.update({'font.size': 8})
    py.rcParams.update({'lines.linewidth': 2})
    py.rcParams.update({'xtick.major.pad': 8})
    py.rcParams.update({'ytick.major.pad': 8})

    #fig, ((ax1, ax2),(ax3,ax4)) = py.subplots(nrows=2,ncols=2)
    #fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = py.subplots(nrows=3,ncols=2)
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = py.subplots(nrows=2,ncols=3)
    # plot flux ratio vs flux:
    rmin = 0.0
    rmax = 2.0
    ax1.text(0.05,0.9,'a)', horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)
    ax1.scatter(np.log10(SDSS_flux_sum[igood]),ratio[igood],marker='x',facecolors='black', edgecolors='black',s=100,linewidths=1.0)
    ax1.set_ylabel('(SAMI Flux)/(SDSS flux)',labelpad=1)
    ax1.set_xlabel('log(SDSS Flux/microJy)',labelpad=1)
    if (fluxlim > 0.0):
        ax1.axvline(np.log10(fluxlim),linestyle=':',color='k')
        
    ax1.axhline(ratio_mean,linestyle='--',color='r')
    ax1.axhline(ratio_mean+outlier_flux,linestyle=':',color='r')
    ax1.axhline(ratio_mean-outlier_flux,linestyle=':',color='r')
    ax1.set_ylim(ymin=rmin,ymax=rmax)

    # plot flux ratio vs. WCS offset
    ax4.scatter(np.log10(r_off[igood]),ratio[igood],marker='x',facecolors='black', edgecolors='black',s=100,linewidths=1.0)
    ax4.set_xlabel('log(WCS offset/arcsec)',labelpad=1)
    ax4.set_ylabel('(SAMI Flux)/(SDSS flux)',labelpad=1)
    ax4.set_ylim(ymin=rmin,ymax=rmax)
    ax4.text(0.05,0.9,'b)', horizontalalignment='center',verticalalignment='center',transform=ax4.transAxes)

    # plot histogram of flux ratio:
    ax2.text(0.05,0.9,'c)', horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes)
    ax2.hist(ratio[igood],bins=40,range=(rmin,rmax),histtype='step',color='k')
    ax2.set_xlabel('(SAMI Flux)/(SDSS flux)',labelpad=1)
    ax2.set_ylabel('Number')
    ax2.axvline(ratio_median,linestyle=':',color='k')
    #ax1.set_xlim(xmin = xmin, xmax = xmax)
    #ax1.set_ylim(ymin = xmin, ymax = xmax)

    # plot histogram of SB at S/N=5:
    ax5.text(0.05,0.9,'d)', horizontalalignment='center',verticalalignment='center',transform=ax5.transAxes)
    ax5.hist(sb_sn5[igood],bins=40,range=(21,24),histtype='step',color='k')
    ax5.set_xlabel('g-band SB for S/N=5',labelpad=1)
    ax5.set_ylabel('Number',labelpad=1)
    ax5.axvline(sb_median,linestyle=':',color='k')
    
    # plot histogram of S/N for different SB:
    ax3.text(0.05,0.9,'e)', horizontalalignment='center',verticalalignment='center',transform=ax3.transAxes)
    ax3.hist(sn220[igood],bins=24,range=(0,12),histtype='step',color='k',label='SB = 22.0')
    ax3.axvline(sn220_median,linestyle=':',color='k')
    ax3.hist(sn225[igood],bins=24,range=(0,12),histtype='step',color='r',label='SB = 22.5')
    ax3.axvline(sn225_median,linestyle=':',color='r')
    ax3.hist(sn230[igood],bins=24,range=(0,12),histtype='step',color='g',label='SB = 23.0')
    ax3.axvline(sn230_median,linestyle=':',color='g')
    ax3.hist(sn235[igood],bins=24,range=(0,12),histtype='step',color='b',label='SB = 23.5')
    ax3.axvline(sn235_median,linestyle=':',color='b')
    ax3.hist(sn240[igood],bins=24,range=(0,12),histtype='step',color='c',label='SB = 24.0')
    ax3.axvline(sn240_median,linestyle=':',color='c')
    ax3.set_xlabel('S/N',labelpad=1)
    ax3.set_ylabel('Number',labelpad=1)
    ax3.legend(loc='upper right',prop={'size':8})
    
    if (dopdf):
        py.savefig(pdf, format='pdf')        
        pdf.close()

    # plot the pdf histogram of flux ratio:
    if (dopdf):
        pdf = PdfPages('fluxratio_hist.pdf')
        fig = py.figure(4)
        ax1 = fig.add_subplot(1,1,1)
            # plot histogram of flux ratio:
        # good size for single plot in 1 col of MNRAS:
        py.rcParams.update({'font.size': 16})
        py.rcParams.update({'lines.linewidth': 2})
        rmin = 0.5
        rmax = 1.5
        #ax1.text(0.05,0.9,'c)', horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)
        weights = np.ones_like(ratio[igood])/float(len(ratio[igood]))
        ax1.hist(ratio[igood],bins=20,range=(rmin,rmax),histtype='step',color=col,linestyle=lstyle,weights=weights,linewidth=2)
        ax1.set_xlabel('(SAMI g-band flux)/(SDSS g-band flux)',labelpad=1)
        ax1.set_ylabel('Fraction')
        ax1.axvline(ratio_median,linestyle=':',color=col)
        ax1.set_xlim(xmin = 0.5, xmax = 1.5)
        #ax1.set_ylim(ymin = 0.0, ymax = 100)

        py.savefig(pdf, format='pdf')        
        pdf.close()
 

    return


##################################################################################
# simple short function to reject outliers:
def reject_outliers(data, m=3):

    return data[abs(data - np.mean(data)) < m * np.std(data)]

##################################################################################
# function to reject outliers iteratively:
def reject_outliers_iter(data, m=3,verbose=False,max_iter=10):

    ndata1 = np.size(data)
    data_tmp = np.copy(data)
    nrej = 1
    niter = 0
    while (nrej > 0):
        new_data = reject_outliers(data_tmp,m=m)
        ndata2 = np.size(new_data)
        nrej = ndata1 - ndata2
        niter = niter+1
        
        if (verbose):
            print('iteration ',niter,', n_rej = ',nrej,ndata1,ndata2)
            
        data_tmp = np.copy(new_data)
        ndata1 = ndata2

        if (niter > max_iter):
            if (verbose):
                print("Maximum iterations reached:",niter)
                
            break 
        
    return new_data

############### END OF FILE ###############
