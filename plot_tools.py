###############################################################
# various tools make generic plots for SAMI data

import shutil
import os
import numpy as np
import pylab as py
import scipy as sp
import astropy.io.fits as fits
from scipy import stats
import glob
import warnings

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dr_tools.sami_fluxcal import sami_read_apspec
from dr_tools.sami_utils import spectres
from racs_cutout_tools import get_racs_image_cutout

from urllib.parse import urlencode
from urllib.request import urlretrieve

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u

from PIL import Image

from astroquery.casda import Casda  
from astroquery.utils.tap.core import TapPlus



# CASDA OPAL username to access RACS data:
OPAL_USER = "mspa4681@uni.sydney.edu.au"

# Folder location with all SAMI cubes:
ifs_path = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/ifs"

# Location of SAMI AGN Summary Catalogue
AGN_Summary_path = "shared_catalogues/SAMI_AGN_matches.fits"

# some emission/absorption lines to plot:
spectra_features_dict = {'$H\\alpha$': 6564.61, '$H\\beta$':4862.68, '$NII$': 6549.86, '$OIII$': 5008.240, '$SII$': 6718.29}


###############################################################
def plot_line_lam(ax,z):

    """Plot location of main emission and abs lines at redshift
    of current object."""
    
    # get ylims
    ymin, ymax = ax.set_ylim()
    yrange = ymax - ymin

    h=0.1
    #z=0
    # Ha
    Ha_lam = (1+z) * 6564.61
    ax.axvline(Ha_lam, color='grey', zorder=-1)
    ax.text(Ha_lam+10, h*yrange+ymin, '$H\\alpha$')
    
    # Hb
    Hb_lam = (1+z) * 4862.68
    ax.axvline(Hb_lam, color='grey', zorder=-1)
    ax.text(Hb_lam+10, h*yrange+ymin, '$H\\beta$')
    
    # NII
    NII_lam = (1+z) * 6549.86
    ax.axvline(NII_lam, color='grey', zorder=-1)
    ax.text(NII_lam-90, h*yrange+ymin, '$NII$')
    
    # OIII
    OIII_lam = (1+z) * 5008.24
    ax.axvline(OIII_lam, color='grey', zorder=-1)
    ax.text(OIII_lam+10, h*yrange+ymin, '$OIII$')
    
    # SII
    SII_lam = (1+z) * 6718.29
    ax.axvline(SII_lam, color='grey', zorder=-1)
    ax.text(SII_lam+10, h*yrange+ymin, '$SII$')

    
    
    return None
    

    

###############################################################

def plot_sov_many(catfile,bin='default'):

    """Plot many SOV plots, reading list from a FITS file"""

    hdulist = fits.open(catfile)
    
    tab = hdulist[1].data
    catids = tab['CATID_1']
    print(catids)
    mstar = tab['Mstar']
    redshift = tab['z_tonry']
    xrayflux = tab['XRAY_ML_FLUX_0']


    pdf = PdfPages('dr3_sov_pdfs/sov_many.pdf')

    n = 0
    for catid in catids:

        dl = cosmo.luminosity_distance(redshift[n])
        dlcm = dl.to(u.cm)
        lum = xrayflux[n]*(dlcm/u.cm)**2
        print(dl,dlcm,lum)

        
        # define label:
        label = 'log(M*) = {0:5.2f}, z = {1:5.3f}, L(0.2-6kev)={2:6.2e} erg/s'.format(mstar[n],redshift[n],lum)

        print(label)
        
        usebin=bin
        # fix adaptive bining for this object:
        if (catid == 203609):
            usebin = 'default'
            
        plot_dr3_sov(catid,bin=usebin,dopdf=False,label=label)

        py.draw()
        # pause for input if plotting all the spectra:
        #yn = input('Continue? (y/n):')
        py.savefig(pdf, format='pdf')

        n=n+1
        
    pdf.close()
     

###############################################################

def plot_sov_many_new(catfile, specific_catids= 'All', save_name = 'sov_many.pdf', bin='default', radio_sources=True, only_radio=False, snlim=3.0, OPAL_USER=OPAL_USER, do_printstatement=False, save_folder=None):
    """
    Plot many SOV plots, reading list from the Matched AGN FITS file
    """


    # get casda credentials if needed (only if isradio)
    if radio_sources:
        casda=Casda()
        casda.login(username=OPAL_USER, store_password=False)
    else:
        casda=None
        
        
    hdulist = fits.open(catfile)
    
    tab = hdulist[1].data
    
    if radio_sources:
        isradio_ls = tab['IS_RADIOSOURCE']==1
        if only_radio:
            tab = tab[isradio_ls]
        
    else:
        isradio_ls = np.zeros(len(tab['IS_RADIOSOURCE']))


    if type(specific_catids) != str:
        tab = tab[np.isin(tab['CATID'], specific_catids)] # redefine tab to only include specific_catids
        
        #check if any missing catids:
        missing_catids = np.setdiff1d(specific_catids, tab['CATID'])

        
        if len(missing_catids) >0:
            warnings.warn(f"{len(missing_catids)} CATIDs not in given catfile: \n {missing_catids}", UserWarning)
            
    catids = tab['CATID']

    mstar = tab['M_STAR']
    redshift = tab['Z_SPEC']
    bpt_classification = tab['CATEGORY_BPT_AGN']
    xrayflux = tab['eROSITA_TOTALFLUX_1'] # from eROSITA
    radioflux = tab['RACS_TOTALFLUX'] *1e-3 * 1e-23 *u.erg/u.s * u.cm**(-2) /u.Hz # in mJy, now in erg/s /cm^2 /Hz


    # set up pdf plotting:
    if save_folder==None:
        pdf_path = os.path.join('dr3_sov_pdfs', save_name)
    else:
        pdf_path = os.path.join(save_folder, save_name)
        
    pdf = PdfPages(pdf_path)
        

    n = 0
    for catid in catids:
        print(f"\nCATID: {catid: <12} ({n+1}/{len(catids)})")

        dl = cosmo.luminosity_distance(redshift[n])
        dlcm = dl.to(u.cm)
        lum = xrayflux[n]*(dlcm/u.cm)**2

                #define label:
        label = 'log(M*) = {0:5.2f}, z = {1:5.3f}, L(0.2-2.3kev)={2:6.2e} erg/s, BPT classification: {3}'.format(mstar[n],redshift[n],lum, bpt_classification[n])


        if radio_sources:
            # Convert to Luminosity, Sectiion 3.2 of Pracy et al. 2016, in W/Hz
            alpha_spectral_index = -0.7
            lum =  radioflux[n] / (u.erg/u.s/(u.cm**2)/u.Hz) * 4 * np.pi * ( (dl.to(u.cm)/u.cm) **2) * 1 / ( (1+redshift[n])** (1+alpha_spectral_index) ) 

            
                    #define label:
            label = 'log(M*) = {0:5.2f}, z = {1:5.3f}, L(1367.5 MHz)={2:6.2e} erg/s /Hz, BPT classification: {3}'.format(mstar[n],redshift[n],lum.decompose(), bpt_classification[n])



       
        #print(label)
        
        usebin=bin
        # fix adaptive bining for this object:
        if (catid == 203609):
            usebin = 'default'
            
        plot_dr3_sov(catid,bin=usebin,dopdf=False,label=label, casda=casda, isradio=isradio_ls[n], redshift=redshift[n], do_printstatement=do_printstatement)

        py.draw()
        # pause for input if plotting all the spectra:
        #yn = input('Continue? (y/n):')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            py.savefig(pdf, format='pdf')

        n=n+1
        
    pdf.close()
    print(f"Many SOV PDF saved to: {pdf_path}")
     

###############################################################

def plot_dr3_sov(catid,bin='default',dopdf=True,snlim=3.0,label=None, isradio=False, casda = None, redshift=None, OPAL_USER=OPAL_USER, do_printstatement=False, save_folder=None):

    """Make a summary plot of all the main data from DR3, much like a single
    object viewer.  Assumes the format of DR3."""
    
    # get casda credentials if needed (only if isradio)
    if isradio and casda==None:
        casda=Casda()
        casda.login(username=OPAL_USER, store_password=False)

    # set up formating:
    py.rc('text', usetex=False)
    py.rcParams.update({'font.size': 5})
    py.rcParams.update({'lines.linewidth': 1})
    py.rcParams.update({'figure.autolayout': True})
    # this to get sans-serif latex maths:
    py.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \usepackage{sansmath} \sansmath'               # <- tricky! -- gotta actually tell tex to use!


    py.subplots_adjust(hspace = 0.0, wspace = 0.0)
    
    # set up pdf plotting:
    if save_folder==None:
        pdf_path = os.path.join('dr3_sov_pdfs', f'dr3_sov_{catid}.pdf')
    else:
        pdf_path = os.path.join(save_folder, f'dr3_sov_{catid}.pdf')
        
    if (dopdf):
        pdf = PdfPages(pdf_path)

    # set up grid:
    fig1 = py.figure(1,constrained_layout=True)
    fig1.clf()
    gs = fig1.add_gridspec(4, 4,wspace=0.3,hspace=0.3,top=0.95,bottom=0.1)

    # first set up aperture spectrum plot:
    ax1 = fig1.add_subplot(gs[0,:])

    # read aperture spectrum:
    apspecfile_blue = os.path.join(ifs_path, str(catid),str(catid)+'_A_spectrum_3-arcsec_blue.fits')
    apspecfile_red = os.path.join(ifs_path, str(catid),str(catid)+'_A_spectrum_3-arcsec_red.fits')

    hdulist = fits.open(apspecfile_blue)
    sami_flux_blue,sami_lam_blue = sami_read_apspec(hdulist,0,doareacorr=False)
    ra = hdulist[0].header['CATARA']
    dec = hdulist[0].header['CATADEC']
    hdulist.close()
    hdulist = fits.open(apspecfile_red)
    sami_flux_red,sami_lam_red = sami_read_apspec(hdulist,0,doareacorr=False)
    
    hdulist.close()

    # read SDSS spectrum, if available:
    sdss_spec_files = glob.glob('/Users/scroom/data/sami/fluxcal/sdss_spec2/'+str(catid)+'*.fits')

    nsdss = np.size(sdss_spec_files)
    if (nsdss > 0):
        print('SDSS spectra found:',sdss_spec_files)
        hdulist = fits.open(sdss_spec_files[0])
        sdss_spec_table = hdulist['COADD'].data
        sdss_flux = sdss_spec_table['flux']
        sdss_loglam = sdss_spec_table['loglam']
        sdss_lam = 10.0**sdss_loglam
        psf2fib = 10.0**(-0.4*0.35)
        # scale tp fib mag and factor of 10 diff in normalization compared to SAMI
        sdss_flux = sdss_flux * psf2fib / 10.0
        # transform SDSS spectrum from vac to air:
        sdss_lam_air = sdss_lam/(1.0 +2.735182e-4 + 131.4182/sdss_lam**2 + 2.76249e8/sdss_lam**4)

        # scale to normalise to SAMI (take into account seeing):
        sdss_flux_blue = spectres(sami_lam_blue,sdss_lam_air,sdss_flux)
        ratio = np.nanmedian(sami_flux_blue/sdss_flux_blue)
        print('median blue SAMI/SDSS flux ratio:',ratio)
        sdss_flux = sdss_flux * ratio
        

        hdulist.close()
        ax1.plot(sdss_lam_air,sdss_flux,color='k',label='SDSS scaled')
        
        print(nsdss)
    
    ax1.plot(sami_lam_blue,sami_flux_blue,color='b',label='SAMI blue arm')
    ax1.plot(sami_lam_red,sami_flux_red,color='r',label='SAMI red arm')
    ax1.set(xlim=[3700.0,7500.0])
    ax1.text(0.05, 0.9,str(catid),fontsize=10,horizontalalignment='left',verticalalignment='center', transform=ax1.transAxes)
    if (label != None):
        
        ax1.set_title(label,fontsize=10)
    
    ax1.xaxis.labelpad=0
    #ax1.legend()

    ax1.set(ylabel='Flux',xlabel='Wavelength (\AA)')
    #ax1.set(ylabel='Flux (1E-17 erg/cm$^2$/s/Ang)')
    
    ax1_ylims = ax1.set_ylim()

   
    # add some spectral line indicators
    if redshift == None: # first get the redshift
        AGN_summary_table = fits.open(AGN_Summary_path)[1].data
        redshift = AGN_summary_table['Z_SPEC'][AGN_summary_table['CATID'] == catid]
    
    plot_line_lam(ax1, redshift)

        
        
    
    # download large SDSS RGB:
    large_image_scale = 30
    impix = large_image_scale * 50
    imsize = large_image_scale * 0.4166*u.arcmin
    cutoutbaseurl = 'https://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
    query_string = urlencode(dict(ra=ra,dec=dec, 
                                     width=impix, height=impix, 
                                     scale=imsize.to(u.arcsec).value/impix))
    url = cutoutbaseurl + '?' + query_string
    
    # this downloads the image to your disk
    urlretrieve(url, 'SDSS_cutout.jpg')
    image = Image.open('SDSS_cutout.jpg')

    ax21 = fig1.add_subplot(gs[1,0])
    ax21.imshow(np.flipud(image),origin='lower',interpolation='nearest')
    ax21.text(0.05, 0.05,'SDSS',color='w',horizontalalignment='left',verticalalignment='center', transform=ax21.transAxes)

    # download SDSS RGB:
    medium_image_scale = 3
    impix = medium_image_scale * 50
    imsize = medium_image_scale * 0.4166*u.arcmin
    cutoutbaseurl = 'https://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
    query_string = urlencode(dict(ra=ra,dec=dec, 
                                     width=impix, height=impix, 
                                     scale=imsize.to(u.arcsec).value/impix))
    url = cutoutbaseurl + '?' + query_string
    
    # this downloads the image to your disk
    urlretrieve(url, 'SDSS_cutout.jpg')
    image = Image.open('SDSS_cutout.jpg')

    ax22 = fig1.add_subplot(gs[1,1])
    ax22.imshow(np.flipud(image),origin='lower',interpolation='nearest')
    ax22.text(0.05, 0.05,'SDSS',color='w',horizontalalignment='left',verticalalignment='center', transform=ax22.transAxes)

    # download SDSS RGB of same size as SAMI IFU:
    impix = 50
    #imsize = 0.25*u.arcmin
    imsize = 0.4166*u.arcmin
    cutoutbaseurl = 'https://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
    query_string = urlencode(dict(ra=ra,dec=dec, 
                                     width=impix, height=impix, 
                                     scale=imsize.to(u.arcsec).value/impix))
    url = cutoutbaseurl + '?' + query_string
    # this downloads the image to your disk
    urlretrieve(url, 'SDSS_cutout_small.jpg')
    image = Image.open('SDSS_cutout_small.jpg')

    ax23 = fig1.add_subplot(gs[1,2])
    ax23.imshow(np.flipud(image),origin='lower',interpolation='nearest')
    ax23.text(0.05, 0.05,'SDSS zoom',color='w',horizontalalignment='left',verticalalignment='center', transform=ax23.transAxes)

    #image.show()
    
    # plot radio contours on SDSS images if isradio
    if isradio:
        # download RACS Radio image cutout:
        
        impix = large_image_scale * 50
        imsize = large_image_scale * 0.4166*u.arcmin
        
        cutout_file = get_racs_image_cutout(ra, dec, imsize, casda=casda)
        image = fits.open(cutout_file)[0].data.squeeze()
        
        ax21.contour(np.flipud(image), colors='white', linewidths=0.5, alpha=0.25, extent=(0,impix, 0, impix))
        
        
        impix = medium_image_scale * 50
        imsize = medium_image_scale * 0.4166*u.arcmin
        
        cutout_file = get_racs_image_cutout(ra, dec, imsize, casda=casda)
        image = fits.open(cutout_file)[0].data.squeeze()
        
        ax22.contour(np.flipud(image), colors='white', linewidths=0.5, alpha=0.25, extent=(0,impix, 0, impix))


        # again for same size as SAMI IFU:
        impix = 65
        #imsize = 0.25*u.arcmin
        imsize = 0.4166*u.arcmin
        cutout_file = get_racs_image_cutout(ra, dec, imsize, casda=casda)

        image = fits.open(cutout_file)[0].data.squeeze()
        
        ax23.contour(np.flipud(image), colors='white', linewidths=0.5, alpha=0.25, extent=(0,impix, 0, impix))


    ax24 = fig1.add_subplot(gs[1,3])

    # velocity fields:
    ax31 = fig1.add_subplot(gs[2,0])
    ax31.set_aspect('equal', 'box')
    stelvel_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_stellar-velocity_'+bin+'_two-moment.fits')
    stelvel = fits.getdata(stelvel_file, ext=0)
    stelflux = fits.getdata(stelvel_file, extname='FLUX')
    vmin = -150.0
    vmax=150.0
    im31 = ax31.imshow(stelvel,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu_r,vmin=vmin,vmax=vmax)
    axins31 = inset_axes(ax31,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im31, cax=axins31, orientation="horizontal")
    ax31.text(0.05, 0.05,'vel stel', horizontalalignment='left',verticalalignment='center', transform=ax31.transAxes)

    # plot flux from SAMI:
    im24 = ax24.imshow(stelflux,origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=0.0,vmax=np.nanpercentile(stelflux,98.0))
    axins24 = inset_axes(ax24,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im24, cax=axins24, orientation="horizontal")
    ax24.text(0.05, 0.05,'flux stel', horizontalalignment='left',verticalalignment='center', transform=ax24.transAxes)
    
    # stellar sigma:
    ax32 = fig1.add_subplot(gs[2,1])
    ax32.set_aspect('equal', 'box')
    stelsig_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_stellar-velocity-dispersion_'+bin+'_two-moment.fits')
    stelsig = fits.getdata(stelsig_file, ext=0)
    vmin = 0.0
    vmax=np.nanpercentile(stelsig,95.0)
    im32 = ax32.imshow(stelsig,origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=vmin,vmax=vmax)
    axins32 = inset_axes(ax32,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im32, cax=axins32, orientation="horizontal")
    ax32.text(0.05, 0.05,'sig stel', horizontalalignment='left',verticalalignment='center', transform=ax32.transAxes)

    # get Halpha early, so we can use it for S/N cuts:
    haflux_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_Halpha_'+bin+'_1-comp.fits')
    haflux = fits.getdata(haflux_file, ext=0)[0,:,:]
    haerr =  fits.getdata(haflux_file, extname='HALPHA_ERR')[0,:,:]
    hasn = haflux/haerr
    ha_snflag = np.where((hasn > snlim),0,1)

    # gas velocity fields:
    ax33 = fig1.add_subplot(gs[2,2])
    ax33.set_aspect('equal', 'box')
    gasvel_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_gas-velocity_'+bin+'_1-comp.fits')
    gasvel = fits.getdata(gasvel_file, ext=0)[0,:,:]
    gasvel_masked = np.ma.masked_array(gasvel,(ha_snflag>0))
    vmin = -150.0
    vmax=150.0
    im33 = ax33.imshow(gasvel_masked,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu_r,vmin=vmin,vmax=vmax)
    axins33 = inset_axes(ax33,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im33, cax=axins33, orientation="horizontal")
    ax33.text(0.05, 0.05,'vel gas', horizontalalignment='left',verticalalignment='center', transform=ax33.transAxes)

    
    # gas velocity dispersion fields:
    ax34 = fig1.add_subplot(gs[2,3])
    ax34.set_aspect('equal', 'box')
    gassig_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_gas-vdisp_'+bin+'_1-comp.fits')
    gassig = fits.getdata(gassig_file, ext=0)[0,:,:]
    gassig_masked = np.ma.masked_array(gassig,(ha_snflag>0))
    vmin = 0.0
    vmax=np.nanpercentile(gassig,95.0)
    im34 = ax34.imshow(gassig_masked,origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=vmin,vmax=vmax)
    axins34 = inset_axes(ax34,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im34, cax=axins34, orientation="horizontal")
    ax34.text(0.05, 0.05,'sig gas', horizontalalignment='left',verticalalignment='center', transform=ax34.transAxes)

    # line fluxes and ratios:
    ax41 = fig1.add_subplot(gs[3,0])
    ax41.set_aspect('equal', 'box')
    haflux_masked = np.ma.masked_array(haflux,(ha_snflag>0))
    vmin = np.log10(np.nanpercentile(haflux,5.0))
    vmax=np.log10(np.nanpercentile(haflux,95.0))
    im41 = ax41.imshow(np.log10(haflux_masked),origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=vmin,vmax=vmax)
    axins41 = inset_axes(ax41,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im41, cax=axins41, orientation="horizontal")
    ax41.text(0.05, 0.05,'log(Ha)', horizontalalignment='left',verticalalignment='center', transform=ax41.transAxes)

    # NII/Ha
    ax42 = fig1.add_subplot(gs[3,1])
    ax42.set_aspect('equal', 'box')
    n2flux_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_NII6583_'+bin+'_1-comp.fits')
    n2flux = fits.getdata(n2flux_file, ext=0)
    n2err =  fits.getdata(n2flux_file, extname='NII6583_ERR')
    n2sn = n2flux/n2err
    n2ha = np.log10(n2flux/haflux)
    n2ha_snflag = np.where(((hasn > snlim) & (n2sn > snlim)),0,1)
    n2ha_masked = np.ma.masked_array(n2ha,(n2ha_snflag>0))
    vmin = np.nanpercentile(n2ha,5.0)
    vmax=np.nanpercentile(n2ha,95.0)
    im42 = ax42.imshow(n2ha_masked,origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=vmin,vmax=vmax)
    axins42 = inset_axes(ax42,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im42, cax=axins42, orientation="horizontal")
    ax42.text(0.05, 0.05,'log([NII]/Ha)', horizontalalignment='left',verticalalignment='center', transform=ax42.transAxes)

    # get Hbeta and OIII:
    ax43 = fig1.add_subplot(gs[3,2])
    ax43.set_aspect('equal', 'box')
    o3flux_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_OIII5007_'+bin+'_1-comp.fits')
    o3flux = fits.getdata(o3flux_file, ext=0)
    o3err =  fits.getdata(o3flux_file, extname='OIII5007_ERR')
    o3sn = o3flux/o3err
    hbflux_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_Hbeta_'+bin+'_1-comp.fits')
    hbflux = fits.getdata(hbflux_file, ext=0)
    hberr =  fits.getdata(hbflux_file, extname='HBETA_ERR')
    hbsn = hbflux/hberr
    o3hb = np.log10(o3flux/hbflux)
    o3hb_snflag = np.where(((hbsn > snlim) & (o3sn > snlim)),0,1)
    o3hb_masked = np.ma.masked_array(o3hb,(o3hb_snflag>0))
    vmin = np.nanpercentile(o3hb,5.0)
    vmax=np.nanpercentile(o3hb,95.0)
    im43 = ax43.imshow(o3hb_masked,origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=vmin,vmax=vmax)
    axins43 = inset_axes(ax43,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im43, cax=axins43, orientation="horizontal")
    ax43.text(0.05, 0.05,'log([OIII]/Hb)', horizontalalignment='left',verticalalignment='center', transform=ax43.transAxes)
    
    # plot BPT
    snflag = np.where(((hasn > snlim) & (hbsn > snlim) & (o3sn > snlim) & (n2sn > snlim)),0,1)
                
    o3hb_masked = np.ma.masked_array(o3hb,(snflag>0))
    n2ha_masked = np.ma.masked_array(n2ha,(snflag>0))

    # set up grid to get distance from centre (in arcsec):
    x = y = np.arange(0.0,50.0,1.0)
    X, Y = np.meshgrid(x, y)
    xcent = 25.0
    ycent = 25.0
    rdist = np.sqrt((X-xcent)**2 + (Y-ycent)**2)/2.0

    ax44 = fig1.add_subplot(gs[3,3])
    im44 = ax44.scatter(n2ha_masked,o3hb_masked,c=rdist,marker='.',vmin=0.0,vmax=8.0,cmap=py.cm.rainbow)
    ax44.text(0.05, 0.05,'BPT vs radius', horizontalalignment='left',verticalalignment='center', transform=ax44.transAxes)
    ax44.xaxis.labelpad=0
    ax44.yaxis.labelpad=0
    # plot Kauffmann line:
    k03_bpt_line(colour='r',line=':')
    k01_bpt_line(colour='g',line=':')
    k03_bpt_Seyfert_LINER_line(colour='grey', line=':')
    axins44 = inset_axes(ax44,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im44, cax=axins44, orientation="horizontal")

    ax44.set(xlim=[-1.5,0.5],ylim=[-1.2,1.5],xlabel='log([NII]/Ha)',ylabel='log([OIII]/Hb)')
    


    if (dopdf):
    	py.savefig(pdf, format='pdf')
    	pdf.close()
    	
    	print(f"{catid} dr3 sov saved to: {pdf_path}")

    
############################################################################
# plot Kauffmann BPT line:
def k03_bpt_line(colour='r',line=':'):
        xbpt = np.arange(-2.0,-0.1,0.01)
        ybpt = 0.61/(xbpt-0.05) + 1.3
        py.plot(xbpt,ybpt,color=colour,linestyle=line)

############################################################################
# plot Kewley 2001 BPT line:
def k01_bpt_line(colour='r',line=':'):
        xbpt = np.arange(-2.0,0.3,0.01)
        ybpt = 0.61/(xbpt-0.47) + 1.19
        py.plot(xbpt,ybpt,color=colour,linestyle=line)
        
###########################################################################
# plot Kauffmann Seyfert/LINER seperation line:      
def k03_bpt_Seyfert_LINER_line(colour='r', line =':'):
        xbpt = np.arange(-0.45, 0.5, 0.01)
        ybpt = np.tan(65 *np.pi/180) * (xbpt+0.45) - 0.5
        py.plot(xbpt, ybpt, color=colour, linestyle=line)

