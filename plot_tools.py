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
import EW_tools

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
def plot_line_lam(ax,z, annotations=True):

    """Plot location of main emission and abs lines at redshift
    of current object. (http://astronomy.nmsu.edu/drewski/tableofemissionlines.html)"""
    
    # get ylims
    ymin, ymax = ax.set_ylim()
    yrange = ymax - ymin

    h=0.1
    #z=0
    # Ha
    Ha_lam = (1+z) * 6562.819
    ax.axvline(Ha_lam, color='grey', zorder=-1)
    if annotations:
        ax.text(Ha_lam-90, h*yrange+ymin, '$H\\alpha$')
    
    # Hb
    Hb_lam = (1+z) * 4861.333
    ax.axvline(Hb_lam, color='grey', zorder=-1)
    if annotations:
        ax.text(Hb_lam-90, h*yrange+ymin, '$H\\beta$')
    
    # NII
    #NII_lam1 = (1+z) * 6548.050
    #ax.axvline(NII_lam1, color='grey', zorder=-1, linestyle='--')
    #ax.text(NII_lam1-90, h*yrange+ymin, '$NII$')
    
    NII_lam2 = (1+z) * 6583.50
    ax.axvline(NII_lam2, color='grey', zorder=-1)
    if annotations:
        ax.text(NII_lam2+10, h*yrange+ymin, '$NII$')
    
    
    # OIII
    OIII_lam = (1+z) * 5006.843
    ax.axvline(OIII_lam, color='grey', zorder=-1)
    if annotations:
        ax.text(OIII_lam+10, h*yrange+ymin, '$OIII$')
    
    # SII
    SII_lam1 = (1+z) * 6716.29
    ax.axvline(SII_lam1, color='grey', zorder=-1)
    if annotations:
        ax.text(SII_lam1+10, h*yrange+ymin, '$SII$')
    
    SII_lam2 = (1+z) * 6730.810
    ax.axvline(SII_lam2, color='grey', zorder=-1)
    #ax.text(SII_lam2+10, h*yrange+ymin, '$SII$')

    
    
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

def plot_sov_many_new(catfile, specific_catids= 'All', save_name = 'sov_many.pdf', bin='default', radio_sources=True, only_radio=False, snlim=3.0, OPAL_USER=OPAL_USER, do_printstatement=False, save_folder=None, advanced=False):
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
   

    if only_radio:
        tab = tab[tab['IS_RADIOSOURCE']==1]
        
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
    if radio_sources:
        isradio_ls = tab['IS_RADIOSOURCE']==1
    
    else:
        isradio_ls = np.zeros(len(tab['IS_RADIOSOURCE']))

    
    if advanced:
        save_name = save_name[:-4] +"_advanced.pdf"


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
            

            
        plot_dr3_sov(catid,bin=usebin,dopdf=False,label=label, casda=casda, isradio=isradio_ls[n], redshift=redshift[n], do_printstatement=do_printstatement,
        advanced=advanced)

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

def plot_dr3_sov(catid,bin='default',dopdf=True,snlim=3.0,label=None, isradio=False, casda = None, redshift=None, OPAL_USER=OPAL_USER, do_printstatement=False, save_folder=None, advanced=False):

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
    plot_pointsize = 3 # set the size of the points on the BPT, WHAN and gassig v n2ha plots
    
    # set up pdf plotting:
    if advanced:
        save_name = f'dr3_sov_{catid}_advanced.pdf'
    else:
        save_name = f'dr3_sov_{catid}.pdf'
    
    if save_folder==None:
        pdf_path = os.path.join('dr3_sov_pdfs', save_name)
    else:
        pdf_path = os.path.join(save_folder, save_name)
        
    if (dopdf):
        pdf = PdfPages(pdf_path)

    # set up grid:
    fig1 = py.figure(1,constrained_layout=True)
    fig1.clf()
    
    
    if not advanced:
        gs = fig1.add_gridspec(4, 4,wspace=0.3,hspace=0.3,top=0.95,bottom=0.1)
        
        # row 0
        axspectra = fig1.add_subplot(gs[0,:])
        
        # row 1
        ax_image_large = fig1.add_subplot(gs[1,0])
        ax_image = fig1.add_subplot(gs[1,1])
        ax_image_small = fig1.add_subplot(gs[1,2])
        ax_stelflux = fig1.add_subplot(gs[1,3])
        
        # row 2
        ax_stelvel = fig1.add_subplot(gs[2,0])
        ax_stelsig = fig1.add_subplot(gs[2,1])
        ax_gasvel = fig1.add_subplot(gs[2,2])
        ax_gassig = fig1.add_subplot(gs[2,3])
        
        # row 3
        ax_haflux = fig1.add_subplot(gs[3,0])
        ax_n2ha = fig1.add_subplot(gs[3,1])
        ax_o3hb = fig1.add_subplot(gs[3,2])
        ax_bpt = fig1.add_subplot(gs[3,3])
    
    
    else: # advanced == True
        gs = fig1.add_gridspec(4, 5,wspace=0.3,hspace=0.3,top=0.95,bottom=0.1, left=0.05, right=0.95)
        
        # row 0
        axspectra = fig1.add_subplot(gs[0,:])
        
        # row 1
        ax_image_large = fig1.add_subplot(gs[1,0])
        ax_image = fig1.add_subplot(gs[1,1])
        ax_image_small = fig1.add_subplot(gs[1,2])
        ax_stelflux = fig1.add_subplot(gs[1,3])
        
        ax_bpt = fig1.add_subplot(gs[1,4])
        
        # row 2
        ax_stelvel = fig1.add_subplot(gs[2,0])
        ax_stelsig = fig1.add_subplot(gs[2,1])
        ax_gasvel = fig1.add_subplot(gs[2,2])
        ax_gassig = fig1.add_subplot(gs[2,3])
        
        ax_gassign2ha = fig1.add_subplot(gs[2,4])
        
        
        # row 3
        ax_haflux = fig1.add_subplot(gs[3,0])
        ax_n2ha = fig1.add_subplot(gs[3,1])
        ax_o3hb = fig1.add_subplot(gs[3,2])
        
        ax_ha_ew = fig1.add_subplot(gs[3,3])
        ax_whan = fig1.add_subplot(gs[3,4])


    # first set up aperture spectrum plot:
    #axspectra = fig1.add_subplot(gs[0,:])

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
        axspectra.plot(sdss_lam_air,sdss_flux,color='k',label='SDSS scaled')
        
        print(nsdss)
    
    axspectra.plot(sami_lam_blue,sami_flux_blue,color='b',label='SAMI blue arm')
    axspectra.plot(sami_lam_red,sami_flux_red,color='r',label='SAMI red arm')
    axspectra.set(xlim=[3700.0,7500.0])
    axspectra.text(0.05, 0.9,str(catid),fontsize=10,horizontalalignment='left',verticalalignment='center', transform=axspectra.transAxes)
    if (label != None):
        
        axspectra.set_title(label,fontsize=10)
        
            # add some spectral line indicators
    if redshift == None: # first get the redshift
        AGN_summary_table = fits.open(AGN_Summary_path)[1].data
        redshift = AGN_summary_table['Z_SPEC'][AGN_summary_table['CATID'] == catid]
    
    plot_line_lam(axspectra, redshift)
    
    axspectra.xaxis.labelpad=0
    #axspectra.legend()

    axspectra.set(ylabel='Flux',xlabel='Wavelength (\AA)')
    #axspectra.set(ylabel='Flux (1E-17 erg/cm$^2$/s/Ang)')
    
    axspectra_ylims = axspectra.set_ylim()

    
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
    urlretrieve(url, 'SDSS_cutout_large.jpg')
    image = Image.open('SDSS_cutout_large.jpg')

    #ax_image_large = fig1.add_subplot(gs[1,0])
    ax_image_large.imshow(np.flipud(image),origin='lower',interpolation='nearest')
    ax_image_large.text(0.05, 0.05,'SDSS',color='w',horizontalalignment='left',verticalalignment='center', transform=ax_image_large.transAxes)

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

    #ax_image = fig1.add_subplot(gs[1,1])
    ax_image.imshow(np.flipud(image),origin='lower',interpolation='nearest')
    ax_image.text(0.05, 0.05,'SDSS',color='w',horizontalalignment='left',verticalalignment='center', transform=ax_image.transAxes)

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

    #ax_image_small = fig1.add_subplot(gs[1,2])
    ax_image_small.imshow(np.flipud(image),origin='lower',interpolation='nearest')
    ax_image_small.text(0.05, 0.05,'SDSS zoom',color='w',horizontalalignment='left',verticalalignment='center', transform=ax_image_small.transAxes)

    #image.show()
    
    # plot radio contours on SDSS images if isradio
    if isradio:
        # download RACS Radio image cutout:
        
        impix = large_image_scale * 50
        imsize = large_image_scale * 0.4166*u.arcmin
        
        cutout_file = get_racs_image_cutout(ra, dec, imsize, casda=casda)
        image = fits.open(cutout_file)[0].data.squeeze()
        
        ax_image_large.contour(np.flipud(image), colors='white', linewidths=0.5, alpha=0.25, extent=(0,impix, 0, impix))
        
        
        
        impix = medium_image_scale * 50
        imsize = medium_image_scale * 0.4166*u.arcmin
        
        cutout_file = get_racs_image_cutout(ra, dec, imsize, casda=casda)
        image = fits.open(cutout_file)[0].data.squeeze()
        
        ax_image.contour(np.flipud(image), colors='white', linewidths=0.5, alpha=0.25, extent=(0,impix, 0, impix))


        # again for same size as SAMI IFU:
        impix = 65
        #imsize = 0.25*u.arcmin
        imsize = 0.4166*u.arcmin
        cutout_file = get_racs_image_cutout(ra, dec, imsize, casda=casda)

        image = fits.open(cutout_file)[0].data.squeeze()
        
        ax_image_small.contour(np.flipud(image), colors='white', linewidths=0.5, alpha=0.25, extent=(0,impix, 0, impix))


    #ax_stelflux = fig1.add_subplot(gs[1,3])

    # velocity fields:
    #ax_stelvel = fig1.add_subplot(gs[2,0])
    ax_stelvel.set_aspect('equal', 'box')
    stelvel_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_stellar-velocity_'+bin+'_two-moment.fits')
    stelvel = fits.getdata(stelvel_file, ext=0)
    stelflux = fits.getdata(stelvel_file, extname='FLUX')
    vmin = -150.0
    vmax=150.0
    im_stelvel = ax_stelvel.imshow(stelvel,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu_r,vmin=vmin,vmax=vmax)
    axins_stelvel = inset_axes(ax_stelvel,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im_stelvel, cax=axins_stelvel, orientation="horizontal")
    ax_stelvel.text(0.05, 0.05,'vel stel', horizontalalignment='left',verticalalignment='center', transform=ax_stelvel.transAxes)

    # plot flux from SAMI:
    im_stelflux = ax_stelflux.imshow(stelflux,origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=0.0,vmax=np.nanpercentile(stelflux,98.0))
    axins_stelflux = inset_axes(ax_stelflux,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im_stelflux, cax=axins_stelflux, orientation="horizontal")
    ax_stelflux.text(0.05, 0.05,'flux stel', horizontalalignment='left',verticalalignment='center', transform=ax_stelflux.transAxes)
    
    # stellar sigma:
    #ax_stelsig = fig1.add_subplot(gs[2,1])
    ax_stelsig.set_aspect('equal', 'box')
    stelsig_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_stellar-velocity-dispersion_'+bin+'_two-moment.fits')
    stelsig = fits.getdata(stelsig_file, ext=0)
    vmin = 0.0
    vmax=np.nanpercentile(stelsig,95.0)
    im_stelsig = ax_stelsig.imshow(stelsig,origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=vmin,vmax=vmax)
    axins_stelsig = inset_axes(ax_stelsig,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im_stelsig, cax=axins_stelsig, orientation="horizontal")
    ax_stelsig.text(0.05, 0.05,'sig stel', horizontalalignment='left',verticalalignment='center', transform=ax_stelsig.transAxes)

    # get Halpha early, so we can use it for S/N cuts:
    haflux_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_Halpha_'+bin+'_1-comp.fits')
    haflux = fits.getdata(haflux_file, ext=0)[0,:,:]
    haerr =  fits.getdata(haflux_file, extname='HALPHA_ERR')[0,:,:]
    hasn = haflux/haerr
    ha_snflag = np.where((hasn > snlim),0,1)

    # gas velocity fields:
    #ax_gasvel = fig1.add_subplot(gs[2,2])
    ax_gasvel.set_aspect('equal', 'box')
    gasvel_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_gas-velocity_'+bin+'_1-comp.fits')
    gasvel = fits.getdata(gasvel_file, ext=0)[0,:,:]
    gasvel_masked = np.ma.masked_array(gasvel,(ha_snflag>0))
    vmin = -150.0
    vmax=150.0
    im_gasvel = ax_gasvel.imshow(gasvel_masked,origin='lower',interpolation='nearest',cmap=py.cm.RdYlBu_r,vmin=vmin,vmax=vmax)
    axins_gasvel = inset_axes(ax_gasvel,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im_gasvel, cax=axins_gasvel, orientation="horizontal")
    ax_gasvel.text(0.05, 0.05,'vel gas', horizontalalignment='left',verticalalignment='center', transform=ax_gasvel.transAxes)

    
    # gas velocity dispersion fields:
    #ax_gassig = fig1.add_subplot(gs[2,3])
    ax_gassig.set_aspect('equal', 'box')
    gassig_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_gas-vdisp_'+bin+'_1-comp.fits')
    gassig = fits.getdata(gassig_file, ext=0)[0,:,:]
    gassig_masked = np.ma.masked_array(gassig,(ha_snflag>0))
    vmin = 0.0
    vmax=np.nanpercentile(gassig,95.0)
    im_gassig = ax_gassig.imshow(gassig_masked,origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=vmin,vmax=vmax)
    axins_gassig = inset_axes(ax_gassig,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im_gassig, cax=axins_gassig, orientation="horizontal")
    ax_gassig.text(0.05, 0.05,'sig gas', horizontalalignment='left',verticalalignment='center', transform=ax_gassig.transAxes)

    # line fluxes and ratios:
    #ax_haflux = fig1.add_subplot(gs[3,0])
    ax_haflux.set_aspect('equal', 'box')
    haflux_masked = np.ma.masked_array(haflux,(ha_snflag>0))
    vmin = np.log10(np.nanpercentile(haflux,5.0))
    vmax=np.log10(np.nanpercentile(haflux,95.0))
    im_haflux = ax_haflux.imshow(np.log10(haflux_masked),origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=vmin,vmax=vmax)
    axins_haflux = inset_axes(ax_haflux,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im_haflux, cax=axins_haflux, orientation="horizontal")
    ax_haflux.text(0.05, 0.05,'log(Ha)', horizontalalignment='left',verticalalignment='center', transform=ax_haflux.transAxes)

    # NII/Ha
    #ax_n2ha = fig1.add_subplot(gs[3,1])
    ax_n2ha.set_aspect('equal', 'box')
    n2flux_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_NII6583_'+bin+'_1-comp.fits')
    n2flux = fits.getdata(n2flux_file, ext=0)
    n2err =  fits.getdata(n2flux_file, extname='NII6583_ERR')
    n2sn = n2flux/n2err
    n2ha = np.log10(n2flux/haflux)
    n2ha_snflag = np.where(((hasn > snlim) & (n2sn > snlim)),0,1)
    n2ha_masked = np.ma.masked_array(n2ha,(n2ha_snflag>0))
    vmin = np.nanpercentile(n2ha,5.0)
    vmax=np.nanpercentile(n2ha,95.0)
    im_n2ha = ax_n2ha.imshow(n2ha_masked,origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=vmin,vmax=vmax)
    axins_n2ha = inset_axes(ax_n2ha,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im_n2ha, cax=axins_n2ha, orientation="horizontal")
    ax_n2ha.text(0.05, 0.05,'log([NII]/Ha)', horizontalalignment='left',verticalalignment='center', transform=ax_n2ha.transAxes)

    # get Hbeta and OIII:
    #ax_o3hb = fig1.add_subplot(gs[3,2])
    ax_o3hb.set_aspect('equal', 'box')
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
    im_o3hb = ax_o3hb.imshow(o3hb_masked,origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=vmin,vmax=vmax)
    axins_o3hb = inset_axes(ax_o3hb,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im_o3hb, cax=axins_o3hb, orientation="horizontal")
    ax_o3hb.text(0.05, 0.05,'log([OIII]/Hb)', horizontalalignment='left',verticalalignment='center', transform=ax_o3hb.transAxes)
    
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

    #ax_bpt = fig1.add_subplot(gs[3,3])
    im_bpt = ax_bpt.scatter(n2ha_masked,o3hb_masked,c=rdist,marker='.',vmin=0.0,vmax=8.0,cmap=py.cm.rainbow, 
    s=plot_pointsize)
    ax_bpt.text(0.05, 0.05,'BPT vs radius', horizontalalignment='left',verticalalignment='center', transform=ax_bpt.transAxes)
    ax_bpt.xaxis.labelpad=0
    ax_bpt.yaxis.labelpad=0
    # plot Kauffmann line:
    k03_bpt_line(ax_bpt, colour='r',line=':')
    k01_bpt_line(ax_bpt, colour='g',line=':')
    k03_bpt_Seyfert_LINER_line(ax_bpt, colour='grey', line=':')
    axins_bpt = inset_axes(ax_bpt,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im_bpt, cax=axins_bpt, orientation="horizontal")

    ax_bpt.set(xlim=[-1.5,0.5],ylim=[-1.2,1.5],xlabel='log([NII]/Ha)',ylabel='log([OIII]/Hb)')
    
    
    if advanced:
        ax_ha_ew.set_aspect('equal', 'box')

        ha_ew, ha_ew_err= EW_tools.get_Halpha_EW_image(catid, estimation_method='median', bin=bin, haflux_masked=haflux_masked, haerr=haerr) # haflux_masked=haflux_masked, haerr=haerr, redshift=redshift

        ha_ewsn = ha_ew / ha_ew_err
        ha_ew_snflag = np.where((ha_ewsn > snlim),0,1)
        ha_ew_masked = np.ma.masked_array(ha_ew,(ha_ew_snflag>0))

        vmin = np.log10( np.nanpercentile(ha_ew,5.0) )
        vmax = np.log10(np.nanpercentile(ha_ew,95.0))


        im_ha_ew = ax_ha_ew.imshow(np.log10(ha_ew_masked),origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=vmin,vmax=vmax)
        axins_ha_ew = inset_axes(ax_ha_ew,width="90%",height="5%",loc='upper center')
        fig1.colorbar(im_ha_ew, cax=axins_ha_ew, orientation="horizontal")
        ax_ha_ew.text(0.05, 0.05,'log(EW(Ha))', horizontalalignment='left',verticalalignment='center', transform=ax_ha_ew.transAxes)
        fig1.show()



        # plot WHaN diagram
        snflag = np.where(((hasn > snlim) & (n2sn > snlim) & (ha_ewsn > snlim)),0,1)
                    
        ha_ew_masked = np.ma.masked_array(ha_ew,(snflag>0))
        n2ha_masked = np.ma.masked_array(n2ha,(snflag>0))

        # set up grid to get distance from centre (in arcsec):
        x = y = np.arange(0.0,50.0,1.0)
        X, Y = np.meshgrid(x, y)
        xcent = 25.0
        ycent = 25.0
        rdist = np.sqrt((X-xcent)**2 + (Y-ycent)**2)/2.0

        # ax_whan = fig1.add_subplot(gs[3,4])

        im_whan = ax_whan.scatter(n2ha_masked,np.log10(ha_ew_masked),c=rdist,marker='.',vmin=0.0,vmax=8.0,cmap=py.cm.rainbow, s=plot_pointsize)
        ax_whan.text(0.05, 0.05,'WHAN vs radius', horizontalalignment='left',verticalalignment='center', transform=ax_whan.transAxes)
        ax_whan.xaxis.labelpad=0
        ax_whan.yaxis.labelpad=0



        axins_whan = inset_axes(ax_whan,width="90%",height="5%",loc='upper center')
        fig1.colorbar(im_whan, cax=axins_whan, orientation="horizontal")

        ax_whan.set(xlim=[-1.5,0.9],ylim=[-1.2,3],xlabel='log([NII]/Ha)',ylabel='log(EW(Ha))')
        # plot WHAN lines:
        EW_tools.plot_WHAN_lines(ax_whan, fontsize=10, region_labels=False, paper='Fernandes (2011) - strong/weak')

        
        # add dispersion v n2ha plot

        snflag = np.where(((hasn > snlim) & (n2sn > snlim)),0,1)
                    
        gassig_masked = np.ma.masked_array(gassig,(snflag>0))
        n2ha_masked = np.ma.masked_array(n2ha,(snflag>0))

        # set up grid to get distance from centre (in arcsec):
        x = y = np.arange(0.0,50.0,1.0)
        X, Y = np.meshgrid(x, y)
        xcent = 25.0
        ycent = 25.0
        rdist = np.sqrt((X-xcent)**2 + (Y-ycent)**2)/2.0

        #ax_gassign2ha = fig1.add_subplot(gs[3,3])
        im_gassign2ha = ax_gassign2ha.scatter(n2ha_masked,gassig_masked,c=rdist,marker='.',vmin=0.0,vmax=8.0,cmap=py.cm.rainbow, s=plot_pointsize)
        ax_gassign2ha.text(0.05, 0.05,'Shock plot vs radius', horizontalalignment='left',verticalalignment='center', transform=ax_gassign2ha.transAxes)
        ax_gassign2ha.xaxis.labelpad=0
        ax_gassign2ha.yaxis.labelpad=0

        axins_gassign2ha = inset_axes(ax_gassign2ha,width="90%",height="5%",loc='upper center')
        fig1.colorbar(im_gassign2ha, cax=axins_gassign2ha, orientation="horizontal")

        ax_gassign2ha.set(xlim=[-1.5,0.5],ylim=[0,np.max([260, np.max(gassig_masked)])], xlabel='log([NII]/Ha)',ylabel='Velocity Disperson [km/s]')  
        
        


    if (dopdf):
    	py.savefig(pdf, format='pdf')
    	pdf.close()
    	
    	print(f"{catid} dr3 sov saved to: {pdf_path}")

    
############################################################################
# plot Kauffmann BPT line:
def k03_bpt_line(ax, colour='r',line=':'):
        xbpt = np.arange(-2.0,-0.1,0.01)
        ybpt = 0.61/(xbpt-0.05) + 1.3
        ax.plot(xbpt,ybpt,color=colour,linestyle=line)

############################################################################
# plot Kewley 2001 BPT line:
def k01_bpt_line(ax, colour='r',line=':'):
        xbpt = np.arange(-2.0,0.3,0.01)
        ybpt = 0.61/(xbpt-0.47) + 1.19
        ax.plot(xbpt,ybpt,color=colour,linestyle=line)
        
###########################################################################
# plot Kauffmann Seyfert/LINER seperation line:      
def k03_bpt_Seyfert_LINER_line(ax, colour='r', line =':'):
        xbpt = np.arange(-0.45, 0.5, 0.01)
        ybpt = np.tan(65 *np.pi/180) * (xbpt+0.45) - 0.5
        ax.plot(xbpt, ybpt, color=colour, linestyle=line)

