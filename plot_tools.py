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

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dr_tools.sami_fluxcal import sami_read_apspec
from dr_tools.sami_utils import spectres

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

###############################################################
def plot_line_lam(ax,z):

    """Plot location of main emission and abs lines at redshift
    of current object."""

    
    

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

    
    pdf = PdfPages('sov_many.pdf')

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

def plot_sov_many_new(catfile,bin='default', isradio=True):

    """Plot many SOV plots, reading list from the Matched AGN FITS file"""

    # get casda credentials if needed (only if isradio)
    if isradio:
        casda=Casda()
        casda.login(username=OPAL_USER, store_password=False)
        
        
    hdulist = fits.open(catfile)
    
    tab = hdulist[1].data
    
    tab = tab[np.isin(tab['CATID'], (9011900430, 9388000001))] # just now for testing so not rendering
    catids = tab['CATID']
    print(catids)
    #mstar = tab['Mstar']
    #redshift = tab['z_tonry']
    #xrayflux = tab['XRAY_ML_FLUX_0']
    isradio_ls = tab['IS_RADIOSOURCE']
    isxray_ls = tab['IS_XRAYSOURCE']

    
    pdf = PdfPages('sov_many.pdf')

    n = 0
    for catid in catids:

        #dl = cosmo.luminosity_distance(redshift[n])
        #dlcm = dl.to(u.cm)
        #lum = xrayflux[n]*(dlcm/u.cm)**2
        #print(dl,dlcm,lum)
        

        
        # define label:
        #label = 'log(M*) = {0:5.2f}, z = {1:5.3f}, L(0.2-6kev)={2:6.2e} erg/s'.format(mstar[n],redshift[n],lum)
        label = 'test'

        #print(label)
        
        usebin=bin
        # fix adaptive bining for this object:
        if (catid == 203609):
            usebin = 'default'
            
        plot_dr3_sov(catid,bin=usebin,dopdf=False,label=label, casda=casda, isradio=isradio_ls[n])

        py.draw()
        # pause for input if plotting all the spectra:
        #yn = input('Continue? (y/n):')
        py.savefig(pdf, format='pdf')

        n=n+1
        
    pdf.close()
     

###############################################################

def plot_dr3_sov(catid,bin='default',dopdf=True,snlim=3.0,label=None, isradio=False, casda = None):

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
    if (dopdf):
        pdf = PdfPages(f'dr3_sov_pdfs/dr3_sov_{catid}.pdf')

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
    print('SDSS spectra found:',sdss_spec_files)
    nsdss = np.size(sdss_spec_files)
    if (nsdss > 0):
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

    # download SDSS RGB:
    impix = 3 * 50
    imsize = 3 * 0.4166*u.arcmin
    cutoutbaseurl = 'https://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
    query_string = urlencode(dict(ra=ra,dec=dec, 
                                     width=impix, height=impix, 
                                     scale=imsize.to(u.arcsec).value/impix))
    url = cutoutbaseurl + '?' + query_string
    
    # this downloads the image to your disk
    urlretrieve(url, 'SDSS_cutout.jpg')
    image = Image.open('SDSS_cutout.jpg')

    ax21 = fig1.add_subplot(gs[1,0])
    ax21.imshow(np.fliplr(image),origin='lower',interpolation='nearest')
    ax21.text(0.05, 0.05,'SDSS',color='w',horizontalalignment='left',verticalalignment='center', transform=ax21.transAxes)

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

    ax22 = fig1.add_subplot(gs[1,1])
    ax22.imshow(np.fliplr(image),origin='lower',interpolation='nearest')
    ax22.text(0.05, 0.05,'SDSS zoom',color='w',horizontalalignment='left',verticalalignment='center', transform=ax22.transAxes)

    #image.show()
    
    # plot radio contours on SDSS images if isradio
    if isradio:
        # download RACS Radio image cutout:
        impix = 3 * 50
        imsize = 3 * 0.4166*u.arcmin
        
        # get args
        centre = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        
        
        query = "select * from ivoa.obscore "\
            "where filename LIKE 'RACS-DR1%' "\
            "AND filename LIKE '%A.fits' "\
            f"AND 1 = CONTAINS(POINT('ICRS',{ra},{dec}),s_region)"
        # open connection to TAP service and run query
        casdatap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
        job = casdatap.launch_job_async(query)
        table = job.get_results()
        
        # request a cutout of the images returned by the query and download
        url_list = casda.cutout(table, coordinates=centre, radius=imsize) 
        print(url_list)
        cutout_file = casda.download_files(url_list[:2])[0].removesuffix('.checksum')
        print(cutout_file)
        image = fits.open(cutout_file)[0].data.squeeze()
        
        ax21.contour(np.fliplr(image), colors='white', linewidths=0.5, alpha=0.25, extent=(0,impix, 0, impix))
        
        
        # again for same size as SAMI IFU:
        impix = 65
        #imsize = 0.25*u.arcmin
        imsize = 0.4166*u.arcmin
        url_list = casda.cutout(table, coordinates=centre, radius=imsize) 
        
        cutout_file = casda.download_files(url_list[:2])[0].removesuffix('.checksum')

        image = fits.open(cutout_file)[0].data.squeeze()
        
        ax22.contour(np.fliplr(image), colors='white', linewidths=0.5, alpha=0.25, extent=(0,impix, 0, impix))
    

    ax23 = fig1.add_subplot(gs[1,2])

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
    im23 = ax23.imshow(stelflux,origin='lower',interpolation='nearest',cmap=py.cm.YlOrRd,vmin=0.0,vmax=np.nanpercentile(stelflux,98.0))
    axins23 = inset_axes(ax23,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im23, cax=axins23, orientation="horizontal")
    ax23.text(0.05, 0.05,'flux stel', horizontalalignment='left',verticalalignment='center', transform=ax23.transAxes)
    
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
    axins44 = inset_axes(ax44,width="90%",height="5%",loc='upper center')
    fig1.colorbar(im44, cax=axins44, orientation="horizontal")

    ax44.set(xlim=[-1.5,0.5],ylim=[-1.2,1.5],xlabel='log([NII]/Ha)',ylabel='log([OIII]/Hb)')
    


    if (dopdf):
    	py.savefig(pdf, format='pdf')
    	pdf.close()

    
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

