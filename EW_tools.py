
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

import astropy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits
from astropy.table import join
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u

from PIL import Image

from astroquery.casda import Casda  
from astroquery.utils.tap.core import TapPlus

import Code.all_fctns as all_fctns

# Folder location with all SAMI cubes:
ifs_path = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/ifs"
catalogue_filepath = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/catalogues/"


# Location of SAMI AGN Summary Catalogue
AGN_Summary_path = "shared_catalogues/SAMI_AGN_matches.fits"




def get_redshift_corrected_spectra(CATID, ifs_path=ifs_path, catalogue_filepath=catalogue_filepath, spectra_colour='red'):
    apspecfile_red = os.path.join(ifs_path,str(CATID), str(CATID)+f'_A_spectrum_1-4-arcsec_{spectra_colour}.fits')
    hdulist = fits.open(apspecfile_red)
    sami_flux_red,sami_lam_red = sami_read_apspec(hdulist,0,doareacorr=False)
    hdulist.close()
    redshift = all_fctns.get_z_best(catalogue_filepath, [CATID], only_spec=True)
    sami_lam_red_zcorr = sami_lam_red / (1 + redshift)
    return sami_flux_red, sami_lam_red_zcorr


def get_spectra_region_flux(sami_lam, sami_flux, region, estimation_method='median'):
    """Get the mean flux in a given wavelength region."""
    region_mask = (sami_lam >= region[0]) & (sami_lam <= region[1])
    region_flux = np.mean(sami_flux[region_mask])
    region_flux_err = np.std(sami_flux[region_mask]) # check this
    return region_flux, region_flux_err


def get_continuum_flux(CATID, regions, em_line = 'H Alpha', spectra_filepath=None, estimation_method='median', sami_flux_red=None, sami_lam_red=None, already_zcorr=False, catalogue_filepath=catalogue_filepath):
    """Get the continuum flux at Halpha for a given CATID.\\
    Two continuum estimation methods implemented: 'median' and 'linefit'.\\
    This function can take in a pre-loaded spectra or relevant CATID.
    
    returns continuum_flux, continuum_flux_err
    """
    if em_line == 'H Alpha':
        em_line_lam = 6562.819
        spectra_colour = 'red'
    elif em_line == 'O III':
        em_line_lam = 5006.843
        spectra_colour = 'blue'
    else:
        raise NotImplementedError("Only 'H Alpha' and 'O III' emission lines are implemented.")
    
    # check regions has two arrays of length 2
    assert len(regions) == 2, TypeError("Regions must be an iterable object of two arrays.")
    region1, region2 = regions

    assert len(region1)==2 and len(region2)==2, TypeError("Each region must be an array of length 2.")

    # check that spectra are provided or spectra_filepath is provided
    if (sami_flux_red is None or sami_lam_red is None) and spectra_filepath is None:
        raise ValueError("Either spectra or spectra_filepath must be provided.")
    
    # first check if spectra are provided, also check if they are already redshift corrected
    elif sami_flux_red is not None and sami_lam_red is not None and already_zcorr:
        sami_flux_red = sami_flux_red
        sami_lam_red_zcorr = sami_lam_red

    elif sami_flux_red is not None and sami_lam_red is not None and not already_zcorr:
        redshift = all_fctns.get_z_best(catalogue_filepath, [CATID], only_spec=True)
        sami_lam_red_zcorr = sami_lam_red / (1 + redshift)

    else: # only other possible case is that spectra_filepath is provided
            # read in the spectra
        try:
            sami_flux_red, sami_lam_red_zcorr = get_redshift_corrected_spectra(CATID, ifs_path=spectra_filepath, catalogue_filepath=catalogue_filepath, spectra_colour=spectra_colour)
        except FileNotFoundError:
            print(f"No valid red spectra found for {CATID}")
            return np.nan, np.nan
            


    # now estimate the continuum flux
    if estimation_method == 'median': # two regions, median of both, midpoint of those is the continuum flux error from std of regions
        region_fluxes = np.zeros(2)
        region_flux_errs = np.zeros(2)

        for i, region in enumerate(regions):
            region_mask = (sami_lam_red_zcorr >= region[0]) & (sami_lam_red_zcorr <= region[1])
            region_fluxes[i] = np.median(sami_flux_red[region_mask])
            region_flux_errs[i] = np.std(sami_flux_red[region_mask])

        continuum_flux = np.mean(region_fluxes)
        continuum_flux_err = continuum_flux * np.sqrt(np.sum( (region_flux_errs/region_fluxes)**2 )) #


    elif estimation_method == 'linefit': #linear fit to the two regions, evaluate at em_line_lam to get continuum flux, error from fit errors
        # get the relevant points for line fitting
        regions_mask = (sami_lam_red_zcorr >= regions[0][0]) & (sami_lam_red_zcorr <= regions[0][1]) | (sami_lam_red_zcorr >= regions[1][0]) & (sami_lam_red_zcorr <= regions[1][1])
        regions_sami_lam_red_zcorr = sami_lam_red_zcorr[regions_mask]
        regions_sami_flux_red = sami_flux_red[regions_mask]

        # fit a line to these points
        res = sp.stats.linregress(regions_sami_lam_red_zcorr, regions_sami_flux_red)
        slope, intercept, slope_err, intercept_err = res.slope, res.intercept, res.stderr, res.intercept_stderr
        continuum_flux = slope*em_line_lam + intercept
        continuum_flux_err = np.sqrt( (slope_err*em_line_lam)**2 + intercept_err**2 )
    else:
        raise NotImplementedError("Only 'median' and 'linefit' estimation methods are implemented.")
    
    #print(f"CATID {CATID}: Continuum flux at {em_line} ({estimation_method}): {continuum_flux} +/- {continuum_flux_err}")

    return continuum_flux, continuum_flux_err
    

def get_EW(CATID, em_line='H Alpha', catalogue_filepath=catalogue_filepath, ifs_path=ifs_path, estimation_method='median', 
                  SAMI_spectra_table_hdu=None,  sami_flux_red=None, sami_lam_red=None, already_zcorr=False, 
                  em_line_flux=None, em_line_error=None):
    """Get the Halpha equivalent width for a given CATID in the 1.4 arcsec aperture.\\
    Two continuum estimation methods implemented: 'median' and 'linefit'.\\
    This function can take in a pre-loaded SAMI_spectra_table (EmissionLine1compDR3.fits) to avoid re-loading the catalogue for each CATID.\\
    This function can also take in a pre-loaded spectra or relevant CATID. to feed into the continuum flux calculation. (see get_continuum_flux function)\\

    returns em_line_EW, em_line_EW_err
    """
    # set the regions for collecting continuum flux, should be symmetric around the emission line
    if em_line == 'H Alpha':
        region_width = 65
        region_separation = 140
        em_line_lam = 6562.819
    elif em_line == 'O III':
        region_width = 60
        region_separation = 140
        em_line_lam = 5006.843
    else:
        raise NotImplementedError("Only 'H Alpha' and 'O III' emission lines are implemented.")
    
    region1 = np.array([-region_width, 0]) + em_line_lam - region_separation/2  
    region2 = np.array([0, region_width]) + em_line_lam + region_separation/2  


    # check if em_line_flux and em_line_error are provided
    if em_line_flux is not None and em_line_error is not None:
        em_line_flux = em_line_flux
        em_line_error = em_line_error

    else:
        # read in the emission line catalogue if not provided
        if SAMI_spectra_table_hdu is None:
            SAMI_spectra_catalogue = "EmissionLine1compDR3.fits"
            with fits.open( os.path.join(catalogue_filepath + SAMI_spectra_catalogue) ) as SAMI_spectra_hdul:
                SAMI_spectra_table_hdu = Table(SAMI_spectra_hdul[1].data)
        else:
            SAMI_spectra_table_hdu = SAMI_spectra_table_hdu

        em_line_flux, em_line_error = all_fctns.get_flux_and_error_1_4_ARCSEC(SAMI_spectra_table_hdu[SAMI_spectra_table_hdu['CATID'] == CATID], em_line)
        
        # use the first value only and check if there is an actual value
    try:
        em_line_flux = em_line_flux[0]
        em_line_error = em_line_error[0]
    except IndexError:
        em_line_flux = np.nan
        em_line_error = np.nan

    # print(f"Halpha flux: {HAlpha_flux} +/- {HAlpha_error}")

    # get the continuum flux
    continuum_flux, continuum_flux_err = get_continuum_flux(CATID, (region1, region2), em_line=em_line, spectra_filepath=ifs_path, estimation_method=estimation_method, sami_flux_red=sami_flux_red, sami_lam_red=sami_lam_red, already_zcorr=already_zcorr, catalogue_filepath=catalogue_filepath)
    # print(f"Continuum flux at Halpha: {continuum_flux} +/- {continuum_flux_err}")
    # calculate the EW
    em_line_EW = em_line_flux / continuum_flux
    em_line_EW_err = em_line_EW * np.sqrt((em_line_error / em_line_flux)**2 + (continuum_flux_err / continuum_flux)**2)
    
    #print(em_line_flux)
        
    return em_line_EW, em_line_EW_err


def get_EW_table(CATIDs, em_line = 'H Alpha', catalogue_filepath=catalogue_filepath, ifs_path=ifs_path):
    """Get the Halpha equivalent width for a given CATID.
    
    Uses the EmissionLine1compDR3.fits catalogue, and individual spectra.
    """
    em_line_underscored = em_line.replace(' ', '_')

    # read in the emission line catalogue
    SAMI_spectra_catalogue = "EmissionLine1compDR3.fits"
    with fits.open(catalogue_filepath + SAMI_spectra_catalogue) as SAMI_spectra_hdul:
        SAMI_spectra_table_hdu = Table(SAMI_spectra_hdul[1].data)

    # initialise HA_EW astropy table
    em_line_EW_table = Table(names=['CATID', f'{em_line_underscored}_EW_MedianFit', f'{em_line_underscored}_EW_err_MedianFit', f'{em_line_underscored}_EW_LineFit', f'{em_line_underscored}_EW_err_LineFit'], dtype=[int, float, float, float, float])

    
    # iterate over CATIDs
    for CATID in CATIDs:
        em_line_flux, em_line_error = all_fctns.get_flux_and_error_1_4_ARCSEC(SAMI_spectra_table_hdu[SAMI_spectra_table_hdu['CATID'] == CATID], em_line)

        em_line_EW_medianfit, em_line_EW_medianfit_err = get_EW(CATID, em_line=em_line, catalogue_filepath=catalogue_filepath, ifs_path=ifs_path, estimation_method='median', SAMI_spectra_table_hdu=SAMI_spectra_table_hdu, em_line_flux=em_line_flux, em_line_error=em_line_error)
        em_line_EW_linefit, em_line_EW_linefit_err = get_EW(CATID, em_line=em_line, catalogue_filepath=catalogue_filepath, ifs_path=ifs_path, estimation_method='linefit', SAMI_spectra_table_hdu=SAMI_spectra_table_hdu, em_line_flux=em_line_flux, em_line_error=em_line_error)
        # add row to table

        em_line_EW_table.add_row([CATID, em_line_EW_medianfit, em_line_EW_medianfit_err, em_line_EW_linefit, em_line_EW_linefit_err])
        
    return em_line_EW_table

def read_cube(cubefile):

    hdu = fits.open(cubefile)
    primary_header=hdu['PRIMARY'].header
    # print(primary_header)
    
    # Get the information needed to create the wavelength array
    crval3=primary_header['CRVAL3']
    cdelt3=primary_header['CDELT3']
    crpix3=primary_header['CRPIX3']
    naxis3=primary_header['NAXIS3']
    
    x=np.arange(naxis3)+1
    L0=crval3-crpix3*cdelt3       
    lam=L0+x*cdelt3 # wavelengths array

    flux = hdu[0].data # flux array
    variance = hdu['VARIANCE'].data # variance array

    hdu.close()

    (zs,ys,xs) = flux.shape
    #print('The shape of the cube: ',zs,ys,xs)

    return lam, flux, variance


def get_Halpha_EW_spectra_investigation_plot(sami_flux_red, sami_lam_red_zcorr, HAlpha_flux, HAlpha_error, region_plot=True):
    """Basic setup to investigate the spectra and continuum fitting around Halpha for a given CATID.
    Just made for quick debugging and visualisation.
    """
    fig, ax = py.subplots(1,1, figsize=(7,4))
    # set regions for collecting continuum flux symmetric about 6562
    region_width = 65
    region_separation = 140
    Ha_lam = 6562.819
    region1 = np.array([-region_width, 0]) + Ha_lam - region_separation/2  
    region2 = np.array([0, region_width]) + Ha_lam + region_separation/2  
    #print(region1, region2)

    # plot these regions
    ax.axvspan(region1[0], region1[1], color='grey', alpha=0.3, linestyle='--')
    ax.axvspan(region2[0], region2[1], color='grey', alpha=0.3, linestyle='--')


    # plot the spectra:
    ax.plot(sami_lam_red_zcorr, sami_flux_red)
    # plot_line_lam(ax, z=0, annotations=False)

    # ax.set_title(f"SAMI ID: {test_CATID}")
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_xlim(6400,6700)

    py.show()

    if region_plot:
        fig1, ax1 = py.subplots(1,1, figsize=(7,4))
        region_mask =( (sami_lam_red_zcorr >= region1[0]) & (sami_lam_red_zcorr <= region1[1]) ) | ( (sami_lam_red_zcorr >= region2[0]) & (sami_lam_red_zcorr <= region2[1]) )
        ax1.scatter(sami_lam_red_zcorr[region_mask], sami_flux_red[region_mask])
        ax1.set_xlabel(r"Wavelength ($\AA$)")
        ax1.set_xlim(6400,6700)

        continuum_flux, continuum_flux_err = get_continuum_flux(1, (region1, region2), estimation_method='linefit', sami_flux_red=sami_flux_red, sami_lam_red=sami_lam_red_zcorr, already_zcorr=True)

        ax1.errorbar(Ha_lam, continuum_flux, yerr=continuum_flux_err, color='orange', fmt='.', label='Continuum flux at $H\\alpha$', capsize=3)
        py.show()

    return 


def plot_WHAN_lines(ax, paper='Fernandes (2011) - Seyfert/LINER', region_labels=True, fontsize=20):
    xrange = ax.get_xlim()
    yrange = ax.get_ylim()

    xs = np.linspace(xrange[0], xrange[1], 1000)

    if paper == 'Fernandes (2011) - Seyfert/LINER':
        m_line = (np.log10(5)-np.log10(0.5))/(-1-0)
        c_line = np.log10(0.5)

        # uncertain lines
        ax.plot(xs[xs<0], m_line*xs[xs<0] + c_line, 'k:')
        ax.plot([0, xrange[1]], [np.log10(0.5)]*2, 'k:')

        # passive galaxy lines
        ax.plot(xs[xs>0], m_line*xs[xs>0] + c_line, 'k--')
        ax.plot([xrange[0], 0], [np.log10(0.5)]*2, 'k--')

        ax.plot([-0.4, xrange[1]], [np.log10(6)]*2, color='k') # K06 Seyfert/LINER line
        ax.axvline(-0.4, color='k') # S06 SF/AGN line

        if region_labels:
            ax.text(0.2, 1.5, 'Seyferts', size=fontsize)
            ax.text(0.4, 0.5, 'LINERs', size=fontsize)
            ax.text(-0.9, 1.9, 'SF', size=fontsize)
            ax.text(-0.3, -0.8, 'Passive galaxies', size=fontsize)

    elif paper == 'Fernandes (2011) - strong/weak':
        m_line = (np.log10(5)-np.log10(0.5))/(-1-0)
        c_line = np.log10(0.5)

        # Radio galaxy lines
        ax.plot([(np.log10(3)-c_line)/m_line, 0], [np.log10(3), np.log10(0.5)], 'k:')
        # ax.plot(xs[xs<0], m_line*xs[xs<0] + c_line, 'k:')
        ax.plot([0, xrange[1]], [np.log10(0.5)]*2, 'k:')

        # # passive galaxy lines
        ax.plot(xs[xs>0], m_line*xs[xs>0] + c_line, 'k--')
        ax.plot([xrange[0], 0], [np.log10(0.5)]*2, 'k--')

        ax.plot([xrange[0], xrange[1]], [np.log10(3)]*2, 'k--') # weak/radio line
        ax.plot([-0.4, xrange[1]], [np.log10(6)]*2, color='k') # K06 Seyfert/LINER line - weak/strong line

        ax.plot([-0.4]*2, [0.5, yrange[1]], 'k--') # S06 SF/AGN line


        if region_labels:
            ax.text(0.2, 1.5, 'sAGN', size=fontsize)
            ax.text(0.45, 0.6, 'wAGN', size=fontsize)
            ax.text(0.4, 0.1, 'RG', size=fontsize)
            ax.text(-0.9, 1.9, 'SF', size=fontsize)
            ax.text(-0.3, -0.8, 'Passive galaxies', size=fontsize)


    return ax



def get_Halpha_EW_image(catid, ifs_path=ifs_path, estimation_method='median', haflux_masked=None, haerr=None, ha_SN_lim=3, redshift_spec=None, bin='default'):
    """Get the Halpha EW at each spaxel in the cube.
    Can take in a pre-masked Halpha flux array to avoid re-calculating the mask.
    Can take in a specific estimation method (e.g., 'median', 'mean') for the EW calculation.

    Returns Ha_EW_image, Ha_EW_err_image
    """

    # first read in the cube - check that this is alwats catid_A
    lam, flux, variance = read_cube(os.path.join(ifs_path, str(catid), f"{catid}_A_cube_red.fits.gz"))

    # redshift correct the wavelength array
    if redshift_spec is None:
        redshift_spec = all_fctns.get_z_best(catalogue_filepath, [catid], only_spec=True)
    
    lam = lam / (1 + redshift_spec)
    


    if haflux_masked is None:
        # copied from scotts original dr3_sov code:
        haflux_file = os.path.join(ifs_path, str(catid),str(catid)+'_A_Halpha_'+bin+'_1-comp.fits')
        haflux = fits.getdata(haflux_file, ext=0)[0,:,:]
        haerr =  fits.getdata(haflux_file, extname='HALPHA_ERR')[0,:,:]
        hasn = haflux/haerr
        ha_snflag = np.where((hasn > ha_SN_lim),0,1)
        haflux_masked = np.ma.masked_array(haflux,(ha_snflag>0))

    HAlpha_EW_image = np.zeros(haflux_masked.shape)
    HAlpha_EW_err_image = np.zeros(haflux_masked.shape)
    counter = 0

    # now iterate over each spaxel
    for i in range(haflux_masked.shape[0]):
        for j in range(haflux_masked.shape[1]):
            # if not (i in (16,17,18,19,20,21) and (j in (16,17,18,19,20,21))):
                # continue

            if not haflux_masked.mask[i,j]:
                # print(np.sum(~np.isnan(flux[:,i,j])))
                HAlpha_EW_image[i,j], HAlpha_EW_err_image[i,j] = get_EW(catid, ifs_path=ifs_path, estimation_method=estimation_method, 
                                                                                sami_flux_red=flux[:,i,j], sami_lam_red=lam, already_zcorr=True,
                                                                                em_line_flux=haflux_masked[i,j], em_line_error=haerr[i,j])
            else:
                HAlpha_EW_image[i,j] = np.nan
                HAlpha_EW_err_image[i,j] = np.nan
            
            # print(f"Spaxel ({i},{j}): EW = {HAlpha_EW_image[i,j]}, EW_err = {HAlpha_EW_err_image[i,j]}")
            # EW_Ha_tools.get_Halpha_EW_spectra_investigation_plot(flux[:,i,j], lam, haflux_masked[i,j], haerr[i,j])

            if HAlpha_EW_image[i,j] > 0:
                counter += 1
                
    # print(f"Counter: {counter}")

    # print(np.sum(HAlpha_EW_image>0))

    return HAlpha_EW_image, HAlpha_EW_err_image
