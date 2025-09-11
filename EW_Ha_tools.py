
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




def get_redshift_corrected_spectra(CATID, ifs_path=ifs_path):
    apspecfile_red = os.path.join(ifs_path,str(CATID), str(CATID)+'_A_spectrum_1-4-arcsec_red.fits')
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


def get_continuum_flux(CATID, regions, spectra_filepath=None, estimation_method='median', sami_flux_red=None, sami_lam_red=None, already_zcorr=False, catalogue_filepath=catalogue_filepath):
    """Get the continuum flux at Halpha for a given CATID.\\
    Two continuum estimation methods implemented: 'median' and 'linefit'.\\
    This function can take in a pre-loaded spectra or relevant CATID.
    
    returns continuum_flux, continuum_flux_err
    """
    Ha_lam = 6562.819

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
            sami_flux_red, sami_lam_red_zcorr = get_redshift_corrected_spectra(CATID, spectra_filepath)
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


    elif estimation_method == 'linefit': #linear fit to the two regions, evaluate at Ha_lam to get continuum flux, error from fit errors
        # get the relevant points for line fitting
        regions_mask = (sami_lam_red_zcorr >= regions[0][0]) & (sami_lam_red_zcorr <= regions[0][1]) | (sami_lam_red_zcorr >= regions[1][0]) & (sami_lam_red_zcorr <= regions[1][1])
        regions_sami_lam_red_zcorr = sami_lam_red_zcorr[regions_mask]
        regions_sami_flux_red = sami_flux_red[regions_mask]

        # fit a line to these points
        res = sp.stats.linregress(regions_sami_lam_red_zcorr, regions_sami_flux_red)
        slope, intercept, slope_err, intercept_err = res.slope, res.intercept, res.stderr, res.intercept_stderr
        continuum_flux = slope*Ha_lam + intercept
        continuum_flux_err = np.sqrt( (slope_err*Ha_lam)**2 + intercept_err**2 )
    else:
        raise NotImplementedError("Only 'median' and 'linefit' estimation methods are implemented.")
    
    return continuum_flux, continuum_flux_err
    

def get_Halpha_EW(CATID, catalogue_filepath=catalogue_filepath, ifs_path=ifs_path, estimation_method='median', SAMI_spectra_table_hdu=None,  sami_flux_red=None, sami_lam_red=None, already_zcorr=False):
    """Get the Halpha equivalent width for a given CATID in the 1.4 arcsec aperture.\\
    Two continuum estimation methods implemented: 'median' and 'linefit'.\\
    This function can take in a pre-loaded SAMI_spectra_table (EmissionLine1compDR3.fits) to avoid re-loading the catalogue for each CATID.\\
    This function can also take in a pre-loaded spectra or relevant CATID. to feed into the continuum flux calculation. (see get_continuum_flux function)\\

    returns Ha_EW, Ha_EW_err
    """
    # set the regions for collecting continuum flux, should be symmetric around Halpha
    region_width = 65
    region_separation = 140
    Ha_lam = 6562.819
    region1 = np.array([-region_width, 0]) + Ha_lam - region_separation/2  
    region2 = np.array([0, region_width]) + Ha_lam + region_separation/2  


    # read in the emission line catalogue if not provided
    if SAMI_spectra_table_hdu is None:
        SAMI_spectra_catalogue = "EmissionLine1compDR3.fits"
        with fits.open( os.path.join(catalogue_filepath + SAMI_spectra_catalogue) ) as SAMI_spectra_hdul:
            SAMI_spectra_table_hdu = Table(SAMI_spectra_hdul[1].data)
    else:
        SAMI_spectra_table_hdu = SAMI_spectra_table_hdu

    
    HAlpha_flux, HAlpha_error = all_fctns.get_flux_and_error_1_4_ARCSEC(SAMI_spectra_table_hdu[SAMI_spectra_table_hdu['CATID'] == CATID], 'H Alpha')
    
    # use the first value only and check if there is an actual value
    try:
        HAlpha_flux = HAlpha_flux[0]
        HAlpha_error = HAlpha_error[0]
    except IndexError:
        HAlpha_flux = np.nan
        HAlpha_error = np.nan
            

    # get the continuum flux
    continuum_flux, continuum_flux_err = get_continuum_flux(CATID, (region1, region2), ifs_path, estimation_method=estimation_method, sami_flux_red=sami_flux_red, sami_lam_red=sami_lam_red, already_zcorr=already_zcorr, catalogue_filepath=catalogue_filepath)

    # calculate the EW
    Ha_EW = HAlpha_flux / continuum_flux
    Ha_EW_err = Ha_EW * np.sqrt((HAlpha_error / HAlpha_flux)**2 + (continuum_flux_err / continuum_flux)**2)

        
    return Ha_EW, Ha_EW_err


def get_Halpha_EW_table(CATIDs, catalogue_filepath=catalogue_filepath, ifs_path=ifs_path):
    """Get the Halpha equivalent width for a given CATID.
    
    Uses the EmissionLine1compDR3.fits catalogue, and individual spectra.
    """
    # set the regions for collecting continuum flux, should be symmetric around Halpha
    region_width = 65
    region_separation = 140
    Ha_lam = 6562.819
    region1 = np.array([-region_width, 0]) + Ha_lam - region_separation/2  
    region2 = np.array([0, region_width]) + Ha_lam + region_separation/2  


    # read in the emission line catalogue
    SAMI_spectra_catalogue = "EmissionLine1compDR3.fits"
    with fits.open(catalogue_filepath + SAMI_spectra_catalogue) as SAMI_spectra_hdul:
        SAMI_spectra_table_hdu = Table(SAMI_spectra_hdul[1].data)

    # initialise HA_EW astropy table
    Ha_EW_table = Table(names=['CATID', 'HAlpha_EW_MedianFit', 'HAlpha_EW_err_MedianFit', 'HAlpha_EW_LineFit', 'HAlpha_EW_err_LineFit'], dtype=[int, float, float, float, float])

    
    # iterate over CATIDs
    for CATID in CATIDs:
        HAlpha_flux, HAlpha_error = all_fctns.get_flux_and_error_1_4_ARCSEC(SAMI_spectra_table_hdu[SAMI_spectra_table_hdu['CATID'] == CATID], 'H Alpha')

        Ha_EW_medianfit, Ha_EW_medianfit_err = get_Halpha_EW(CATID, catalogue_filepath=catalogue_filepath, ifs_path=ifs_path, estimation_method='median', SAMI_spectra_table_hdu=SAMI_spectra_table_hdu)
        Ha_EW_linefit, Ha_EW_linefit_err = get_Halpha_EW(CATID, catalogue_filepath=catalogue_filepath, ifs_path=ifs_path, estimation_method='linefit', SAMI_spectra_table_hdu=SAMI_spectra_table_hdu)
        # add row to table

        Ha_EW_table.add_row([CATID, Ha_EW_medianfit, Ha_EW_medianfit_err, Ha_EW_linefit, Ha_EW_linefit_err])
        
    return Ha_EW_table
