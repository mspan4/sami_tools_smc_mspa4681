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

import Code.all_fctns as all_fctns

# CASDA OPAL username to access RACS data:
OPAL_USER = "mspa4681@uni.sydney.edu.au"

# Folder location with all SAMI cubes:
ifs_path = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/ifs"
catalogue_filepath = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/catalogues/"


# Location of SAMI AGN Summary Catalogue
AGN_Summary_path = "shared_catalogues/SAMI_AGN_matches.fits"



def get_redshift_corrected_spectra(CATID, spectra_filepath=None):
    if spectra_filepath== None:
        spectra_filepath = os.path.join(ifs_path, str(CATID))
        
    apspecfile_red = os.path.join(spectra_filepath,str(CATID)+'_A_spectrum_1-4-arcsec_red.fits')
    hdulist = fits.open(apspecfile_red)
    sami_flux_red,sami_lam_red = sami_read_apspec(hdulist,0,doareacorr=False)
    hdulist.close()
    redshift = all_fctns.get_z_best(catalogue_filepath, [CATID])
    sami_lam_red_zcorr = sami_lam_red / (1 + redshift)
    return sami_flux_red, sami_lam_red_zcorr


def get_spectra_region_flux(sami_lam, sami_flux, region):
    """Get the mean flux in a given wavelength region."""
    region_mask = (sami_lam >= region[0]) & (sami_lam <= region[1])
    region_flux = np.mean(sami_flux[region_mask])
    region_flux_err = np.std(sami_flux[region_mask]) # check this
    return region_flux, region_flux_err

def get_Halpha_EW(CATIDs, catalogue_filepath=catalogue_filepath, spectra_filepath=None):
    """Get the Halpha equivalent width for a given CATID.
    
    Uses the EmissionLine1compDR3.fits catalogue, and individual spectra to find mean of two continuum either side of the Halpha, NII peak.
    """
    
    # set the regions for collecting continuum flux
    region1 = (6400, 6500)
    region2 = (6600, 6700)


    # read in the emission line catalogue
    SAMI_spectra_catalogue = "EmissionLine1compDR3.fits"
    with fits.open( os.path.join(catalogue_filepath, SAMI_spectra_catalogue) ) as SAMI_spectra_hdul:
        SAMI_spectra_table_hdu = Table(SAMI_spectra_hdul[1].data)

    # initialise HA_EW astropy table
    Ha_EW_table = Table(names=['CATID', 'HAlpha_EW', 'HAlpha_EW_err'], dtype=[int, float, float])

    
    # iterate over CATIDs
    for CATID in CATIDs:
        HAlpha_flux, HAlpha_error = all_fctns.get_flux_and_error_1_4_ARCSEC(SAMI_spectra_table_hdu[SAMI_spectra_table_hdu['CATID'] == CATID], 'H Alpha')
        
        # use the first value only and check if there is an actual value
        try:
            HAlpha_flux = HAlpha_flux[0]
            HAlpha_error = HAlpha_error[0]
        except IndexError:
            HAlpha_flux = np.nan
            HAlpha_error = np.nan

        # read in the spectra
        try:
            sami_flux_red, sami_lam_red_zcorr = get_redshift_corrected_spectra(CATID, spectra_filepath)
        except FileNotFoundError:
            print(f"No valid red spectra found for {CATID}")
            Ha_EW = np.nan
            Ha_EW_err = np.nan
            Ha_EW_table.add_row([CATID, Ha_EW, Ha_EW_err])
            continue
            

        # get the continuum flux
        region1_flux, region1_flux_err = get_spectra_region_flux(sami_lam_red_zcorr, sami_flux_red, region1)
        region2_flux, region2_flux_err = get_spectra_region_flux(sami_lam_red_zcorr, sami_flux_red, region2)
        continuum_flux = (region1_flux + region2_flux) / 2
        continuum_flux_err = continuum_flux * np.sqrt( (region1_flux_err/region1_flux)**2 + (region2_flux_err/region2_flux)**2) #

        # calculate the EW
        Ha_EW = HAlpha_flux / continuum_flux
        Ha_EW_err = Ha_EW * np.sqrt((HAlpha_error / HAlpha_flux)**2 + (continuum_flux_err / continuum_flux)**2)

        # add row to table
        #print(HAlpha_flux)
        #print([CATID, Ha_EW, Ha_EW_err])
        Ha_EW_table.add_row([CATID, Ha_EW, Ha_EW_err])
        
    return Ha_EW_table
    

# Code to generate list
CATIDs = fits.getdata(AGN_Summary_path)['CATID']

Ha_EW_table = get_Halpha_EW(CATIDs)

overwrite=True

Ha_EW_table_filename = f"HAlpha_EW_Table.fits"
git_Ha_EW_table_filepath = os.path.join("shared_catalogues", Ha_EW_table_filename)
Ha_EW_table.write(git_Ha_EW_table_filepath , overwrite=overwrite)
print(f"Summary Table Saved to: {git_Ha_EW_table_filepath}")

    




