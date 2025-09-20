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
import EW_tools

# CASDA OPAL username to access RACS data:
OPAL_USER = "mspa4681@uni.sydney.edu.au"

# Folder location with all SAMI cubes:
ifs_path = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/ifs"
catalogue_filepath = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/catalogues/"


# Location of SAMI AGN Summary Catalogue
AGN_Summary_path = "shared_catalogues/SAMI_AGN_matches.fits"
    

# Code to generate list
CATIDs = fits.getdata(AGN_Summary_path)['CATID']

Ha_EW_table = EW_tools.get_EW_table(CATIDs)

overwrite=True

Ha_EW_table_filename = f"HAlpha_EW_Table.fits"
git_Ha_EW_table_filepath = os.path.join("shared_catalogues", Ha_EW_table_filename)
Ha_EW_table.write(git_Ha_EW_table_filepath , overwrite=overwrite)
print(f"Summary Table Saved to: {git_Ha_EW_table_filepath}")

O3_EW_table = EW_tools.get_EW_table(CATIDs, em_line='O III')

O3_EW_table_filename = f"O_III_EW_Table.fits"
git_O3_EW_table_filepath = os.path.join("shared_catalogues", O3_EW_table_filename)
O3_EW_table.write(git_O3_EW_table_filepath , overwrite=overwrite)
print(f"Summary Table Saved to: {git_O3_EW_table_filepath}")



