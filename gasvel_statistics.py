# Goal is to construct code to output a table for all CATIDs with relevant vel_disp statistics that can be used to compare AGNs and non-AGN containing galaxies


import shutil
import os
import numpy as np
import pylab as py
import scipy as sp
from scipy import stats
import glob
import sys


import astropy
from astropy.io import fits
from astropy.table import Table
from astropy.table import join
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.coordinates import SkyCoord

import cube_fctns

# Folder location with all SAMI cubes:
ifs_path = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/ifs"

# Location of SAMI AGN Summary Catalogue
AGN_Summary_path = "SAMI_AGN_matches.fits"

AGN_Summary_table = fits.open(AGN_summary_path)[1].data
all_CATIDS = AGN_Summary_table['CATID']

def get_vel_statistics_table(catids=all_CATIDs, ifs_path=ifs_path, SN_lim=5.0):
    # setup table
    vel_statistics_table = Table(
    
    
    
