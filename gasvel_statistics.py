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

AGN_Summary_table = fits.open(AGN_Summary_path)[1].data
all_CATIDs = AGN_Summary_table['CATID']


def get_gassig_statistics_table(catids=all_CATIDs, save_filepath=os.path.join('shared_catalogues','gassig_statistics.fits'), ifs_path=ifs_path, snlim=5.0, save_file = True):
    """
    Create a table containing velocity dispersion statistics for each given CATID.
    """
    # setup table
    vel_statistics_table = Table(names=['CATID', 'MEDIAN_VEL_DISP', 'VEL_DISP_SEM'])

    # get the gassig cube for each CATID
    for catid in catids:
        # first get ha cube for S/N mask
        haflux_file = cube_fctns.get_specific_cube_file(catid, 'haflux', ifs_path=ifs_path)
        try:
            haflux = fits.getdata(haflux_file, ext=0)[0,:,:]
        except Exception as e:
            continue
        haerr =  fits.getdata(haflux_file, extname='HALPHA_ERR')[0,:,:]
        hasn = haflux/haerr
        ha_snflag = np.where((hasn > snlim),0,1)


        gassig_file = cube_fctns.get_specific_cube_file(catid, 'gassig', ifs_path=ifs_path)

        gassig = fits.getdata(gassig_file, ext=0)[0,:,:]
        gassig_err = fits.getdata(gassig_file, extname='VDISP_ERR')[0,:,:]

        gassig_masked = np.ma.masked_array(gassig,(ha_snflag>0))
        gassig_err_masked = np.ma.masked_array(gassig_err,(ha_snflag>0))


        # get the median velocity dispersion (and SEM)
        median_vel_disp = np.ma.median(gassig_masked)
        vel_disp_sem = stats.sem(gassig_masked.compressed())
    
        # add to table
        vel_statistics_table.add_row([catid, median_vel_disp, vel_disp_sem])
        
        
    if save_file == True:
        vel_statistics_table.write(save_filepath)
        
    return vel_statistics_table
    
    
get_gassig_statistics_table()
