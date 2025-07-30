# Goal is to construct code to output a table for all CATIDs with relevant vel_disp statistics that can be used to compare AGNs and non-AGN containing galaxies


import shutil
import os
import numpy as np
import pylab as py
import scipy as sp
from scipy import stats
import glob
import sys
from matplotlib.backends.backend_pdf import PdfPages

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


def get_gassig_statistics_table(catids=all_CATIDs, save_filepath=os.path.join('shared_catalogues','gassig_statistics.fits'), 
                                ifs_path=ifs_path, snlim=10.0, save_file = True, centroid_radius=3.0):
    """
    Create a table containing velocity dispersion statistics for each given CATID.
    """
    # setup table
    vel_statistics_table = Table(names=['CATID', 'MEDIAN_VEL_DISP', 'VEL_DISP_SEM', '5SPAXEL_MEDIAN_VEL_DISP', '5SPAXEL_VEL_DISP_SEM'], dtype=[int, float, float, float, float])

    # get the gassig cube for each CATID
    for catid in catids:
        print(catid)
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
        
        axs[0].imshow(gassig_masked, origin='lower')
        axs[0].set_title('gassig_masked')
        
        gassig_err_masked = np.ma.masked_array(gassig_err,(ha_snflag>0))

        # get the median velocity dispersion (and SEM)
        median_vel_disp = np.ma.median(gassig_masked)
        vel_disp_sem = stats.sem(gassig_masked.compressed())



        # Now, get the centre of the cube (centroid based on flux)
        # first need stelflux cube
        stelvel_file = cube_fctns.get_specific_cube_file(catid, 'stelvel', ifs_path=ifs_path)
        #stelvel = fits.getdata(stelvel_file, ext=0)[0,:,:]
        stelflux = fits.getdata(stelvel_file, extname='FLUX')

        # now get the centroid
        centroid_position = cube_fctns.get_centroid(stelflux, averaging_radius=centroid_radius, array_dim=2)
        
        axs[1].imshow(stelflux, origin='lower')
        axs[1].plot(centroid_position[1], centroid_position[0], '.')
        axs[1].set_title('stelflux')
       

        # get the median velocity dispersion within 5 spaxels of the centroid
        if centroid_position is not None:
            reduced_5spaxel_gassig_masked = cube_fctns.apply_nan_circle_mask(gassig_masked, 5, centroid_position, array_dim = 2)
            median_5spaxel_vel_disp = np.nanmedian(reduced_5spaxel_gassig_masked)

            sem_5spaxel_vel_disp = stats.sem(reduced_5spaxel_gassig_masked, axis=None, nan_policy='omit')
            axs[2].imshow(reduced_5spaxel_gassig_masked, origin='lower')
            axs[2].set_title('5spaxel_gassig')

        else:
            median_5spaxel_vel_disp = np.nan
            sem_5spaxel_vel_disp = np.nan

        # add to table
        vel_statistics_table.add_row([catid, median_vel_disp, vel_disp_sem, median_5spaxel_vel_disp, sem_5spaxel_vel_disp])


    if save_file == True:
        vel_statistics_table.write(save_filepath, overwrite=True)
        
    return vel_statistics_table
    
fig1 = py.figure(1,constrained_layout=True)
fig1.clf()
axs = fig1.subplots(1,3)
axs = axs.flatten()
get_gassig_statistics_table()

pdf = PdfPages(f"test_plots/gassig_cube_testing")
py.draw()
# pause for input if plotting all the spectra:
#yn = input('Continue? (y/n):')
py.savefig(pdf, format='pdf')
pdf.close()
