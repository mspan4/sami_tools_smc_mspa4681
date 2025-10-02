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


def get_gassig_statistics_table(catids=all_CATIDs, save_filepath=os.path.join('shared_catalogues','gassig_statistics.fits'), ifs_path=ifs_path, snlim=10.0, save_file = False, centroid_radius=3.0, min_spaxel_count = 10, cube_plots = False):
    """
    Create a table containing velocity dispersion statistics for each given CATID.
    """
    # setup table
    vel_statistics_table = Table(names=['CATID', 'MEDIAN_VEL_DISP', 'VEL_DISP_SEM', '5SPAXEL_MEDIAN_VEL_DISP', '5SPAXEL_VEL_DISP_SEM', 'OUTSIDE_5SPAXEL_VEL_DISP', 'OUTSIDE_5SPAXEL_VEL_DISP_SEM'], dtype=[int, float, float, float, float, float, float])
    
    if cube_plots:
        pdf = PdfPages(f"test_plots/gassig_cube_testing")

    count =0
    
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
        
        gassig_err_masked = np.ma.masked_array(gassig_err,(ha_snflag>0))

        # check enough valid spaxels
        gassig_masked_num_spaxels = np.sum(np.isfinite(gassig_masked))
        
        # get the median velocity dispersion (and SEM)
        if gassig_masked_num_spaxels > min_spaxel_count:
            median_vel_disp = np.ma.median(gassig_masked)
            vel_disp_sem = stats.sem(gassig_masked.compressed())
        else:
            median_vel_disp = np.nan
            vel_disp_sem = np.nan
            
            
        if cube_plots:
            fig1 = py.figure(1,constrained_layout=True)
            fig1.clf()
            axs = fig1.subplots(1,4)
            axs = axs.flatten()

            axs[0].imshow(gassig_masked, origin='lower')
            axs[0].set_title(f'gassig_masked - {gassig_masked_num_spaxels}')


        # Now, get the centre of the cube (centroid based on flux)
        # first need stelflux cube
        stelvel_file = cube_fctns.get_specific_cube_file(catid, 'stelvel', ifs_path=ifs_path)
        #stelvel = fits.getdata(stelvel_file, ext=0)[0,:,:]
        stelflux = fits.getdata(stelvel_file, extname='FLUX')

        # now get the centroid
        centroid_position = cube_fctns.get_centroid(stelflux, averaging_radius=centroid_radius, array_dim=2)
        
        if cube_plots:
            axs[1].imshow(stelflux, origin='lower')
            axs[1].plot(centroid_position[1], centroid_position[0], '.')
            axs[1].set_title('stelflux')
            
            axs[0].plot(centroid_position[1], centroid_position[0], '.')
            axs[2].plot(centroid_position[1], centroid_position[0], '.')

        # get the median velocity dispersion within 5 spaxels of the centroid and that outside 5 spaxels
        if centroid_position is not None:            
            reduced_5spaxel_gassig_masked = cube_fctns.apply_nan_circle_mask(gassig_masked, 5, centroid_position, array_dim = 2)
            reduced_outisde_5spaxel_gassig_masked = cube_fctns.apply_nan_circle_mask(gassig_masked, 5, centroid_position, array_dim = 2, inverse=True)
            
            reduced_5spaxel_gassig_masked_num_spaxels = np.sum(np.isfinite(reduced_5spaxel_gassig_masked))
            reduced_outisde_5spaxel_gassig_masked_num_spaxels = np.sum(np.isfinite(reduced_outisde_5spaxel_gassig_masked))

            
            if reduced_5spaxel_gassig_masked_num_spaxels > min_spaxel_count:
                median_5spaxel_vel_disp = np.ma.median(reduced_5spaxel_gassig_masked)
                sem_5spaxel_vel_disp = stats.sem(reduced_5spaxel_gassig_masked, axis=None, nan_policy='omit')

            else:
                median_5spaxel_vel_disp = np.nan
                sem_5spaxel_vel_disp = np.nan


            if reduced_outisde_5spaxel_gassig_masked_num_spaxels > min_spaxel_count:
                median_outisde_5spaxel_vel_disp = np.ma.median(reduced_outisde_5spaxel_gassig_masked)
                sem_outisde_5spaxel_vel_disp = stats.sem(reduced_outisde_5spaxel_gassig_masked, axis=None, nan_policy='omit')
            
            else:
                median_outisde_5spaxel_vel_disp = np.nan
                sem_outisde_5spaxel_vel_disp = np.nan


            if cube_plots:
                axs[2].imshow(reduced_5spaxel_gassig_masked, origin='lower')
                axs[2].set_title(f'5spaxel_gassig - {reduced_5spaxel_gassig_masked_num_spaxels}')

                axs[3].imshow(reduced_outisde_5spaxel_gassig_masked, origin='lower')
                axs[3].set_title(f'outside5spaxel_gassig - {reduced_outisde_5spaxel_gassig_masked_num_spaxels}')

        else:
            median_5spaxel_vel_disp = np.nan
            sem_5spaxel_vel_disp = np.nan
            median_outisde_5spaxel_vel_disp = np.nan
            sem_outisde_5spaxel_vel_disp = np.nan

        # add to table
        vel_statistics_table.add_row([catid, median_vel_disp, vel_disp_sem, median_5spaxel_vel_disp, sem_5spaxel_vel_disp, median_outisde_5spaxel_vel_disp, sem_outisde_5spaxel_vel_disp])
        
        if cube_plots:
            py.draw()
            py.savefig(pdf, format='pdf')
            
            

    if save_file == True:
        vel_statistics_table.write(save_filepath, overwrite=True)
    
    if cube_plots:
        pdf.close()
    return vel_statistics_table
    

get_gassig_statistics_table(cube_plots=False, save_file=True)

#checking high gassig cubes
'''
filename = "shared_catalogues/gassig_statistics.fits"
gassig_statistics = fits.open(filename)
gassig_table = gassig_statistics[1].data

mask = gassig_table['MEDIAN_VEL_DISP'] > 400
relevant_CATIDs = gassig_table['CATID'][mask]
relevant_CATIDs = [np.int64(30615), np.int64(214250), np.int64(273336), np.int64(508132), np.int64(9011900034), np.int64(9011900137), np.int64(9388000003), np.int64(9388000041), np.int64(9403800001), np.int64(9403800117)]
print(relevant_CATIDs)

get_gassig_statistics_table(relevant_CATIDs, save_filepath=os.path.join('shared_catalogues','test_gassig_statistics.fits'), cube_plots=True)
'''


