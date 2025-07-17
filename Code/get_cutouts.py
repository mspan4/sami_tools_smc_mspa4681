"""Script to get a cutout of fits tables within SAMI target regions

    returns a fits file with only SAMI target regions remaining with name Source_dir + SAMI_target_regions_cutout_ + fits_filepath
"""
import sys
import numpy as np
import astropy
from astropy.io import ascii
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from matplotlib.colors import LogNorm


def angular_dist(r1, d1, r2, d2):    
    a = np.sin(np.abs(d1 - d2)/2)**2
    b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2
    
    d_rad = 2*np.arcsin((a+b)**(1/2))
    d_deg = np.degrees(d_rad)
    return d_deg

def get_RA_Dec_cutout_fits_table(table_hdu: Table, bounds: tuple, bound_type = 'rect', col_names = ('RA', 'Dec')):
    
    if bound_type == 'rect':
        # Get rid of all not within RA range
        RA_array = np.array(table_hdu[col_names[0]])
        sort_ind_RA_removed = (RA_array > bounds[0][0]) & (RA_array < bounds[1][0] )
        table_hdu_RA_removed = table_hdu[sort_ind_RA_removed]


        # Get rid of all values not within Dec range
        Dec_array = np.array(table_hdu_RA_removed[col_names[1]])
        sort_ind_both_removed = (Dec_array > bounds[0][1]) & (Dec_array < bounds[1][1] )
        table_hdu_both_removed = table_hdu_RA_removed[sort_ind_both_removed]
    
    elif bound_type == 'circ': # need to convert degrees to radians for np.sin and np.cos
        RA_array = np.array(table_hdu[col_names[0]]) * np.pi/180
        Dec_array = np.array(table_hdu[col_names[1]]) * np.pi/180

        r_bound = bounds[1]
        circle_centre_RA = bounds[0][0] * np.pi/180
        circle_centre_Dec = bounds[0][1] * np.pi/180

        mask_within_radius = angular_dist(RA_array, Dec_array, circle_centre_RA, circle_centre_Dec) < r_bound
        table_hdu_both_removed = table_hdu[mask_within_radius]

    else:
        raise TypeError(f"Invalid bound type: {bound_type}. Valid types are \'rect\' and \'circ\'")

    return table_hdu_both_removed


#Source_dir ="C:\\Users\\mspan\\OneDrive - The University of Sydney (Students)\\Honours\\"

# testing a cutout of 09h Gamma Region

#input_files = ["RACS-mid1_components.fits", "RACS-mid1_sources.fits"]


# validate args
if len(sys.argv) != 3 and len(sys.argv) != 5:
    print("Need 2 or 3 args")
    print(f"Usage: python {sys.argv[0]} source_dir fits_filepath RA_col_name Dec_col_name")
    exit()
    
Source_dir = sys.argv[1]
fits_filepath = sys.argv[2]

if len(sys.argv) == 5:
    col_names = (sys.argv[3], sys.argv[4])
else:
    col_names = ('RA', 'Dec')


GAMA_region_bounds = [((129,-2), (141, 3)), ((174, -3), (186, 2)), ((211.5, -2), (223.5, 3))] # bounds of the GAMA target regions in the SAMI data including filler which are rectangular in shape (bottom left to top right coords) all in degrees

Cluster_region_bounds = [((355.397880, -29.236351), 2), ((18.815777, 0.213486), 1.5), ((18.739974, 0.430807), 1), 
                        ((356.937810, -28.140661), 3), ((6.380680, -33.046570), 1.5), ((336.977050, -30.575371), 2), 
                        ((329.372605, -7.795692), 2), ((14.067150, -1.255370), 2), ((10.460211, -9.303184), 3)] # bounds of the Cluster target regions (including filler targets) circular in shape  ((x_centre,y_centre), r_approx) all in degrees


with fits.open(Source_dir + fits_filepath) as hdul:
    cutout_tables = []

    table_hdu = Table(hdul[1].data)


    bound_type = 'rect'

    for bounds in GAMA_region_bounds:
        table_hdu_both_cutout = get_RA_Dec_cutout_fits_table(table_hdu, bounds, bound_type=bound_type, col_names=col_names)
        cutout_tables.append(table_hdu_both_cutout)


    bound_type = 'circ'

    for bounds in Cluster_region_bounds:
        table_hdu_both_cutout = get_RA_Dec_cutout_fits_table(table_hdu, bounds, bound_type=bound_type, col_names=col_names)
        cutout_tables.append(table_hdu_both_cutout)


table_hdu_allcutouts = astropy.table.vstack(cutout_tables) # stack tables together
unique_table_hdu_allcutouts = astropy.table.unique(table_hdu_allcutouts) # remove duplicates created from vstacking

print(f"{len(table_hdu)-len(unique_table_hdu_allcutouts)} objects cutout.")

unique_table_hdu_allcutouts.write(Source_dir+"SAMI_target_regions_cutout_"+fits_filepath)

