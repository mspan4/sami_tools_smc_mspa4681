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
from urllib.parse import urlencode
from urllib.request import urlretrieve
import pylab as py
from PIL import Image
import glob
import os

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u

# Folder location with all SAMI cubes:
ifs_path = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/ifs"

# Location of SAMI AGN Summary Catalogue
AGN_Summary_path = "SAMI_AGN_matches.fits"

# CASDA OPAL username to access RACS data:
OPAL_USER = "mspa4681@uni.sydney.edu.au"
###############################################################

import plot_tools
from racs_cutout_tools import get_racs_image_cutout
import cube_fctns
bin = 'default'
catid = 9011900430


haflux_file = cube_fctns.get_specific_cube_file(catid, 'gassig')
print(fits.open(haflux_file).info())

hdulist = fits.open(AGN_Summary_path)
tab = hdulist[1].data



'''
test_catid = 32362
plot_tools.plot_dr3_sov(test_catid, isradio=True)

#ab_relevant = tab[tab['catid'] == test_catid]

apspecfile_blue = os.path.join(ifs_path, str(test_catid),str(test_catid)+'_A_spectrum_3-arcsec_blue.fits')
hdulist = fits.open(apspecfile_blue)
ra = hdulist[0].header['CATARA']
dec = hdulist[0].header['CATADEC']
print(ra)
print(dec)
'''


#impix = 3 * 50
#imsize = 1*0.4166*u.arcmin

#get_racs_image_cutout(ra, dec, imsize)


'''
BPT_AGN_labels = (6,7,8,9,10,-1)

radio_AGN_catids = tab['CATID'][np.isin(tab['CATEGORY_BPT_AGN'], BPT_AGN_labels)]

plot_tools.plot_sov_many_new(AGN_Summary_path, radio_AGN_catids, save_name = 'many_sov_radio_AGNs.pdf', only_radio=True)
'''




#catid = 9011900430 # highest flux radio source
#plot_tools.plot_dr3_sov(catid, isradio=False)


'''
filename = "shared_catalogues/gassig_statistics.fits"
gassig_statistics = fits.open(filename)
gassig_table = gassig_statistics[1].data

mask = gassig_table['MEDIAN_VEL_DISP'] > 400
relevant_CATIDs = gassig_table['CATID'][mask]

plot_tools.plot_sov_many_new(AGN_Summary_path, relevant_CATIDs, save_name='high_gassig_many_sov.pdf', radio_sources=False)
'''

'''
relevant_CATIDs = [30615, 214250, 273336, 508132, 9011900034, 9011900137,
 9388000003, 9388000041, 9403800001, 9403800117]

plot_tools.plot_sov_many_new(AGN_Summary_path, relevant_CATIDs, save_name='high_gassig_many_sov.pdf', radio_sources=False)
'''


catid = 41274
plot_tools.plot_sov_many_new(AGN_Summary_path, [catid], radio_sources=False,
save_name='test_many_sov.pdf')

#plot_tools.plot_dr3_sov(catid, isradio=False)



