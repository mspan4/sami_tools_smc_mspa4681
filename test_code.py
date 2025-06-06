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

# Folder location with all SAMI cubes:
ifs_path = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/ifs"

###############################################################

import plot_tools

catid = 9011900430 # highest flux radio source
plot_tools.plot_dr3_sov(catid, isradio=True)

#url =     'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx?ra=174.1531666666667&dec=0.815861111111111&width=150&height=150&scale=0.49992'
#url = url.replace("http://", "https://")
#urlretrieve(url, 'SDSS_cutout.jpg')

#cutout_file = "/suphys/mspa4681/.astropy/cache/astroquery/Casda/cutout-936737-imagecube-71340.fits"
#image = fits.open(cutout_file)[0].data.squeeze()
#py.contour(np.fliplr(image), cmap='magma')
#py.show()

