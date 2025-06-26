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


# CASDA OPAL username to access RACS data:
OPAL_USER = "mspa4681@uni.sydney.edu.au"

def get_racs_image_cutout(ra, dec, imsize, casda=None, save_dir="racs_cutouts/", redo_cutout=False, print_statements = True):
    """Saves a cutout of RACS-DR1 images at the given coordinate, with the given size. Returns path at which cutout is located."""
        
    save_str = f"RACS_DR1_cutout_{ra}_{dec}_{imsize}.fits"

    #check if file already exists
    if len(glob.glob(save_dir+save_str))>0 and not redo_cutout:
        
        if print_statements:
            print(f"RACS cutout found: {save_dir}{save_str}")
        return save_str
    
    # doesn't exist:
    
    # first check if logged in to CASDA
    if casda is None:
        casda = Casda()
        casda.login(username=OPAL_USER, store_password=False)

    # get args
    centre = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    
    # if not, get the image cutout
    query = "select * from ivoa.obscore "\
        "where filename LIKE 'RACS-DR1%' "\
        "AND filename LIKE '%A.fits' "\
        f"AND 1 = CONTAINS(POINT('ICRS',{ra},{dec}),s_region)"
    # open connection to TAP service and run query
    casdatap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
    job = casdatap.launch_job_async(query)
    table = job.get_results()

    # request a cutout of the images returned by the query and download
    url_list = casda.cutout(table, coordinates=centre, radius=imsize) 

    #only care about the first url but will have to remove the .checksum
    url_relevant = url_list[0].removesuffix('.checksum')
    cutout_file = casda.download_files([url_relevant], savedir=save_dir)[0]

    #rename the saved file and replace the old one if it exists
    if os.path.exists(save_dir+save_str):
        os.remove(save_dir+save_str)

    os.rename(cutout_file, save_dir+save_str)
    
    if print_statements:
        print(f"New RACS cutout saved: {save_dir}{save_str}")
    
    return save_dir+save_str
