"""Script to get a cutout of RACS-DR1 images of a given coordinate.
"""

import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.casda import Casda  
from astroquery.utils.tap.core import TapPlus

# set these to your own values
OPAL_USER = "mspa4681@uni.sydney.edu.au"
SAVEDIR = "C:\\Users\\mspan\\Documents\\University\\Honours\\Code"

# validate args
if len(sys.argv) != 4:
    print("Need 3 args")
    print(f"Usage: python {sys.argv[0]} ra_deg dec_deg radius_arcmin")
    exit()

# get args
centre = SkyCoord(sys.argv[1], sys.argv[2], unit=(u.deg, u.deg))
ra = centre.ra.degree
dec = centre.dec.degree
radius = float(sys.argv[3]) * u.arcmin

# construct query for catalogued sources in RACS-DR1
# ivoa.obscore is the table name
# filters for RACS-DR1 fits images containing sources overlapping with
#   the given coordinates
query = "select * from ivoa.obscore "\
        "where filename LIKE 'RACS-DR1%' "\
        "AND filename LIKE '%A.fits' "\
       f"AND 1 = CONTAINS(POINT('ICRS',{ra},{dec}),s_region)"

# open connection to TAP service and run query
casdatap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
job = casdatap.launch_job_async(query)
table = job.get_results()

print(table)

# login to CASDA to be able to download images
casda = Casda()
casda.login(username=OPAL_USER, store_password=True)

# request a cutout of the images returned by the query and download
url_list = casda.cutout(table, coordinates=centre, radius=radius)
file_list = casda.download_files(url_list, savedir=SAVEDIR)

print("Downloaded:")
for file in file_list:
    print(f"  {file}")