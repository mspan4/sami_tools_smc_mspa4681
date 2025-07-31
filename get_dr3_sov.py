import sys
import argparse
from astropy.io import fits
import plot_tools  # assuming this is your own module

OPAL_USER = "mspa4681@uni.sydney.edu.au"

# Folder location with all SAMI cubes:
ifs_path = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/ifs"

# Location of SAMI AGN Summary Catalogue
AGN_Summary_path = "SAMI_AGN_matches.fits"

#############################################################

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate DR3 SOV plots for a given CATID.")

# Required argument
parser.add_argument("catid", type=int, help="Galaxy CATID to process")

# Optional arguments
parser.add_argument("--bin", type=str, help="Binning type")
parser.add_argument("--snlim", type=float, help="S/N limit")
parser.add_argument("--label", type=str, help="Label for output")
parser.add_argument("--isradio", action='store_true', help="Flag if source is radio")
parser.add_argument("--casda", type=str, help="CASDA ID")
parser.add_argument("--redshift", type=float, help="Redshift of source")
parser.add_argument("--OPAL_USER", type=str, help="OPAL user login")
parser.add_argument("--save_folder", type=str, help="Output folder")

args = parser.parse_args()

# check catid is valid if its in the summary catalogue
try:
    catids = fits.getdata(AGN_Summary_path)['CATID']
except Exception as e:
    print(f"Error reading summary file: {e}")
    sys.exit(1)

if args.catid not in catids:
    print(f"CATID {args.catid} not found in summary file: {AGN_Summary_path}")
    sys.exit(1)

kwargs = {"bin": args.bin,
    "snlim": args.snlim,
    "label": args.label,
    "isradio": args.isradio,
    "casda": args.casda,
    "redshift": args.redshift,
    "OPAL_USER": args.OPAL_USER,
    "save_folder": args.save_folder}
    
# Remove keys where the value is None so they don't ruin default function values
clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}


plot_tools.plot_dr3_sov(
    args.catid,
    do_printstatement=True,
    dopdf=True,
    **clean_kwargs)

