import shutil
import os
import numpy as np
import pylab as py
import scipy as sp
import astropy.io.fits as fits
from scipy import stats
import glob
# Folder location with all SAMI cubes:
ifs_path = "/import/hortus1/sami/dr3_ingestion_v8/data/sami/dr3/ifs"


def get_specific_cube_file(catid, cube_type, ifs_path = ifs_path, bin = 'default'):
    """
    Returns the filename of the cube for the corresponding catid and cube_type.
    
    Valid cube_types: 'haflux', 'stelvel', 'stelsig', 'gasvel', 'gassig', 'apspec_blue', 'apspec_red', 'n2flux', 'o3flux', 'hbflux'
    """
    if cube_type == 'haflux':
        return os.path.join(ifs_path, str(catid),str(catid)+'_A_Halpha_'+bin+'_1-comp.fits')
    elif cube_type == 'stelvel':
        return os.path.join(ifs_path, str(catid),str(catid)+'_A_stellar-velocity_'+bin+'_two-moment.fits')
    elif cube_type == 'stelsig':
        return os.path.join(ifs_path, str(catid),str(catid)+'_A_stellar-velocity-dispersion_'+bin+'_two-moment.fits')
    elif cube_type == 'gasvel':
        return os.path.join(ifs_path, str(catid),str(catid)+'_A_gas-velocity_'+bin+'_1-comp.fits')
    elif cube_type == 'gassig':
        return os.path.join(ifs_path, str(catid),str(catid)+'_A_gas-vdisp_'+bin+'_1-comp.fits')
    elif cube_type == 'apspec_blue':
        return os.path.join(ifs_path, str(catid),str(catid)+'_A_spectrum_3-arcsec_blue.fits')
    elif cube_type == 'apspec_red':
        return os.path.join(ifs_path, str(catid),str(catid)+'_A_spectrum_3-arcsec_red.fits')
    elif cube_type == 'n2flux':
        return os.path.join(ifs_path, str(catid),str(catid)+'_A_NII6583_'+bin+'_1-comp.fits')
    elif cube_type == 'o3flux':
        return os.path.join(ifs_path, str(catid),str(catid)+'_A_OIII5007_'+bin+'_1-comp.fits')
    elif cube_type == 'hbflux':
        return os.path.join(ifs_path, str(catid),str(catid)+'_A_Hbeta_'+bin+'_1-comp.fits')
    else:
        raise TypeError(f"Cube type: {cube_type} is not valid")
        return None
    
def get_cube_info(cube_type, ifs_path = ifs_path, bin = 'default'):
    catid = 6821 #smallest CATID
    
    cube_file = get_specific_cube_file(catid, cube_type=cube_type, ifs_path=ifs_path, bin=bin)
    cube_hdul = fits.open(cube_file)
    cube_info = cube_hdul.info()
    print(cube_info)
    return None

def get_centroid(flux, averaging_radius=5, array_dim = 3):
    """
    Compute flux-weighted centroid in a circular region around the brightest spaxel.
    Uses NaN masking (not boolean) to ignore unwanted areas.

    Parameters:
        flux: np.ndarray, shape (λ, x, y)
        averaging_radius: radius (in spaxels) around brightest point to consider

    Returns:
        (x_centroid, y_centroid) or None if total flux is zero/NaN
    """
    assert flux.ndim == array_dim, f"Specified array_dim = {array_dim} does not match actual array dimensions: {flux.ndim}"
    
    if array_dim == 3:
        # Collapse to 2D total flux image
        flux_total = np.nansum(flux, axis=0)
    elif array_dim == 2:
        flux_total = flux.copy()
    else:
        raise ValueError(f"Invalid array_dim {array_dim}")        
        

    # Find the brightest spaxel and set as temporary center
    max_index = np.unravel_index(np.nanargmax(flux_total), flux_total.shape)

    # Apply circular NaN mask
    masked_flux = apply_nan_circle_mask(flux, averaging_radius, max_index, array_dim=2)

    # Collapse masked flux along λ again
    flux_masked_total = np.nansum(masked_flux, axis=0)  # shape (x, y)

    # Create coordinate grids
    x_dim, y_dim = flux_total.shape[0], flux_total.shape[1]
    x_grid, y_grid = np.meshgrid(np.arange(x_dim), np.arange(y_dim), indexing='ij')

    total_flux = np.nansum(flux_masked_total)
    if total_flux == 0 or np.isnan(total_flux):
        return None

    x_centroid = np.nansum(flux_masked_total * x_grid) / total_flux
    y_centroid = np.nansum(flux_masked_total * y_grid) / total_flux

    return (x_centroid, y_centroid)



def apply_nan_circle_mask(array, radius, center, array_dim=3):
    """
    Sets values outside a circular region to NaN. Assumes array shape is (λ, x, y).
    
    Parameters:
        array: np.ndarray, shape (λ, x, y)
        radius: circle radius in spaxels
        center: (x, y) center of circle

    Returns:
        masked_array: np.ndarray of same shape with NaNs outside the circle
    """
    if array_dim == 3:
        λ_dim, x_dim, y_dim = array.shape
        
    elif array_dim == 2:
        x_dim, y_dim = array.shape
    else:
        raise ValueError(f"Invalid array_dim {array_dim}")    

    x = np.arange(x_dim)[:, None]
    y = np.arange(y_dim)[None, :]
    cx, cy = center
    mask2d = ((x - cx)**2 + (y - cy)**2) <= radius**2 

    # Broadcast mask to initial array
    mask3d = np.broadcast_to(mask2d, array.shape)

    masked_array = np.where(mask3d, array, np.nan)
    return masked_array

    
