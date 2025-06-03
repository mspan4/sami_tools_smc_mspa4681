
import numpy as np
import math

################################################################################

def median_smooth( spectrum, binsize, givenmad=False ):

    bon2 = binsize/2.
    specsize = spectrum.shape[0]
    result = np.zeros( specsize )
    if givenmad :
        nmad = np.zeros( specsize )
    for i in range( specsize ):
        lo = max( int(i-bon2), 0 )
        hi = min( int(i+bon2+1), specsize )
        med = np.nanmedian( spectrum[ lo:hi ] )
        result[ i ] = med
        if givenmad :
            nmad[ i ] = 1.486 * np.mediannan( np.abs( spectrum[ lo:hi ] - med ) )

    if givenmad :
        return result, nmad
    else :
        return result

###########################################################################
# get rms of array, including sigma clipping:

def med_abs_dev(data):


    ys,xs = np.shape(data)

    diff = np.zeros((ys,xs))
    
    # get median:
    med = np.nanmedian(data,axis=0)

    print(med)
    
    # subtract median from data:
    for i in xrange(ys):
        diff[i,:] = abs(data[i,:] - med)

    # take the median of the difference:
    mad = np.nanmedian(diff,axis=0)

    return mad

##############################################################################
# routine to rebin an array:
        
def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

        
def sqrt_scale(inputArray, scale_min=None, scale_max=None):
	"""Performs sqrt scaling of the input numpy array.

	@type inputArray: numpy array
	@param inputArray: image data array
	@type scale_min: float
	@param scale_min: minimum data value
	@type scale_max: float
	@param scale_max: maximum data value
	@rtype: numpy array
	@return: image data array
	
	"""		
    
	print("img_scale : sqrt")
	imageData=np.array(inputArray, copy=True)
	
	if scale_min == None:
		scale_min = imageData.min()
	if scale_max == None:
		scale_max = imageData.max()

	imageData = imageData.clip(min=scale_min, max=scale_max)
	imageData = imageData - scale_min
	indices = np.where(imageData < 0)
	imageData[indices] = 0.0
	imageData = np.sqrt(imageData)
	imageData = imageData / math.sqrt(scale_max - scale_min)
	
	return imageData

###############################################################################
# spectres routine from: https://github.com/ACCarnall/SpectRes
#
def spectres(new_spec_wavs, old_spec_wavs, spec_fluxes, spec_errs=None):

    """ 
    Function for resampling spectra (and optionally associated uncertainties) onto a new wavelength basis.
    Parameters
    ----------
    new_spec_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the spectrum or spectra.
    old_spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the spectrum or spectra.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in old_spec_wavs, last dimension must correspond to the shape of old_spec_wavs.
        Extra dimensions before this may be used to include multiple spectra.
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties associated with each spectral flux value.
    
    Returns
    -------
    resampled_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same length as new_spec_wavs, other dimensions are the same as spec_fluxes
    resampled_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in resampled_fluxes. Only returned if spec_errs was specified.
    """

    # Generate arrays of left hand side positions and widths for the old and new bins
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_widths = np.zeros(old_spec_wavs.shape[0])
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_lhs[0] = old_spec_wavs[0] - (old_spec_wavs[1] - old_spec_wavs[0])/2
    spec_widths[-1] = (old_spec_wavs[-1] - old_spec_wavs[-2])
    spec_lhs[1:] = (old_spec_wavs[1:] + old_spec_wavs[:-1])/2
    spec_widths[:-1] = spec_lhs[1:] - spec_lhs[:-1]

    filter_lhs = np.zeros(new_spec_wavs.shape[0]+1)
    filter_widths = np.zeros(new_spec_wavs.shape[0])
    filter_lhs[0] = new_spec_wavs[0] - (new_spec_wavs[1] - new_spec_wavs[0])/2
    filter_widths[-1] = (new_spec_wavs[-1] - new_spec_wavs[-2])
    filter_lhs[-1] = new_spec_wavs[-1] + (new_spec_wavs[-1] - new_spec_wavs[-2])/2
    filter_lhs[1:-1] = (new_spec_wavs[1:] + new_spec_wavs[:-1])/2
    filter_widths[:-1] = filter_lhs[1:-1] - filter_lhs[:-2]

    # Check that the range of wavelengths to be resampled_fluxes onto falls within the initial sampling region
    # SMC change, allow new wavelength ranges that are outside the range of the old ones.
    #if filter_lhs[0] < spec_lhs[0] or filter_lhs[-1] > spec_lhs[-1]:
    #    raise ValueError("spectres: The new wavelengths specified must fall within the range of the old wavelength values.")

    #Generate output arrays to be populated
    resampled_fluxes = np.zeros(spec_fluxes[...,0].shape + new_spec_wavs.shape)

    if spec_errs is not None:
        if spec_errs.shape != spec_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape as spec_fluxes.")
        else:
            resampled_fluxes_errs = np.copy(resampled_fluxes)

    start = 0
    stop = 0

    # Calculate the new spectral flux and uncertainty values, loop over the new bins
    for j in range(new_spec_wavs.shape[0]):

        # Find the first old bin which is partially covered by the new bin
        while spec_lhs[start+1] <= filter_lhs[j]:
            start += 1

        # Find the last old bin which is partially covered by the new bin
        while spec_lhs[stop+1] < filter_lhs[j+1]:
            stop += 1

        # if there is not an overlap, then set the value to NaN:
        # (SMC addition):
        if ((start == 0) & (stop == 0)):
            resampled_fluxes[...,j] = np.nan
        # If the new bin falls entirely within one old bin the are the same the new flux and new error are the same as for that bin
        elif stop == start:

            resampled_fluxes[...,j] = spec_fluxes[...,start]
            if spec_errs is not None:
                resampled_fluxes_errs[...,j] = spec_errs[...,start]

        # Otherwise multiply the first and last old bin widths by P_ij, all the ones in between have P_ij = 1 
        else:

            start_factor = (spec_lhs[start+1] - filter_lhs[j])/(spec_lhs[start+1] - spec_lhs[start])
            end_factor = (filter_lhs[j+1] - spec_lhs[stop])/(spec_lhs[stop+1] - spec_lhs[stop])

            spec_widths[start] *= start_factor
            spec_widths[stop] *= end_factor

            # Populate the resampled_fluxes spectrum and uncertainty arrays
            resampled_fluxes[...,j] = np.sum(spec_widths[start:stop+1]*spec_fluxes[...,start:stop+1], axis=-1)/np.sum(spec_widths[start:stop+1])

            if spec_errs is not None:
                resampled_fluxes_errs[...,j] = np.sqrt(np.sum((spec_widths[start:stop+1]*spec_errs[...,start:stop+1])**2, axis=-1))/np.sum(spec_widths[start:stop+1])
            
            # Put back the old bin widths to their initial values for later use
            spec_widths[start] /= start_factor
            spec_widths[stop] /= end_factor


    # If errors were supplied return the resampled_fluxes spectrum and error arrays
    if spec_errs is not None:
        return resampled_fluxes, resampled_fluxes_errs

    # Otherwise just return the resampled_fluxes spectrum array
    else: 
        return resampled_fluxes
