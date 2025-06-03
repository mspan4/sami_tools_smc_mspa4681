
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage as nd

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

    # subtract median from data:
    for i in range(ys):
        diff[i,:] = abs(data[i,:] - med)

    # take the median of the difference:
    mad = np.nanmedian(diff,axis=0)

    return mad

################################################################################
# get rms of array, including sigma clipping:

def clipped_mean_rms(a, nsig,verbose=False,niter_max=10):

    # get size of array:
    n = np.size(a)
    
    # get the mean:
    mean = np.nansum(a)/n
    
    # get rms:
    rms = np.sqrt(np.nansum(((a-mean)**2)/n))

    if (verbose):
        print(mean,rms,n)
        
    nprev = n
    for i in range(niter_max):

        a_clipped = a[np.where(abs(a-mean) < nsig*rms)]
        n_clipped = np.size(a_clipped)

        mean = np.nansum(a_clipped)/n_clipped
        rms = np.sqrt(np.nansum(((a_clipped-mean)**2)/n_clipped))
        if (verbose):
            print(i,mean,rms,n_clipped,nprev)
            
        if (n_clipped == nprev):
            break
        else:
            nprev = n_clipped
        
    return mean,rms,n_clipped

    


###########################################################################
# get rms of array, including sigma clipping:

def clipped_rms(a, nsig):

    # get size of array:
    n = np.size(a)
    # get rms:
    rms = np.sqrt(np.nansum((a**2)/n))

    for i in range(10):

        a_clipped = a[np.where(abs(a) < nsig*rms)]
        n_clipped = np.size(a_clipped)
        rms = np.sqrt(np.nansum((a_clipped**2)/n_clipped))

    return rms,n_clipped

###############################################################################
# 1, single Gaussian
def gaussian(xx, *p):
    
    gpeak, gcent, gsigma = p
    # full normalized version so that variabel gpeak is the total flux:
    #    yy = gpeak / ( math.sqrt(2. * math.pi * abs(gsigma)) )  * np.exp(-0.5*((xx - gcent) / gsigma)**2 )
    # quick un-normalized version to minimize calculations:
    yy = gpeak  * np.exp(-0.5*((xx - gcent) / gsigma)**2 )
    return yy

#############################################################
# routine to do median filter, that handles Nans
#
def gaussian_filter_nan(im,filt):

    V = im.copy()
    V[im!=im]=0
    VV = filters.gaussian_filter(V,(filt))

    W = 0*im.copy()+1
    W[im!=im] = 0
    WW = filters.gaussian_filter(W,(filt))

    im_med = VV/WW

    return im_med

#############################################################
# routine to do median filter, that handles Nans
#
def median_filter_nan(im,filt):

    V = im.copy()
    V[im!=im]=0
    VV = filters.median_filter(V,size=filt)

    W = 0*im.copy()+1
    W[im!=im] = 0
    WW = filters.median_filter(W,size=filt)

    im_med = VV/WW

    return im_med



###############################################################################
    
def polyfitr(x, y, N, s, fev=100, w=None, diag=False, clip='both', \
                 verbose=False, plotfit=False, plotall=False, eps=1e-13, catchLinAlgError=False):
    """Matplotlib's polyfit with weights and sigma-clipping rejection.

    :DESCRIPTION:
      Do a best fit polynomial of order N of y to x.  Points whose fit
      residuals exeed s standard deviations are rejected and the fit is
      recalculated.  Return value is a vector of polynomial
      coefficients [pk ... p1 p0].

    :OPTIONS:
        w:   a set of weights for the data; uses CARSMath's weighted polynomial 
             fitting routine instead of numpy's standard polyfit.

        fev:  number of function evaluations to call before stopping

        'diag'nostic flag:  Return the tuple (p, chisq, n_iter)

        clip: 'both' -- remove outliers +/- 's' sigma from fit
              'above' -- remove outliers 's' sigma above fit
              'below' -- remove outliers 's' sigma below fit

        catchLinAlgError : bool
          If True, don't bomb on LinAlgError; instead, return [0, 0, ... 0].

    :REQUIREMENTS:
       :doc:`CARSMath`

    :NOTES:
       Iterates so long as n_newrejections>0 AND n_iter<fev. 


     """
    # 2008-10-01 13:01 IJC: Created & completed
    # 2009-10-01 10:23 IJC: 1 year later! Moved "import" statements within func.
    # 2009-10-22 14:01 IJC: Added 'clip' options for continuum fitting
    # 2009-12-08 15:35 IJC: Automatically clip all non-finite points
    # 2010-10-29 09:09 IJC: Moved pylab imports inside this function
    # 2012-08-20 16:47 IJMC: Major change: now only reject one point per iteration!
    # 2012-08-27 10:44 IJMC: Verbose < 0 now resets to 0
    # 2013-05-21 23:15 IJMC: Added catchLinAlgError

    from CARSMath import polyfitw
    from numpy import polyfit, polyval, isfinite, ones, array, std
    from numpy.linalg import LinAlgError
    from pylab import plot, legend, title

    if verbose < 0:
        verbose = 0

    xx = array(x, copy=False)
    yy = array(y, copy=False)
    noweights = (w==None)
    if noweights:
        ww = ones(xx.shape, float)
    else:
        ww = array(w, copy=False)

    ii = 0
    nrej = 1

    if noweights:
        goodind = isfinite(xx)*isfinite(yy)
    else:
        goodind = isfinite(xx)*isfinite(yy)*isfinite(ww)
    
    xx2 = xx[goodind]
    yy2 = yy[goodind]
    ww2 = ww[goodind]

    while (ii<fev and (nrej != 0)):
        if noweights:
            p = polyfit(xx2,yy2,N)
            residual = yy2 - polyval(p,xx2)
            stdResidual = std(residual)
            clipmetric = s * stdResidual
        else:
            if catchLinAlgError:
                try:
                    p = polyfitw(xx2,yy2, ww2, N)
                except LinAlgError:
                    p = np.zeros(N+1, dtype=float)
            else:
                p = polyfitw(xx2,yy2, ww2, N)

            p = p[::-1]  # polyfitw uses reverse coefficient ordering
            residual = (yy2 - polyval(p,xx2)) * np.sqrt(ww2)
            clipmetric = s

        if clip=='both':
            worstOffender = abs(residual).max()
            if worstOffender <= clipmetric or worstOffender < eps:
                ind = ones(residual.shape, dtype=bool)
            else:
                ind = abs(residual) <= worstOffender
        elif clip=='above':
            worstOffender = residual.max()
            if worstOffender <= clipmetric:
                ind = ones(residual.shape, dtype=bool)
            else:
                ind = residual < worstOffender
        elif clip=='below':
            worstOffender = residual.min()
            if worstOffender >= -clipmetric:
                ind = ones(residual.shape, dtype=bool)
            else:
                ind = residual > worstOffender
        else:
            ind = ones(residual.shape, dtype=bool)
    
        xx2 = xx2[ind]
        yy2 = yy2[ind]
        if (not noweights):
            ww2 = ww2[ind]
        ii = ii + 1
        nrej = len(residual) - len(xx2)
        if plotall:
            plot(x,y, '.', xx2,yy2, 'x', x, polyval(p, x), '--')
            legend(['data', 'fit data', 'fit'])
            title('Iter. #' + str(ii) + ' -- Close all windows to continue....')

        if verbose:
            print(str(len(x)-len(xx2)) + ' points rejected on iteration #' + str(ii))

    if (plotfit or plotall):
        plot(x,y, '.', xx2,yy2, 'x', x, polyval(p, x), '--')
        legend(['data', 'fit data', 'fit'])
        title('Close window to continue....')

    if diag:
        chisq = ( (residual)**2 / yy2 ).sum()
        p = (p, chisq, ii)

    return p

###############################################################################
# 2, single Gaussian with constant continuum level
def gaussian_cont(xx, *p):
    
    gpeak, gcent, gsigma, cont = p
    yy = gaussian(xx,gpeak,gcent,gsigma)

    #    if (docont):
    yy = yy + cont

    return yy

###############################################################################
# single lorentzian
def lorentzian(xx, *p):
    
    lpeak, lcent, lgamma = p
    # properly normalized version:
    #yy = (lpeak / math.pi /lgamma) / (1+((xx-lcent)/lgamma)**2)
    # un-normalized version:
    yy = lpeak / (1+((xx-lcent)/lgamma)**2)
    return yy

###############################################################################
# single lorentzian witha constant background:
def lorentzian_cont(xx, *p):
    
    lpeak, lcent, lgamma, cont = p
    # properly normalized version:
    #yy = (lpeak / math.pi /lgamma) / (1+((xx-lcent)/lgamma)**2)
    # un-normalized version:
    yy = cont + lpeak / (1+((xx-lcent)/lgamma)**2)
    return yy

#############################################################################
# generate a 2d Gaussian.  This does not have any rotation
# 
def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

###########################################################################
# get block mean of an array:
#
def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    print(sx,sy)
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = nd.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    print(np.shape(res))
    res.shape = (int(sx/fact), int(sy/fact))
    return res
