##########################################################################
# functions to process Hector simulation data and do various things to it

import numpy as np
import astropy.io.fits as fits
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from symfit import parameters, variables, Fit, Poly, Model, cos, sin
from symfit.core.objectives import LogLikelihood
from symfit.core.minimizers import NelderMead,BFGS
from symfit.core.minimizers import DifferentialEvolution

from shutil import copyfile
from .sami_2dfdr_read import find_fibre_table

##########################################################################
# simple scaling of simulated data to move it below saturation
#
def scale_hector_sim(infile):

    # copy to output file:
    outfile = infile.replace('.fits','scale.fits')
    copyfile(infile,outfile)

    # open output file and make changes:
    hdulist_out = fits.open(outfile,mode='update',do_not_scale_image_data=True)

    # get image:
    im = hdulist_out[0].data
    print(hdulist_out[0].header['BITPIX'])
    print(hdulist_out[0].header['BSCALE'])
    print(hdulist_out[0].header['BZERO'])
    print(im.dtype.name)
    # get max value:
    maxval = np.nanmax(im)
    print('Max value in image:',maxval)

    # scale image:
    im = np.floor_divide(im,2)

    print(np.nanmax(im))

    hdulist_out.flush()
    
    hdulist_out.close()
    
    return


##########################################################################
# Function to read in csv files of modelled wavelength vs pixel position
# from Hector sims and fit a 2D surface:
#

#def hector_fit_lampix(xcsvfile,ycsvfile,lamcsvfile):
def hector_fit_lampix(csvfile,sims=False,nfibslitlet=45):

    plt.rc('text', usetex=True)
    
    # get data:
    #xarr = np.genfromtxt(xcsvfile, delimiter=',')
    #yarr = np.genfromtxt(ycsvfile, delimiter=',')
    #lamarr = np.genfromtxt(lamcsvfile, delimiter=',')

    # get actual data from single file:
    (xarr,yarr,lamarr,fibno) = np.genfromtxt(csvfile, delimiter=',',unpack=True)

    nval = np.size(xarr)
    
    # get the number of fibres:
    nfib = int(np.max(fibno))

    # get the number of slitets.  need to round properly as sometime end fibres not
    # used (due to being sky):
    nslitlet = int(np.rint(nfib/nfibslitlet))

    print('Number of fibres:',nfib)
    print('Number of fibres per slitlet:',nfibslitlet)
    print('Number of slitlets:',nslitlet)

    # set up array to contain offsets for slitlets:
    dxslitlet = np.zeros(int(nslitlet))
    dxfibre = np.zeros(int(nfib)) 

    # min/max wavelength range in microns:
    #lmin = 3600.0/10000.0
    #lmax = 6000.0/10000.0
    #lamrange=lmax-lmin
    # flaten arrays and convert to pixels.  ote that x,y are natively in mm
    # but pixels are 15microns
    #x = xarr.flatten()/0.015
    #y = yarr.flatten()/0.015
    #xv = xarr.flatten()/30
    #yv = yarr.flatten()/30
    if (sims):
        xv = xarr.flatten()
        yv = yarr.flatten()
        lam = lamarr.flatten()
    else:
        # pixels to mm
        xv = xarr*0.015
        yv = yarr*0.015
        # wavelength to microns:
        lam = lamarr/10000.0

        # scale x,y to be between zero and 1 (approx)
        xv = xv /100
        yv = yv /100

        # for physical model, centre the coordinates:
        xv = (xarr - 4096.0/2.0)*0.015/100
        yv = (yarr - 4112.0/2.0)*0.015/100

        
        # scale wavelength to be between zero and 1:
        #lam = (lam - lmin)/lamrange

        
        
    print('number of elements in array:',np.size(xv))

    # define the model to fit to the surface using symfit:
    x,y,z = variables('x, y, z')
    # first fit, to take out simple plane:
    c1, c2, c3 = parameters('c1, c2, c3')
    model1_dict = {z: Poly({(0,0): c1, (1,0): c2, (0,1): c3},x,y).as_expr()}
    model1 = Model(model1_dict)

    print('\n Simple linear model:')
    print(model1)
    
    # fit the model:
    fit1 = Fit(model1,x=xv,y=yv,z=lam,minimizer=[NelderMead, BFGS])
    fit1_result = fit1.execute()
    zfit1 = model1(x=xv, y=yv, **fit1_result.params).z
    print('linear model results:')
    print(fit1_result)
    
    # plot the fit and residuals for the simple plane (in 3d):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(xv,yv,lam,label='data')
    ax1.scatter(xv,yv,zfit1,label='linear fit')
    ax1.set(xlabel='x,spectral (mm/100)',ylabel='y, spatial (mm/100)',zlabel='Wavelength (microns)')
    ax1.legend(prop={'size':8})
    plt.show()

    # take the difference between the data and the linear model:
    diff1 = (lam-zfit1)


    
    # now loop over the higher order fits, trying to optimize the offsets between slitlets

    for j in range(3):

        # adjust for offsets based on fibre and slitlet:
        for i in range(nval):
            ifib = int(fibno[i]-1)
            islitlet = int(ifib/nfibslitlet+1)
            #print(i,fibno[i],ifib,islitlet)
            delta = dxslitlet[islitlet-1]
            #delta = 1.0e-6
            #print(i,lam[i],lam[i]-delta,delta)
            lam[i] -= delta
            if (i%1000 == 0):
                print(i,xv[i],yv[i],lam[i],fibno[i],delta,ifib,islitlet)


        # default function used in 2dfdr from commissioning is this (as of sept 2024):
        #[z(x, y; c1, c2, c3, c4, c5, c6, c7) = c1 + c2*x + c3*x**2 + c4*y**2 + c5*x**2*y**2 + c6*x**3 + c7*x*y**2]
        #c1, c2, c3, c4, c5, c6, c7 = parameters('c1, c2, c3, c4, c5, c6, c7')
        #model2_dict = {z: Poly({(0,0): c1, (1,0): c2, (2,0): c3, (0,2): c4, (2,2): c5, (3,0): c6, (1,2): c7},x,y).as_expr()}

        # using a different setup to generate the model, without the
        # Poly generator:
        x, y, z = variables('x, y, z')
        # default model:
        #c1, c2, c3, c4, c5, c6, c7, c8, c9, xc, yc = parameters('c1, c2, c3, c4, c5, c6, c7, c8, c9, xc, yc')
        c1, c2, c3, c4, c5, c6, c7, xc, yc = parameters('c1, c2, c3, c4, c5, c6, c7, xc, yc')
        r2 = (x - xc)**2 + (y - yc)**2
        # default model:
        #model2 = {z: c1 + c2 * (x - xc)  + c3 * (y - yc) + c4 * r2 + c5 * r2 * r2 + c6 * r2 * r2 * r2 + c7 * (x-xc)**3 * (y-yc)**2 + c8 * (x-xc)**3 + c9 * r2 * r2 * r2 *r2}
        # highest r**2, r**4, r**8 (no r**6)
        #model2 = {z: c1 + c2 * (x - xc)  + c3 * (y - yc) + c4 * r2 + c5 * r2 * r2 + c6 * r2 * r2 * r2 * r2 + c7 * (x-xc)**3 * (y-yc)**2 + c8 * (x-xc)**3} 
        # highest r**2, r**4
        #model2 = {z: c1 + c2 * (x - xc)  + c3 * (y - yc) + c4 * r2 + c5 * r2 * r2 + c7 * (x-xc)**3 * (y-yc)**2 + c8 * (x-xc)**3} 

        # highest r**2, r**4
        model2 = {z: c1 + c2 * (x - xc)  + c3 * (y - yc) + c4 * r2 + c5 * r2 * r2 + c6 * (x-xc)**3 * (y-yc)**2 + c7 * (x-xc)**3} 
        #model1_dict = {z: Poly({(0,0): c1, (1,0): c2, (0,1): c3},x,y).as_expr()}
        #model1 = Model(model1_dict)
        
        #c1, c2, c3, c4,c5, c6, c7, c8, c9, c10 = parameters('c1, c2, c3, c4, c5, c6, c7, c8, c9, c10')
        #model_dict = {z: Poly({(0,0): c1, (1,0): c2, (0,1): c3, (1,1): c4, (2,0): c5, (0,2): c6, (2,2): c7, (3,0): c8, (0,3): c9, (1,2): c10},x,y).as_expr()}
        #c1, c2, c3, c4 = parameters('c1, c2, c3, c4')
        #model2_dict = {z: Poly({(0,0): c1, (1,0): c2, (0,1): c3, (4,4): c4},x,y).as_expr()}
        #c1, c2, c3, c4, c5, c6, c7, c8 = parameters('c1, c2, c3, c4, c5, c6, c7, c8')
        #model2_dict = {z: Poly({(0,0): c1, (1,0): c2, (2,0): c3, (0,2): c4, (2,2): c5, (3,0): c6, (1,2): c7, (1,4): c8},x,y).as_expr()}

        # Physical model (currently just testing, can't get this to work yet) :
        #x, y, z = variables('x, y, z')
        #apix, alpha, beta, lpmm = parameters('apix, alpha, beta, lpmm')
        #apix, lpmm, alpha, beta = parameters('apix, lpmm, alpha, beta')
        # LPMM is in units of mm (of course), but we will calculate wavelength in microns:
        #model2 = {z: 1.0e3 * cos(y*apix) * (sin(alpha) + sin(beta))/lpmm   + 1.0e3* (cos(beta)*apix/lpmm) * cos(beta + x*apix) /cos(beta)}
        # cut down versions for tests:
        #model2 = {z: 1.0e3 * cos(y*apix) * (sin(alpha) + sin(beta))/lpmm}
        #salpha = cos(alpha)
        #sbeta = sin(beta)
        #model2 = {z: 1.0e3 * cos(y * apix) * (sin(beta)+sin(alpha)) + x}
        #model2 = Model(model2_dict)
        print(model2)

        # fit the complex model to the full data:
        #fit2 = Fit(model2,x=xv,y=yv,z=lam,minimizer=[DifferentialEvolution, BFGS])
        #fit2 = Fit(model2,x=xv,y=yv,z=lam,minimizer=[NelderMead, BFGS],objective=LogLikelihood)
        #fit2 = Fit(model2,x=xv,y=yv,z=lam)
        fit2 = Fit(model2,x=xv, y=yv,z=lam)
        fit2_result = fit2.execute()
        print(fit2_result)
        # evalulate the model at the location of the data.  The ".output[0]" gets
        # just a numpy array of values, as the full command outputs a structure
        # that is more complicated.
        zfit2 = fit2.model(x=xv, y=yv, **fit2_result.params).output[0]
        #zfit2 = model2(x=xv, y=yv, **fit2_result.params).z
        #print(type(zfit2))
        #print(zfit2)

        # subtract the linear fit from the complex model to be able to plot them more clearly:
        zfit2_diff = zfit2 - zfit1
    
        # take the difference between the data and the complex model:
        diff2 = (lam-zfit2)
        
        
        # calculate the mean offset per fibre:
        for i in range(nfib):
            idx = np.where(fibno == i)
            meandiff = np.nanmedian(diff2[idx])
            # get the slit number:
            islitlet = int(i/nfibslitlet+1)
            dxfibre[i] = meandiff
            #print(i+1,meandiff,islitlet)

        # calculate the median offset per slitlet:
        for i in range(nslitlet):
            i1 = i+1+i*(nfibslitlet-1)
            i2 = i1+nfibslitlet-1
            dxslitlet[i] = np.nanmedian(dxfibre[i1-1:i2-1])
            print(i,i1,i2,dxslitlet[i],dxslitlet[i]/10000.0)
            
    
    
        # plot the fit and residuals for the simple plane:
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.scatter(xv,yv,diff1*1.0e4,alpha=0.1,label='data-(linear fit)')
        #ax3.scatter(xv,yv,zfit2_diff*1.0e4,alpha=0.1,label='fit-(linear fit)')
        ax3.azim = -70
        ax3.dist = 10
        ax3.elev = 20
        ax3.set(xlabel='x,spectral (mm/100)',ylabel='y, spatial (mm/100)',zlabel='$\Delta$Wavelength (Angstroms)',title='data and linear fit')
        ax3.legend(prop={'size':8})
        plt.show()
    
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111, projection='3d')
        ax4.scatter(xv,yv,diff2*1.0e4)
        ax4.azim = -80
        ax4.dist = 10
        ax4.elev = 10
        ax4.set(xlabel='x,spectral (mm/100)',ylabel='y, spatial (mm/100)',zlabel='$\Delta$Wavelength (Angstroms)',title='residual for higher order fit')
        ax4.legend(prop={'size':8})
        plt.show()

        # plot a simple 2D image with the differences for simple linear model:
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(1,1,1)
        cax5 = ax5.scatter(xv,yv,c=diff1*1.0e4,marker='.',cmap=cm.rainbow)
        ax5.set(xlabel='x (mm/100)',ylabel='y(mm/100)',title='data-(linear fit) (in Angstroms)')
        cbar5 = fig5.colorbar(cax5,ax=ax5)
        
        # plot a simple 2D image with the differences:
        fig6 = plt.figure()
        ax6 = fig6.add_subplot(1,1,1)
        cax6 = ax6.scatter(xv,yv,c=diff2*1.0e4,marker='.',cmap=cm.rainbow)
        ax6.set(xlabel='x (mm/100)',ylabel='y(mm/100)',title='data-(higher order fit) (in Angstroms)')
        cbar6 = fig6.colorbar(cax6,ax=ax6)
    
        # get median abs diff and max:
        medabs = np.nanmedian(abs(diff2))*1.0e4
        maxabs = np.nanmax(abs(diff2))*1.0e4
        print('median abs diff:',medabs,' Angstoms')
        print('max abs diff:',maxabs,' Angstroms')

    
    
    return


##########################################################################
# Function to take a simulated flat frame and modify it (both fibre table
# and image) so that some of the sky fibres are set to bad.
#

def hector_sim_blocksky(imfile,tlmfile):

    
    # read in original image:
    hdulist = fits.open(imfile)

    # read in tlm file:
    hdulist_tlm = fits.open(tlmfile)
    tlm = hdulist_tlm[0].data

    xs,ys = np.shape(tlm)
    print('tlm size:',xs,ys)
    
    #fibre numbers for skies:
    skyfib = [1,44,45,46,89,90,91,134,135,136,179,180,181,225,226,269,270,271,314,315,316,359,360,361,404,405,406,449,450,451,494,495,496,539,540,541,584,585,586,630,631,675,676,719,720,721,765,766,809,810,811,855]

    nsky = len(skyfib)
    print('Number of sky fibres:',nsky)

    # randomly set 20% of sky fibres to dead:
    ndead = int(nsky/5.0)
    ranskydead = np.random.randint(low=0, high=nsky, size=(ndead))
    print(ranskydead)
    # hard code the first and last fibre to be bad
    ranskydead[0] = 0
    ranskydead[-1] = nsky-1

    # copy to output file:
    outfile = imfile.replace('.fits','block.fits')
    copyfile(imfile,outfile)

    # open output file and make changes:
    hdulist_out = fits.open(outfile,mode='update')

    # get image:
    im = hdulist_out[0].data

    # get percentile value to define background:
    back = np.percentile(im,5.0)
    print('background counts = ',back)
    
    try: 
        fib_tab_hdu=find_fibre_table(hdulist_out)
        table_data = hdulist_out[fib_tab_hdu].data
        types=table_data.field('TYPE')

    except KeyError:
        print('No fibre table found')

    ntype = np.size(types)
        
    # update the fibre types:
    for i in range(ntype):
        for j in range(ndead):
            if (i == (skyfib[ranskydead[j]]-1)):
                print('fibre ',i,' set to D')
                types[i] = 'D'

    # change the flux in the image to account for fibres being turned off:
    for j in range(ndead):
        fib = skyfib[ranskydead[j]]-1
        # get tlm for this fibre:
        tlmfib = tlm[fib,:]
        # go through each col and reset the flux in the fibre to zero:
        for i in range(ys):
            i1 = int(round(tlmfib[i]))-2
            i2 = int(round(tlmfib[i]))+2
            im[i1:i2,i] = back
                
    # close file:
    hdulist_out.close()

            

    return

####################################################################
# 2D poly fitting:

def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

def poly2Dreco(X, Y, c):
    return (c[0] + X*c[1] + Y*c[2] + X**2*c[3] + X**2*Y*c[4] + X**2*Y**2*c[5] + 
           Y**2*c[6] + X*Y**2*c[7] + X*Y*c[8])
