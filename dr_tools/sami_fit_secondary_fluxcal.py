# routines to process the SAMI secondary flux calibration stars, and in particular fit models to
# them to get a better flux calibration model.  This makes use of PPXF to fit the stars.
#
################################################################################################

import pylab as py
import glob
from os import path
from time import perf_counter as clock

import astropy.io.fits as pf
from scipy import ndimage
import numpy as np
import scipy.ndimage as nd

#import ppxf as ppxf_package
#from ppxf.ppxf import ppxf
#import ppxf.ppxf_util as util

#import healpy as hp
# dependency on main SAMI DR package.
#from sami_tools_smc_mspa4681.dr_tools.fluxcal2 import fit_spline
#import sami.dr.dust as dust

# local imports
from .sami_utils import spectres


###########################################################################
# read in Krurucz from SDSS templates list.
#
def read_kurucz(infile,doplot=False,plot_iter=False,verbose=False):

    # open the file:
    hdulist = pf.open(infile)

    # get the model info and mags:
    table_data = hdulist[1].data
    
    # get individual model data cols:
    model_name=table_data.field('MODEL')
    model_feh =table_data.field('FEH')
    model_teff =table_data.field('TEFF')
    model_g =table_data.field('G')
    model_mag =table_data.field('MAG')
    
    # now read in the actual model spectra:
    model_flux = hdulist[0].data
    nmodel,nlam = np.shape(model_flux)
    if (verbose):
        print('number of models:',nmodel)
        print('number of wavelength bins:',nlam)

    # and get the wavelength array;
    primary_header=hdulist['PRIMARY'].header
    crval1=primary_header['CRVAL1']
    cdelt1=primary_header['CD1_1']
    crpix1=primary_header['CRPIX1']
    naxis1=primary_header['NAXIS1']
    x=np.arange(naxis1)+1
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    lam=L0+x*cdelt1

    # if plot requestd, then do it:
    if (doplot):
        fig1 = py.figure()
        ax1 = fig1.add_subplot(211)
        ax2 = fig1.add_subplot(212)
        for i in range(nmodel):
            ax1.cla()
            ax2.cla()
            ax1.set(xlim=[3600,7000])
            ax2.set(xlim=[3600,7000])
            ax1.plot(lam,model_flux[i,:])
            label = '[Fe/H]='+str(model_feh[i])+' Teff='+str(model_teff[i])+' G='+str(model_g[i]) 
            ax1.text(0.5,1.1,label,verticalalignment='center',transform=ax1.transAxes)
            if (i<(nmodel-1)):
                label2 = '[Fe/H]='+str(model_feh[i+1])+' Teff='+str(model_teff[i+1])+' G='+str(model_g[i+1]) 
                ax2.plot(lam,model_flux[i,:]/model_flux[i+1,:])
                ax2.text(0.5,0.9,label+'/'+label2,verticalalignment='center',horizontalalignment='center',transform=ax2.transAxes)

            py.draw()
            if (plot_iter):
                yn=input('continue?')

    return lam, model_flux, model_feh, model_teff, model_g, model_mag

###############################################################################################
# calibrate a particular field using secondary flux cal stars
#
def fluxcal_secondary(field_path,doplot=True,plotall=False):

    # find all the sci.fits files in this path.
    # these need to also have sufficient expsoure time (>10min)

    ccd1_path = field_path+'/main/ccd_1/'

    scifiles = glob.glob(ccd1_path+'*sci.fits')

    nsci = np.size(scifiles)

    tf_b = np.zeros((nsci,2048))
    tf_r = np.zeros((nsci,2048))

    if (doplot):
        fig1 = py.figure()
        ax1 = fig1.add_subplot(111)

    nread = 0
    for scifile in scifiles:
        print(scifile)

        # read headers:
        hdulist = pf.open(scifile)
        primary_header=hdulist['PRIMARY'].header
        exposed = primary_header['EXPOSED']
        hdulist.close()

        # skip is exposure time too short:
        if (exposed < 900):
            print('skipping :',scifile)
            continue
        # now run fitting of secondary:
        lam_bb,tf_bb,lam_rr,tf_rr = ppxf_fit_secondary(scifile,doplot=plotall,returntf=True)
        tf_b[nread,:] = tf_bb
        tf_r[nread,:] = tf_rr
        
        if (nread == 0):
            lam_b = lam_bb
            lam_r = lam_rr

        nread = nread + 1

        if (doplot):
            ax1.plot(lam_bb,tf_bb,color='b')
            ax1.plot(lam_rr,tf_rr,color='r')

        
    print('done')

        
###############################################################################################
# do a ppxf fit of templates to a secondary standard star.
# to run:
#
# sami_dr_smc.sami_fit_secondary_fluxcal.ppxf_fit_secondary('09feb10044sci.fits',tempfile='/suphys/scroom/lib/python/sami_dr_smc/kurucz_stds_raw_v5.fits') 

def ppxf_fit_secondary(infile,doplot=True,verbose=True,tempfile='/Users/scroom/data/sami/fluxcal/kurucz_stds_raw_v5.fits',mdegree=8,returntf=True):

    # ppxf path in case we use templates packaged with this:
    ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))

    # which templates to use:
    use_vaz = False
    use_kur = True
    
    # get the spectrum:
    lam_t,flux_t,sigma_t, bundlera, bundledec = read_flux_calibration_extension(infile)

    # get first and last good point:
    nlam = np.size(lam_t)
    for i in range(nlam):
        if (np.isfinite(flux_t[i])):
            istart = i
            break
    for i in reversed(range(nlam)):
        if (np.isfinite(flux_t[i])):
            iend = i
            break

    # check for other nans:
    nnan = 0
    for i in range(istart,iend):
        if (np.isnan(flux_t[i])):
            print(i,flux_t[i])
            nnan = nnan+1

    if (nnan > 0):
        print('WARNING: extra NaNs found')

    # specify good spectrum:
    lam = lam_t[istart:iend]
    flux = flux_t[istart:iend]
    sigma = sigma_t[istart:iend]
            
    # divide by median flux to resolve possible numerical issues:
    medflux = np.nanmedian(flux)
    flux = flux/medflux
    sigma = sigma/medflux
    
    # log rebin input spectrum.  Returns the natural log
    # of wavelength values and resampled spectrum.
    lamrange = np.array([lam[0],lam[-1]])
    logflux, loglam, velscale = util.log_rebin(lamrange,flux)
    logsigma = util.log_rebin(lamrange,sigma)[0]
    lam_gal = np.exp(loglam)
    if (verbose):
        print('Velocity scale after log rebinning: ',velscale)
    
    # define instrumental resolution:
    fwhm_gal = 0.1

    # read in templates:
    if (use_kur):
        # define amount to rebin/resample the template spectrum:
        nrebin = 10
        zfact = 1.0/float(nrebin)
        if (verbose):
            print('reading templates from: ',tempfile)
        lam_temp1, temp_kur1, model_feh, model_teff, model_g, model_mag = read_kurucz(tempfile,doplot=False,plot_iter=False)

        n_kur, n_lam_kur = np.shape(temp_kur1)
        templates_kur1 = np.zeros((n_kur,int(n_lam_kur/nrebin)))
        # template resolution (just a rough guess), after convolution:
        fwhm_tem = 0.1
        # convolve templates to lower resolution to better match data:
        lam_temp = nd.zoom(lam_temp1,zfact)
        for i in range(n_kur):
            templates_kur1[i,:] = nd.zoom(nd.gaussian_filter1d(temp_kur1[i,:],float(nrebin)),zfact)

        if (verbose):
            print('template wavelength binsize:',lam_temp[1]-lam_temp[0])

        # get range of wavelength:
        lamRange_temp = [np.min(lam_temp),np.max(lam_temp)]
        
        # now log rebinning of templates.  Do it for one to start with, so we know the number
        # of bins to use:
        tmp_tmp = util.log_rebin(lamRange_temp,templates_kur1[0,:],velscale=velscale)[0]
        templates_kur2 = np.zeros((n_kur,np.size(tmp_tmp)))
        # now log rebin all templates:
        for i in range(n_kur):
            templates_kur2[i,:] = util.log_rebin(lamRange_temp,templates_kur1[i,:],velscale=velscale)[0] 

        # transpose templates:
        templates = np.transpose(templates_kur2)

        # correct templates for galactic extinction.  We do this as these templates are models and
        # the stars that we observe should all be halo stars and so the light from them should
        # pass through the dust in the disk.
        
        temp_n=n_kur
    
    if (use_vaz):
        # Read the list of filenames from the Single Stellar Population library
        # by Vazdekis (2010, MNRAS, 404, 1639) http://miles.iac.es/. A subset
        # of the library is included for this example with permission
        vazdekis = glob.glob(ppxf_dir + '/miles_models/Mun1.30Z*.fits')
        fwhm_tem = 2.51 # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the SDSS galaxy spectrum, to determine
        # the size needed for the array which will contain the template spectra.
        hdu = fits.open(vazdekis[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
        lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
        sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
        templates = np.empty((sspNew.size, len(vazdekis)))

        # read all templates:
        for j, fname in enumerate(vazdekis):
            hdu = fits.open(fname)
            ssp = hdu[0].data
            sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
            templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates

        temp_n = len(vazdekis)


        
    c = 299792.458
    dv = np.log(lam_temp[0]/lam_gal[0])*c    # km/s
    if (verbose):
        print('dv in km/s: ',dv)
    
    # starting guess for velocity and dispersion:
    start = [0,50.0]

    #noise = np.ones_like(logflux)*0.1

    # fit one template at a time:
    chisq = np.zeros(temp_n)
    chimin=1.0e10
    ibest = -9
    for i in range(temp_n):
        pp = ppxf(templates[:,i],logflux,logsigma, velscale, start,
              plot=False, moments=2, mdegree=mdegree,quiet=True,
              degree=-1, vsyst=dv, clean=False, lam=lam_gal)
        print(i,pp.chi2,model_teff[i],model_feh[i],model_g[i])
        chisq[i] = pp.chi2
        if (chisq[i] < chimin):
            ibest = i
            chimin = chisq[i]
        
        
    print('best template:',ibest,chimin)
    print(pp.mpoly)

    # get the next best fits:
    top_5_idx = np.argsort(chisq)[0:5]
    print(top_5_idx)
    print(chisq[top_5_idx])
    
    # redo best fit:
    if (doplot):
        fig2 = py.figure(2)
        ax2_1 = fig2.add_subplot(111)

    pp = ppxf(templates[:,ibest],logflux,logsigma, velscale, start,
              plot=True, moments=2, mdegree=mdegree,
              degree=-1, vsyst=dv, clean=False, lam=lam_gal)


    # get best velocity and sigma:
    best_vel, best_sigma = pp.sol
    print('best velocity solution:',best_vel,best_sigma)
    
    if (doplot):
        fig3 = py.figure(3)
        ax3_1 = fig3.add_subplot(111)
    
    # plot best fit template:
    if (doplot):
        xax= np.linspace(0,np.size(templates[:,0]),np.size(templates[:,0]))
        print(np.shape(pp.bestfit),np.shape(xax))
        ax3_1.plot(lam_gal,pp.bestfit,label='Best fit model')
        ax3_1.plot(lam_gal,pp.mpoly,label='multiplicative polynomial')
        ax3_1.plot(lam_gal,pp.bestfit/pp.mpoly,label='model without polynomial')
        ax3_1.legend()

    # go through each of best 5 fits and look at variation in results:
    if (doplot):
        fig4 = py.figure(4)
        ax4 = fig4.add_subplot(111)
    for i in range(5):
        it = top_5_idx[i]
        print(it,chisq[it])
        pp = ppxf(templates[:,it],logflux,logsigma, velscale, start,
              plot=False, moments=2, mdegree=mdegree,
              degree=-1, vsyst=dv, clean=False, lam=lam_gal)
        if (doplot):
            ax4.plot(lam_gal,pp.mpoly,label='mult polynomial, temp:'+str(it))
            ax4.plot(lam_gal,pp.bestfit/pp.mpoly,label='model (no poly), temp:'+str(it))

    if (doplot):
        ax4.legend()

    # plot dist of chisq in terms of FeH and Teff:
    if (doplot):
        fig5 = py.figure(5)
        ax5 = fig5.add_subplot(111)
        cax5 = ax5.scatter(model_feh,model_teff,c=chisq,marker='o',cmap=py.cm.rainbow)
        ax5.plot(model_feh[ibest],model_teff[ibest],'x',color='r',markersize=12)
        ax5.plot(model_feh[top_5_idx],model_teff[top_5_idx],'+',color='g',markersize=12)
        ax5.set(xlabel='FeH',ylabel='Teff')
        cbar5 = fig5.colorbar(cax5, ax=ax5)

        fig6 = py.figure(6)
        ax6_1 = fig6.add_subplot(211)
        ax6_2 = fig6.add_subplot(212)
        # loop over metallicity bins:
        for i in range(5):
            feh_val = -0.5*float(i)
            idxfeh = np.where((model_feh == feh_val) & (model_g == 4.0))
            cax6_1 = ax6_1.scatter(model_teff[idxfeh],chisq[idxfeh],c=model_feh[idxfeh],marker='o',cmap=py.cm.rainbow,vmin=-2.0,vmax=0.0)
            ax6_1.plot(model_teff[idxfeh],chisq[idxfeh],'-',color='k')

        cbar6_1 = fig6.colorbar(cax6_1,ax=ax6_1)
        
        for i in range(5):
            feh_val = -0.5*float(i)
            idxfeh = np.where((model_feh == feh_val) & (model_g == 4.5))
            cax6_2 = ax6_2.scatter(model_teff[idxfeh],chisq[idxfeh],c=model_feh[idxfeh],marker='o',cmap=py.cm.rainbow,vmin=-2.0,vmax=0.0)
            ax6_2.plot(model_teff[idxfeh],chisq[idxfeh],'-',color='k')

        cbar6_2 = fig6.colorbar(cax6_2,ax=ax6_2)

    # re-read the full red and blue fluxcal extension, so that we can do the correction.
    lam_b,flux_b,sigma_b, bundlera, bundledec = read_flux_calibration_extension(infile)
    infile_r = infile.replace('ccd_1','ccd_2')
    infile_r = infile_r[0:-13]+'2'+infile_r[-12:]
    lam_r,flux_r,sigma_r, bundlera, bundledec, tel = read_flux_calibration_extension(infile_r,gettel=True)

    # correct red arm for telluric:
    flux_r = flux_r * tel

    # shift best fit template to the right velocity and sigma:
    lam_temp_best = lam_temp*(1+best_vel/2.98e5)
    
    # convolve:
    #temp_best_conv = nd.gaussian_filter1d(templates_kur1[ibest,:],best_sigma)
    temp_best_conv = templates_kur1[ibest,:]

    # correct for galactic extinction:
    theta, phi = dust.healpixAngularCoords(bundlera, bundledec )
    print(theta,phi)
    print(dust.MAPS_FILES)
    print(dust.HEALPY_AVAILABLE)
    for name, map_info in dust.MAPS_FILES.items():
        print(name)
        ebv = dust.EBV(name, theta, phi)
        print(name,ebv)
        if name == 'planck':
            correction_t = dust.MilkyWayDustCorrection(lam_temp_best, ebv)
            correction_b = dust.MilkyWayDustCorrection(lam_b, ebv)
            correction_r = dust.MilkyWayDustCorrection(lam_r, ebv)

    temp_best_conv = temp_best_conv/correction_t
            
    # resample onto SAMI wavelength range:
    temp_blue = spectres(lam_b,lam_temp_best,temp_best_conv)
    temp_red = spectres(lam_r,lam_temp_best,temp_best_conv)

    # get relative scaling (normalized to blue):
    medscales = np.nanmedian(flux_b)
    medscalet = np.nanmedian(temp_blue)

    # generate a spline fit to the ratio.  Best to use the same code as the sami manager:
    ratio_sp_b = fit_spline(lam_b,(flux_b/medscales)/(temp_blue/medscalet))
    ratio_sp_r = fit_spline(lam_r,(flux_r/medscales)/(temp_red/medscalet))
    
    # do some plotting:
    if (doplot):
        fig7 = py.figure(7)
        ax7_1 = fig7.add_subplot(311)

        ax7_1.plot(lam_b,flux_b/medscales,'b')
        ax7_1.plot(lam_r,flux_r/medscales,'r')

        
        ax7_1.plot(lam_temp_best,temp_best_conv/medscalet,'k')
        ax7_1.plot(lam_temp_best,(temp_best_conv/medscalet)*correction_t,'k',alpha=0.5)
        ax7_1.plot(lam_b,temp_blue/medscalet,'c')
        ax7_1.plot(lam_r,temp_red/medscalet,'m')
        xmin = np.min(lam_b)-100.0
        xmax = np.max(lam_r)+100.0
        ax7_1.set(xlim=[xmin,xmax],xlabel='Wavelength (\AA)',ylabel='Relative flux')

        ax7_2 = fig7.add_subplot(312)
        ax7_2.axhline(1.0,color='k')
        ax7_2.plot(lam_b,(flux_b/medscales)/(temp_blue/medscalet),'b')
        ax7_2.plot(lam_r,(flux_r/medscales)/(temp_red/medscalet),'r')
        ax7_2.set(xlim=[xmin,xmax],xlabel='Wavelength (\AA)',ylabel='SAMI/template ratio')

        ax7_2.plot(lam_b,ratio_sp_b,'c')
        ax7_2.plot(lam_r,ratio_sp_r,'m')
        
        ax7_3 = fig7.add_subplot(313)
        ax7_3.axhline(1.0,color='k')
        ax7_3.plot(lam_b,ratio_sp_b,'c')
        ax7_3.plot(lam_r,ratio_sp_r,'m')
        
    # take the best fit template, correct it for dispersion and sampling, and then compare it to
    # the red and blue arms:
    
    # plot templates?
    #if (doplot):
    #    cmap = py.get_cmap(py.cm.rainbow)
    #    cols = [cmap(i) for i in np.linspace(0, 1, temp_n)]
    #    fig5 = py.figure(5)
    #    ax5_1 = fig5.add_subplot(111)
    #    for i in range(temp_n):
    #        ax5_1.plot(xax,templates[:,i],color=cols[i])

    # now we have the best template, calculate the transfer function.
    # first take the original model spectrum and apply the galactic dust obscuration
    # so that it matches what we expect from the observed stars.  The stars should all
    # be halo stars, so treating then as iff they are behind the full dust screen of
    # the MW should be reasonable.

    if (returntf):
        return lam_b,ratio_sp_b,lam_r,ratio_sp_r
    else:
        return


    
        
###############################################################################################
# read secondary flux standard data:

def read_flux_calibration_extension(infile,gettf=False,gettel=False):

    # open the file:
    hdulist = pf.open(infile)
    primary_header=hdulist['PRIMARY'].header

    # read WCS:
    crval1=primary_header['CRVAL1']
    cdelt1=primary_header['CDELT1']
    crpix1=primary_header['CRPIX1']
    naxis1=primary_header['NAXIS1']
    
    # define wavelength array:
    x=np.arange(naxis1)+1
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    lam=L0+x*cdelt1

    # get data from FLUX_CALIBRATION extension:
    fdata = hdulist['FLUX_CALIBRATION'].data
    fc_header=hdulist['FLUX_CALIBRATION'].header
    probenum = fc_header['PROBENUM']
    stdname = fc_header['STDNAME']

    # get the coordinates of the star from the fibre table:
    fibtab_data = hdulist['FIBRES_IFU'].data
    
    print('probe number:',probenum,np.size(fibtab_data))
    for i in range(np.size(fibtab_data)):
        if (fibtab_data['GROUP_N'][i] == probenum):
            bundlera = fibtab_data['GRP_MRA'][i]
            bundledec = fibtab_data['GRP_MDEC'][i]
            print('found:',bundlera,bundledec)
            break
        
    if (gettel):
        tel = fdata[5,:]
        
    flux = fdata[0, :]
    sigma = fdata[2, :]
    if (gettf):
        tf = fdata[4,:]
        return lam,flux,sigma, bundlera, bundledec, tf
    elif (gettel):
        return lam,flux,sigma, bundlera, bundledec, tel        
    else:
        return lam,flux,sigma, bundlera, bundledec
    
