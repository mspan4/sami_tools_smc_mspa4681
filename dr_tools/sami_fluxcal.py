# routines for testing spectrophotometry

import pylab as py
import numpy as np
import scipy as sp
import scipy.stats
import scipy.optimize as optimize
import scipy.ndimage as nd
import scipy.ndimage.filters as filters
from scipy.special import legendre
import glob
import astropy.io.fits as pf
import sami_tools_smc_mspa4681.dr_tools.sami_stats_utils as sami_stats_utils
# import sami.dr.fluxcal2 as fluxcal2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import shutil
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd

#import wget


import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from os.path import basename,join,dirname,exists
from os import mkdir
from astroquery.sdss import SDSS
from astropy import wcs

from sami_tools_smc_mspa4681.dr_tools.sami_utils import spectres
from sami_tools_smc_mspa4681.dr_tools.sami_stats_utils import polyfitr, median_filter_nan
from sami_tools_smc_mspa4681.dr_tools.sami_fit_secondary_fluxcal import read_flux_calibration_extension

#from sami.manager import read_stellar_mags
#import sami.dr.fluxcal2 as fluxcal2


# Circular patch.
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

# SAMI package dust correction:
#from sami.dr.dust import  MilkyWayDustCorrection

###############################################################################
# function to correct Kurucz model spectra based on the SDSS empirical
# correction (see kurucz_restore.pro in idlspec2d from SDSS).
#
# usage:
# sami_tools_smc.dr_tools.sami_fluxcal.correct_kurucz('/Users/scroom/code/sdss_spec/idlspec2d/etc/kurucz_stds_v5.fit')
#
def correct_kurucz(infile,infile2,doplot=True):
    
    #open the file and read in data:
    hdulist = pf.open(infile)
    primary_header=hdulist['PRIMARY'].header
    
    # get primary data and header:
    crval = primary_header['CRVAL1']    
    kur_spec = hdulist[0].data
    (nmod,npix) = np.shape(kur_spec)

    # setup wavelength:
    kwave = 10.0**(np.arange(npix) * 1.0e-4 + crval)
    
    # read in legendre poly HDU:
    tab = hdulist[2].data

    poly_pars = tab[0]
    xmin = poly_pars[1]
    xmax = poly_pars[2]
    coeff = poly_pars[3]
    ncoeff = np.size(coeff)

    # set range:
    xmid = 0.5 * (xmin + xmax)
    xrang = xmax - xmin
    
    # define input x.  Note, using same naming as
    # SDSS idl code to reduce confusion...
    xnatural = kwave
    xvec = 2.0 * (xnatural-xmid)/xrang
    yvec = np.zeros(npix)
    #     if (tset.func EQ 'poly') then legarr = fpoly(xvec, ncoeff)
    #     if (tset.func EQ 'legendre') then legarr = flegendre(xvec, ncoeff)
    #     if (tset.func EQ 'chebyshev') then legarr = fchebyshev(xvec, ncoeff)
    #     ypos[*,iTrace] = legarr # [tset.coeff[*,iTrace]]

    if (doplot):
        fig1 = py.figure(1)
        ax1 = fig1.add_subplot(2,1,1)
        ax2 = fig1.add_subplot(2,1,2)
    
    for i in range(ncoeff):

        leg = legendre(i)
        yl = leg(xvec)
        yvec = yvec + yl*coeff[i]
        ax1.plot(xvec,yl)

    ax1.set(ylim=[-2.0,2.0])
        
    ax2.plot(kwave,yvec)
    ax2.set(xlim=[3600,7500],xlabel='Wavelength (Angstroms)',ylabel='correction')
        
    # now read second file and check if spectra are the same
    hdulist2 = pf.open(infile2)
    primary_header2=hdulist2['PRIMARY'].header
    
    # get primary data and header:
    crval2 = primary_header2['CRVAL1']
    cdelt2 = primary_header2['CD1_1']
    crpix2 = primary_header2['CRPIX1']
    kur_spec2 = hdulist2[0].data
    (nmod2,npix2) = np.shape(kur_spec2)
    x=np.arange(npix2)+1
    L0=crval2-crpix2*cdelt2 #Lc-pix*dL        
    lam=L0+x*cdelt2

    # smooth original spectrum to approximately match the SDSS spectrum:
    nrebin = 20
    zfact = 1.0/float(nrebin)
    kur_filt = zoom(gaussian_filter1d(kur_spec2[0,:],float(nrebin)),zfact)
    lam_filt = zoom(lam,zfact)
    # plot spectra and compare:

    fig2 = py.figure(2)
    ax21 = fig2.add_subplot(1,1,1)
    ax21.plot(lam_filt,kur_filt)
    ax21.plot(kwave,kur_spec[0,:])
    
    

###############################################################################
# test the difference in dust correction between two different E(B-V)
# distributions read from fits binary table
#
def ebv_diff_list(infile,plot=True):

    # open file:
    hdulist = pf.open(infile)
    tab = hdulist[1].data
    ebv_planck = tab['EBVPLNCK']
    ebv_sfd98 = tab['EBVSFD98']

    med_sfd98 = np.nanmedian(ebv_sfd98)
    med_planck = np.nanmedian(ebv_planck)
    print('median SFD98 E(B-V):',np.nanmedian(ebv_sfd98))
    print('median Planck E(B-V):',np.nanmedian(ebv_planck))
    
    # set up wavelength vector:

    lam = np.linspace(3600.0,7600.0,1000)

    correction1 = MilkyWayDustCorrection(lam,med_sfd98)
    correction2 = MilkyWayDustCorrection(lam,med_planck)

    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(2,1,1)
    ax1.plot(lam,correction1,label='E(B-V) 1')
    ax1.plot(lam,correction2,label='E(B-V) 2')
    ax1.legend()
    ax2 = fig1.add_subplot(2,1,2)
    ax2.plot(lam,correction1/correction2,label='1/2')
    ax2.legend()

    
    

###############################################################################
# test the difference in dust correction between two different E(B-V) values
#
def ebv_diff(ebv1,ebv2,plot=True):

    # set up wavelength vector:

    lam = np.linspace(3600.0,7600.0,1000)

    correction1 = MilkyWayDustCorrection(lam,ebv1)
    correction2 = MilkyWayDustCorrection(lam,ebv2)

    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(2,1,1)
    ax1.plot(lam,correction1,label='E(B-V) 1')
    ax1.plot(lam,correction2,label='E(B-V) 2')
    ax1.legend()
    ax2 = fig1.add_subplot(2,1,2)
    ax2.plot(lam,correction1/correction2,label='1/2')
    ax2.legend()

##############################################################################
# get mean E(V-V) over all cubes.

def get_ebv_allcubes(filelist):

    cubelist = glob.glob(filelist)

    nmax = len(cubelist)
    ebv_plnck = np.zeros(nmax)
    ebv_sfd98 = np.zeros(nmax)

    i = 0

    for cube in cubelist:

        print('Getting dust info from ',cube)
        ebv_plnck[i] = pf.getval(cube,'EBVPLNCK',extname='DUST')
        ebv_sfd98[i] = pf.getval(cube,'EBVSFD98',extname='DUST')
        i = i + 1

    col1 = pf.Column(name='Cube', format='256A', array=cubelist[0:i])
    col2 = pf.Column(name='EBVPLNCK', format='D', array=ebv_plnck[0:i])
    col3 = pf.Column(name='EBVSFD98', format='D', array=ebv_sfd98[0:i])
    
    cols = pf.ColDefs([col1,col2,col3])
        #Now, create a new binary table HDU object:
    tbhdu = pf.BinTableHDU.from_columns(cols)

    # finally write the table HDU to a file:
    outfile = 'all_ebv.fits'
    print('Writing results to ',outfile)
    tbhdu.writeto(outfile,overwrite=True)

    print('median SFD98 E(B-V):',np.nanmedian(ebv_sfd98[0:i]))
    print('median Planck E(B-V):',np.nanmedian(ebv_plnck[0:i]))
    

###############################################################################
# script to look at distribution of delta mags for secondary standard stars
# comparing templates to actual values.  Runs on list of TRANSFER2combined.fits
# files
#
# usage:
# sami_dr_smc.sami_fluxcal.compare_delta_mag('2016_02_08-2016_02_14_test2/reduced/*/*/*/main/ccd_1/TRANSFER2combined.fits')
#

def compare_delta_mag(inlist):

    py.rc('text', usetex=True)
    py.rc('font', family='sans-serif')
    py.rcParams.update({'font.size': 12})
    py.rcParams.update({'lines.linewidth': 1})    
 

    infiles = glob.glob(inlist)

    nmax = np.size(infiles)

    dmag = np.zeros((nmax,5))
    dui =  np.zeros(nmax)
    star = np.zeros(nmax,dtype='U256')
    
    nf = 0
    for infile in infiles:

        hdulist=pf.open(infile)
        hdr = hdulist['TEMPLATE_OPT'].header

        dmag[nf,0] = hdr['CATMAGU'] - hdr['TEMPMAGU']
        dmag[nf,1] = hdr['CATMAGG'] - hdr['TEMPMAGG']
        dmag[nf,2] = hdr['CATMAGR'] - hdr['TEMPMAGR']
        dmag[nf,3] = hdr['CATMAGI'] - hdr['TEMPMAGI']
        dmag[nf,4] = hdr['CATMAGZ'] - hdr['TEMPMAGZ']
        star[nf] = hdr['STDNAME']

        dui[nf] =  (hdr['CATMAGU'] - hdr['CATMAGI']) - (hdr['TEMPMAGU'] - hdr['TEMPMAGI'])
        
        nf = nf + 1

    fig1 = py.figure()
    
    fig1.subplots_adjust(hspace=0,wspace=0)

    axlist = []
    
    rmin=-0.2
    rmax=0.2
    nbin=20

    # list stars that have large offsets
    for i in range(nf):
        maxdelta = np.max(np.abs(dmag[i,:]))
        if (maxdelta > 0.06):
            print(star[i],dmag[i,:])
    
    bands = ['u','g','r','i','z']
    # loop over axes:
    print('band mag_mean  mag_sigma  flux_mean')
    for i in range(5):
        axlist.append(fig1.add_subplot(2,3,i+1))
        axlist[i].hist(dmag[0:nf,i],bins=nbin,range=(rmin,rmax),histtype='step',color='k')
        mean = np.nanmean(dmag[0:nf,i])
        std = np.nanstd(dmag[0:nf,i])
        axlist[i].axvline(mean,linestyle=':',color='k')
        lab1 = '$\langle \Delta m \\rangle $='+'{0:6.3f}'.format(mean)
        lab2 = '$\sigma_{\Delta m}$ ='+'{0:6.3f}'.format(std)
        axlist[i].text(0.05, 0.95,bands[i]+'-band', horizontalalignment='left',verticalalignment='center',transform=axlist[i].transAxes)
        axlist[i].text(0.05, 0.85,lab1, horizontalalignment='left',verticalalignment='center',transform=axlist[i].transAxes)
        axlist[i].text(0.05, 0.75,lab2, horizontalalignment='left',verticalalignment='center',transform=axlist[i].transAxes)
        print(bands[i],mean,std,10.0**(-0.4*mean))
        if (i<2):
            axlist[i].xaxis.set_major_formatter(py.NullFormatter())
        if ((i != 0) and (i != 3)):
            axlist[i].yaxis.set_major_formatter(py.NullFormatter())
        if (i == 3):
            py.setp(axlist[i].get_yticklabels()[-1], visible=False)


    # finally plot u-i colour diff:
    axlist.append(fig1.add_subplot(2,3,6))
    axlist[5].hist(dui[0:nf],bins=nbin,range=(rmin,rmax),histtype='step',color='k')
    mean = np.nanmean(dui[0:nf,])
    std = np.nanstd(dui[0:nf])
    axlist[5].axvline(mean,linestyle=':',color='k')
    lab1 = '$\langle \Delta m \\rangle$ ='+'{0:6.3f}'.format(mean)
    lab2 = '$\sigma_{\Delta m}$ ='+'{0:6.3f}'.format(std)
    axlist[5].text(0.05, 0.95,'u-i colour', horizontalalignment='left',verticalalignment='center',transform=axlist[5].transAxes)
    axlist[5].text(0.05, 0.85,lab1, horizontalalignment='left',verticalalignment='center',transform=axlist[5].transAxes)
    axlist[5].text(0.05, 0.75,lab2, horizontalalignment='left',verticalalignment='center',transform=axlist[5].transAxes)
    axlist[5].yaxis.set_major_formatter(py.NullFormatter())
    print('u-i',mean,std)
    
    ax = fig1.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis that is just for labels
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set(xlabel='difference (mag)', ylabel='Number')


###############################################################################
# script to use astroquery to get SDSS spectra
#
# istart is the index in the list file to start on, if you know this and
# you know wmych files have been got already.
def get_sdss_spec(listfile,istart=0):

    #read list file:
    catid,ra,dec = np.loadtxt(listfile,usecols=(0,2,3),unpack=True)

    # get the current files that are present in the directory:
    curr_files = glob.glob('*.fits')
    
    # get object names from current file list:
    curr_catid = []
    for cfile in curr_files:
        tmp = cfile.split('_')
        curr_catid.append(tmp[0])

    ncoord = np.size(ra)

    print('Number of coords read:',ncoord)
    
    # loop through list:
    #nstart = 10
    for i in range(istart,ncoord):

        # check if a file aready exists:
        if str(int(catid[i])) in curr_catid:
            print('spectrum already found, skipping...')
            continue
        
        print('getting SDSS spectrum for ',str(int(catid[i])),', ',i,' of ',ncoord)

        # define an astropy coordinate object:
        coord = SkyCoord(ra[i],dec[i], unit="deg")

        # query the region
        #xid = SDSS.query_region(coord, spectro=True)
        #print(xid)

        # query for spectrum:
        try:
            spec = SDSS.get_spectra(coord)
        #except (ConnectionError) as err:
        except:
            print('ERROR getting galaxy ',int(catid[i]))
            continue
        
        print('shape of spec returned:',np.shape(spec))
        
        # construct the file name if something if found:
        if (spec != None):

            # loop through recovered spectra:
            ns = 0
            for spec1 in spec:
                hdr = spec1[0].header
                mjd = hdr['MJD']
                plateid = hdr['PLATEID']
                fiberid = hdr['FIBERID']

                ns = ns +1

                fname = str(int(catid[i]))+'_'+str(ns)+'_spec-'+str(plateid)+'-'+str(mjd)+'-'+str(fiberid)+'.fits'
                print(fname)
                
                spec1.writeto(fname,overwrite=True)
        else:
            print('No spectrum found for ',int(catid[i]))

        
###############################################################################
# function to go to each SAMI cube (blue only) and pull out ra/dec and
# object name.  This is used to get other data (e.g. SDSS spectra).
#
def get_cube_radec(inlist):

    # glob list:
    infiles = glob.glob(inlist)

    nlist = np.size(infiles)
    print('number of files in list:',nlist)

    # open output file:
    out = open('cube_radec.txt',mode='w')

    # go through all cubes:
    nf = 0
    for cubefile in infiles:

        cfile = basename(cubefile)
        hdr = pf.getheader(cubefile, 0)
        ra = hdr['CATARA']
        dec = hdr['CATADEC']
        catid = hdr['NAME']

        outstr = '{0:s} {1:s} {2:10.6f} {3:10.6f}\n'.format(catid,cfile,ra,dec)
        out.write(outstr)
        print(cfile,nlist,nf)
        nf = nf + 1 

        

################################################################################
# function to check repeat SDSS spectra
#
def check_rep_sdss_spec(inlist,plotall=True):

    # glob files:
    infiles = glob.glob(inlist)

    # set up arrays:
    nfiles = np.size(infiles)

    matchfiles = np.zeros(20,dtype='U256')
    
    # loop through files:
    nf = 0
    nmtot = 0
    for i in range(nfiles):

        infile = infiles[i]
        
        fname = basename(infile)
        tmpstr = fname.split('_')
        catid = tmpstr[0]


        nm = 0
        # now loop through the list again and find any matches
        for j in range(i+1,nfiles):

            infile2 = infiles[j]
            
            # skip unless this is our favourite bad field:
            #if '285-51663' not in infile2:
            #    continue
            # ignore identical files (i.e. self match):
            if (infile == infile2):
                continue
    
            fname2 = basename(infile2)
            tmpstr2 = fname2.split('_')
            catid2 = tmpstr2[0]

            if (catid == catid2):
                matchfiles[nm] = infile2
                nm =  nm + 1
                nmtot = nmtot + 1
                print(infile)
                print(infile2,nm,nmtot)

        # skip if no match:
        if (nm == 0):
            continue

        # read the original file:
        hdulist = pf.open(infile)
        print('SDSS spectrum matched: ',infile)
        sdss_spec_table = hdulist['COADD'].data
        sdss_flux = sdss_spec_table['flux']
        sdss_loglam = sdss_spec_table['loglam']
        sdss_lam = 10.0**sdss_loglam
        hdulist.close()

        lidx = np.where((sdss_lam>3850) & (sdss_lam<9000.0))
        
        
        if (plotall):
            py.figure(1)
            py.clf()
            py.subplot(3,1,1)
            py.plot(sdss_lam[lidx],sdss_flux[lidx],color='k')

        print(sdss_lam)
            
        # loop through matches:
        for im in range(nm):
            print('match file:',matchfiles[im])
            hdulist = pf.open(matchfiles[im])
            sdss_spec_table = hdulist['COADD'].data
            sdss_flux2t = sdss_spec_table['flux']
            sdss_loglam2 = sdss_spec_table['loglam']
            sdss_lam2 = 10.0**sdss_loglam2
            hdulist.close()
            sdss_flux2 = spectres(sdss_lam[lidx],sdss_lam2,sdss_flux2t)
            #print(sdss_lam2)
        
            if (plotall):
                py.subplot(3,1,1)
                py.plot(sdss_lam[lidx],sdss_flux2,color='b')
                py.title(infile)

                # plot ratios:
                py.subplot(3,1,2)
                py.plot(sdss_lam[lidx],sdss_flux2/sdss_flux[lidx],color='b')
                py.axhline(1.0,color='k',linestyle=':')
                py.ylim(ymin=0.5,ymax=1.5)

                # plot scaled ratios (by median):                
                py.subplot(3,1,3)
                scale = np.nanmedian(sdss_flux2/sdss_flux[lidx])
                py.plot(sdss_lam[lidx],(sdss_flux2/sdss_flux[lidx])/scale,color='b')
                py.axhline(1.0,color='k',linestyle=':')
                py.ylim(ymin=0.5,ymax=1.5)
                
        if (plotall): 
            py.draw()
        # pause for input if plotting all the spectra:
            yn = input('Continue? (y/n):')

    print('total number of matches: ',nmtot)
            
################################################################################
# function to check repeat aperture spectra for flux calibration.
#
#
# usage:
# sami_dr_smc.sami_fluxcal.check_rep_spec('dr3_aperture_spec_v1/all_apspec_byrun/20??_??_??-20??_??_??/*blue*apspec.fits')
#

def check_rep_spec(inlist,apname='3_ARCSECOND',plotall=True):

    # glob files:
    infiles = glob.glob(inlist)

    # set up arrays:
    nfiles = np.size(infiles)

    matchfiles = np.zeros(20,dtype='U256')
    
    # loop through files:
    nf = 0
    nmtot = 0
    for i in range(nfiles):

        infile = infiles[i]
        
        fname = basename(infile)
        tmpstr = fname.split('_')
        catid = tmpstr[0]
        fieldname = '_'.join(tmpstr[3:6])

        nm = 0
        # now loop through the list again and find any matches
        for j in range(i+1,nfiles):

            infile2 = infiles[j]
            
            # skip if this is a red file, will handle those later...
            if (infile2.find('red') > 0):
                continue

            # ignore identical files (i.e. self match):
            if (infile == infile2):
                continue
            
            fname2 = basename(infile2)
            tmpstr2 = fname2.split('_')
            catid2 = tmpstr2[0]
            fieldname2 = '_'.join(tmpstr2[3:6])

            if (catid == catid2):
                matchfiles[nm] = infile2
                nm =  nm + 1
                nmtot = nmtot + 1
                print(infile)
                print(infile2,nm,nmtot)

        # skip if no match:
        if (nm == 0):
            continue

        # now read the original file and the matched files and plot them:
        hdulist = pf.open(infile)
        sami_flux_blue1,sami_lam_blue1 = sami_read_apspec(hdulist,apname,doareacorr=False)
        hdulist.close()

        infile_red = infile.replace('blue','red')
        hdulist = pf.open(infile_red)
        sami_flux_red1,sami_lam_red1 = sami_read_apspec(hdulist,apname,doareacorr=False)
        hdulist.close()

        
        if (plotall):
            py.figure(1)
            py.clf()
            py.subplot(2,1,1)
            #py.plot(sdss_lam_air,sdss_flux,color='k')
            #py.plot(sami_lam_blue,sdss_flux_blue,color='g')
            #py.plot(sami_lam_red,sdss_flux_red,color='m')
            py.plot(sami_lam_blue1,sami_flux_blue1,color='b')
            py.plot(sami_lam_red1,sami_flux_red1,color='r')


            
        # loop through matches:
        for im in range(nm):
            print('match file:',matchfiles[im])
            hdulist = pf.open(matchfiles[im])
            sami_flux_blue,sami_lam_blue = sami_read_apspec(hdulist,apname,doareacorr=False)
            hdulist.close()

            infile_red = matchfiles[im].replace('blue','red')
            hdulist = pf.open(infile_red)
            sami_flux_red,sami_lam_red = sami_read_apspec(hdulist,apname,doareacorr=False)
            hdulist.close()
            
            if (plotall):
                py.plot(sami_lam_blue,sami_flux_blue,color='b')
                py.plot(sami_lam_red,sami_flux_red,color='r')

                # plot ratios:
                py.subplot(2,1,2)
                py.plot(sami_lam_blue,sami_flux_blue/sami_flux_blue1,color='b')
                py.plot(sami_lam_red,sami_flux_red/sami_flux_red1,color='r')
                py.axhline(1.0,color='k',linestyle=':')
                py.ylim(ymin=0.5,ymax=1.5)

        if (plotall): 
            py.draw()
        # pause for input if plotting all the spectra:
            yn = input('Continue? (y/n):')

        #hdulist = pf.open(infile)

    print('total number of matches: ',nmtot)

################################################################################
# function to check zenith distance estimates when fitting std psf.  Take the
# keyword from the FLUX_CALIBRATION extension and then compare to a directly
# calculated ZD.
#
def check_zdfit(filelist):

    # glob files:
    infiles = glob.glob(filelist)

    # set up arrays:
    nfiles = np.size(infiles)
    zd_start = np.zeros(nfiles)
    zd_fit = np.zeros(nfiles)
    zd_eff = np.zeros(nfiles)
    fnum = np.zeros(nfiles)

    nf = 0
    for infile in infiles:

        hdulist = pf.open(infile)

        # check if FLUX_CALIBRATION extension is there:
        try:
            fdata = hdulist['FLUX_CALIBRATION'].data
            fc_header=hdulist['FLUX_CALIBRATION'].header
        except KeyError:
            continue

        print('found FLUX_CALIBRATION HDU in ',infile)

        # get main header:
        header = hdulist[0].header
        # get the direct ZDSTART keyword:
        zd_start[nf] = header['ZDSTART']
        # get the effective ZD based on header
        zd_eff[nf] = fluxcal2.calc_eff_airmass(header,return_zd=True)

        # get the fitted ZD from the FLUX_CALIBRATION extension.  This
        # is in radians, so convert to degrees:
        zd_fit[nf] = fc_header['ZENDIST']*180.0/np.pi
        
        print('ZD start, eff, fit:',zd_start[nf],zd_eff[nf],zd_fit[nf])

        fnum[nf] = nf
        
        nf = nf + 1
        

    # do plot:
    fig1 = py.figure()
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)
    ax1.plot(fnum[0:nf],zd_fit[0:nf]/zd_eff[0:nf],'o')
    ax1.axhline(1.0,linestyle=':',color='k')
    ax1.set(xlabel='frame number',ylabel='zd_fit/zd_eff')
    
    ax2.plot(zd_fit[0:nf],zd_eff[0:nf],'o')
    ax2.plot([0.0,65.0],[0.0,65.0],linestyle='-',color='k')
    ax2.set(xlabel='zd_fit',ylabel='zd_eff',xlim=[0.0,65.0],ylim=[0.0,65.0])
    

        


        
    


################################################################################
# function to plot TF from secondary flux standards for a particular field
#
def test_sec_fluxcal(fldpath,doplot=True,interactive=True,minexp=600):

    # first need to use glob to get all the sci.fits files:

    scifiles = glob.glob(fldpath+'/main/ccd_1/*sci.fits')

    # define the filename for the TRANSFER2combined.fits file:

    tf2file_blue = fldpath+'/main/ccd_1/TRANSFER2combined.fits'
    tf2file_red = fldpath+'/main/ccd_2/TRANSFER2combined.fits'

    # read in TF2 files to get optimal template:
    hdulist_tfblue = pf.open(tf2file_blue)
    lam_temp_opt = hdulist_tfblue['TEMPLATE_OPT'].data['wavelength']
    flux_temp_opt = hdulist_tfblue['TEMPLATE_OPT'].data['flux']
    
    lam_tf_mean_b = hdulist_tfblue['TF_MEAN'].data['wavelength']
    tf_mean_b = hdulist_tfblue['TF_MEAN'].data['tf_average']
    # also get the mags:
    opt_header = hdulist_tfblue['TEMPLATE_OPT'].header
    tempmag = np.zeros(5)
    tempmag[0] = opt_header['TEMPMAGU']
    tempmag[1] = opt_header['TEMPMAGG']
    tempmag[2] = opt_header['TEMPMAGR']
    tempmag[3] = opt_header['TEMPMAGI']
    tempmag[4] = opt_header['TEMPMAGZ']
    hdulist_tfblue.close()
    
    hdulist_tfred = pf.open(tf2file_red)
    lam_tf_mean_r = hdulist_tfred['TF_MEAN'].data['wavelength']
    tf_mean_r = hdulist_tfred['TF_MEAN'].data['tf_average']

    # convert the SDSS mags to flux in same units as SAMI spectra:
    maglam = np.array([3557.0,4702.0,6175.0,7491.0,8946.0])
    tempmag_flux = np.zeros(5)
    tempmag_flux[0] = (10.0**(-0.4*(tempmag[0]-8.90)))/3.34e4/(maglam[0]**2)*1.0e16
    tempmag_flux[1] = (10.0**(-0.4*(tempmag[1]-8.90)))/3.34e4/(maglam[1]**2)*1.0e16
    tempmag_flux[2] = (10.0**(-0.4*(tempmag[2]-8.90)))/3.34e4/(maglam[2]**2)*1.0e16
    tempmag_flux[3] = (10.0**(-0.4*(tempmag[3]-8.90)))/3.34e4/(maglam[3]**2)*1.0e16
    tempmag_flux[4] = (10.0**(-0.4*(tempmag[4]-8.90)))/3.34e4/(maglam[4]**2)*1.0e16

    # read in photometry of stars from catalogue:
    catalogue = read_stellar_mags()


    print(tempmag_flux)

      
    if (doplot):
        fig1 = py.figure()
        ax1_1 = fig1.add_subplot(311)
        ax1_2 = fig1.add_subplot(312)
        ax1_3 = fig1.add_subplot(313)

        # then the template with extinction applied:
        ax1_1.plot(lam_temp_opt,flux_temp_opt,'k')

        
    
    # read the spectra from individual files:
    nframe = 0
    for scifile_b in scifiles:
        imfile_t = scifile_b.replace('ccd_1','ccd_2')
        scifile_r = imfile_t[:-13]+'2'+imfile_t[-12:]        
        print(scifile_b,scifile_r)

        hdulist_b = pf.open(scifile_b)
        hdulist_r = pf.open(scifile_r)

        # check exposure time
        exposed = hdulist_b[0].header['EXPOSED']
        if (exposed < minexp):
            continue

        
        # first get cat mags from header:
        fluxcal_header = hdulist_b['FLUX_CALIBRATION'].header 
        catmag = np.zeros(5)
        # get object name:
        std_name = fluxcal_header['STDNAME']
        std_parameters = catalogue[std_name]
        print('getting star parameters for: ',std_name)
        # get the ab mags for the stars,  There are small offsets between SDSS and ab
        # that ar given here:
        catmag[0] = std_parameters['u']-0.04
        catmag[1] = std_parameters['g']
        catmag[2] = std_parameters['r']
        catmag[3] = std_parameters['i']
        catmag[4] = std_parameters['z']-0.02
        catmag_flux = np.zeros(5)
        catmag_flux[0] = (10.0**(-0.4*(catmag[0]-8.90)))/3.34e4/(maglam[0]**2)*1.0e16        
        catmag_flux[1] = (10.0**(-0.4*(catmag[1]-8.90)))/3.34e4/(maglam[1]**2)*1.0e16        
        catmag_flux[2] = (10.0**(-0.4*(catmag[2]-8.90)))/3.34e4/(maglam[2]**2)*1.0e16        
        catmag_flux[3] = (10.0**(-0.4*(catmag[3]-8.90)))/3.34e4/(maglam[3]**2)*1.0e16        
        catmag_flux[4] = (10.0**(-0.4*(catmag[4]-8.90)))/3.34e4/(maglam[4]**2)*1.0e16        
        print('catmag:',catmag)
        print('tempmag:',tempmag)
        print('catmag-tempmag = ',catmag-tempmag)
        print('flux ratio from mags (cat/temp)= ',10**(-0.4*(catmag-tempmag)))
        
        # now read in the TFs and individual spectra:
        lam_star_b = hdulist_b['FLUX_CALIBRATION2'].data['wavelength']
        flux_star_b = hdulist_b['FLUX_CALIBRATION2'].data['flux']
        temp_star_b = hdulist_b['FLUX_CALIBRATION2'].data['template']
        tf_star_b = hdulist_b['FLUX_CALIBRATION2'].data['transfer_fn']
        
        lam_star_r = hdulist_r['FLUX_CALIBRATION2'].data['wavelength']
        flux_star_r = hdulist_r['FLUX_CALIBRATION2'].data['flux']
        temp_star_r = hdulist_r['FLUX_CALIBRATION2'].data['template']
        tf_star_r = hdulist_r['FLUX_CALIBRATION2'].data['transfer_fn']

        # plot the TFs etc:
        if (doplot):
            if (interactive):
                # then the template with extinction applied:
                ax1_1.cla()
                ax1_2.cla()
                ax1_1.plot(lam_temp_opt,flux_temp_opt,'k')


            ax1_1.plot(lam_star_b,flux_star_b,color='b')
            ax1_1.plot(lam_star_r,flux_star_r,color='r')

        ratio_b = flux_star_b/temp_star_b
        ratio_r = flux_star_r/temp_star_r
        
        if (doplot):
            ax1_2.plot(lam_star_b,ratio_b,color='b')
            ax1_2.plot(lam_star_r,ratio_r,color='r')
            ax1_2.plot(lam_star_b,tf_star_b,color='c')
            ax1_2.plot(lam_star_r,tf_star_r,color='m')
            ax1_2.axhline(1.0,color='k')
            
            ax1_3.plot(lam_star_b,tf_star_b,'c')
            ax1_3.plot(lam_star_r,tf_star_r,'m')
            ax1_3.axhline(1.0,color='k')
            ax1_3.plot(lam_tf_mean_b,tf_mean_b,'b')
            ax1_3.plot(lam_tf_mean_r,tf_mean_r,'r')
            #ax1_3.set(xlim=[xmin,xmax],xlabel='Wavelength (Ang.)',ylabel='SAMI/template')


            if (interactive):
                ax1_1.plot(maglam,catmag_flux,'o',color='m')
                ax1_1.plot(maglam,tempmag_flux,'o',color='g')
                fig1.canvas.draw_idle()
                yn = input("Continue? (Y/N)")
                
            
    # plot the mags:
    if (nframe == 0):
        ax1_1.plot(maglam,catmag_flux,'o',color='m')
        ax1_1.plot(maglam,tempmag_flux,'o',color='g')



            
            
            

            
        nframe = nframe + 1

################################################################################
# test SAMI psf fitting for extraction of flux cal stars by running the main
# SAMI pipeline code, but in a stand alone mode
#
# usage:
# sami_dr_smc.sami_fluxcal.test_psf_fit(['reduced/130306/Y13SAR1_P002_09T004_15T006/Y13SAR1_P002_15T006/main/ccd_1/06mar10050red.fits','reduced/130306/Y13SAR1_P002_09T004_15T006/Y13SAR1_P002_15T006/main/ccd_2/06mar20050red.fits'],7)
#
def test_psf_fit(frames,probenum,doplot=True,plotall=False,zdhdr=False):


    # get header for first frame
    hdulist = pf.open(frames[0])
    header = hdulist[0].header

    # get airmass:
    eff_airmass = fluxcal2.calc_eff_airmass(header)
    print('eff_airmass:',eff_airmass)
    
    # type of model.  If zdhdr is true, take the ZD from the header, do not
    # calculate it from the star.
    if (zdhdr):
        model_name = 'ref_centre_alpha_circ_hdratm'
    else:
        model_name = 'ref_centre_alpha_dist_circ_hdratm'

    rtod = 180.0/np.pi
    
    # chunk the data:
    chunked_data = fluxcal2.read_chunked_data(frames,probenum)

    # adjust the variances to account for systematic errors in the data, plus lower limit on noise:
    #chunked_data['variance'] = chunked_data['variance']+(0.01*chunked_data['data'])**2 + 4.0
    # adding a constant only works for bright std, as the flux is scaled to exp time, so
    # for regular data frames 4.0 would be too large
    #chunked_data['variance'] = chunked_data['variance']+(0.01*chunked_data['data'])**2
    print(chunked_data['variance'])
    
    nfib, nchunk = np.shape(chunked_data['data'])
    print('size of chunked data, fibs x chunks:',nfib,nchunk)
    
    # fix parameters in the model:
    fixed_parameters = fluxcal2.set_fixed_parameters(frames,model_name,probenum=probenum)
    print(fixed_parameters)

    # do the fit:
    psf_parameters = fluxcal2.fit_model_flux(chunked_data['data'],chunked_data['variance'],chunked_data['xfibre'],chunked_data['yfibre'],chunked_data['wavelength'],model_name,fixed_parameters=fixed_parameters)

    psf_parameters = fluxcal2.insert_fixed_parameters(psf_parameters, fixed_parameters)

    print(psf_parameters)
    # output zenith distance in degrees:
    print('zenith distance:',psf_parameters['zenith_distance'],psf_parameters['zenith_distance']*rtod)
    
    # convert from a dictionary to a vector of psf model parameters:
    #par_vector = fluxcal2.parameters_dict_to_vector(psf_parameters,'ref_centre_alpha_dist_circ_hdratm')

    # generate the model:
    model = fluxcal2.model_flux(psf_parameters,chunked_data['xfibre'],chunked_data['yfibre'],chunked_data['wavelength'],model_name)

    print(np.shape(model))

    parameters_dict = psf_parameters
    
    residual = chunked_data['data'] - model

    chi = residual/np.sqrt(chunked_data['variance'])
    
    alpha_ref_global = psf_parameters['alpha_ref']
    beta_global = psf_parameters['beta']

    # get the actual centres from the model for each chunk.  Uses the same lines as in fluxcal2
    xcen_lam = np.zeros(nchunk)
    ycen_lam = np.zeros(nchunk)
    for i in range(nchunk):
        xcen_lam[i] = (
            parameters_dict['xcen_ref'] +
            np.sin(parameters_dict['zenith_direction']) * 
            fluxcal2.dar(chunked_data['wavelength'][i], parameters_dict['zenith_distance'],
                temperature=parameters_dict['temperature'],
                pressure=parameters_dict['pressure'],
                vapour_pressure=parameters_dict['vapour_pressure']))
        ycen_lam[i] = (
            parameters_dict['ycen_ref'] +
            np.cos(parameters_dict['zenith_direction']) * 
            fluxcal2.dar(chunked_data['wavelength'][i], parameters_dict['zenith_distance'],
                temperature=parameters_dict['temperature'],
                pressure=parameters_dict['pressure'],
                vapour_pressure=parameters_dict['vapour_pressure']))


    # set up arrays to get individual chunk fits:
    chisq_chunk = np.zeros(nchunk)
    xcen_chunk = np.zeros(nchunk)
    ycen_chunk = np.zeros(nchunk)
    alpha_chunk = np.zeros(nchunk)
    alpha_ref_chunk = np.zeros(nchunk)
    beta_chunk = np.zeros(nchunk)
    flux_chunk = np.zeros(nchunk)
    background_chunk = np.zeros(nchunk)
        
    # do some plots:
    if (doplot):
        mycolormap=py.get_cmap('YlGnBu_r')
        if (plotall):
            fig1 = py.figure(1)
            fig4 = py.figure(4)
            fig6 = py.figure(6)
            ax6_1 = fig6.add_subplot(111)
            #ax6_2 = fig6.add_subplot(222)
        for i in range(nchunk):
            if (plotall):
                fig1.clf()
                fig4.clf()
                title = 'Wavelength: {0:7.1f} A'.format(chunked_data['wavelength'][i])
                fig4.suptitle(title)
                ax1_1 = fig1.add_subplot(231)
                ax1_2 = fig1.add_subplot(232)
                ax1_3 = fig1.add_subplot(233)
                ax1_4 = fig1.add_subplot(234)
                ax1_5 = fig1.add_subplot(235)
                ax1_6 = fig1.add_subplot(236)
                ax4_1 = fig4.add_subplot(221)
                ax4_2 = fig4.add_subplot(222)
                ax4_3 = fig4.add_subplot(223)
                ax4_4 = fig4.add_subplot(224)
            fibres=[]
            fibresm=[]
            fibresd=[]
            fibressd=[]
            fibresmc=[]
            fibresl=[]

            # try fitting the data for a single chunk:
            psf_parameters_chunk = fluxcal2.fit_model_flux(chunked_data['data'][:,i:i+1],chunked_data['variance'][:,i:i+1],chunked_data['xfibre'],chunked_data['yfibre'],chunked_data['wavelength'][i:i+1],model_name,fixed_parameters=fixed_parameters)

            psf_parameters_chunk = fluxcal2.insert_fixed_parameters(psf_parameters_chunk, fixed_parameters)
                
            print(psf_parameters_chunk)

            model_chunk = fluxcal2.model_flux(psf_parameters_chunk,chunked_data['xfibre'],chunked_data['yfibre'],chunked_data['wavelength'][i:i+1],model_name)

            wl = chunked_data['wavelength'][i:i+1]
            # get the centres based on the individual fits:

            xcen_chunk[i] = psf_parameters_chunk['xcen_ref']+np.sin(psf_parameters_chunk['zenith_direction']) * fluxcal2.dar(wl, psf_parameters_chunk['zenith_distance'],temperature=psf_parameters_chunk['temperature'],pressure=psf_parameters_chunk['pressure'],vapour_pressure=psf_parameters_chunk['vapour_pressure'])
            ycen_chunk[i] = psf_parameters_chunk['ycen_ref']+np.cos(psf_parameters_chunk['zenith_direction']) * fluxcal2.dar(wl, psf_parameters_chunk['zenith_distance'],temperature=psf_parameters_chunk['temperature'],pressure=psf_parameters_chunk['pressure'],vapour_pressure=psf_parameters_chunk['vapour_pressure'])


            chi_chunk =  (chunked_data['data'][:,i:i+1] - model_chunk)/np.sqrt(chunked_data['variance'][:,i:i+1])

            chisq_chunk[i] = np.sum(chi_chunk**2)

            alpha_ref_chunk[i] = psf_parameters_chunk['alpha_ref']
            alpha_chunk[i] = fluxcal2.alpha(chunked_data['wavelength'][i:i+1],alpha_ref_chunk[i])
            beta_chunk[i] =  psf_parameters_chunk['beta']
            flux_chunk[i] =  psf_parameters_chunk['flux'][0]
            background_chunk[i] =  psf_parameters_chunk['background'][0]



            diff = residual[:,i]
            sigdiff = chi[:,i]
            if (plotall):
                # Iterate over the x, y positions making a circle patch for each fibre, with the appropriate color.
                nf_count = 0
                vmin = np.nanmin(chunked_data['data'][:,i])
                vmax = np.nanmax(chunked_data['data'][:,i])
                for xval, yval, dataval in zip(chunked_data['xfibre'],chunked_data['yfibre'],chunked_data['data'][:,i]):
                    #Add the fibre patch.
                    fibre=Circle(xy=(xval,yval), radius=0.8)
                    fibres.append(fibre)
                    nf_count = nf_count+1
                    col = 'w'
                    if (dataval > 0.7*vmax):
                        col = 'k'
                    ax1_1.text(xval,yval,str(nf_count),color=col,horizontalalignment='center',verticalalignment='center',fontsize=9)
                
                for xval, yval, dataval in zip(chunked_data['xfibre'],chunked_data['yfibre'],model[:,i]):
                    #Add the fibre patch.
                    fibre=Circle(xy=(xval,yval), radius=0.8)
                    fibresm.append(fibre)
                    ax1_2.plot(xval,yval,',')
                
                for xval, yval, dataval in zip(chunked_data['xfibre'],chunked_data['yfibre'],diff):
                    #Add the fibre patch.
                    fibre=Circle(xy=(xval,yval), radius=0.8)
                    fibresd.append(fibre)
                    ax1_3.plot(xval,yval,',')
                
                for xval, yval, dataval in zip(chunked_data['xfibre'],chunked_data['yfibre'],sigdiff):
                    #Add the fibre patch.
                    fibre=Circle(xy=(xval,yval), radius=0.8)
                    fibressd.append(fibre)
                    ax1_4.plot(xval,yval,',')
                    
                for xval, yval, dataval in zip(chunked_data['xfibre'],chunked_data['yfibre'],model_chunk):
                    #Add the fibre patch.
                    fibre=Circle(xy=(xval,yval), radius=0.8)
                    fibresmc.append(fibre)
                    ax1_5.plot(xval,yval,',')

                for xval, yval, dataval in zip(chunked_data['xfibre'],chunked_data['yfibre'],np.log10(chunked_data['data'][:,i])):
                    #Add the fibre patch.
                    fibre=Circle(xy=(xval,yval), radius=0.8)
                    fibresl.append(fibre)
                    ax1_6.plot(xval,yval,',')

                    
                allpatches=PatchCollection(fibres, cmap=mycolormap, edgecolors='none') 
                allpatches.set_array(chunked_data['data'][:,i])
            
                allpatchesm=PatchCollection(fibresm, cmap=mycolormap, edgecolors='none') 
                allpatchesm.set_array(model[:,i])
            
                allpatchesd=PatchCollection(fibresd, cmap=mycolormap, edgecolors='none') 
                allpatchesd.set_array(diff)
            
                allpatchessd=PatchCollection(fibressd, cmap=mycolormap, edgecolors='none') 
                allpatchessd.set_array(sigdiff)
                
                allpatchesmc=PatchCollection(fibresmc, cmap=mycolormap, edgecolors='none') 
                allpatchesmc.set_array(np.ravel(model_chunk))
                
                allpatchesl=PatchCollection(fibresl, cmap=mycolormap, edgecolors='none')
                allpatchesl.set_array(np.log10(chunked_data['data'][:,i]))
                # need to reset limits as Nan's seem to stuff it up:
                lmin = np.nanmin(np.log10(chunked_data['data'][:,i]))
                lmax = np.nanmax(np.log10(chunked_data['data'][:,i]))
                allpatchesl.set_clim([lmin,lmax])
                
                cax1_1 = ax1_1.add_collection(allpatches)
                ax1_1.set(title='data')
                cbar1_1 = fig1.colorbar(cax1_1,ax=ax1_1)
                
                cax1_2 = ax1_2.add_collection(allpatchesm)
                ax1_2.set(title='full model')
                cbar1_2 = fig1.colorbar(cax1_2,ax=ax1_2)
                
                cax1_3 = ax1_3.add_collection(allpatchesmc)
                ax1_3.set(title='chunked model')
                cbar1_3 = fig1.colorbar(cax1_3,ax=ax1_3)

                cax1_4 = ax1_4.add_collection(allpatchesd)
                cbar1_4 = fig1.colorbar(cax1_4,ax=ax1_4)
                
                cax1_5 = ax1_5.add_collection(allpatchessd)
                cbar1_5 = fig1.colorbar(cax1_5,ax=ax1_5)
                
                cax1_6 = ax1_6.add_collection(allpatchesl)
                ax1_6.set(title='log(data)')
                cbar1_6 = fig1.colorbar(cax1_6,ax=ax1_6)


            # use the moffat function fitted parameters to integrate the function out to a given radius and
            # see what fraction of the flux is contained within the bundle.
            # make a grid of points:
            bsize = 0.1
            xax = np.linspace(-50.0,50.0,1000)
            yax = np.linspace(-50.0,50.0,1000)
            xg, yg = np.meshgrid(xax,yax)
            radg = np.sqrt(xg**2 + yg**2)
            print('size',np.shape(xg))
            # define parameters in the right format:
            moffat_pars = {'xcen':0.0,
                            'ycen':0.0,
                            'alphax':psf_parameters_chunk['alpha_ref'],
                            'alphay':psf_parameters_chunk['alpha_ref'],
                            'beta':psf_parameters_chunk['beta'],
                            'rho':0.0}
            moffat_fn = fluxcal2.moffat_normalised(moffat_pars,xg,yg,simple=True)*bsize**2/(np.pi * fluxcal2.FIBRE_RADIUS**2)

            moffat_fn_sum = np.sum(moffat_fn)
            print('sum of best fit moffat fn:',moffat_fn_sum)

            #get a curve of growth:
            nr = 100
            rax = np.linspace(0,50.0,nr)
            moffat_cog = np.zeros(nr)
            for ir in range(nr):
                idx = np.where(radg<rax[ir])
                moffat_cog[ir] = np.sum(moffat_fn[idx])
                
            idx = np.where(radg<7.5)
            print('fractional moffat fn flux within 7.5 arcsec:',np.sum(moffat_fn[idx]))
            
            if (plotall):
                #cax6_1 = ax6_1.imshow(np.log10(moffat_fn),origin='lower',interpolation='nearest')
                #cbar1=fig6.colorbar(cax6_1,ax=ax6_1)
                ax6_1.plot(rax,moffat_cog)
                ax6_1.set(xlabel='Radius (arcsec)',ylabel='fractional flux')
            
            if (plotall):
                # plot location of model centre:
                ax1_1.plot(xcen_lam,ycen_lam,'.',color='r',markersize=2)
                ax1_2.plot(xcen_lam,ycen_lam,'.',color='r',markersize=2)
                ax1_1.plot(xcen_lam[i],ycen_lam[i],'+',color='g',markersize=15)
                ax1_2.plot(xcen_lam[i],ycen_lam[i],'+',color='g',markersize=15)
                ax1_3.plot(xcen_lam[i],ycen_lam[i],'+',color='g',markersize=15)
                ax1_4.plot(xcen_lam[i],ycen_lam[i],'+',color='g',markersize=15)
                ax1_5.plot(xcen_lam[i],ycen_lam[i],'+',color='g',markersize=15)
                ax1_6.plot(xcen_lam[i],ycen_lam[i],'+',color='g',markersize=15)

                ax1_3.plot(xcen_chunk[i],ycen_chunk[i],'x',color='b',markersize=15)

                # plot the radial profile:
                rad = np.sqrt((chunked_data['xfibre']-xcen_lam[i])**2+(chunked_data['yfibre']-ycen_lam[i])**2)
                #ax4.plot(rad,chunked_data['data'][:,i],'o',label='data')
                ax4_1.errorbar(rad,chunked_data['data'][:,i],np.sqrt(chunked_data['variance'][:,i]),fmt='o',color='r',label='data')
                ax4_1.plot(rad,model[:,i],'o',color='b',label='model')
                ax4_1.plot(rad,model_chunk,'o',color='g',label='model_chunk')
                ax4_1.axhline(0,linestyle=':',color='k')
                ax4_1.axhline(psf_parameters['background'][i],linestyle='-',color='b')
                ax4_1.axhline(psf_parameters_chunk['background'][0],linestyle='-',color='g')
                ax4_1.set(xlabel='Radius',ylabel='Flux')
                ax4_1.legend()
                
                ax4_2.plot(rad,np.log10(chunked_data['data'][:,i]),'o',color='r',label='data')
                ax4_2.plot(rad,np.log10(model[:,i]),'o',color='b',label='model')
                ax4_2.plot(rad,np.log10(model_chunk),'o',color='g',label='model_chunk')
                ax4_2.plot(rad,np.log10(-1.0*chunked_data['data'][:,i]),'o',color='r',label='data (-ve points)',mfc='none')
                ax4_2.set(xlabel='Radius',ylabel='log(Flux)')
                ax4_2.legend()

                fibax = np.linspace(1.0,61.0,61)
                ax4_3.errorbar(fibax,chunked_data['data'][:,i],np.sqrt(chunked_data['variance'][:,i]),fmt='o',color='r',label='data')
                ax4_3.plot(fibax,model[:,i],'o',color='b',label='model')
                ax4_3.plot(fibax,model_chunk,'o',color='g',label='model_chunk')
                ax4_3.axhline(0,linestyle=':',color='k')
                ax4_3.axhline(psf_parameters['background'][i],linestyle='-',color='g')
                ax4_3.set(xlabel='fib number',ylabel='Flux')
                
                ax4_4.plot(fibax,np.log10(chunked_data['data'][:,i]),'o',color='r',label='data')
                ax4_4.plot(fibax,np.log10(model[:,i]),'o',color='b',label='model')
                ax4_4.plot(fibax,np.log10(model_chunk),'o',color='g',label='model_chunk')
                ax4_4.plot(fibax,np.log10(-1.0*chunked_data['data'][:,i]),'o',color='r',label='model',mfc='none')
                ax4_4.set(xlabel='fib number',ylabel='log(Flux)')

                
                fig1.canvas.draw_idle()
                fig4.canvas.draw_idle()
                fig6.canvas.draw_idle()
                #py.draw()
                chisq_lam = np.sum(sigdiff**2)
                print(i,chunked_data['wavelength'][i],chisq_lam,chisq_chunk[i],psf_parameters['flux'][i],flux_chunk[i])
                yn = input("Continue? (Y/N)")

        # plot total residual flux vs wavelength:
        ressum = np.sum(residual,axis=0)
        chisqsum = np.sum(chi**2,axis=0)
        sumflux = np.sum(chunked_data['data'],axis=0)
        summodel = np.sum(model,axis=0)
        fig2 = py.figure(2)
        ax2_1 = fig2.add_subplot(311)
        #ax2_1.plot(chunked_data['wavelength'],ressum,label='summed fib residual')
        ax2_1.plot(chunked_data['wavelength'],psf_parameters['flux'],label='model flux',color='r')
        ax2_1.plot(chunked_data['wavelength'],sumflux,label='summed fib flux',color='g')
        #ax2_1.plot(chunked_data['wavelength'],summodel,label='summed fib model')
        ax2_1.plot(chunked_data['wavelength'],flux_chunk,label='model flux (chunks)',color='b')
        #ax2_1.plot(chunked_data['wavelength'],background_chunk,label='background (chunks)')
        #ax2_1.plot(chunked_data['wavelength'],psf_parameters['background'],label='background model')
        ax2_1.legend()
        
        ax2_2 = fig2.add_subplot(312)
        ax2_2.plot(chunked_data['wavelength'],psf_parameters['flux']/sumflux,label='model/(fib flux)')
        ax2_2.plot(chunked_data['wavelength'],flux_chunk/sumflux,label='model_chunk/(fib flux)')
        ax2_2.legend()
        
        ax2_3 = fig2.add_subplot(313)
        ax2_3.plot(chunked_data['wavelength'],chisqsum,label='chisq')
        ax2_3.plot(chunked_data['wavelength'],chisq_chunk,label='chisq chunk')
        ax2_3.legend()

        fig3 = py.figure(3)
        ax3_1 = fig3.add_subplot(311)
        ax3_1.plot(chunked_data['wavelength'],alpha_ref_chunk,label='alpha ref per chunk')
        ax3_1.plot(chunked_data['wavelength'],alpha_chunk,label='alpha per chunk')
        ax3_1.axhline(alpha_ref_global,label='alpha ref global')
        ax3_1.legend()
        ax3_2 = fig3.add_subplot(312)
        ax3_2.plot(chunked_data['wavelength'],beta_chunk,label='beta per chunk')
        ax3_2.axhline(beta_global,label='beta global')
        ax3_2.legend()
        fwhm = 2.0 * alpha_chunk * np.sqrt(2.0**(1/beta_chunk)-1) 
        ax3_3 = fig3.add_subplot(313)
        ax3_3.plot(chunked_data['wavelength'],fwhm,label='fwhm per chunk')
        fwhm_global = 2.0 * alpha_ref_global * np.sqrt(2.0**(1/beta_global)-1) 
        fwhm_model = fwhm_global*(chunked_data['wavelength']/fluxcal2.REFERENCE_WAVELENGTH)**(-0.2)
        fwhm_model2 = fwhm_global*(chunked_data['wavelength']/fluxcal2.REFERENCE_WAVELENGTH)**(-0.4)
        fwhm_model3 = fwhm_global*(chunked_data['wavelength']/fluxcal2.REFERENCE_WAVELENGTH)**(-0.6)
        ax3_3.plot(chunked_data['wavelength'],fwhm_model,label='fwhm model -0.2')
        #ax3_3.plot(chunked_data['wavelength'],fwhm_model2,label='fwhm model -0.4')
        #ax3_3.plot(chunked_data['wavelength'],fwhm_model3,label='fwhm model -0.6')
        ax3_3.legend()

        fig5 = py.figure(5)
        ax5_1 = fig5.add_subplot(211)
        ax5_1.plot(chunked_data['wavelength'],xcen_lam,label='xcen')
        ax5_1.plot(chunked_data['wavelength'],xcen_chunk,label='xcen chunk')
        ax5_1.legend()
        
        ax5_2 = fig5.add_subplot(212)
        ax5_2.plot(chunked_data['wavelength'],ycen_lam,label='ycen')
        ax5_2.plot(chunked_data['wavelength'],ycen_chunk,label='ycen chunk')
        ax5_2.legend()

        
################################################################################
# plot an extinction curve:
#
def plot_ext(infile,docorr=False,corrfile='/Users/scroom/data/sami/fluxcal/ext_curves/extin23mnewcorr.tab',airmass=1.0):

     lam, ext  = np.loadtxt(infile, usecols=(0, 1), unpack=True, comments=['*','set'])

     # read in correction file:
     lamc, extc  = np.loadtxt(corrfile, usecols=(0, 1), unpack=True, comments=['*','set'])

     if (docorr):
         ext = ext + extc
     
     fig1 = py.figure(1)
     ax1 = fig1.add_subplot(2,1,1)
     ax1.plot(lam,ext,label=infile)
     ax1.set(xlabel='wavelength',ylabel='extinction (mags)',xlim=[3500.0,7500.0])
     ax1.legend()

     # convert to flux and replot:
     ext_am = ext * airmass
     ext_f = 10.0**(-0.4 * ext_am)
     ax2 = fig1.add_subplot(2,1,2)
     ax2.plot(lam,ext_f,label=infile)
     ax2.set(xlabel='wavelength',ylabel='extinction',xlim=[3500.0,7500.0])

     

################################################################################
# script to copy aperture spectra, but keep them in directories corresponding
# to particular runs.
#
def cp_apspec(runlist):

    # glob the run list to start with:
    runs = glob.glob(runlist)

    # loop over runs:
    for runpath in runs:
        print('copying aperture spectra from run: ',runpath)

        # split the path up, and just keep last bit:
        # remove trailing '/' if needed:
        if runpath[-1] == '/':
            runpath = runpath[:-1]
        pathdirs = runpath.split('/')
        run = pathdirs[-1]
        print(pathdirs)
        print(run)
        
        # need to make a copy of this folder in the local directory:
        try: 
            mkdir(run)
            print('making new directory:',run) 
        except FileExistsError:
            print('directory already exists:',run)

        # now find all the aperture spectra files:
        #for filename in glob.glob(join(runpath, 'cubed/*/*_apspec.fits')):
        for filename in glob.glob(join(runpath, 'cubed/*/*.fits')):
            print('copying ',filename)
            shutil.copy(filename,run)



        



################################################################################
# compare many TFs from individual primary star observations and check against
# various parameters.
#
def check_tf_many(inlist):

    # glob the list:
    infiles = glob.glob(inlist)

    # set up arrays:
    nmax = np.size(infiles)
    allflux = np.zeros((2048,nmax))
    alllam = np.zeros((2048,nmax))
    alltf = np.zeros((2048,nmax))
    allairmass = np.zeros(nmax)
    allexposed = np.zeros(nmax)
    allsn = np.zeros(nmax)
    allrbratio = np.zeros(nmax)
    allname = np.empty(nmax,dtype='U256')
    
    # set up plotting:
    fig1 = py.figure(1)
    ax1_1 = fig1.add_subplot(311)
    ax1_2 = fig1.add_subplot(312)
    ax1_3 = fig1.add_subplot(313)
    
    # loop over files
    nf = 0
    for infile in infiles:

        # first check some header keywords.  In particular, make sure
        # this is a flux calibration star.
        try:
            hdulist = pf.open(infile)
            primary_header=hdulist['PRIMARY'].header
        except FileNotFoundError:
            print('file not found: ',infile)
            continue
            
        # flag set if spectrophotometric std:
        try: 
            stdflag = primary_header['MNGRSPMS']
            stdname = primary_header['MNGRNAME']
            zdstart = primary_header['ZDSTART']
            exposed = primary_header['EXPOSED']
            # calculate airmass:
            alt = float(90.0 - zdstart)
            airmass = 1./ ( np.sin( ( alt + 244. / ( 165. + 47 * alt**1.1 )
                            ) / 180. * np.pi ) )
            
        except KeyError:
            continue
 
        if (stdflag):
            # check for data from FLUX_CALIBRATION extension:
            try:
                fdata = hdulist['FLUX_CALIBRATION'].data
                fc_header=hdulist['FLUX_CALIBRATION'].header
            except KeyError:
                continue

            # if found, read in the data:
            lam, flux, sigma, tf = read_flux_calibration_extension(infile,gettf=True)
            print('reading ',infile)
            allflux[:,nf] = flux
            alllam[:,nf] = lam
            alltf[:,nf] = tf
            allexposed[nf] = exposed
            allairmass[nf] = airmass
            allname[nf] = infile
            allsn[nf] = np.nanmedian(flux)/np.nanmedian(sigma)
            
            nf = nf +1

    cmap = py.get_cmap(py.cm.rainbow)
    cols = [cmap(i) for i in np.linspace(0, 1, nf)]

    # get mean TF:
    meantf = np.mean(alltf[:,0:nf],axis=1)

    #exposure time range:
    exp_max = np.max(allexposed[0:nf])
    exp_min = np.min(allexposed[0:nf])
    airmass_max = np.max(allairmass[0:nf])
    airmass_min = np.min(allairmass[0:nf])
    for i in range(nf):
        allrbratio[i] = (alltf[200,i]/meantf[200])/(alltf[1900,i]/meantf[1900])

    # set up colourmaps:
    colormap = py.cm.rainbow
    colorparams1 = allexposed[0:nf]
    normalize1 = mcolors.Normalize(vmin=np.min(colorparams1), vmax=np.max(colorparams1))
    colorparams2 = allairmass[0:nf]
    normalize2 = mcolors.Normalize(vmin=np.min(colorparams2), vmax=np.max(colorparams2))
    colorparams3 = allsn[0:nf]
    normalize3 = mcolors.Normalize(vmin=np.min(colorparams3), vmax=np.max(colorparams3))

    for i in range(nf):
        # set color based on exposure time range:
        color1 = colormap(normalize1(allexposed[i]))
        # set color based on airmass range:
        color2 = colormap(normalize2(allairmass[i]))
        # set color based on S/N:
        color3 = colormap(normalize3(allsn[i]))
        ax1_1.plot(alllam[:,i],alltf[:,i]/meantf,color=color1)
        ax1_2.plot(alllam[:,i],alltf[:,i]/meantf,color=color2)
        ax1_3.plot(alllam[:,i],alltf[:,i]/meantf,color=color3)
        print(i,allexposed[i],allairmass[i],allsn[i],allname[i])

    #ax1_1.plot(alllam[:,i],meantf,color='k',label='mean TF')

    print('airmass range: ',airmass_min,airmass_max)
    print('exposure time range: ',exp_min,exp_max)

    # set up exposure time colourbar:
    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize1, cmap=colormap)
    s_map.set_array(colorparams1)
    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams1[1] - colorparams1[0])/2.0
    # Use this to show a continuous colorbar
    cbar1 = fig1.colorbar(s_map, ax=ax1_1,spacing='proportional', format='%5.1f')
    cbarlabel = 'Exp (s)'
    cbar1.set_label(cbarlabel, fontsize=10)
    
    # set up airmass colourbar:
    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize2, cmap=colormap)
    s_map.set_array(colorparams2)
    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams2[1] - colorparams2[0])/2.0
    # Use this to show a continuous colorbar
    cbar2 = fig1.colorbar(s_map, ax=ax1_2,spacing='proportional', format='%4.2f')
    cbarlabel = 'Airmass'
    cbar2.set_label(cbarlabel, fontsize=10)

    # set up S/N colourbar:
    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize3, cmap=colormap)
    s_map.set_array(colorparams3)
    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams3[1] - colorparams3[0])/2.0
    # Use this to show a continuous colorbar
    cbar3 = fig1.colorbar(s_map, ax=ax1_3,spacing='proportional', format='%6.1f')
    cbarlabel = 'S/N'
    cbar3.set_label(cbarlabel, fontsize=10)

    # second set of plots for different parameters:
    fig2 = py.figure(2)
    ax2_1 = fig2.add_subplot(221)
    #ax2_1.plot(allairmass[0:nf],allrbratio[0:nf],'o')
    cax2_1 = ax2_1.scatter(allairmass[0:nf],allrbratio[0:nf],c=allexposed[0:nf],marker='o',cmap=colormap)
    ax2_1.set(xlabel='Airmass (s)',ylabel='blue/red TF residual')
    cbar2_1 = fig2.colorbar(cax2_1,spacing='proportional', format='%5.1f')
    cbar2_1.set_label('Exposure time (s)', fontsize=10)

    ax2_2 = fig2.add_subplot(222)
    #ax2_2.plot(allexposed[0:nf],allrbratio[0:nf],'o')
    cax2_2 = ax2_2.scatter(allexposed[0:nf],allrbratio[0:nf],c=allairmass[0:nf],marker='o',cmap=colormap)
    ax2_2.set(xlabel='exposure time (s)',ylabel='blue/red TF residual')
    cbar2_2 = fig2.colorbar(cax2_2,spacing='proportional', format='%5.1f')
    cbar2_2.set_label('Airmass', fontsize=10)

    ax2_3 = fig2.add_subplot(223)
    #ax2_3.plot(allsn[0:nf],allrbratio[0:nf],'o')
    cax2_3 = ax2_3.scatter(allsn[0:nf],allrbratio[0:nf],c=allairmass[0:nf],marker='o',cmap=colormap)
    ax2_3.set(xlabel='S/N',ylabel='blue/red TF residual')
    cbar2_3 = fig2.colorbar(cax2_3,spacing='proportional', format='%5.1f')
    cbar2_3.set_label('Airmass', fontsize=10)

    ax2_4 = fig2.add_subplot(224)
    nax = np.linspace(1.0,float(nf),num=nf)
    cax2_4 = ax2_4.scatter(nax,allrbratio[0:nf],c=allairmass[0:nf],marker='o',cmap=colormap)
    ax2_4.set(xlabel='frame number',ylabel='blue/red TF residual')
    cbar2_4 = fig2.colorbar(cax2_4,spacing='proportional', format='%5.1f')
    cbar2_4.set_label('Airmass', fontsize=10)

    
    #ax1_1.legend()
            

################################################################################
# plot primary standards and compare to flux standard data
#
# sami_dr_smc.sami_fluxcal.comp_primary_std('13may10013red.fits','/Users/scroom/code/sami/python/sami/standards/ESO/fltt3218.dat')    
#
def comp_primary_std(infile,stdfile):

    # open and read the file:
    lam, flux, sigma, tf = read_flux_calibration_extension(infile,gettf=True)

    # plot the spectrum:
    fig1 = py.figure(1)
    ax1_1 = fig1.add_subplot(211)
    ax1_2 = fig1.add_subplot(212)

    ax1_1.plot(lam,flux*tf,label='obs flux * TF')
    #ax1.plot(lam,tf)

    # get the std data:
    lam_std, flux_std  = np.loadtxt(stdfile, usecols=(0, 1), unpack=True)

    ax1_1.plot(lam_std,flux_std,label='standard flux')
    ax1_1.set(xlabel='Wavelength (A)',ylabel='Flux')

    # resample onto obs lam scale:
    flux_std_r = spectres(lam,lam_std,flux_std)
    ax1_1.plot(lam,flux_std_r)
    
    ax1_2.plot(lam,flux*tf/flux_std_r)
    ax1_2.set(xlabel='Wavelength (A)',ylabel='obs*TF/std Flux')
    
    

    

################################################################################
# reformat coords for ADC cutout server:
def reform_starcat(infile):
    
    ra, dec = np.loadtxt(infile, usecols=(2, 3), comments = ['#', '*', 'set'], unpack=True)

    n = np.size(ra)
    
    ftab = open('coord.txt',mode='w')

    for i in range(n):
        ftab.write(str(ra[i])+'d,'+str(dec[i])+'d\n')
    
    
    
################################################################################
# plot the spectrophotometric calibrators in turn (using the published fluxes
#

def plot_std_data(inlist,lam1=3700,lam2=7500):

    # get file names:
    specfiles = glob.glob(inlist)

    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(111)

    # define locations of telluric abs bands:
    bandstart = np.array([6850.0,7130.0,7560.0,8100.0])
    bandend = np.array([6960.0,7360.0,7770.0,8360.0])

    ftab = open('specphot_check.txt',mode='w')
    # go through files:
    for filename in specfiles:
        print('opening: ',filename)
        lam, flux = np.loadtxt(filename, usecols=(0, 1), comments = ['#', '*', 'set'], unpack=True)
        idx = np.where((lam>lam1) & (lam < lam2))
        ymin = np.min(flux[idx])
        ymax = np.max(flux[idx])
        ax1.plot(lam[idx],flux[idx])
        ax1.set(xlabel='Wavelength',ylabel='Flux',title=filename,ylim=[ymin,ymax],xlim=[lam1,lam2])
        for ib in range(4):
            ax1.axvline(bandstart[ib],linestyle=':',color='b')
            ax1.axvline(bandend[ib],linestyle=':',color='b')
        
        py.draw()
        yn = input("Good? (Y/N)")
        if ((yn == 'y') | (yn == 'Y')):
            ftab.write(filename+' good\n')
        elif ((yn == 'n') | (yn == 'N')):
            ftab.write(filename+' bad\n')
        else:
            print('not classified:',filename)
                
        ax1.cla()

    ftab.close()
        
################################################################################
# fit a spectrophotometric standard using a black body dist. 
#

def fit_std_spec(filename,lam1=5200,lam2=8500,conv_ab=False):


    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(111)

    fig2 = py.figure(2)
    ax2 = fig2.add_subplot(111)

    # define locations of telluric abs bands:
    bandstart = np.array([6850.0,7130.0,7560.0,8100.0])
    bandend = np.array([6960.0,7360.0,7770.0,8360.0])
    nband = np.size(bandstart)

    outfile = filename+'_telcor'

    print('opening: ',filename)

    # the third col does not look like its actually the error:
    lam, flux, err = np.loadtxt(filename, usecols=(0, 1, 2), comments = ['#', '*', 'set'], unpack=True)

    # check if need to covert from AB mags to flux:
    if (conv_ab):
        # first get fnu:
        fnu = 3631 * 10.0**(-0.4*flux)
        # now get flam (erg/cm^2/s/A)
        flam = fnu /3.34e4/lam**2
        #flam = fnu * 299792458.0 / (lam/1.0e10)**2
        flux = flam
        
    newflux = np.copy(flux)
    
    binsize = (lam[-1]-lam[0])/(np.size(lam)-1)
    hbinsize = binsize/2.0
    print('mean bin size:',binsize)
    
    npix = np.size(lam)
    good = np.ones(npix)
    inband = np.zeros(npix)
    # set flag to useful or bad lam values
    for ip in range(npix):
        if ((lam[ip]<lam1) | (lam[ip]>lam2)):
            good[ip]=0

        for ib in range(nband):
            if ((lam[ip]>(bandstart[ib]-hbinsize)) & (lam[ip]<(bandend[ib]+hbinsize))):
                good[ip]=0
                inband[ip]=1

        # also ignore around Halpha:
        if (abs(lam[ip]-6563)<140.0):
            good[ip]=0
            
    idx = np.where((good==1))
    idx_band = np.where((inband==1))

    # Fit the data using a black body
    pstart = (10000.0,1.0e-13)
    (popt,cov)=sp.optimize.curve_fit(bb_fn,lam[idx],flux[idx],p0=pstart)
    print('black body fit, (temp, normalization):',popt)
    bb_model = bb_fn(lam,*popt)
    ratio = flux/bb_model

    # get RMS scatter of fractional difference in good regions:
    diff = (ratio-1.0)
    rms = np.sqrt((np.sum(diff[idx]**2))/np.size(lam[idx]))
    print('fractional RMS scatter from model: ',rms)

    # get RMS in the bands:
    rms_inband = np.sqrt((np.sum(diff[idx_band]**2))/np.size(lam[idx_band]))
    print('fractional RMS scatter from model: ',rms_inband)
    
    ymin = np.min(flux[idx])
    ymax = np.max(flux[idx])
    yrange = ymax-ymin
    ymin=ymin-0.2*yrange
    ymax=ymax+0.2*yrange
    ax1.plot(lam[idx],flux[idx],'o',color='g')

    # loop through points and replace those in the bands:
    for ip in range(npix):
        for ib in range(nband):
            if ((lam[ip]>(bandstart[ib]-hbinsize)) & (lam[ip]<(bandend[ib]+hbinsize))):
                newflux[ip] = bb_fn(lam[ip],*popt)

    ax1.plot(lam,flux,'o',color='r',label='original flux')
#    ax1.errorbar(lam,flux,err,fmt='o',color='r',label='original flux')
    ax1.plot(lam,bb_model,color='b',label='black body fit')
    ax1.plot(lam,newflux,'o',color='g',label='corrected fluxes')
    ax1.set(xlabel='Wavelength',ylabel='Flux',title=filename,ylim=[ymin,ymax],xlim=[lam1,lam2])
    for ib in range(4):
        if (ib == 0):
            ax1.axvspan(bandstart[ib],bandend[ib], alpha=0.3, color='m',label='telluric bands')
        else:
            ax1.axvspan(bandstart[ib],bandend[ib], alpha=0.3, color='m')
#        ax1.axvline(bandstart[ib],linestyle=':',color='k')
#        ax1.axvline(bandend[ib],linestyle=':',color='k')

    py.legend(prop={'size':8})

# also plot the data divided by the model to see how much it varies from the model:
    ax2.plot(lam,ratio,'o',color='r',label='(original flux)/(BB fit)')
    ax2.set(xlabel='Wavelength',ylabel='Flux/model',title=filename,xlim=[lam1,lam2])
    ax2.axhline(1.0,color='b')
    for ib in range(4):
        if (ib == 0):
            ax2.axvspan(bandstart[ib],bandend[ib], alpha=0.3, color='m',label='telluric bands')
        else:
            ax2.axvspan(bandstart[ib],bandend[ib], alpha=0.3, color='m')
    
    

    print('best fit BB temp:',popt[0]) 

    # finally reopen the file and read one line at a time, correcting if we need to:
    fin = open(filename,mode='r')
    fout = open(outfile,mode='w')

    line = fin.readline()
    fout.write(line)
    nlines=0
    while line:
        line=fin.readline()
        # need to check for comments here...
        if (('#' in line)):
            fout.write(line+'\n')
            continue
        # if its the end of the file, quit the loop
        if not line:
            break
        vals =  line.split()
        newflux1=vals[1]
        fflux1 = float(vals[1])
        lam1 = float(vals[0])
        # check if need to covert from AB mags to flux:
        if (conv_ab):
            # first get fnu:
            fnu = 3631 * 10.0**(-0.4*fflux1)
            # now get flam:
            flam = fnu * 299792458.0 / (lam1/1.0e10)**2
            flux = flam
            
        for ib in range(nband):
            if ((float(lam1)>(bandstart[ib]-hbinsize)) & (float(lam1)<(bandend[ib]+hbinsize))):
                fnewflux1 = bb_fn(lam1,*popt)
                newflux1 = '{0:e}'.format(fnewflux1)
                print('new flux at :',lam1,vals[1],newflux1)
                vals[1] = '{0:e}'.format(fnewflux1)
        nlines=nlines+1
        vals.append('\n')
        fout.write('     '.join(vals))

    fout.close()
    fin.close()
    print('corrected fluxes written to ',outfile)
    
##########################################################
# plot a test black body fn:
#
def plot_bb(temp=10000, peak = 10):

    x = np.linspace(1000,10000,100)

    bb = bb_fn(x,temp,peak)

    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.plot(x,bb)
    
        
###########################################################
# simple black body function (B_lam) with arbitrary scaling.
#
def bb_fn(x, *p):

    temp, peak = p

    k=1.380649e-23
    c = 299792458.0
    h = 6.62607015e-34

    hc_k = 6.62607015e-34 * 299792458.0 / 1.380649e-23

    # convert from angstroms to m
    lam = x/1.0e10
    
#    bb1 = peak * 2 * h * c**2 / lam**5
    
#    bb2 = 1 / (np.exp(hc_k/(lam*temp))-1)

    bb = peak * 2 * h * c**2 / (np.exp(hc_k/(lam*temp))-1) / lam**5

    return bb
    
###############################################################################
# 2, single Gaussian with constant continuum level
def gaussian_cont(xx, *p):
    
    gpeak, gcent, gsigma, cont = p
    yy = gaussian(xx,gpeak,gcent,gsigma)

    #    if (docont):
    yy = yy + cont

    return yy


###########################################################################
# check secondary calibration for many fields
#
def check_flux_cal_secondary_many(path,cor_ext=False,dopdf=True):

    # search through the directory structure for all the fields

    pathlist = glob.glob(path+'/reduced/[0-9]?????/Y*/Y*/main/ccd_1')

    nfields = np.size(pathlist)
    print('total number of fields to check:',nfields)
    br_rms = np.zeros(nfields)
    br_mad = np.zeros(nfields)
    brr_rms = np.zeros(nfields)
    brr_mad = np.zeros(nfields)
    airmass_min = np.zeros(nfields)
    airmass_max = np.zeros(nfields)
    nframes = np.zeros(nfields)
    rescale_range = np.zeros(nfields)
    fwhm = np.zeros(nfields)
    fpaths = np.empty(nfields,dtype='S256')

    nf = 0
    # for each path in the list, run the test:
    for path in pathlist:
        airmass_min[nf],airmass_max[nf],fwhm[nf],rescale_range[nf],br_rms[nf],br_mad[nf],brr_rms[nf],brr_mad[nf],nframes[nf] = check_flux_cal_secondary(path+'/??????????sci.fits',cor_ext=cor_ext,verbose=False,doplot=False)
        print(path,airmass_min[nf],airmass_max[nf],br_rms[nf],br_mad[nf],brr_rms[nf],brr_mad[nf],fwhm[nf],rescale_range[nf],nframes[nf])
        nf = nf + 1

    print('number of fields checked:',nf)

    if (dopdf):
        pdf = PdfPages('br_rms.pdf')

    # make a plot:
    idx = np.where(nframes>6)
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(231)
    ax1.plot(airmass_min[idx],br_rms[idx],'o')
    ax1.set(xlabel='airmass min',ylabel='rms in B/R flux')
    ax2 = fig1.add_subplot(232)
    ax2.plot(airmass_max[idx],br_rms[idx],'o')
    ax2.set(xlabel='airmass max',ylabel='rms in B/R flux')
    ax3 = fig1.add_subplot(233)
    ax3.plot(airmass_max[idx]-airmass_min[idx],br_rms[idx],'o')
    ax3.set(xlabel='airmass diff',ylabel='rms in B/R flux')
    ax4 = fig1.add_subplot(234)
    ax4.plot(fwhm[idx],br_rms[idx],'o')
    ax4.set(xlabel='seeing FWHM (arcsec)',ylabel='rms in B/R flux')
    ax5 = fig1.add_subplot(235)
    ax5.plot(rescale_range[idx],br_rms[idx],'o')
    ax5.set(xlabel='rescale range',ylabel='rms in B/R flux')

    if (dopdf):
        py.savefig(pdf, format='pdf')        
        pdf.close()

    if (dopdf):
        pdf = PdfPages('br_mad.pdf')
    
    fig2 = py.figure(2)
    ax2_1 = fig2.add_subplot(231)
    ax2_1.plot(airmass_min[idx],br_mad[idx],'o')
    ax2_1.set(xlabel='airmass min',ylabel='rms in B/R flux')
    ax2_2 = fig2.add_subplot(232)
    ax2_2.plot(airmass_max[idx],br_mad[idx],'o')
    ax2_2.set(xlabel='airmass max',ylabel='rms in B/R flux')
    ax2_3 = fig2.add_subplot(233)
    ax2_3.plot(airmass_max[idx]-airmass_min[idx],br_mad[idx],'o')
    ax2_3.set(xlabel='airmass diff',ylabel='rms in B/R flux')
    ax2_4 = fig2.add_subplot(234)
    ax2_4.plot(fwhm[idx],br_mad[idx],'o')
    ax2_4.set(xlabel='seeing FWHM (arcsec)',ylabel='rms in B/R flux')
    ax2_5 = fig2.add_subplot(235)
    ax2_5.plot(rescale_range[idx],br_mad[idx],'o')
    ax2_5.set(xlabel='rescale range',ylabel='rms in B/R flux')

    if (dopdf):
        py.savefig(pdf, format='pdf')        
        pdf.close()

    print('number of fields read:',nf)
    # finally, write a binary table of the results so we can keep them
    col1 = pf.Column(name='FieldPath', format='256A', array=fpaths[0:nf])
    col2 = pf.Column(name='AirmassMin', format='D', array=airmass_min[0:nf])
    col3 = pf.Column(name='AirmassMax', format='D', array=airmass_max[0:nf])
    col4 = pf.Column(name='Seeing', format='D', array=fwhm[0:nf],unit='arcsec')
    col5 = pf.Column(name='RescaleRange', format='D', array=rescale_range[0:nf])
    col6 = pf.Column(name='brRMS', format='D', array=br_rms[0:nf])
    col7 = pf.Column(name='brMAD', format='D', array=br_mad[0:nf])
    col8 = pf.Column(name='brrRMS', format='D', array=brr_rms[0:nf])
    col9 = pf.Column(name='brrMAD', format='D', array=brr_mad[0:nf])

    #cols = pf.ColDefs([col1, col2, col3, col4, col5, col6,col7,col8,col9])
    cols = pf.ColDefs([col2,col3,col4,col5,col6,col7,col8,col9])

    #Now, create a new binary table HDU object:
    tbhdu = pf.BinTableHDU.from_columns(cols)

    # finally write the table HDU to a file:
    outfile = 'fluxcal_secondary_check.fits'
    print('Writing results to ',outfile)
    tbhdu.writeto(outfile,overwrite=True)
 
    
###########################################################################
# get secondary flux standard spectra and do some comparisons:
#
# run on opus:
#  sami_dr_smc.sami_fluxcal.check_flux_cal_secondary('ccd_1/*sci.fits',sso_ext_tab='/suphys/scroom/lib/python/sami/standards/extinsso.tab') 
#
def check_flux_cal_secondary(infiles,cor_ext=False,verbose=True,doplot=True,sso_ext_tab='/Users/scroom/data/sami/test_data_v1/standards/extinsso.tab',dairmass=0.0):

    py.rc('text', usetex=False)

    # get the list of image files:
    files = glob.glob(infiles)
    nfiles = np.size(files)

    # set up colours:
    #cols=py.cm.rainbow(np.linspace(0,1,10)))
    cmap = py.get_cmap(py.cm.rainbow)
    cols = [cmap(i) for i in np.linspace(0, 1, 8)]

    
    # set up plotting:
    if (doplot):
        fig1 = py.figure(1)
        ax1_1 = fig1.add_subplot(311)
        ax1_2 = fig1.add_subplot(312)
        fig3 = py.figure(3)
        ax3_1 = fig3.add_subplot(211)
        ax3_2 = fig3.add_subplot(212)

    if (verbose):    
        print('number of files to read:',nfiles)

    spectra = np.zeros((2048,nfiles))
    spectra_r = np.zeros((2048,nfiles))
    specnames = np.empty(nfiles, dtype=object)
    med_b = np.zeros(nfiles)
    med_g = np.zeros(nfiles)
    med_r = np.zeros(nfiles)
    med_rr = np.zeros(nfiles)
    airmass_cent =  np.zeros(nfiles)
    airmass_1 =  np.zeros(nfiles)
    delta_airmass =  np.zeros(nfiles)
    fwhm =  np.zeros(nfiles)
    rescale =  np.zeros(nfiles)
    xcenref =  np.zeros(nfiles)
    ycenref =  np.zeros(nfiles)
    
    nf = 0
    for imfile in files:
        if (verbose):
            print('reading '+imfile)

        # this is running on the blue arm data by default, but also get the name
        # for the red arm:
        imfile_t = imfile.replace('ccd_1','ccd_2')
        imfile_red = imfile_t[:-13]+'2'+imfile_t[-12:]        
            
        hdulist = pf.open(imfile)
        hdulist_r = pf.open(imfile_red)
        primary_header=hdulist['PRIMARY'].header
        primary_header_r=hdulist_r['PRIMARY'].header
        utdate = primary_header['UTDATE']
        utstart = primary_header['UTSTART']
        utend = primary_header['UTEND']
        zdstart = primary_header['ZDSTART']
        hastart = primary_header['HASTART']
        lat_obs = primary_header['LAT_OBS']
        long_obs = primary_header['LONG_OBS']
        alt_obs = primary_header['ALT_OBS']
        exposed = primary_header['EXPOSED']
        meanra =  primary_header['MEANRA']
        meandec =  primary_header['MEANDEC']
        # skip if short exposure:
        if (exposed < 900):
            continue
        crval1=primary_header['CRVAL1']
        cdelt1=primary_header['CDELT1']
        crpix1=primary_header['CRPIX1']
        naxis1=primary_header['NAXIS1']
        x=np.arange(naxis1)+1
        L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
        lam=L0+x*cdelt1
        
        crval1_r=primary_header_r['CRVAL1']
        cdelt1_r=primary_header_r['CDELT1']
        crpix1_r=primary_header_r['CRPIX1']
        naxis1_r=primary_header_r['NAXIS1']
        x_r=np.arange(naxis1_r)+1
        L0_r=crval1_r-crpix1_r*cdelt1_r #Lc-pix*dL        
        lam_r=L0_r+x_r*cdelt1_r
        
        # if this is the first frame and we've now got the lam array, then get the extinction vector:
        # read in extinction file:
        wavelength_extinction, extinction_mags = fluxcal2.read_atmospheric_extinction(sso_extinction_table=sso_ext_tab)
        # interpolate extinction file to correct wavelength range
        extinction_mags_b = np.interp(lam, wavelength_extinction, 
                                extinction_mags)
        extinction_mags_r = np.interp(lam_r, wavelength_extinction, 
                                extinction_mags)

        # define observatory:
        siding_spring_loc = EarthLocation(lat=lat_obs*u.deg, lon=long_obs*u.deg, height=alt_obs*u.m)

        # define time:
        date = utdate.replace(':','-')
        time_start = date+' '+utstart
        time_end = date+' '+utend
        time1 = Time(time_start) 
        time2 = Time(time_end) 

        time_diff = time2-time1
        time_mid = time1 + time_diff/2.0
        
        # define coordinates:
        coords = SkyCoord(meanra*u.deg,meandec*u.deg) 

        # calculate zenith distance:
        altazpos1 = coords.transform_to(AltAz(obstime=time1,location=siding_spring_loc))   
        altazpos2 = coords.transform_to(AltAz(obstime=time2,location=siding_spring_loc))   
        altazpos_mid = coords.transform_to(AltAz(obstime=time_mid,location=siding_spring_loc))   

        #
        zd1 = 90.0-altazpos1.alt/u.deg
        zd2 = 90.0-altazpos2.alt/u.deg
        zd_mid = 90.0-altazpos_mid.alt/u.deg

        alt1 = float(90.0 - zd1) 
        alt2 = float(90.0 - zd2)
        alt_mid = float(90.0 - zd_mid)
        airmass1 = 1./ ( np.sin( ( alt1 + 244. / ( 165. + 47 * alt1**1.1 )
                            ) / 180. * np.pi ) )
        airmass2 = 1./ ( np.sin( ( alt2 + 244. / ( 165. + 47 * alt2**1.1 )
                            ) / 180. * np.pi ) )
        airmass_mid = 1./ ( np.sin( ( alt_mid + 244. / ( 165. + 47 * alt_mid**1.1 )
                            ) / 180. * np.pi ) )

        # geteffective airmass by simpsons rule integration:
        airmass = ( airmass1 + 4. * airmass_mid + airmass2 ) / 6.

        airmass_1[nf] = airmass1
        airmass_cent[nf] = airmass
        delta_airmass[nf] = airmass2-airmass

        specnames[nf] = imfile
        
        if (verbose):
            print('airmass at start, midpoint and finish: ',airmass1,airmass_mid,airmass2)
            print('effective airmass: ',airmass)
            print('difference between start and effective airmass: ',airmass2-airmass)

        # get the extinction for the start and cent airmass (blue):
        ext_mag_1_b = extinction_mags_b * airmass_1[nf]
        ext_mag_cent_b = extinction_mags_b * (airmass_cent[nf])
        ext_1_b = 10.0**(-0.4 * ext_mag_1_b)
        ext_cent_b = 10.0**(-0.4 * ext_mag_cent_b)
        # get the extinction for the start and cent airmass (red):
        ext_mag_1_r = extinction_mags_r * airmass_1[nf]
        ext_mag_cent_r = extinction_mags_r * (airmass_cent[nf])
        ext_1_r = 10.0**(-0.4 * ext_mag_1_r)
        ext_cent_r = 10.0**(-0.4 * ext_mag_cent_r)
        
        #plot extinction:
        if (doplot):
            ax3_1.plot(lam,ext_1_b,label=imfile+' start')
            ax3_1.plot(lam_r,ext_1_r)
            ax3_1.plot(lam,ext_cent_b,label=imfile+' cent')
            ax3_1.plot(lam_r,ext_cent_r)
            ax3_2.plot(lam,ext_1_b/ext_cent_b,label=imfile)
            ax3_2.plot(lam_r,ext_1_r/ext_cent_r)
        
        # get data from FLUX_CALIBRATION extension:
        try:
            fdata = hdulist['FLUX_CALIBRATION'].data
            fc_header=hdulist['FLUX_CALIBRATION'].header
        except KeyError:
            print('FLUX_CALIBRATION extension not found, skipping ',imfile)
            continue

        # get data from FLUX_CALIBRATION extension from red arm:
        try:
            fdata_r = hdulist_r['FLUX_CALIBRATION'].data
            fc_header_r=hdulist_r['FLUX_CALIBRATION'].header
        except KeyError:
            print('FLUX_CALIBRATION extension not found, skipping ',imfile_red)
            continue

        # try getting rescale values:
        try:
            probenum[nf]=fc_header['PROBENUM']
            rescale[nf]=fc_header['RESCALE']
            fwhm[nf]=fc_header['FWHM']
            xcenref[nf]=fc_header['XCENREF']
            ycenref[nf]=fc_header['YCENREF']
        except KeyError:
            print('RESCALE keyword not found for ',imfile)
            probenum[nf] = -1.0
            rescale[nf] = 1.0
            fwhm[nf] = 0.0
            xcenref[nf] = 0.0
            ycenref[nf] = 0.0

            
        ys,xs = np.shape(fdata)
        flux = fdata[0, :]
        sigma = fdata[1, :]
        flux_r = fdata_r[0, :]
        sigma_r = fdata_r[1, :]

        # if correcting extinction to central value, do it.  This
        # assumes the correction done was to the starting zd: 
        if (cor_ext):
            flux = flux * ext_1_b/ext_cent_b
            flux_r = flux_r * ext_1_r/ext_cent_r
            
        spectra[:,nf] = flux
        spectra_r[:,nf] = flux_r
        
        #medf_blue = get_med_flux_lam(lam,flux,3780.0,3920.0)
        medf_blue = get_med_flux_lam(lam,flux,3700.0,3900.0)
        medf = get_med_flux_lam(lam,flux,4500.0,4700.0)
        medf_red = get_med_flux_lam(lam,flux,5400.0,5550.0)
        medf_rred = get_med_flux_lam(lam_r,flux_r,6620.0,6820.0)
        med_b[nf] = medf_blue
        med_g[nf] = medf
        med_r[nf] = medf_red
        med_rr[nf] = medf_rred

        if (doplot):
            ax1_1.plot(lam,flux,label=imfile,color=cols[nf])
            ax1_1.plot(lam_r,flux_r,color=cols[nf])
            #ax1_1.set(xlim=[3600.0,5800.0])
            ax1_2.plot(lam,flux/medf,color=cols[nf])
            ax1_2.plot(lam_r,flux_r/medf,color=cols[nf])
            #ax1_2.set(xlim=[3600.0,5800.0])
        
            
        nf=nf+1

    if (doplot):
        ax1_1.legend(prop={'size':8})
        ax3_1.legend(prop={'size':8})
        ax3_2.legend(prop={'size':8})
    
    # estimate median flux:
    if (nf>0):
        medflux = np.nanmedian(spectra,axis=1)
        medflux_r = np.nanmedian(spectra_r,axis=1)
        medf_med = get_med_flux_lam(lam,medflux,4500.0,4700.0)


    # now plot ratios:
    if (doplot):
        ax1_3 = fig1.add_subplot(313)

    if (verbose):
        print('Filename       cen_am    del_am flux_b flux_g flux_r flux_rr   b/r    fwhm    rescale')
        
    for i in range(nf):
        if (doplot):
            med_spec = sami_stats_utils.median_filter_nan((spectra[:,i]/med_g[i])/(medflux/medf_med),51)
            med_spec_r = sami_stats_utils.median_filter_nan((spectra_r[:,i]/med_g[i])/(medflux_r/medf_med),51)
            #ax2.plot(lam,(spectra[:,i]/med_g[i])/(medflux/medf_med))
            ax1_3.plot(lam,med_spec,color=cols[i],label=files[i])
            ax1_3.plot(lam_r,med_spec_r,color=cols[i],label=files[i])
        if (verbose):    
            print('{0} {1:5.3f} {2:5.3f} {3:6.2f} {4:6.2f} {5:6.2f} {6:6.2f} {7:6.3f} {8:5.3f} {9:6.3f}'.format(specnames[i],airmass_cent[i],delta_airmass[i],med_b[i],med_g[i],med_r[i],med_rr[i],med_b[i]/med_r[i],fwhm[i],rescale[i]))
        

    # plot airmass vs differences, and other distributions.
    if (doplot):
        fig2 = py.figure(2)

        ax2_1 = fig2.add_subplot(2,3,1)
        ax2_1.plot(airmass_cent[0:nf],med_b[0:nf]/med_r[0:nf],'o')
        ax2_1.set(xlabel='central airmass',ylabel='(blue flux)/(red flux)')
    
        ax2_2 = fig2.add_subplot(2,3,2)
        ax2_2.plot(delta_airmass[0:nf],med_b[0:nf]/med_r[0:nf],'o')
        ax2_2.set(xlabel='delta airmass',ylabel='(blue flux)/(red flux)')

        ax2_3 = fig2.add_subplot(2,3,3)
        ax2_3.hist(med_b[0:nf],histtype='step')
        ax2_3.set(xlabel='blue flux',ylabel='Number')

        ax2_4 = fig2.add_subplot(2,3,4)
        ax2_4.plot(rescale[0:nf],med_b[0:nf]/med_r[0:nf],'o')
        ax2_4.set(xlabel='rescale',ylabel='(blue flux)/(red flux)')

        ax2_5 = fig2.add_subplot(2,3,5)
        ax2_5.plot(fwhm[0:nf],med_b[0:nf]/med_r[0:nf],'o')
        ax2_5.set(xlabel='seeinf FWHM (arcsec)',ylabel='(blue flux)/(red flux)')
        
        ax2_6 = fig2.add_subplot(2,3,6)
        ax2_6.scatter(xcenref[0:nf],ycenref[0:nf],c=med_b[0:nf]/med_r[0:nf],marker='o')
        ax2_6.set(xlabel='X cen',ylabel='Y cen')

        
    if (verbose):
        print('number of files read: ',nf)
        print('mean and std of blue flux:',np.nanmean(med_b[0:nf]),np.nanstd(med_b[0:nf]))
        print('mean and std of green flux:',np.nanmean(med_g[0:nf]),np.nanstd(med_g[0:nf]))
        print('mean and std of red flux:',np.nanmean(med_r[0:nf]),np.nanstd(med_r[0:nf]))
        print('mean and std of blue/red flux:',np.nanmean(med_b[0:nf]/med_r[0:nf]),np.nanstd(med_b[0:nf]/med_r[0:nf]))
        print('mean and std of blue/rred flux:',np.nanmean(med_b[0:nf]/med_rr[0:nf]),np.nanstd(med_b[0:nf]/med_rr[0:nf]))
        
        
    if (nf>0):
        br_rms = np.nanstd(med_b[0:nf]/med_r[0:nf])
        brr_rms = np.nanstd(med_b[0:nf]/med_rr[0:nf])
        rescale_range = np.max(rescale[0:nf])-np.min(rescale[0:nf])
        br_mad = sp.stats.median_absolute_deviation((med_b[0:nf]/med_r[0:nf]))
        brr_mad = sp.stats.median_absolute_deviation((med_b[0:nf]/med_rr[0:nf]))
        if (verbose):
            print('MAD blue/red flux: ',br_mad)
            print('MAD blue/rred flux: ',brr_mad)
        
        return np.min(airmass_cent[0:nf]),np.max(airmass_cent[0:nf]),np.nanmean(fwhm[0:nf]),rescale_range,br_rms,br_mad,brr_rms,brr_mad,nf
    else:
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,nf
    
    
    
###########################################################################
# get secondary flux standard spectra and do some comparisons:
#
def flux_cal_secondary_vs_pickles(infiles):

    # get the list of image files:
    files = glob.glob(infiles)

    # read in Pickles stars:
    pickles_path = '/Users/scroom/data/sami/fluxcal/pickles/'
    pickles_list = ['f0v.dat','f2v.dat','f5v.dat','wf5v.dat','f6v.dat','rf6v.dat','f8v.dat','wf8v.dat','rf8v.dat']

    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)

    pn = len(pickles_list)

    
    ip = 0
    for pfile in pickles_list:
        itemp = np.loadtxt(pickles_path+pfile)
        (nx,ny) = np.shape(itemp)
         # define the size of arrays:
        if (ip == 0):
            plam = np.zeros((nx,pn))
            pflux = np.zeros((nx,pn))

        
        plam[:,ip] = itemp[:,0]
        pflux[:,ip] = itemp[:,1]
        print(pfile,np.shape(itemp))
        
        ax1.plot(plam[:,ip],pflux[:,ip],label=pfile)
        ax1.set(xlim=[3600.0,7500.0])

        if (ip > 0):
            ax2.plot(plam[:,ip],pflux[:,ip]/pflux[:,ip-1])
            ax2.set(xlim=[3600.0,7500.0],ylim=[0.8,1.2])
            
        ip = ip +1
    
    
    ax1.legend(prop={'size':8})
        
    # set up plotting:
    fig2 = py.figure(2)

    nf = 0
    for imfile in files:
        print('reading '+imfile)
        hdulist = pf.open(imfile)
        primary_header=hdulist['PRIMARY'].header
        utstart = primary_header['UTSTART']
        exposed = primary_header['EXPOSED']
        # skip if short exposure:
        if (exposed < 900):
            continue
        crval1=primary_header['CRVAL1']
        cdelt1=primary_header['CDELT1']
        crpix1=primary_header['CRPIX1']
        naxis1=primary_header['NAXIS1']
        x=np.arange(naxis1)+1
        L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
        lam=L0+x*cdelt1

        # get data from FLUX_CALIBRATION extension:
        try:
            fdata = hdulist['FLUX_CALIBRATION'].data
        except KeyError:
            print('FLUX_CALIBRATION extension not found, skipping ',imfile)
            continue

        flux = fdata[0, :]
        sigma = fdata[1, :]

        #plot
        ax22 = fig2.add_subplot(2,1,1)

        # get median flux in a wavelength range:
        medf = get_med_flux_lam(lam,flux,4500.0,4700.0)
        print(medf)
        
        # loop, plotting different templates
        while True:
        #
            #ax22.draw()
            tnum = input('template number? (y/n):')
            itnum=int(tnum)
            medf_temp = get_med_flux_lam(plam[:,itnum],pflux[:,itnum],4500.0,4700.0)
            print(medf_temp)
            ax22.clear()
            ax22.plot(lam,flux)
            ax22.set(xlim=[3600.0,5800.0])
            ax22.plot(plam[:,itnum],pflux[:,itnum]*medf/medf_temp)
            py.draw()
            
            if (itnum == 99):
                break

        
        
        if (nf == 0):
            flux1 = np.copy(flux)
            lam1 = np.copy(lam)

        ratio = flux/flux1
        #plot
        py.subplot(2,1,2)
        py.plot(lam,ratio)

            
        nf=nf+1
        


###########################################################################
# script to get name, ra, dec from aperture spectra:
#
def get_apspec_coords(inlist):

    # glob the list
    samifiles = glob.glob(inlist)

    # open output file:
    ftab = open('apspec_coords.txt',mode='w')
    
    for infile in samifiles:

        hdulist = pf.open(infile)
        primary_header=hdulist['PRIMARY'].header
        ra_sami = primary_header['CRVAL1']
        dec_sami = primary_header['CRVAL2']
        catid = primary_header['NAME']

        print(catid,ra_sami,dec_sami)
        outstr = '{0:12s} {1:11.7f} {2:11.7f}\n'.format(catid,ra_sami,dec_sami)
        ftab.write(outstr)
        hdulist.close()
        
    ftab.close()
    
        



###########################################################################
# get the median flux ove a given wavelength range
#
def get_med_flux_lam(lam,flux,l1,l2):

    n = 0
    nx = np.size(lam)
    medarr = np.zeros(nx)
    for i in range(nx):
        if ((lam[i] > l1) & (lam[i] < l2)):
                medarr[n] = flux[i]
                n = n + 1

    medflux = np.nanmedian(medarr[0:n])

    return medflux
    



###########################################################################
# plot aperture spectra:
#
def plot_ap_spec(apspeclist,doareacorr=False,apname='3_ARCSECOND',check=False,lmin=3600.0,lmax=7500.0):

    """Plot a list of aperture spectra.  If check is true, then write a file
    with a flag for each spectrum to say if there is a problem or some other
    issue is present.  Only run on the blue files, and the code will automatically
    find the red file."""
    
    py.rc('text', usetex=True)
    py.rc('font', family='sans-serif')
    py.rcParams.update({'font.size': 12})
    py.rcParams.update({'lines.linewidth': 1})    

    samifiles = sorted(glob.glob(apspeclist))

    # open the list file:
    if (check):
        outfile = 'ap_spec_flag.txt'
        # check if exists:
        if (exists(outfile)):
            yn = input('Old input file exists, do you want to append new data to this (y/n)? (if not will overwrite the file)')
            if (yn.lower() == 'y'):
                f = open(outfile,'a')
            else:
                f = open(outfile,'w')
        else:
            f = open(outfile,'w')
        
    
    nsami=0
    i = 0
    nfiles = np.size(samifiles)
    while i < nfiles:
        samifile = samifiles[i]
        # skip if this is a red file:
        if (samifile.find('red') > 0):
            continue

        hdulist = pf.open(samifile)
        # get just the file name (no path):
        fname = basename(samifile)
        # get the object plus field name for this spectrum:
        tmpstr = fname.split('_')
        objfname = '_'.join(tmpstr[0:6])
        fieldname = '_'.join(tmpstr[3:6])
        print('spectrum ',nsami,' of ',nfiles,'   ',samifile)
        print(i,objfname,fieldname)
        
        primary_header=hdulist['PRIMARY'].header

        # get keywords:
        ifuprobe = primary_header['IFUPROBE']
        
        sami_flux_blue,sami_lam_blue = sami_read_apspec(hdulist,apname,doareacorr=doareacorr)
        hdulist.close()

        # now read in the matching SAMI red arm data:
        samifile_red = samifile.replace('blue','red')
        hdulist = pf.open(samifile_red)
        sami_flux_red,sami_lam_red = sami_read_apspec(hdulist,apname,doareacorr=doareacorr)
        hdulist.close()

        py.figure(1)
        py.clf()
        py.subplot(1,1,1)
            
        # plot SAMI:
        py.plot(sami_lam_blue,sami_flux_blue,color='b')
        py.plot(sami_lam_red,sami_flux_red,color='r')
        #py.ylabel('Fslux (1E-17 erg/cm$^2$/s/Ang)')
        py.xlabel('Wavelength (Angstroms)')
        py.axvline(3700.0,linestyle=':',color='k')
        py.axvline(5750.0,linestyle=':',color='k')
        py.axvline(6300.0,linestyle=':',color='k')
        py.axvline(7400.0,linestyle=':',color='k')
        ax = py.gca()
        label = samifile.replace('_','\_')
        py.text(0.05, 0.95,label, horizontalalignment='left',verticalalignment='center',transform=ax.transAxes)
        py.axhline(0.0,linestyle=':',color='k')
        py.xlim(lmin,lmax)
        #py.title(samifile+' Probe:'+ifuprobe)
        py.draw()


        # if check is true, than allow different flags to be set:
        # flags:
        # q: quasar

        flag = 0
        comment = ''
        
        if (check):
            wr = True
            rep = True
            while (rep):
                yn =  input('enter flags or data quality:')

                if (yn.lower() == 'h'):
                    print('options:')
                    print('q: quasar, object has broad AGN lines, e.g. type 1 AGN.')
                    print('f: apparent general flux cal error.') 
                    print('s: apparent step between red and blue flux cal.')
                    print('p: some other problem.')
                    print('a: all okay, nothing to report.')
                    print('b: go back by 1')
                    print('c: add comment')
                    print('h: help, print this list')
                    print('e: exit and save')
                elif (yn.lower() == 'b'):
                    flag = 0
                    rep = False
                    wr = False
                    i = i - 2
                    if (i < -1):
                        i = -1
                elif (yn.lower() == 'a'):
                    flag = 0
                    rep = False
                elif (yn.lower() == 'q'):
                    flag = 1
                    rep = False
                elif (yn.lower() == 'f'):
                    flag = 2
                    rep = False
                elif (yn.lower() == 's'):
                    flag = 3
                    rep = False
                elif (yn.lower() == 'p'):
                    flag = 4
                    rep = False
                elif (yn.lower() == 'c'):
                    comment = input('Add comment:')
                    rep = False
                elif (yn.lower() == 'e'):
                    f.close()
                    return
                else:
                    print('Input not recognised...')
                    rep = True

                    
            # now write the file:
            if (wr):
                outstring = samifile+' '+str(flag)+' '+comment+'\n'
                print(outstring+'\n')
                f.write(outstring)

            
        else:
    # pause for input if plotting all the spectra:
            yn = input('Continue? (y/n):')

        # increment counter:
        i = i + 1
        nsami = nsami + 1

    f.close()

######################################################
# handy function to round to given number of sig fig:
def Round_n_sig_dig(x, n):

    xr = (np.floor(np.log10(np.abs(x)))).astype(int)
    xr=10.**xr*np.around(x/10.**xr,n-1)   
    return xr

###########################################################################
# fit SDSS vs SAMI flux ratio with a polynomial:
#
def fit_sami_sdss_fluxratio(infile,compfile='/Users/scroom/data/sami/fluxcal/spec_over_model_residuals.fits',doplot=True,order=5,dopdf=True):

    # set up plotting:
    py.rc('text', usetex=True)
    py.rcParams.update({'font.size': 14})
    py.rcParams.update({'figure.autolayout': True})
    # this to get sans-serif latex maths:
    py.rcParams['text.latex.preamble'] = [
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmathfonts}'  # load up the sansmath so that math -> helvet
        ]  

    

    
    df = pd.read_csv(infile)
    lam = df['Wavelength'].to_numpy()
    ratio = df['Ratio'].to_numpy()
    ratio_err_m =  df['Ratio_err_minus'].to_numpy()
    ratio_err_p =  df['Ratio_err_plus'].to_numpy()
    err = ((ratio_err_p-ratio_err_m)/2.0)/30.0

    # read in the comparison file, that is FITS format and generated by Sam Vaughan.
    # this is from a comparison to stellar population models.
    hdulist = pf.open(compfile)
    comp_lam = hdulist['WAVE'].data
    comp_med = hdulist['MEDIAN'].data
    comp_p16 = hdulist['PERCENTILE_16'].data
    comp_p84 = hdulist['PERCENTILE_84'].data

    good_comp = np.where((comp_med>0.97) & (comp_med < 1.04))

    comp_med_smooth = median_filter_nan(comp_med,21)
    # only define the smoothed case over a fixed wavelength range:
    comp_med_smooth = np.where(((comp_lam>3710.0) & (comp_lam < 5760.0) | ((comp_lam>6305.0) & (comp_lam<7420.0))),comp_med_smooth,np.nan)

    # get the standard SAMI wavelength solution from an aperture spectrum:
    apspec_blue = '/Users/scroom/data/sami/fluxcal/dr3_aperture_spec_v9/99795_blue_6_Y16SAR4_P002_12T100_2017_04_19-2017_05_01_apspec.fits'
    apspec_red = '/Users/scroom/data/sami/fluxcal/dr3_aperture_spec_v9/99795_red_6_Y16SAR4_P002_12T100_2017_04_19-2017_05_01_apspec.fits'

    hdulist = pf.open(apspec_blue)
    sami_flux_blue,sami_lam_blue = sami_read_apspec(hdulist,'3_ARCSECOND',doareacorr=False)
    # get WCS:
    w_blue_header = hdulist['3_ARCSECOND'].header
    hdulist.close()
    
    hdulist = pf.open(apspec_red)
    sami_flux_red,sami_lam_red = sami_read_apspec(hdulist,'3_ARCSECOND',doareacorr=False)
    # get WCS:
    w_red_header = hdulist['3_ARCSECOND'].header
    hdulist.close()

    # interpolate onto the default SAMI wavelength solution:
    comp_med_smooth_blue = spectres(sami_lam_blue,comp_lam,comp_med_smooth)
    comp_med_smooth_red = spectres(sami_lam_red,comp_lam,comp_med_smooth)

    # plot new model on SAMI scale, just to check its okay:
    if (doplot):
        fig3 = py.figure(3)
        ax3 = fig3.add_subplot(1,1,1)
        ax3.plot(sami_lam_blue,comp_med_smooth_blue,'-',color='b')
        ax3.plot(sami_lam_red,comp_med_smooth_red,'-',color='r')
    
    # make new HDU container:
    new_hdul = pf.HDUList()
    new_hdul.append(pf.ImageHDU())
    new_hdul.append(pf.ImageHDU(data=comp_med_smooth_blue,name='BLUECORR'))
    new_hdul.append(pf.ImageHDU(data=comp_med_smooth_red,name='REDCORR'))
    # add wcs:
    new_hdul['BLUECORR'].header['CRPIX1'] = w_blue_header['CRPIX1']
    new_hdul['BLUECORR'].header['CRVAL1'] = w_blue_header['CRVAL1']
    new_hdul['BLUECORR'].header['CDELT1'] = w_blue_header['CDELT1']
    new_hdul['BLUECORR'].header['CUNIT1'] = w_blue_header['CUNIT1']
    new_hdul['BLUECORR'].header['CTYPE1'] = w_blue_header['CTYPE1']
    new_hdul['BLUECORR'].header['WCSAXES'] = w_blue_header['WCSAXES']
    new_hdul['REDCORR'].header['CRPIX1'] = w_red_header['CRPIX1']
    new_hdul['REDCORR'].header['CRVAL1'] = w_red_header['CRVAL1']
    new_hdul['REDCORR'].header['CDELT1'] = w_red_header['CDELT1']
    new_hdul['REDCORR'].header['CUNIT1'] = w_red_header['CUNIT1']
    new_hdul['REDCORR'].header['CTYPE1'] = w_red_header['CTYPE1']
    new_hdul['REDCORR'].header['WCSAXES'] = w_red_header['WCSAXES']

    
    new_hdul.writeto('sami_fcal_ripple_corr.fits',overwrite=True)
    
    #calculate percentiles of the comp spectrum:
    m1sig = np.nanpercentile(comp_med[good_comp],15.9)
    p1sig = np.nanpercentile(comp_med[good_comp],84.1)
    m2sig = np.nanpercentile(comp_med[good_comp],2.27)
    p2sig = np.nanpercentile(comp_med[good_comp],97.7)
    med = np.nanmedian(comp_med[good_comp])
    print(m2sig,m1sig,med,p1sig,p2sig)
    
    # define indices that are good:
    idx = np.where((np.isfinite(ratio) & (lam > 3900))) 
    idx_b = np.where((np.isfinite(ratio) & (lam > 3900) & (lam < 6000))) 
    idx_r = np.where((np.isfinite(ratio) & (lam > 6000))) 

    # fit a simply poylnomial:
    z,z_cov=np.polyfit(lam[idx]/5500,ratio[idx],order,w=1/err[idx],cov='unscaled')
    p = np.poly1d(z)
    z_rnd = Round_n_sig_dig(z,6)
    p_rnd = np.poly1d(z_rnd)
    print(z)
    print(z_cov)
    for i in range(np.size(p)):
        print(i,z[i],z_rnd[i],z_cov[i,i])

    print('ratio at 4000A:',p(4000/5500))
    print('ratio at 4500A:',p(4500/5500))
    print('ratio at 5000A:',p(5000/5500))

    # generate a ratio corrected by the polynomial:
    ratio_corr = ratio/p(lam/5500.0)
    
    if (doplot):
        if (dopdf):
            pdf = PdfPages('sami_sdss_flux_ratio_fit.pdf')
        fig1 = py.figure(1)
        ax1 = fig1.add_subplot(1,1,1)
        ax1.plot(lam[idx_b],ratio[idx_b],'-',color='r',label='Median flux ratio')
        ax1.plot(lam[idx_r],ratio[idx_r],'-',color='r')
        ax1.axhline(1.0,linestyle='--',color='k')
        ax1.plot(lam[idx_b],ratio_err_m[idx_b],':',color='r',label='68th percentile range')
        ax1.plot(lam[idx_b],ratio_err_p[idx_b],':',color='r')
        ax1.plot(lam[idx_r],ratio_err_m[idx_r],':',color='r')
        ax1.plot(lam[idx_r],ratio_err_p[idx_r],':',color='r')
        lam_m = np.linspace(3600.0,7600.0,10.0)
        ax1.plot(lam_m,p(lam_m/5500),'-',color='b',label='Polynomial fit')
        #ax1.plot(lam_m,p_rnd(lam_m/5500),'-',color='g')
        ax1.set(xlim=[3700.0,7500.0],ylim=[0.75,1.1],xlabel='Wavelength [\AA]',ylabel='SAMI/SDSS 3-arcsec fibre flux')
        ax1.legend(loc='lower right')
        if (dopdf):
            py.savefig(pdf, format='pdf')        
            pdf.close()

        # plot a second plot comparing with the stellar pop comparison
        # spectrum:
        if (dopdf):
            pdf = PdfPages('sami_sdss_flux_ratio_fit_sp.pdf')
        fig2 = py.figure(2)
        ax2 = fig2.add_subplot(1,1,1)
        ax2.plot(lam[idx_b],ratio_corr[idx_b],'-',color='r',label='Median SAMI/SDSS')
        ax2.plot(lam[idx_r],ratio_corr[idx_r],'-',color='r')
        ax2.axhline(1.0,linestyle='--',color='k')
        ax2.plot(comp_lam,comp_med,'-',color='b',label='Median SAMI/(SSP fits)')
        #ax2.plot(comp_lam,comp_p16,':',color='b')
        #ax2.plot(comp_lam,comp_p84,':',color='b')
        ax2.plot(comp_lam,comp_med_smooth,'-',color='cyan',label='Median SAMI/(SSP fits) smoothed')
        ax2.set(xlim=[3650.0,7500.0],ylim=[0.96,1.04],xlabel='Wavelength [\AA]',ylabel='Flux ratio')

        ax2.legend(loc='upper right')
        if (dopdf):
            py.savefig(pdf, format='pdf')        
            pdf.close()
            
###########################################################################
# SDSS vs. SAMI spectral flux calibration tests comparing aperture spectra
# from SAMI and fibre spectra from SDSS:
#
# sami_dr_smc.sami_fluxcal.sami_sdss_specflux_comp('dr3_aperture_spec_v2/2016_02/*.fits','sdss_spec/*.fits',samilist2='dr2_aperture_spec_v2/*.fits',plotall=True)  
#
# sami_dr_smc.sami_fluxcal.sami_sdss_specflux_comp('dr3_aperture_spec_v2/2016_02/*P004_12T086*.fits','sdss_spec/*.fits',samilist2='dr2_aperture_spec_v2/*.fits',plotall=True,fibmagsfile='sami_gama_fibpsfmags_scroom.fits')                  
#
# sami_tools_smc.dr_tools.sami_fluxcal.sami_sdss_specflux_comp('dr3_aperture_spec_v8/*2015*.fits','sdss_spec2/*.fits',samilist2='dr2_aperture_spec_v2/*.fits',plotall=True,fibmagsfile='sami_gama_fibpsfmags_scroom.fits')
#                
#
#
def sami_sdss_specflux_comp(samilist,sdsslist,samilist2=None,plotall=True,plot=True,doareacorr=False,apname='3_ARCSECOND',fibmagsfile=None,dopdf=True,sdssplate=None,verbose=False,smooth=51):

    # set up formating for pdfs for paper:
    if (dopdf):
        py.rc('text', usetex=True)
        py.rcParams.update({'font.size': 14})
        py.rcParams.update({'lines.linewidth': 1})
        py.rcParams.update({'figure.autolayout': True})
        # this to get sans-serif latex maths:
        py.rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
            r'\usepackage{helvet}',    # set the normal font here
            r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
            r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
            ]  
    else:
        py.rc('text', usetex=True)
        py.rc('font', family='sans-serif')
        py.rcParams.update({'font.size': 8})
        py.rcParams.update({'lines.linewidth': 1})

    
    # define some parameters:
    # max radial separation for matching (in arcsec):
    maxrad = 3.0
    maxrad = maxrad/3600 # (convert to degrees)
    # min/max wavelength range for plotting:
    xmin=3650
    xmax=7500
    
    # glob the file lists:
    samifiles = glob.glob(samilist)
    sdssfiles = glob.glob(sdsslist)

    # get the directory name
    dir = dirname(samifiles[0])
    
    # if the second SAMI list isn't 'None', then glob that as well:
    if (samilist2 != None):
        dosami2 = True
        samifiles2 = glob.glob(samilist2)
    else:
        dosami2 = False

    # open output files that will list bad objects identified.
    out = open('sami_sdss_comp.txt',mode='w')
    out_bad = open('sami_sdss_bad.txt',mode='w')
        

    if (fibmagsfile != None):
        dofibmags = True
        # read in the fibre mags:
        hdutab = pf.open(fibmagsfile)
        tbdata = hdutab[1].data
        fibmag_catid = tbdata['CATID']
        fibmag_u = tbdata['fiberMag_u']
        fibmag_g = tbdata['fiberMag_g']
        fibmag_r = tbdata['fiberMag_r']
        fibmag_i = tbdata['fiberMag_i']
        fibmag_z = tbdata['fiberMag_z']
        psfmag_u = tbdata['psfMag_u']
        psfmag_g = tbdata['psfMag_g']
        psfmag_r = tbdata['psfMag_r']
        psfmag_i = tbdata['psfMag_i']
        psfmag_z = tbdata['psfMag_z']

        # convert fib mags (that are approx AB) to f_lam fluxes:
        fibmag_flux_u = (10.0**(-0.4*(fibmag_u-8.90)))/3.34e4/(3557.0**2)*1.0e17
        fibmag_flux_g = (10.0**(-0.4*(fibmag_g-8.90)))/3.34e4/(4702.0**2)*1.0e17
        fibmag_flux_r = (10.0**(-0.4*(fibmag_r-8.90)))/3.34e4/(6175.0**2)*1.0e17
        fibmag_flux_i = (10.0**(-0.4*(fibmag_i-8.90)))/3.34e4/(7491.0**2)*1.0e17
        fibmag_flux_z = (10.0**(-0.4*(fibmag_z-8.90)))/3.34e4/(8946.0**2)*1.0e17
        psfmag_flux_u = (10.0**(-0.4*(psfmag_u-8.90)))/3.34e4/(3557.0**2)*1.0e17
        psfmag_flux_g = (10.0**(-0.4*(psfmag_g-8.90)))/3.34e4/(4702.0**2)*1.0e17
        psfmag_flux_r = (10.0**(-0.4*(psfmag_r-8.90)))/3.34e4/(6175.0**2)*1.0e17
        psfmag_flux_i = (10.0**(-0.4*(psfmag_i-8.90)))/3.34e4/(7491.0**2)*1.0e17
        psfmag_flux_z = (10.0**(-0.4*(psfmag_z-8.90)))/3.34e4/(8946.0**2)*1.0e17

        nfm = np.size(fibmag_u)
        
        
        
    else:
        dofibmags = False
        
    maxsami = np.size(samifiles)
    maxsdss = np.size(sdssfiles)

    print('Number of SAMI files:',maxsami)
    print('Number of SDSS files:',maxsdss)

    
    # define arrays:
    ra_sdss = np.zeros(maxsdss)
    dec_sdss = np.zeros(maxsdss)
    plate_sdss = np.empty(maxsdss,dtype='U256')
    # type here is U for unicode, if we use S then this makes a byte array
    # in python 3.
    filename_sdss = np.empty(maxsdss,dtype='U256')
    ra_sami = np.zeros(maxsami)
    dec_sami = np.zeros(maxsami)
    filename_sami = np.empty(maxsami,dtype='U256')
    fieldname_sami = np.empty(maxsami,dtype='U256')
    plate_sdss_sami = np.empty(maxsami,dtype='U256')
    filename_sdss_sami = np.empty(maxsami,dtype='U256')
    
    ratio_med = np.zeros(maxsami)
    ratio_med2 = np.zeros(maxsami)
    ratio_blue_med = np.zeros(maxsami)
    ratio_red_med = np.zeros(maxsami)
    ratio_blue_med2 = np.zeros(maxsami)
    ratio_red_med2 = np.zeros(maxsami)
    posdiff = np.zeros(maxsami)
    ifuprobe_sami = np.zeros(maxsami)
    psffwhm_sami = np.zeros(maxsami)

    rdiff = np.zeros(maxsami)
    rdiff2 = np.zeros(maxsami)
    bluegrad = np.zeros(maxsami)
    bluegrad2 = np.zeros(maxsami)
    bluegradbr = np.zeros(maxsami)
    bluegradbr2 = np.zeros(maxsami)
    ratio4000 =  np.zeros(maxsami)
    ratio5450 =  np.zeros(maxsami)
    ratio7000 =  np.zeros(maxsami)
    ratio4000_2 =  np.zeros(maxsami)
    ratio5450_2 =  np.zeros(maxsami)
    ratio7000_2 =  np.zeros(maxsami)
    sdss_seeing = np.zeros(maxsami)
    ra_plate_sdss =  np.zeros(maxsami)
    dec_plate_sdss =  np.zeros(maxsami)
    
    # define arrays to hold all ratios:
    ratio_all_blue = np.zeros((maxsami,2048))
    ratio_all_red = np.zeros((maxsami,2048))
    ratio_all_blue_probe = np.zeros((maxsami,13,2048))
    ratio_all_red_probe = np.zeros((maxsami,13,2048))
    nprobe = np.zeros(13,dtype='i4')
    
    # we need to match the SDSS file to a SAMI file, so do this
    # based on RA/Dec.  First read in all SDSS files and get the ra,dec:
    nsdss=0
    for sdssfile in sdssfiles:
        hdulist = pf.open(sdssfile)
        primary_header=hdulist['PRIMARY'].header
        filename_sdss[nsdss] = sdssfile
        ra_sdss[nsdss] = primary_header['PLUG_RA']
        dec_sdss[nsdss] = primary_header['PLUG_DEC']
        plateid =  primary_header['PLATEID']
        mjd =  primary_header['MJD']
        plate_sdss[nsdss] = str(plateid)+'-'+str(mjd)
        if (verbose):
            print(nsdss, filename_sdss[nsdss],ra_sdss[nsdss],dec_sdss[nsdss],plate_sdss[nsdss])
        nsdss = nsdss + 1
        
    #
    # now loop through each of the SAMI frames.  We will only loop over
    # blue frames, but assume the red is also there and read it.
    nsami=0
    for samifile in samifiles:
        # skip if this is a red file:
        if (samifile.find('red') > 0):
            continue

        print('processing: ',nsami,' of ',maxsami,'  ',samifile)
        hdulist = pf.open(samifile)
        # get just the file name (no path):
        fname = basename(samifile)
        # get the object plus field name for this spectrum:
        tmpstr = fname.split('_')
        objfname = '_'.join(tmpstr[0:6])
        fieldname = '_'.join(tmpstr[3:6])
        fieldname_sami[nsami] = fieldname
        
        primary_header=hdulist['PRIMARY'].header
        filename_sami[nsami] = samifile
        ra_sami[nsami] = primary_header['CRVAL1']
        dec_sami[nsami] = primary_header['CRVAL2']
        catid = primary_header['NAME']
        # get seeing and IFU probe number:
        tmp_probe =  primary_header['IFUPROBE']
        try:
            float(tmp_probe)
            ifuprobe_sami[nsami] = float(tmp_probe)
        except ValueError:
            print('Probe Not a float: ',tmp_probe)
            ifuprobe_sami[nsami] = -1.0

        try:
            psffwhm_sami[nsami] = primary_header['PSFFWHM']
        except KeyError:
            psffwhm_sami[nsami] = -1.0

        # find the SDSS fiber mags:
        if (dofibmags):
            foundmag = False
            for ifm in range(nfm):
                if (str(fibmag_catid[ifm]) == str(catid)):
                    fu = fibmag_flux_u[ifm]
                    fg = fibmag_flux_g[ifm]
                    fr = fibmag_flux_r[ifm]
                    fi = fibmag_flux_i[ifm]
                    if (verbose):
                        print('fibre mags (ugri):',fibmag_flux_u[ifm],fibmag_flux_g[ifm],fibmag_flux_r[ifm],fibmag_flux_i[ifm])
                    foundmag = True
            
            
        # loop over SDSS spectra and find an RA/Dec match:
        sel = -1
        mindiff = 1.0e10
        for i in range(nsdss):
            diff = np.sqrt((ra_sami[nsami]-ra_sdss[i])**2 + (dec_sami[nsami]-dec_sdss[i])**2)
            if (diff < maxrad):
                mindiff = diff
                sel = i
                break

        posdiff[nsami] = mindiff
        # only process if we find a match:
        if (sel == -1):
            if (verbose):
                print('No match found, skipping...')
            continue
        # read in the data for the SAMI spectrum:
        # SAMI data has flux units of 1E-16 erg/s/cm**2/ang/pixel
        sami_flux_blue,sami_lam_blue = sami_read_apspec(hdulist,apname,doareacorr=doareacorr)
        hdulist.close()


        # now read in the matching SAMI red arm data:
        samifile_red = samifile.replace('blue','red')
        hdulist = pf.open(samifile_red)
        sami_flux_red,sami_lam_red = sami_read_apspec(hdulist,apname,doareacorr=doareacorr)
        hdulist.close()


        # get extinction curve if this is the first file to be checked:
        if (nsami == 0):
            sso_ext_tab='/Users/scroom/data/sami/test_data_v1/standards/extinsso.tab'
            wavelength_extinction, extinction_mags = fluxcal2.read_atmospheric_extinction(sso_extinction_table=sso_ext_tab)
            # interpolate extinction file to correct wavelength range
            extinction_mags_b = np.interp(sami_lam_blue, wavelength_extinction, 
                                    extinction_mags)
            extinction_mags_r = np.interp(sami_lam_red, wavelength_extinction, 
                                extinction_mags)

            # assume rough airmass of 1.2:
            ext_b = 10**(-0.4*extinction_mags_b * 2.0)
            ext_r = 10**(-0.4*extinction_mags_r * 2.0)


        # get the second set of sami spectra if needed:
        s2match=False
        if (dosami2):
            for samifile2 in samifiles2:
                if (objfname in samifile2):
                    hdulist = pf.open(samifile2)
                    sami_flux_blue2,sami_lam_blue2 = sami_read_apspec(hdulist,apname,doareacorr=doareacorr)
                    hdulist.close()

                    # now read in the matching SAMI red arm data:
                    samifile_red2 = samifile2.replace('blue','red')
                    hdulist = pf.open(samifile_red2)
                    sami_flux_red2,sami_lam_red2 = sami_read_apspec(hdulist,apname,doareacorr=doareacorr)
                    hdulist.close()
                    s2match=True

        if (verbose):
            if(s2match):
                print('found match for second set of SAMI spectra')
            else:
                print('no match found for second set of SAMI spectra')
        
        # scale SAMI to SDSS flux scale.  No longer need to do this, as
        # we identified the issues between the scaling.
        #scale = 0.5 # WHY?????
        scale = 1.0 # remove now  fixd by fibre->psf correction.
        sami_flux_blue = sami_flux_blue * 10.0 * scale
        sami_flux_red = sami_flux_red * 10.0 * scale
        if (dosami2 and s2match):
            sami_flux_blue2 = sami_flux_blue2 * 10.0 * scale
            sami_flux_red2 = sami_flux_red2 * 10.0 * scale
            

        # set SAMI fluxes to Nan where they are exactly zero:
        sami_flux_blue[sami_flux_blue==0.0]=np.nan
        sami_flux_red[sami_flux_red==0.0]=np.nan
        if (dosami2 and s2match):
            sami_flux_blue2[sami_flux_blue2==0.0]=np.nan
            sami_flux_red2[sami_flux_red2==0.0]=np.nan
            
        # flag the first and last "good" pixels in each arm as bad, as they often has sudden steps:
        nonan_idx = np.where(np.isfinite(sami_flux_blue))
        ngood = np.size(nonan_idx)
        if (verbose):
            print('blue arm, number of good pixels:',ngood)
        # if less than half good pixels, then skip:
        if (ngood < np.size(sami_flux_blue)/2):
            continue
        if (verbose):
            print('first good pixel:',nonan_idx[0][0])
            print('last good pixel:',nonan_idx[0][-1])
        sami_flux_blue[nonan_idx[0][0]] = np.nan
        sami_flux_blue[nonan_idx[0][1]] = np.nan
        sami_flux_blue[nonan_idx[0][-1]] = np.nan
        sami_flux_blue[nonan_idx[0][-2]] = np.nan
        
        nonan_idx = np.where(np.isfinite(sami_flux_red))
        ngood = np.size(nonan_idx)
        # if less than half good pixels, then skip:
        if (ngood < np.size(sami_flux_red)/2):
            continue
        if (verbose):
            print('red arm, number of good pixels:',ngood)
            print('first good pixel:',nonan_idx[0][0])
            print('last good pixel:',nonan_idx[0][-1])
        sami_flux_red[nonan_idx[0][0]] = np.nan
        sami_flux_red[nonan_idx[0][1]] = np.nan
        sami_flux_red[nonan_idx[0][-1]] = np.nan
        sami_flux_red[nonan_idx[0][-2]] = np.nan
        
            
        # read in the matched SDSS spectrum:
        # SDSS has flux units of 1E-17 erg/cm^2/s/Ang
        sdssfile = filename_sdss[sel]
        hdulist = pf.open(sdssfile)
        if (verbose):
            print('SDSS spectrum matched: ',sdssfile)
        sdss_spec_table = hdulist['COADD'].data
        sdss_flux = sdss_spec_table['flux']
        sdss_loglam = sdss_spec_table['loglam']
        sdss_lam = 10.0**sdss_loglam

        plate_sdss_sami[nsami] = plate_sdss[sel]
        filename_sdss_sami[nsami] = sdssfile

        # get the mean SDSS seeing:
        nhdu = len(hdulist)
        print('number of HDUs:',nhdu)
        seeinglist = []
        for i in range(nhdu):
            try:
                seeing_t = hdulist[i].header['SEEING50']
                seeinglist.append(seeing_t)
            except KeyError:
                continue
        sdss_seeing[nsami] = np.nanmean(seeinglist)
        
        # get the SDSS plate info:
        ra_plate_sdss[nsami] = hdulist[0].header['RADEG']
        dec_plate_sdss[nsami] = hdulist[0].header['DECDEG']

        # SDSS spectra (as of DR6) are normalized to PSF mags
        # these have more flux in them than fibre mags, so there is a correction.
        # the difference is 0.35 mags (see http://classic.sdss.org/dr6/products/spectra/spectrophotometry.html)
        # this correction is only true in the average, as it also depends on seeing.
        psf2fib = 10.0**(-0.4*0.35)
        if (verbose):
            print('scale from SDSS PSF to fiber mags:',psf2fib)
        sdss_flux = sdss_flux * psf2fib
        
        # transform SDSS spectrum from vac to air:
        sdss_lam_air = sdss_lam/(1.0 +2.735182e-4 + 131.4182/sdss_lam**2 + 2.76249e8/sdss_lam**4)

        # resample SDSS spectrum onto SAMI scale:
        sdss_flux_blue = spectres(sami_lam_blue,sdss_lam_air,sdss_flux)
        sdss_flux_red = spectres(sami_lam_red,sdss_lam_air,sdss_flux)
        if (dosami2 and s2match):
            sdss_flux_blue2 = spectres(sami_lam_blue2,sdss_lam_air,sdss_flux)
            sdss_flux_red2 = spectres(sami_lam_red2,sdss_lam_air,sdss_flux)

        # derive the ratios:
        ratio_blue = sami_flux_blue/sdss_flux_blue
        ratio_red = sami_flux_red/sdss_flux_red
        if (dosami2 and s2match):
            ratio_blue2 = sami_flux_blue2/sdss_flux_blue2
            ratio_red2 = sami_flux_red2/sdss_flux_red2

        # derive the median ratio (over all of red and blue):
        ratio_med[nsami] = np.nanmedian(np.concatenate((ratio_blue,ratio_red)))
        ratio_blue_med[nsami] = np.nanmedian(ratio_blue)
        ratio_red_med[nsami] = np.nanmedian(ratio_red)
        if (dosami2 and s2match):
            ratio_med2[nsami] = np.nanmedian(np.concatenate((ratio_blue2,ratio_red2)))
            ratio_blue_med2[nsami] = np.nanmedian(ratio_blue2)
            ratio_red_med2[nsami] = np.nanmedian(ratio_red2)
            

        # we want to find the median ratio at each wavelength.  We will do this after
        # normalizing for the median ratio for the whole spectrum and seperately for
        # both red and blue normalization.  We will also smooth the ratio spectrum before
        # doing this, to reduce the contribution from shot noise and bad pixels.  As we
        # want to get rid of bad pixels we do a median filter:
        ratio_blue_conv = median_filter_nan(ratio_blue,smooth)
        ratio_red_conv = median_filter_nan(ratio_red,smooth)

        ratio_all_blue[nsami,:] = ratio_blue_conv/ratio_med[nsami]
        ratio_all_red[nsami,:] = ratio_red_conv/ratio_med[nsami]

        # also put the ratios in an array indexed by probe number:
        iprobe = int(ifuprobe_sami[nsami])-1
        if (iprobe >= 0):
            if (verbose):
                print(nprobe[iprobe],iprobe,nsami)
            ratio_all_blue_probe[nprobe[iprobe],iprobe,:] = ratio_blue_conv/ratio_med[nsami]
            ratio_all_red_probe[nprobe[iprobe],iprobe,:] = ratio_red_conv/ratio_med[nsami]
            nprobe[iprobe] = nprobe[iprobe] + 1
            
        
        rdiff[nsami] = (ratio_blue_med[nsami]-ratio_red_med[nsami])/(0.5*(ratio_blue_med[nsami]+ratio_red_med[nsami]))
        if (dosami2 and s2match):
            rdiff2[nsami] = (ratio_blue_med2[nsami]-ratio_red_med2[nsami])/(0.5*(ratio_blue_med2[nsami]+ratio_red_med2[nsami]))
        else:
            rdiff2[nsami] = np.nan
            ratio_med2[nsami] = np.nan
            ratio_blue_med2[nsami] = np.nan
            ratio_red_med2[nsami] = np.nan
            ratio4000_2[nsami] = np.nan
            ratio5450_2[nsami] = np.nan
            ratio7000_2[nsami] = np.nan

        if (verbose):
            print('{0:s} {1:6.3f} {2:6.3f} {3:6.3f} {4:6.3f} {5:6.3f} {6:6.3f} {7:6.3f}'.format(filename_sami[nsami],ratio_med[nsami],ratio_blue_med[nsami],ratio_red_med[nsami],rdiff[nsami],ratio_blue_med2[nsami],ratio_red_med2[nsami],rdiff2[nsami]))

        # find the gradient of the blue ratio, to see if its flat or tilted.
        #pfit = polyfitr(sami_lam_blue,sami_flux_blue/sdss_flux_blue,1,5.0)
        # find the median ratio around 4000A:
        idxb = np.where((sami_lam_blue>3900.0) & (sami_lam_blue<4100))
        medratio_4000 = np.nanmedian(sami_flux_blue[idxb]/sdss_flux_blue[idxb])
        idxr = np.where((sami_lam_blue>5400.0) & (sami_lam_blue<5500))
        medratio_5500 = np.nanmedian(sami_flux_blue[idxr]/sdss_flux_blue[idxr])
        idxrr = np.where((sami_lam_red>6900.0) & (sami_lam_red<7100))
        medratio_7000 = np.nanmedian(sami_flux_red[idxrr]/sdss_flux_red[idxrr])
        if (verbose):
            print('median ratio at 4000A:',medratio_4000)
            print('median ratio at 5450A:',medratio_5500)
            print('median ratio at 7000A:',medratio_7000)
        #print('pfit:',pfit)
        # don't use the blue gradient directly, but derive a ratio based on
        # the gradient at two different wavelengths.  This gives us a better
        # idea of the actual differences red-to-blue:
        #lam1 = 3700.0
        #lam2 = 5700.0
        #bluegrad[nsami] = (pfit[0]*lam1+pfit[1])/(pfit[0]*lam2+pfit[1])
        bluegrad[nsami] = medratio_4000/medratio_5500
        bluegradbr[nsami] = medratio_5500/medratio_7000

        ratio4000[nsami] = medratio_4000
        ratio5450[nsami] = medratio_5500
        ratio7000[nsami] = medratio_7000
        
        if (dosami2 and s2match):
            #pfit2 = polyfitr(sami_lam_blue2,sami_flux_blue2/sdss_flux_blue2,1,5.0)
            #print('pfit2:',pfit2)
            medratio2_4000 = np.nanmedian(sami_flux_blue2[idxb]/sdss_flux_blue[idxb])
            medratio2_5500 = np.nanmedian(sami_flux_blue2[idxr]/sdss_flux_blue[idxr])
            medratio2_7000 = np.nanmedian(sami_flux_red2[idxrr]/sdss_flux_red[idxrr])
            #bluegrad2[nsami] = (pfit2[0]*lam1+pfit2[1])/(pfit2[0]*lam2+pfit2[1])
            bluegrad2[nsami] = medratio2_4000/medratio2_5500
            bluegradbr2[nsami] = medratio2_5500/medratio2_7000
            ratio4000_2[nsami] = medratio2_4000
            ratio5450_2[nsami] = medratio2_5500
            ratio7000_2[nsami] = medratio2_7000
        
        
        # plot the spectra:
        if (plotall):
            py.figure(1)
            py.clf()
            py.subplot(3,1,1)
            # plot SDSS:
            py.plot(sdss_lam_air,sdss_flux,color='k')
            py.plot(sami_lam_blue,sdss_flux_blue,color='g')
            py.plot(sami_lam_red,sdss_flux_red,color='m')
            
            # plot SAMI:
            py.plot(sami_lam_blue,sami_flux_blue,color='b')
            py.plot(sami_lam_red,sami_flux_red,color='r')
            #py.plot(sami_lam_blue,sami_flux_blue/ext_b,color='b',alpha=0.5)
            #py.plot(sami_lam_red,sami_flux_red/ext_r,color='r',alpha=0.5)
            if (dosami2 and s2match):
                py.plot(sami_lam_blue2,sami_flux_blue2,color='k',alpha=0.5)
                py.plot(sami_lam_red2,sami_flux_red2,color='k',alpha=0.5)

            if (dofibmags and foundmag):
                py.plot(3557.0,fu,'o',color='k')
                py.plot(4702.0,fg,'o',color='k')
                py.plot(6175.0,fr,'o',color='k')
                py.plot(7491.0,fi,'o',color='k')
            
            py.ylabel('Flux (1E-17 erg/cm$^2$/s/Ang)')
            title_plot = samifile.replace('_','\_')
            py.title(title_plot)
            ax = py.gca()
            lab1 = 'positional offset: {0:5.2f} arcsec'.format(mindiff*3600)
            py.text(0.05, 0.95,lab1, horizontalalignment='left',verticalalignment='center',transform=ax.transAxes)
            lab2 = 'median SAMI/SDSS flux ratio: {0:6.3f}'.format(ratio_med[nsami])
            py.text(0.05, 0.85,lab2, horizontalalignment='left',verticalalignment='center',transform=ax.transAxes)
            lab3 = 'SAMI Seeing: {0:5.2f}'.format(psffwhm_sami[nsami])
            py.text(0.05, 0.75,lab3, horizontalalignment='left',verticalalignment='center',transform=ax.transAxes)
            lab4 = 'SAMI Probe: {0:2d}'.format(int(ifuprobe_sami[nsami]))
            py.text(0.05, 0.65,lab4, horizontalalignment='left',verticalalignment='center',transform=ax.transAxes)

            # get scaling and range:
            py.xlim(xmin=3500.0,xmax=xmax)
            ymin=-5.0
            ymax=np.nanmax(np.concatenate((sami_flux_blue,sami_flux_red,sdss_flux)))*1.05
            py.ylim(ymin=ymin,ymax=ymax)

            # plot the ratio:
            py.subplot(3,1,2)
            # plot SAMI/SDSS:
            py.axhline(1.0,color='k',linestyle=':')
            py.plot(sami_lam_blue,sami_flux_blue/sdss_flux_blue,color='b')
            py.plot(sami_lam_red,sami_flux_red/sdss_flux_red,color='r')
            #py.plot(sami_lam_blue,(sami_flux_blue/sdss_flux_blue)/ext_b,color='b',alpha=0.5)
            #py.plot(sami_lam_red,(sami_flux_red/sdss_flux_red)/ext_r,color='r',alpha=0.5)
            # plot the linear fit to the ratio:
            axb = np.linspace(3500.0,6000.0,10)
            #py.plot(axb,pfit[0]*axb+pfit[1],':',color='b')
            
            if (dosami2 and s2match):
                py.plot(sami_lam_blue2,sami_flux_blue2/sdss_flux_blue2,color='k',alpha=0.5)
                py.plot(sami_lam_red2,sami_flux_red2/sdss_flux_red2,color='k',alpha=0.5)
            py.xlim(xmin=xmin,xmax=xmax)
            py.ylim(ymin=0.0,ymax=2.0)
            # plot a horizontal line for the median ratio:
            py.axhline(ratio_blue_med[nsami],color='b')
            py.axhline(ratio_red_med[nsami],color='r')
            if (dosami2 and s2match):
                py.axhline(ratio_blue_med2[nsami],color='k',alpha=0.5)
                py.axhline(ratio_red_med2[nsami],color='k',alpha=0.5)
            
            #py.xlabel('Wavelength (Angstroms)')
            py.ylabel('SAMI/SDSS flux ratio')
            ax = py.gca()
            lab1 = "ratio$_{{4000}}$/ratio$_{{5450}}$: {0:6.3f}".format(bluegrad[nsami])
            py.text(0.25, 0.90,lab1, horizontalalignment='left',verticalalignment='center',transform=ax.transAxes)
            lab2 = "ratio$_{{5450}}$/ratio$_{{7000}}$: {0:6.3f}".format(bluegradbr[nsami])
            py.text(0.25, 0.80,lab2, horizontalalignment='left',verticalalignment='center',transform=ax.transAxes)
            
            # plot the ratio normalized by median:
            py.subplot(3,1,3)
            # plot SAMI/SDSS:
            py.axhline(1.0,color='k',linestyle=':')
            py.plot(sami_lam_blue,sami_flux_blue/sdss_flux_blue/ratio_med[nsami],color='b')
            py.plot(sami_lam_red,sami_flux_red/sdss_flux_red/ratio_med[nsami],color='r')
            if (dosami2 and s2match):
                py.plot(sami_lam_blue2,sami_flux_blue2/sdss_flux_blue2/ratio_med2[nsami],color='k',alpha=0.5)
                py.plot(sami_lam_red2,sami_flux_red2/sdss_flux_red2/ratio_med2[nsami],color='k',alpha=0.5)
            py.xlim(xmin=xmin,xmax=xmax)
            py.ylim(ymin=0.0,ymax=2.0)
            py.xlabel('Wavelength (Angstroms)')
            py.ylabel('SAMI/SDSS flux ratio (normalized)')
            py.draw()
        
        # pause for input if plotting all the spectra:
            yn = input('Good? ret/y = Yes, n = No:')

            if (yn.lower() == 'n'):
                out_bad.write(filename_sami[nsami])

            
            
        # increment counter:
        nsami = nsami + 1

        # skip out after a small amount for development:
        #if (nsami > 200):
        #    break

    # close output file that contains bad objects
    out_bad.close()
    out.close()
        
    # derive the median "median flux ratio" over all spectra:
    med_ratio_med = np.nanmedian(ratio_med[0:nsami])
    rms_ratio_med = np.nanstd(ratio_med[0:nsami])
    med_ratio_blue_med = np.nanmedian(ratio_blue_med[0:nsami])
    med_ratio_red_med = np.nanmedian(ratio_red_med[0:nsami])
    rms_ratio_blue_med = np.nanstd(ratio_blue_med[0:nsami])
    rms_ratio_red_med = np.nanstd(ratio_red_med[0:nsami])
    
    if (dosami2):
        med_ratio_med2 = np.nanmedian(ratio_med2[0:nsami])
        rms_ratio_med2 = np.nanstd(ratio_med2[0:nsami])
        med_ratio_blue_med2 = np.nanmedian(ratio_blue_med2[0:nsami])
        med_ratio_red_med2 = np.nanmedian(ratio_red_med2[0:nsami])
        rms_ratio_blue_med2 = np.nanstd(ratio_blue_med2[0:nsami])
        rms_ratio_red_med2 = np.nanstd(ratio_red_med2[0:nsami])

    print('new:')
    print('median flux ratios and sigma (all):',med_ratio_med,rms_ratio_med)
    print('median flux ratios and sigma (blue):',med_ratio_blue_med,rms_ratio_blue_med)
    print('median flux ratios and sigma (red):',med_ratio_red_med,rms_ratio_red_med)
    
    if (dosami2):
        print('old:')
        print('median flux ratios and sigma (all):',med_ratio_med2,rms_ratio_med2)
        print('median flux ratios and sigma (blue):',med_ratio_blue_med2,rms_ratio_blue_med2)
        print('median flux ratios and sigma (red):',med_ratio_red_med2,rms_ratio_red_med2)

    print('median blue grad and sigma (new):',np.nanmedian(bluegrad[0:nsami]),np.nanstd(bluegrad[0:nsami]))
    print('median blue grad and sigma (old):',np.nanmedian(bluegrad2[0:nsami]),np.nanstd(bluegrad2[0:nsami]))
    

    # output for each object some of the key results:
    for i in range(nsami):
        diff = ratio_blue_med[i]/ratio_red_med[i]
        if (verbose):
            print('{0:s} {1:6.3f} {2:6.3f} {3:6.3f} {4:6.3f}'.format(filename_sami[i],ratio_med[i],ratio_blue_med[i],ratio_red_med[i],diff))
        # write measurements to a file:
        #outstr = '{0:s} {1:6.3f} {2:6.3f} {3:6.3f} {4:6.3f} {5:6.3f} {6:6.3f}\n'.format(filename_sami[i],ratio_med[i],ratio_blue_med[i],ratio_red_med[i],ratio_med2[i],ratio_blue_med2[i],ratio_red_med2[i])

            
    # write the results to a FITS binary table.
    # First define the columns to use...
    # file names and field/plate names:
    col1 = pf.Column(name='SAMIFileName',format='100A',array=filename_sami[0:nsami])
    col2 = pf.Column(name='SDSSFileName',format='64A',array=filename_sdss_sami[0:nsami])
    col3 = pf.Column(name='SAMIFieldName',format='32A',array=fieldname_sami[0:nsami])
    col4 =  pf.Column(name='SDSSPlateName',format='16A',array=plate_sdss_sami[0:nsami])
    # global median ratios of SAMI/SDSS for total, red and blue:
    col5 = pf.Column(name='RatioMed',format='D',array=ratio_med[0:nsami])
    col6 = pf.Column(name='RatioMedBlue',format='D',array=ratio_blue_med[0:nsami])
    col7 = pf.Column(name='RatioMedRed',format='D',array=ratio_red_med[0:nsami])
    col8 = pf.Column(name='RatioMed2',format='D',array=ratio_med2[0:nsami])
    col9 = pf.Column(name='RatioMedBlue2',format='D',array=ratio_blue_med2[0:nsami])
    col10 = pf.Column(name='RatioMedRed2',format='D',array=ratio_red_med2[0:nsami])
    # ratios at different wavelengths:
    col11 = pf.Column(name='Ratio4000',format='D',array=ratio4000[0:nsami])
    col12 = pf.Column(name='Ratio5450',format='D',array=ratio5450[0:nsami])
    col13 = pf.Column(name='Ratio7000sami_tools_smc.dr3paper_plots',format='D',array=ratio7000[0:nsami])
    col14 = pf.Column(name='Ratio40002',format='D',array=ratio4000_2[0:nsami])
    col15 = pf.Column(name='Ratio54502',format='D',array=ratio5450_2[0:nsami])
    col16 = pf.Column(name='Ratio70002',format='D',array=ratio7000_2[0:nsami])
    # seeing etc:
    col17 = pf.Column(name='SAMIPSFFWHM',format='D',array=psffwhm_sami[0:nsami])
    col18 = pf.Column(name='SDSSPSFFWHM',format='D',array=sdss_seeing[0:nsami])
    col19 = pf.Column(name='RA',format='D',array=ra_sami[0:nsami])
    col20 = pf.Column(name='DEC',format='D',array=dec_sami[0:nsami])
    col21 = pf.Column(name='RAPLATESDSS',format='D',array=ra_plate_sdss[0:nsami])
    col22 = pf.Column(name='DECPLATESDSS',format='D',array=dec_plate_sdss[0:nsami])

    cols = pf.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22])
    
    #Now, create a new binary table HDU object:
    tbhdu = pf.BinTableHDU.from_columns(cols)

    # finally write the table HDU to a file:
    outfile = 'sami_sdss_comp_tab.fits'
    print('Writing results to ',outfile)
    tbhdu.writeto(outfile,overwrite=True)
 

    # get stats of red/blue ratio:
    print('mean of (blue-red)/mean ratio: ',np.nanmean(rdiff[0:nsami]))
    print('rms of  (blue-red)/mean ratio: ',np.nanstd(rdiff[0:nsami]))
    if (dosami2):
        print('mean of (blue-red)/mean ratio (old): ',np.nanmean(rdiff2[0:nsami]))
        print('rms of  (blue-red)/mean ratio (old): ',np.nanstd(rdiff2[0:nsami]))
        
    # plot distributions of ratio and positional offsets:
    if (plot):
       if (dopdf):
           pdf = PdfPages(dir+'/sdss_fluxcomp_ratios.pdf')

       py.figure(2)
       # plot histogram of flux ratios:
       py.subplot(2,3,1)
       py.hist(ratio_med[0:nsami],bins=50,range=(0.0,2.0),histtype='step',color='k')
       py.hist(ratio_blue_med[0:nsami],bins=50,range=(0.0,2.0),histtype='step',color='b')
       py.hist(ratio_red_med[0:nsami],bins=50,range=(0.0,2.0),histtype='step',color='r')
       py.axvline(med_ratio_med,color='k',linestyle=':')
       py.axvline(med_ratio_blue_med,color='b',linestyle=':')
       py.axvline(med_ratio_red_med,color='r',linestyle=':')
       py.xlabel('SAMI/SDSS flux ratio')
       py.ylabel('Number')

       # plot flux ratio vs. positional offset:
       py.subplot(2,3,2)
       py.plot(posdiff[0:nsami]*3600.0,ratio_med[0:nsami],'x')
       py.axhline(1.0,color='k',linestyle=':')
       py.xlabel('positional offset (arcsec)')
       py.ylabel('SAMI/SDSS flux ratio')

       # plot flux ratio vs. positional offset:
       py.subplot(2,3,3)
       py.plot(psffwhm_sami[0:nsami],ratio_med[0:nsami],'x')
       py.axhline(1.0,color='k',linestyle=':')
       py.xlabel('PSF FWHM (arcsec)')
       py.ylabel('SAMI/SDSS flux ratio')

       # plot flux ratio vs. probe:
       py.subplot(2,3,4)
       py.plot(ifuprobe_sami[0:nsami],ratio_med[0:nsami],'x')
       py.axhline(1.0,color='k',linestyle=':')
       py.xlabel('SAMI IFU probe number')
       py.ylabel('SAMI/SDSS flux ratio')

       # plot distribution of red to blue ratio:
       py.subplot(2,3,5)
       py.hist(rdiff[0:nsami],bins=30,range=(-0.3,0.3),histtype='step',color='k',label='new')
       py.hist(rdiff2[0:nsami],bins=30,range=(-0.3,0.3),histtype='step',color='m',label='old')
       py.legend(prop={'size':10})
       py.xlabel('(blue ratio - red ratio)/mean')
       py.ylabel('Number')
       
       py.subplot(2,3,6)
       py.plot(ifuprobe_sami[0:nsami],rdiff[0:nsami],'x',color='k')
       py.plot(ifuprobe_sami[0:nsami],rdiff2[0:nsami],'x',color='m')
       py.ylabel('(blue ratio - red ratio)/mean')
       py.xlabel('SAMI IFU probe number')

       py.tight_layout()

       if (dopdf):
           py.savefig(pdf, format='pdf')        
           pdf.close()

       
    # plot the median ratio and scatter as a function of wavelength:
    med_ratio_spec_blue = np.nanmedian(ratio_all_blue[0:nsami,:],axis=0)
    pec_p1_ratio_spec_blue = np.nanpercentile(ratio_all_blue[0:nsami,:],(100-68.27)/2,axis=0)
    pec_m1_ratio_spec_blue = np.nanpercentile(ratio_all_blue[0:nsami,:],100-(100-68.27)/2,axis=0)

    med_ratio_spec_red = np.nanmedian(ratio_all_red[0:nsami,:],axis=0)
    pec_p1_ratio_spec_red = np.nanpercentile(ratio_all_red[0:nsami,:],(100-68.27)/2,axis=0)
    pec_m1_ratio_spec_red = np.nanpercentile(ratio_all_red[0:nsami,:],100-(100-68.27)/2,axis=0)

    
    if (plot):
       if (dopdf):
           pdf = PdfPages(dir+'/sdss_fluxcomp_median.pdf')
            
       py.figure(3)
       py.axhline(1.0,color='k',linestyle='--')
       # get the index of the pixel nearest to 3800A:
       i3800 = (np.abs(sami_lam_blue-3850.0)).argmin()
       py.plot(sami_lam_blue[i3800:],med_ratio_spec_blue[i3800:],color='b')
       py.plot(sami_lam_blue[i3800:],pec_p1_ratio_spec_blue[i3800:],color='b',linestyle=':')
       py.plot(sami_lam_blue[i3800:],pec_m1_ratio_spec_blue[i3800:],color='b',linestyle=':')
       py.plot(sami_lam_red,med_ratio_spec_red,color='r')
       py.plot(sami_lam_red,pec_p1_ratio_spec_red,color='r',linestyle=':')
       py.plot(sami_lam_red,pec_m1_ratio_spec_red,color='r',linestyle=':')
       py.xlim(xmin=xmin,xmax=xmax)
       py.ylim(ymin=0.75,ymax=1.1)
       py.xlabel('Wavelength [\\AA]')
       py.ylabel('SAMI/SDSS 3-arcsec fibre flux')

       # write out to  text file.  Easiest using Pandas:
       all_lam = np.concatenate((sami_lam_blue,sami_lam_red))
       all_ratio = np.concatenate((med_ratio_spec_blue,med_ratio_spec_red))
       all_minus = np.concatenate((pec_m1_ratio_spec_blue,pec_m1_ratio_spec_red))
       all_plus = np.concatenate((pec_p1_ratio_spec_blue,pec_p1_ratio_spec_red))
       df = pd.DataFrame({'Wavelength':all_lam, 'Ratio':all_ratio, 'Ratio_err_minus':all_minus, 'Ratio_err_plus':all_plus})
       df.to_csv('med_ratio_sami_sddss.csv')

       if (dopdf):
           py.savefig(pdf, format='pdf')        
           pdf.close()
           
    # get the median ratio for each probe:
    if (plot):
        if (dopdf):
            pdf = PdfPages(dir+'/sdss_fluxcomp_medianprobe.pdf')
        py.figure(4)
        cols=iter(py.cm.rainbow(np.linspace(0,1,13)))
        py.axhline(1.0,color='k',linestyle='--')
        for i in range(12):
            med_ratio_spec_blue = np.nanmedian(ratio_all_blue_probe[0:nprobe[i],i,:],axis=0)
            med_ratio_spec_red = np.nanmedian(ratio_all_red_probe[0:nprobe[i],i,:],axis=0)
            c = next(cols)
            py.plot(sami_lam_blue,med_ratio_spec_blue,color=c,label='Probe'+str(int(i+1)))
            py.plot(sami_lam_red,med_ratio_spec_red,color=c)

        py.xlim(xmin=xmin,xmax=xmax)
        py.ylim(ymin=0.5,ymax=1.5)
        py.xlabel('Wavelength')
        py.ylabel('SAMI/SDSS flux')
        py.legend(loc='upper right',prop={'size':10})

        if (dopdf):
           py.savefig(pdf, format='pdf')        
           pdf.close()


    # get the unique field names and plot/list stats based on these:
    unique_fnames = np.unique(fieldname_sami[0:nsami])
    print('unique SAMI field names:')
    print(unique_fnames)
    print('number of objects:',nsami)

    nu = np.size(unique_fnames)
    fsig = np.zeros(nu)
    fsigbr = np.zeros(nu)
    fnum = np.zeros(nu)
    
    # want to look at red/blue ratio overall zp and blue tilt
    fn = 0
    print('f_no f_name 4000/5500_med 5500/7000_med 4000/5500_sig 5500/7000_sig 4000/5500_med2 5500/7000_med2 nobj')
    #print('f_no f_name rdiff_med rdiff_med2 bluegrad_med bluegrad_med2 nobj')
    for fname in unique_fnames:
        idx = np.where(fieldname_sami==fname)
        bluegrad_med = np.nanmedian(bluegrad[idx])
        bluegrad_medbr = np.nanmedian(bluegradbr[idx])
        bluegrad_sig = np.nanstd(bluegrad[idx])
        bluegrad_sigbr = np.nanstd(bluegradbr[idx])
        bluegrad_med2 = np.nanmedian(bluegrad2[idx])
        bluegrad_medbr2 = np.nanmedian(bluegradbr2[idx])
        fsig[fn] = bluegrad_sig
        fsigbr[fn] = bluegrad_sigbr
        fnum[fn] = np.size(fieldname_sami[idx])
        fnax = np.ones(np.size(plate_sdss_sami[idx])) * fn
        print('{0:3d} {1:s} {2:6.3f} {3:6.3f} {4:6.3f} {5:6.3f} {6:6.3f} {7:6.3f} {8:3d}'.format(fn,fname,bluegrad_med,bluegrad_medbr,bluegrad_sig,bluegrad_sigbr,bluegrad_med2,bluegrad_medbr2,np.size(fieldname_sami[idx])))

        if (plot):
            if (dopdf):
                pdf = PdfPages(dir+'/sdss_fluxcomp_fieldnum.pdf')
            py.figure(5)
            py.subplot(2,2,1)
            py.plot(fnax,rdiff[idx],'x',color='k')
            if (dosami2):
                py.plot(fnax,rdiff2[idx],'x',color='m')
            py.ylabel('(blue - red)/mean ratio')
            py.xlabel('field number')
            py.axhline(np.nanmedian(rdiff[0:nsami]),color='k',linestyle=':')
            if (dosami2):
                py.axhline(np.nanmedian(rdiff2[0:nsami]),color='m',linestyle=':')
            
            py.subplot(2,2,2)
            py.plot(fnax,ratio_med[idx],'x',color='k')
            if (dosami2):
                py.plot(fnax,ratio_med2[idx],'x',color='m')
            py.ylabel('median SAMI/SDSS ratio')
            py.xlabel('field number')

            py.subplot(2,2,3)
            py.plot(fnax,bluegrad[idx],'x',color='k')
            if (dosami2):
                py.plot(fnax,bluegrad2[idx],'x',color='m')
            py.axhline(np.nanmedian(bluegrad[0:nsami]),color='k',linestyle=':')
            if (dosami2):
                py.axhline(np.nanmedian(bluegrad2[0:nsami]),color='m',linestyle=':')
            py.ylabel('ratio$_{4000}$/ratio$_{5450}$')
            py.xlabel('field number')
            
            py.subplot(2,2,4)
            py.plot(fnax,bluegradbr[idx],'x',color='k')
            py.axhline(np.nanmedian(bluegradbr[0:nsami]),color='k',linestyle=':')
            if (dosami2):
                py.plot(fnax,bluegradbr2[idx],'x',color='m')
                py.axhline(np.nanmedian(bluegradbr2[0:nsami]),color='m',linestyle=':')
            py.ylabel('ratio$_{5450}$/ratio$_{7000}$')
            py.xlabel('field number')

            py.tight_layout()
            
            if (dopdf):
                py.savefig(pdf, format='pdf')        
                pdf.close()


        fn = fn + 1

    # plot phot offsets vs ra/dec
    if (plot):
        vmin=0.6
        vmax=1.4
        fig6 = py.figure(6)
        ax6_1 = fig6.add_subplot(3,1,1)
        cax6_1 = ax6_1.scatter(ra_sami[0:nsami],dec_sami[0:nsami],c=bluegrad[0:nsami],marker='o',vmin=vmin,vmax=vmax)
        cbar6_1 = fig6.colorbar(cax6_1,spacing='proportional', format='%5.1f')
        ax6_1.set(xlim=[129.0,141.0],ylim=[-1.0,3.0])

        ax6_2 = fig6.add_subplot(3,1,2)
        cax6_2 = ax6_2.scatter(ra_sami[0:nsami],dec_sami[0:nsami],c=bluegrad[0:nsami],marker='o',vmin=vmin,vmax=vmax)
        cbar6_2 = fig6.colorbar(cax6_2,spacing='proportional', format='%5.1f')
        ax6_2.set(xlim=[174.0,186.0],ylim=[-2.0,2.0])
        
        ax6_3 = fig6.add_subplot(3,1,3)
        cax6_3 = ax6_3.scatter(ra_sami[0:nsami],dec_sami[0:nsami],c=bluegrad[0:nsami],marker='o',vmin=vmin,vmax=vmax)
        cbar6_3 = fig6.colorbar(cax6_3,spacing='proportional', format='%5.1f')
        ax6_3.set(xlim=[211.5,223.5],ylim=[-2.0,2.0])

    # get unique SDSS plates:
    unique_pnames = np.unique(plate_sdss_sami[0:nsami])
    print('unique SDSS plate names:')
    print(unique_pnames)

    nup = np.size(unique_pnames)
    psig = np.zeros(nup)
    psigbr = np.zeros(nup)
    pnum = np.zeros(nup)
    
    # loop through unique plate names:
    fnp = 0
    print('f_no f_name 4000/5500_med 5500/7000_med 4000/5500_sig 5500/7000_sig 4000/5500_med2 5500/7000_med2 nobj')
    for fname in unique_pnames:
        idx = np.where(plate_sdss_sami==fname)
        bluegrad_med = np.nanmedian(bluegrad[idx])
        bluegrad_medbr = np.nanmedian(bluegradbr[idx])
        bluegrad_sig = np.nanstd(bluegrad[idx])
        bluegrad_sigbr = np.nanstd(bluegradbr[idx])
        bluegrad_med2 = np.nanmedian(bluegrad2[idx])
        bluegrad_medbr2 = np.nanmedian(bluegradbr2[idx])
        psig[fnp] = bluegrad_sig
        psigbr[fnp] = bluegrad_sigbr
        pnum[fnp] = np.size(plate_sdss_sami[idx])
        fnax = np.ones(np.size(plate_sdss_sami[idx])) * fnp
        print('{0:3d} {1:s} {2:6.3f} {3:6.3f} {4:6.3f} {5:6.3f} {6:6.3f} {7:6.3f} {8:3d}'.format(fnp,fname,bluegrad_med,bluegrad_medbr,bluegrad_sig,bluegrad_sigbr,bluegrad_med2,bluegrad_medbr2,np.size(fieldname_sami[idx])))

        if (plot):
            if (dopdf):
                pdf = PdfPages(dir+'/sdss_fluxcomp_platenum.pdf')
            py.figure(7)
            py.subplot(2,2,1)
            py.plot(fnax,rdiff[idx],'x',color='k')
            if (dosami2):
                py.plot(fnax,rdiff2[idx],'x',color='m')
            py.ylabel('(blue - red)/mean ratio')
            py.xlabel('field number')
            py.axhline(np.nanmedian(rdiff[0:nsami]),color='k',linestyle=':')
            if (dosami2):
                py.axhline(np.nanmedian(rdiff2[0:nsami]),color='m',linestyle=':')
            
            py.subplot(2,2,2)
            py.plot(fnax,ratio_med[idx],'x',color='k')
            if (dosami2):
                py.plot(fnax,ratio_med2[idx],'x',color='m')
            py.ylabel('median SAMI/SDSS ratio')
            py.xlabel('field number')

            py.subplot(2,2,3)
            py.plot(fnax,bluegrad[idx],'x',color='k')
            if (dosami2):
                py.plot(fnax,bluegrad2[idx],'x',color='m')
            py.axhline(np.nanmedian(bluegrad[0:nsami]),color='k',linestyle=':')
            if (dosami2):
                py.axhline(np.nanmedian(bluegrad2[0:nsami]),color='m',linestyle=':')
            py.ylabel('ratio$_{4000}$/ratio$_{5450}$')
            py.xlabel('field number')
            
            py.subplot(2,2,4)
            py.plot(fnax,bluegradbr[idx],'x',color='k')
            py.axhline(np.nanmedian(bluegradbr[0:nsami]),color='k',linestyle=':')
            if (dosami2):
                py.plot(fnax,bluegradbr2[idx],'x',color='m')
                py.axhline(np.nanmedian(bluegradbr2[0:nsami]),color='m',linestyle=':')
            py.ylabel('ratio$_{5450}$/ratio$_{7000}$')
            py.xlabel('field number')

            py.tight_layout()
            
            if (dopdf):
                py.savefig(pdf, format='pdf')        
                pdf.close()



        
        fnp = fnp + 1
     
    # plot hist of scatter in plates:

    fig8 = py.figure(8)
    ax8_1 = fig8.add_subplot(1,2,1)
    fidx = np.where(fnum>1.0)
    pidx = np.where(pnum>1.0)
    ax8_1.hist(fsig[fidx],bins=50,range=(0.0,0.5),histtype='step',color='k',label='SAMI fields')
    ax8_1.hist(psig[pidx],bins=50,range=(0.0,0.5),histtype='step',color='b',label='SDSS plates')
    ax8_1.legend(prop={'size':10})
    ax8_1.set(xlabel='sigma(4000/5500)',ylabel='Number')

    print('median sigma sami fields:',np.nanmedian(fsig[fidx]))
    print('median sigma sdss plates:',np.nanmedian(psig[pidx]))
    
    ax8_2 = fig8.add_subplot(1,2,2)
    ax8_2.hist(fsigbr[fidx],bins=50,range=(0.0,0.5),histtype='step',color='k',label='SAMI fields')
    ax8_2.hist(psigbr[pidx],bins=50,range=(0.0,0.5),histtype='step',color='b',label='SDSS plates')
    ax8_2.set(xlabel='sigma(5500/7000)',ylabel='Number')
        
#        sys.exit()

#########################################################
# plot SDSS-SAMI spectral comparisons from a catalogue of
# previously determined measurements
#
def plot_sami_sdss_comp(catfile1):

    # plotting setup:
    #py.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    py.rc('font',**{'family':'serif','serif':['Times']})
    py.rc('text', usetex=True)
    py.rcParams.update({'font.size': 14})
    py.rcParams.update({'lines.linewidth': 1})


    # read table:
    hdulist1 = pf.open(catfile1)
    tab1 = hdulist1[1].data
    ratio_blue1 = tab1['RatioMedBlue']
    ratio_red1 = tab1['RatioMedRed']

    ratio4000_1 = tab1['Ratio4000']
    ratio5450_1 = tab1['Ratio5450']
    ratio7000_1 = tab1['Ratio7000']

    seeing = tab1['SAMIPSFFWHM']
    num = np.size(ratio_blue1)

    # plot histograms of normalization:
    pdf = PdfPages('samisdss_spec_fluxratio_hist.pdf')
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.hist(ratio_blue1,bins=50,range=(0.0,2.0),histtype='step',color='b')
    ax1.hist(ratio_red1,bins=50,range=(0.0,2.0),histtype='step',color='r')
    ax1.set(xlabel='SAMI/SDSS flux ratio',ylabel='Number',xlim=[0.25,1.75])
    # get median:
    med_blue = np.nanmedian(ratio_blue1)
    med_red = np.nanmedian(ratio_red1)
    ax1.axvline(med_blue,color='b',linestyle=':')
    ax1.axvline(med_red,color='r',linestyle=':')

    p1_ratio_blue = np.nanpercentile(ratio_blue1,(100-68.27)/2)
    m1_ratio_blue = np.nanpercentile(ratio_blue1,100-(100-68.27)/2)
    rms_blue = abs(p1_ratio_blue-m1_ratio_blue)/2.0
    med_err_blue = rms_blue/np.sqrt(num)
    print('blue median, rms, std_err, num: ',med_blue,rms_blue,med_err_blue,num)
    
    p1_ratio_red = np.nanpercentile(ratio_red1,(100-68.27)/2)
    m1_ratio_red = np.nanpercentile(ratio_red1,100-(100-68.27)/2)
    rms_red = abs(p1_ratio_red-m1_ratio_red)/2.0
    med_err_red = rms_red/np.sqrt(num)
    print('red median, rms, std_err, num: ',med_red,rms_red,med_err_red,num)
    
    # save PDF:
    py.savefig(pdf, format='pdf')        
    pdf.close()

    # plot distributions as a function of seeing.
    fig2 = py.figure(2)
    ax21 = fig2.add_subplot(211)
    ax21.plot(seeing,ratio_blue1,'.',color='b')
    ax21.set(xlim=[1.0,3.5],ylim=[0.5,2.0],xlabel='Seeing (arcsec)',ylabel='SAMI/SDSS flux ratio')

    # bin stats and get median:
    bin_means_blue, bin_edges, binnumber = sp.stats.binned_statistic(seeing,ratio_blue1,statistic='median', bins=15,range=[1.0,4.0])
    bin_stds_blue, bin_edges, binnumber = sp.stats.binned_statistic(seeing,ratio_blue1,statistic='std', bins=15,range=[1.0,4.0])    
    bin_num_blue, bin_edges, binnumber = sp.stats.binned_statistic(seeing,ratio_blue1,statistic='count', bins=15,range=[1.0,4.0])    
    bin_stds_blue = bin_stds_blue/np.sqrt(bin_num_blue)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    ax21.plot(bin_centers,bin_means_blue,'.',color='k')
    ax21.axhline(1.0,color='k',linestyle=':')

    ax22 = fig2.add_subplot(212)
    ax22.plot(seeing,ratio_red1,'.',color='r')
    ax22.set(xlim=[1.0,3.5],ylim=[0.5,2.0])
    
    # bin stats and get median:
    bin_means_red, bin_edges, binnumber = sp.stats.binned_statistic(seeing,ratio_red1,statistic='median', bins=15,range=[1.0,4.0])
    bin_stds_red, bin_edges, binnumber = sp.stats.binned_statistic(seeing,ratio_red1,statistic='std', bins=15,range=[1.0,4.0])
    bin_num_red, bin_edges, binnumber = sp.stats.binned_statistic(seeing,ratio_red1,statistic='count', bins=15,range=[1.0,4.0])
    bin_stds_red = bin_stds_red/np.sqrt(bin_num_red)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    ax22.plot(bin_centers,bin_means_red,'.',color='k')
    ax22.axhline(1.0,color='k',linestyle=':')

    print('seeing ratio_blue ratio_red ratio_blue_err ratio_red_err')
    for i in range(np.size(bin_means_blue)):
        print('{0:3.1f}   {1:5.3f}   {2:5.3f}   {3:5.3f}   {4:5.3f}'.format(bin_centers[i],bin_means_blue[i],bin_means_red[i],bin_stds_blue[i],bin_stds_red[i]))
    
    
    # plot histograms of gradients.
    grad1 = ratio4000_1/ratio5450_1
    grad2 = ratio5450_1/ratio7000_1
    
    pdf = PdfPages('samisdss_spec_fluxgrad_hist.pdf')
    fig3 = py.figure(3)
    ax3 = fig3.add_subplot(111)
    ax3.hist(grad1,bins=50,range=(0.0,2.0),histtype='step',color='b')
    ax3.hist(grad2,bins=50,range=(0.0,2.0),histtype='step',color='r')
    ax3.set(xlabel='SAMI/SDSS flux ratio',ylabel='Number',xlim=[0.25,1.75])
    # get median:
    med_blue = np.nanmedian(grad1)
    med_red = np.nanmedian(grad2)
    ax3.axvline(med_blue,color='b',linestyle=':')
    ax3.axvline(med_red,color='r',linestyle=':')

    p1_ratio_blue = np.nanpercentile(grad1,(100-68.27)/2)
    m1_ratio_blue = np.nanpercentile(grad1,100-(100-68.27)/2)
    rms_blue = abs(p1_ratio_blue-m1_ratio_blue)/2.0
    med_err_blue = rms_blue/np.sqrt(num)
    print('blue median, rms, std_err, num: ',med_blue,rms_blue,med_err_blue,num)
    
    p1_ratio_red = np.nanpercentile(grad2,(100-68.27)/2)
    m1_ratio_red = np.nanpercentile(grad2,100-(100-68.27)/2)
    rms_red = abs(p1_ratio_red-m1_ratio_red)/2.0
    med_err_red = rms_red/np.sqrt(num)
    print('red median, rms, std_err, num: ',med_red,rms_red,med_err_red,num)
    
    
    
    
#########################################################
# do a quick sum of flux up columns for bright flux standards
# The main aim is to see the rough level of flux in the data.
#
def quick_col_sum(imfile1,imfile2,scale=1.0,shift=0):

    sso_ext_tab='/Users/scroom/data/sami/test_data_v1/standards/extinsso.tab'

    # get data:
    hdulist1 = pf.open(imfile1)
    hdulist2 = pf.open(imfile2)

    # get image:
    imdata1 = hdulist1[0].data
    imdata2 = hdulist2[0].data

    # get exposure time:
    imheader = hdulist1[0].header
    exposed = imheader['EXPOSED']
    zdstart = imheader['ZDSTART']
    crval1= imheader['CRVAL1']
    cdelt1= imheader['CDELT1']
    crpix1= imheader['CRPIX1']
    naxis1= imheader['NAXIS1']
    x=np.arange(naxis1)+1
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    lam=L0+x*cdelt1

    # calc airmass - assume STD obs is short enough that we can just take the start:
    alt = float(90.0 - zdstart)
    airmass = 1./ ( np.sin( ( alt + 244. / ( 165. + 47 * alt**1.1 )
                            ) / 180. * np.pi ) )

    wavelength_extinction, extinction_mags = fluxcal2.read_atmospheric_extinction(sso_extinction_table=sso_ext_tab)
    # interpolate extinction file to correct wavelength range
    extinction_mags_b = np.interp(lam, wavelength_extinction, 
                                extinction_mags)

    ext_mag_b = extinction_mags_b * airmass
    ext_b = 10.0**(-0.4 * ext_mag_b)
    
    ys,xs = np.shape(imdata1)
    print('shape:',ys,xs)
    
    #do the sum, but correct for exposure time and extinction:
    spec1 = np.nansum(imdata1,axis=0)/exposed/ext_b
    spec2 = np.nansum(imdata2,axis=0)/exposed/ext_b
    xax = np.linspace(1,xs,xs)
    
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)
    ax1.plot(xax+shift,spec1*scale,label='simple sum '+imfile1)
    ax1.plot(xax+shift,spec2,label='simple sum '+imfile2)

    # plot ratio of two simple sums:
    ax2.plot(xax,spec1*scale/spec2)
    ax2.set(xlabel='wavelength (pixels)',ylabel='fit/sum',ylim=[0.0,2.0])
    ax2.axhline(1.0,linestyle=':')
    
    # now try to get the flux calibration extension
    try:
        fdata = hdulist1['FLUX_CALIBRATION'].data
        fc_header=hdulist1['FLUX_CALIBRATION'].header
            
        ys,xs = np.shape(fdata)
        flux = fdata[0, :] * scale
        sigma = fdata[1, :]
        background = fdata[2, :]
        
        ax1.plot(xax,flux,label='moffat profile fit')
        ax1.plot(xax,background,label='moffat profile background')
        ax2.plot(xax,flux/spec1)
        
        # generate a smoothed ratio:
        ratio = flux/spec1
        smooth_sigma=10.0
        V = ratio.copy()
        V[ratio!=ratio]=0
        VV = nd.filters.gaussian_filter(V,(smooth_sigma))

        W = 0*ratio.copy()+1
        W[ratio!=ratio] = 0
        WW = nd.filters.gaussian_filter(W,(smooth_sigma))

        ratio_conv = VV/WW
        ax2.plot(xax,ratio_conv)
        print('median ratio: ',np.nanmedian(ratio))
    except KeyError:
        print('FLUX_CALIBRATION extension not found, skipping ',imfile1)

        

        
    ax1.legend(prop={'size':10})
        

#########################################################
# function to read a SAMI aperture spectrum:
#
def sami_read_apspec(hdulist,extname,doareacorr=False,bugfix=True):

    sami_flux_blue = hdulist[extname].data
    header = hdulist[extname].header
    crval1=header['CRVAL1']
    cdelt1=header['CDELT1']
    crpix1=header['CRPIX1']
    naxis1=header['NAXIS1']
    x=np.arange(naxis1)+1
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    sami_lam_blue=L0+x*cdelt1

    if (doareacorr):
        # fix bug in ap spec code:
        if (bugfix):
            areacorr=areacorr/2.0
        areacorr = header['AREACORR']
        sami_flux_blue = sami_flux_blue * areacorr

    return sami_flux_blue,sami_lam_blue
    

##############################################################################
# script to compare transfer functions

def compare_tf(inlist):
    """Function to compare multiple SAMI transfer functions
    Usage:
    sami_2dfdr_reduction_tests.compare_tf('130306/*/*/*/ccd_1/TRANSFERcombined.fits')
    """

    # initialize lists:
    tf_blue=[]
    tf_red=[]
    tf_blue_hdu=[]
    tf_red_hdu=[]


    # glob to get files:
    infiles = glob.glob(inlist)

    nmax = np.size(infiles)
    print('max files to read:',nmax)

    tf_blue = np.empty(nmax,dtype='U256')
    tf_red = np.empty(nmax,dtype='U256')

    #
    nmaxtf = 1000
    nn = np.zeros(nmaxtf)
    probenum = np.zeros(nmaxtf)
    stdname = np.zeros(nmaxtf,dtype='U256')
    medflux_blue = np.zeros(nmaxtf)
    medflux_red = np.zeros(nmaxtf)
    tf3800 =  np.zeros(nmaxtf)
    tf5700 =  np.zeros(nmaxtf)
    
    # get the paths input:
    nset = 0
    for infile in infiles:
        tf_blue[nset] = infile
        tf_red[nset] = tf_blue[nset].replace('ccd_1','ccd_2')
        # put paths in list with file name:
        #tf_blue.append(path+'/ccd_1/TRANSFERcombined.fits')
        #tf_red.append(path+'/ccd_2/TRANSFERcombined.fits')
        # temp fix to use local files:
        #tf_blue.append(path+'_ccd1_TRANSFERcombined.fits')
        #tf_red.append(path+'_ccd2_TRANSFERcombined.fits')
        print(tf_blue[nset],tf_red[nset])
        nset = nset +1         

    cmap = py.get_cmap(py.cm.rainbow)
    cols = [cmap(i) for i in np.linspace(0, 1, nset)]

    ntf = 0
        
    # open fits files:
    for i in range(nset):
        print('reading ',tf_blue[i])
        tf_blue_hdu=pf.open(tf_blue[i])
        tf_red_hdu=pf.open(tf_red[i])

        #averaged TF for this file
        tf_blue_spec=tf_blue_hdu[0].data
        tf_red_spec=tf_red_hdu[0].data


        primary_header=tf_blue_hdu['PRIMARY'].header
        crval1=primary_header['CRVAL1']
        cdelt1=primary_header['CDELT1']
        crpix1=primary_header['CRPIX1']
        naxis1=primary_header['NAXIS1']
        x=np.arange(naxis1)+1
        L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
        lam_b=L0+x*cdelt1

        primary_header=tf_red_hdu['PRIMARY'].header
        crval1=primary_header['CRVAL1']
        cdelt1=primary_header['CDELT1']
        crpix1=primary_header['CRPIX1']
        naxis1=primary_header['NAXIS1']
        x=np.arange(naxis1)+1
        L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
        lam_r=L0+x*cdelt1

        # get array sizes:
        xs = tf_blue_spec.size
        # x axis:
        #axis = range(xs)
        axis_b = lam_b
        axis_r = lam_r

        # invert TF to get response:
        tf_blue_spec = 1.0/tf_blue_spec
        tf_red_spec = 1.0/tf_red_spec

        # in out test case there are 4 difference obs in the data:
        numext = get_numext(tf_blue_hdu)

        # set up array to hold all data:
        # indexes are:
        # for 4D arrays:
        # 0th 5 different spectrum types per obs (flux, background, sigma_flux, sigma_background, TF)
        # 1st wavelength
        # 2nd number of independent sets to test
        # 3rd number of extensions (i.e. observations) in this set
        # we set the max number of extensions to 100 (should be plenty)
        if (i == 0):
            tf_blue_spec_all=np.zeros((xs,nset))
            tf_red_spec_all=np.zeros((xs,nset))
            tf_blue_ind_all=np.zeros((5,xs,nset,100))
            tf_red_ind_all=np.zeros((5,xs,nset,100))
            
        # get individual TF data by going through each of the extensions.
        # however, only use the extensions called 'FLUX_CALIBRATION'
        jext = 0
        for j in range(1,numext+1):
            ext_header=tf_blue_hdu[j].header
            extname = ext_header['EXTNAME']
            if (extname == 'FLUX_CALIBRATION'):
                origfile = ext_header['ORIGFILE']
                stdname[ntf] = ext_header['STDNAME']
                probenum[ntf] = ext_header['PROBENUM']
                
                tf_blue_ind_all[:,:,i,jext] = tf_blue_hdu[j].data
                tf_red_ind_all[:,:,i,jext] = tf_red_hdu[j].data

                # get median flux:
                medflux_blue[ntf] = np.nanmedian(tf_blue_ind_all[0,:,i,jext])
                medflux_red[ntf] = np.nanmedian(tf_red_ind_all[0,:,i,jext])
                # get the TF value at twho points:
                tf3800[ntf] = tf_blue_ind_all[4,100,i,jext]
                tf5700[ntf] = tf_blue_ind_all[4,1950,i,jext]
                nn[ntf] = ntf
                print(j,basename(origfile),tf3800[ntf],tf5700[ntf],medflux_blue[ntf],medflux_red[ntf],stdname[ntf],probenum[ntf])

                ntf = ntf+1
                jext = jext +1
                
        tf_blue_spec_all[:,i] = tf_blue_spec
        tf_red_spec_all[:,i] = tf_red_spec

        # finish up by closing files:
        tf_blue_hdu.close()
        tf_red_hdu.close()

    # calculate some global average TFs:
    tf_blue_av = np.nanmean(tf_blue_spec_all,axis=1)
    tf_red_av = np.nanmean(tf_red_spec_all,axis=1)
    tf3800_av = tf_blue_av[100]
    tf5700_av = tf_blue_av[1950]
        
    # now that we have read all the files, go through each of them and plot the data.
    py.figure(1)
    for i in range(nset):
        
        # plot the average TF for each set (i.e. each TRANSFERcombined.fits file) and compare between sets.
        py.subplot(5,1,1)
        lab = 'TF '+str(i)+' '+basename(origfile)
        py.plot(axis_b,tf_blue_spec_all[:,i],label=lab,color=cols[i])
        if (i == 0):
            py.plot(axis_b,tf_blue_av,label='average',color='k')
        py.xlim(xmin=axis_b[0],xmax=axis_b[xs-1])
        py.xlabel('wavelength (pixels)')
        py.ylabel('thput')
        print('TF ',i,': ',tf_blue[i])
        py.legend(prop={'size':10})
        
        # plot ratio of TFs, normalized to median:
        medtf = np.nanmedian(tf_blue_spec_all[:,i]/ tf_blue_av)
        ratio = (tf_blue_spec_all[:,i]/ tf_blue_av)/medtf
        py.subplot(5,1,2)
        py.plot(axis_b,ratio,color=cols[i])
        py.xlim(xmin=axis_b[0],xmax=axis_b[xs-1])
        py.ylabel('thput/<thput>')

        # plot individual star spectra for every std exposure:
        py.subplot(5,1,3)
        for j in range(1,numext+1):
            py.plot(axis_b, tf_blue_ind_all[0,:,i,j-1],label=lab,color=cols[i])
        py.xlim(xmin=axis_b[0],xmax=axis_b[xs-1])
        py.ylabel('Flux')

        # plot the individual throughputs for each spectrum:
        py.subplot(5,1,4)
        for j in range(1,numext+1):
            py.plot(axis_b, 1/tf_blue_ind_all[4,:,i,j-1],label=lab,color=cols[i])            
        py.xlim(xmin=axis_b[0],xmax=axis_b[xs-1])
        py.ylabel('thput')

        # plot normalized throughputs for each spectrum, but divide through by mean TP, so
        # we just see the spectrum to spectrum variation:
        py.subplot(5,1,5)
        for j in range(1,numext+1):
            medtp = np.nanmedian((1/tf_blue_ind_all[4,:,i,j-1])/tf_blue_av)
            py.plot(axis_b, ((1/tf_blue_ind_all[4,:,i,j-1]))/tf_blue_av/medtp,label=lab,color=cols[i])            
        py.xlim(xmin=axis_b[0],xmax=axis_b[xs-1])
        py.xlabel('Wavelength (pixels)')
        py.ylabel('normalized thput')
        
        # next plot the individual TFs for observation which are in extensions to the main TF.
    for i in range(nset):
        # repeat for red arm:
        # plot the average TF and compare between sets.
        py.figure(2)
        py.subplot(5,1,1)
        py.plot(axis_r,tf_red_spec_all[:,i],color=cols[i])
        if (i == 0):
            py.plot(axis_r,tf_red_av,color='k')
        py.xlim(xmin=axis_r[0],xmax=axis_r[xs-1])
        py.ylabel('Thput')

        # plot ratio of TFs
        medtf = np.nanmedian(tf_red_spec_all[:,i]/ tf_red_av)
        ratio = (tf_red_spec_all[:,i]/ tf_red_av)/medtf
        py.subplot(5,1,2)
        py.plot(axis_r,ratio,color=cols[i])
        py.xlim(xmin=axis_r[0],xmax=axis_r[xs-1])
        py.ylabel('thput/thput$_0$')

        # plot individual star spectra
        py.subplot(5,1,3)
        for j in range(1,numext+1):
            py.plot(axis_r, tf_red_ind_all[0,:,i,j-1],label=lab,color=cols[i])        
        py.xlim(xmin=axis_r[0],xmax=axis_r[xs-1])
        py.ylabel('Flux')

        py.subplot(5,1,4)
        for j in range(1,numext+1):
            py.plot(axis_r, 1/tf_red_ind_all[4,:,i,j-1],label=lab,color=cols[i])        
        py.xlim(xmin=axis_r[0],xmax=axis_r[xs-1])
        py.ylabel('thput')

        py.subplot(5,1,5)
        for j in range(1,numext+1):
            medtp = np.nanmedian((1/tf_red_ind_all[4,:,i,j-1])/tf_red_av)
            py.plot(axis_r, (1/tf_red_ind_all[4,:,i,j-1])/tf_red_av/medtp,label=lab,color=cols[i])            
        py.xlim(xmin=axis_r[0],xmax=axis_r[xs-1])
        py.xlabel('Wavelength (pixels)')
        py.ylabel('normalized thput')

        
    # now plot some trends for different TFs.
    py.figure(3)
    py.subplot(2,2,1)
    py.plot(nn[0:ntf],tf3800[0:ntf],'o',color='b')
    py.plot(nn[0:ntf],tf5700[0:ntf],'o',color='r')
    py.xlabel('TF number (date order)')
    py.ylabel('TF value red/blue')
    
    py.subplot(2,2,2)
    py.plot(medflux_blue[0:ntf],tf3800[0:ntf],'o',color='b')
    py.plot(medflux_blue[0:ntf],tf5700[0:ntf],'o',color='r')
    py.xlabel('median blue std flux')
    py.ylabel('TF value red/blue')

    py.subplot(2,2,3)
    py.plot(probenum[0:ntf],tf3800[0:ntf],'o',color='b')
    py.plot(probenum[0:ntf],tf5700[0:ntf],'o',color='r')
    py.xlabel('probe number')
    py.ylabel('TF value red/blue')

    
    

##############################################################################

def get_numext(hdulist):
    """Returns the number of extensions in a FITS file"""

    ihdu = 0
    while (True):
        try:
            hdu = hdulist[ihdu]
        except IndexError:
            nhdu = ihdu - 1
            print('number of extensions found:',nhdu)
            return nhdu
        ihdu = ihdu+1

    return ihdu



        
