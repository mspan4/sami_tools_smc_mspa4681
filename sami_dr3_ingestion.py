import glob,os,shutil,gzip
import numpy as np
import astropy.io.fits as fits
from astropy.io import ascii
from astropy import table
from astropy.table import Table
from astropy.time import Time
from datetime import datetime
from os.path import basename
import pylab as py
import subprocess
import re
import logging
import multiprocessing as mp

'''
Based on sami_dr2_ingestion.py written by Nic Scott.  
Updated by Scott Croom for DR3, including converting from python2 to 
python3 using 2to3.

To create an ingestion ready file system with sami dr3 data:
create_directory_structure(root_directory)
create_basic_metadata_files(root_directory)
create_coordinate_meta_files(root_directory,input_cat_filepath,ids_list_filepath)
reformat_many(ids_list_filepath,root_directory,cube_directory,aperture_directory)

IMPORTANT: when passing the root_directory, you need to have a trailing '/' at the
end of the string you give the code, e.g:  This should be fixed by combining file
names and paths more robustly, but is not currently.

Here is the current correct set of commands:

1) first pull all the QC info from the cubes:

> sami_dr3_ingestion.extract_qc_data('/import/bill1/sami_data/data_releases/v0.12/cubes/*.fits')

2) next apply QC selection to the different file to generate the file list.

> sami_dr3_ingestion.create_cubelist('sami_dr3_cube_qc.fits')  

3) create directory structure

> sami_dr3_ingestion.create_directory_structure('dr3_ingestion/')

4) create basic metadata files (no longer need to make coord meta separately):

> sami_dr3_ingestion.create_basic_metadata_files('dr3_ingestion/')  

6) reformat data files for DR:

> sami_dr3_ingestion.ingest_all_data('dr3_ingestion/',testrun=True,dataflag=1)    

Note testrun=True only generates 10 objects.  dataflag=1 only does cubes.

7) make full product ingestion metadata file:

> sami_dr3_ingestion.create_product_ingestion('dr3_ingestion/') 

8) make the CubeObs catalogue with quality flags and verify data products present:

> sami_dr3_ingestion.verify_ingestion_data('dr3_ingestion_v2/') 


9) ingest catalogue files:

> sami_dr3_ingestion.add_all_tables('dr3_ingestion/') 

###########
old commands:
> sami_dr3_ingestion.create_directory_structure('dr3_ingestion_v1/')

Here are specific formats for calls, that will be updated as we get close to 
the final run:

> sami_dr3_ingestion.create_coordinate_meta_files('dr3_ingestion_v1/','/import/bill1/sami/dr3_storage/sami_sel_20140911_v2.0_FINALobs.dat','/import/bill1/sami/dr3_storage/DR2_ids_final_v2.dat') 



>sami_dr3_ingestion.reformat_many('dr3_ingestion_v1/','/export/bill1/sami/dr3_storage/DR2_ids_final_v2.dat','/import/bill1/sami_data/data_releases/v0.11/','/import/bill1/sami_data/data_releases/v0.11/aperture_spectra/','/import/bill1/sami_data/data_products/DR2/StellarKinematics/dr2_wcs_corrected_files_stelkin/','/import/bill1/sami_data/data_products/DR2/lzifu_newcont_fits/1_comp/','/import/bill1/sami_data/data_products/DR2/lzifu_newcont_fits/recom_comp/','/import/bill1/sami_data/data_products/DR2/lzifu_binned_fits/',testrun=True) 

The above is actually equivalent to:

> sami_dr3_ingestion.ingest_all_data('dr3_ingestion_v1/',testrun=True,dataflag=1) 





'''

# global variables that define some top level things for the data release:

DR = 'dr3'
DRVER = 'V3'
REDVER = 'V0.12.1'
DRDATE = '2020-07-01'

# global paths:
SAMIDATAREL = '/import/bill1/sami_data/data_releases/v0.12/'
SAMIDATAPROD = '/import/bill1/sami_data/data_products/v0.12/'
LISTFILE = '/export/bill1/sami/dr3_storage/cube_list.fits'
QCFILE = '/export/bill1/sami/dr3_storage/sami_dr3_cube_qc.fits'
#LISTFILE = '/export/bill1/sami/dr3_storage/DR2_ids_final_v2.dat'
CATFILE = '/export/bill1/sami/dr3_storage/sami_sel_20140911_v2.0_FINALobs.dat'
CATFILEPATH = '/export/bill1/sami/dr3_storage/'
# temp path for cube testing:
#CUBEPATH = '/export/bill1/sami/dr3_ingestion_tests/test_cubes/'
CUBEPATH = SAMIDATAREL+'cubes/'
APERPATH = SAMIDATAREL+'aperture_spectra/'
STELKINPATH = SAMIDATAPROD+'StellarKinematics/kinematic_results_M2/'
STELKINPATH4 = SAMIDATAPROD+'StellarKinematics/kinematic_results_M4/'
LZIFUPATH = os.path.join(SAMIDATAPROD,'LZIFU/')
LZIFU1COMPPATH = SAMIDATAPROD+'lzifu_newcont_fits/1_comp/'
LZIFURECOMPATH = SAMIDATAPROD+'lzifu_newcont_fits/recom_comp/'
LZIFUBINNEDPATH = SAMIDATAPROD+'lzifu_binned_fits/'


################################################
# check the number of extensions in files, particularly
# for LZIFU files
def check_ext(inlist):
    
    # glob the list:
    infiles = sorted(glob.glob(inlist))
    nfiles = len(infiles)

    # Define a dictionary to hold count of extension names
    ext_names ={}
    
    # list of expected extensions:
    expected_ext = ['PRIMARY', 'B_CONTINUUM', 'R_CONTINUUM', 'B_LINE', 'R_LINE', 
                    'B_LINE_COMP1', 'R_LINE_COMP1', 'V', 'V_ERR', 'VDISP', 'VDISP_ERR', 
                    'CHI2', 'DOF', 'OII3726', 'OII3726_ERR', 'OII3729', 'OII3729_ERR', 
                    'HBETA', 'HBETA_ERR', 'OIII5007', 'OIII5007_ERR', 'OI6300', 'OI6300_ERR', 
                    'HALPHA', 'HALPHA_ERR', 'NII6583', 'NII6583_ERR','SII6716','SII6716_ERR',
                    'SII6731','SII6731_ERR']

    nexpected = len(expected_ext)
    found_ext = np.zeros(nexpected, dtype=np.int64)

    # empty list for problems files:
    prob_files = []

    # loop through files:
    ifile = 0
    for infile in infiles:
        print('checking ',infile,ifile,nfiles)
        hdulist = fits.open(infile)
        # loop through all HDUs in the file:
        nhdu = 0
        # reset all values of found array to false:
        found_ext.fill(0)
        # use hdulist.info to get all the details we need:
        hdu_info = hdulist.info(output=False)
        # use zip to extract just the extension names:
        extnames = list(zip(*hdu_info))[1]
        for extname in extnames:

            for i in range(nexpected):
                if (extname == expected_ext[i]):
                    found_ext[i] = 1
            # does this ext name exist in the dictionary
            if (extname in ext_names):
                ext_names[extname] = ext_names[extname] + 1
            else:
                ext_names[extname] = 1

        nfound = np.sum(found_ext)
        if (nfound != nexpected):
            print('extension missing in',infile)
            prob_files.append(infile)
            for i in range(nexpected):
                if (found_ext[i] == 0):
                    print(expected_ext[i])

        ifile=ifile+1

    for ext in ext_names:
        print(ext,ext_names[ext])
                
    for files in prob_files:
        print(files)

#################################################
# fix CRPIX values for ra,dec WCS.
#
def fix_crpix(inlist,sim=False):
    
    # glob the list:
    infiles = sorted(glob.glob(inlist))
    
    # loop through files:
    for infile in infiles:
        print('updating ',infile)
        hdulist = fits.open(infile,mode='update')
        # loop through all HDUs in the file:
        nhdu = 0
        for hdu in hdulist:
            hdr = hdu.header
            if (nhdu != 0):
                # try to get the extension name, but if it 
                # isn't there, skip it
                try:
                    extname = hdr['EXTNAME']
                except KeyError:
                    continue
            else:
                extname = 'Primary HDU'
                
            # check to see if WCS info is in the header.  Check
            # for CRVAL2, as this means the data must be 2D, e.g.
            # can't be a 1D spectrum.
            if 'CRPIX2' in hdr:
                print('Checking WCS in extension: ',extname,', CRPIX1,CRPIX2 = ',hdr['CRPIX1'],hdr['CRPIX2'])
                # only update if not in simulation mode:
                if (not sim):
                    if (abs(hdr['CRPIX1']-25.0) < 0.05):
                        print (hdr['CRPIX1'],'->',25.5)
                        print (hdr['CRPIX2'],'->',25.5)
                        hdr['CRPIX1'] = 25.5
                        hdr['CRPIX2'] = 25.5
                        print('Updated WCS in extension ',extname) 
                                
                    nhdu = nhdu+1
                    
        # close the file and fix any verification problems.  This is usually
        # due to problems with the location of particular header cards.
        hdulist.close(output_verify='fix')


#################################################
# fix WCS CRVAL1,CRVAL2 for objects with BAD_CLASS=5 that have offsets and also
# adjust for WCS offsets that Sree Oh has defined.  Finally, also fix up w
#
def fix_wcs_all(inlist,sim=False):

    """Use the coordinates from the catalogue file to fix the WCS values
    for objects that have BAD_CLASS=5.  These have coords that are based
    of the IFU pointing, but they should have values based on the object 
    coordinates, as the IFU centre is based on the gaussing fit to the 
    centre of the object.

    catfile - input catalogue file with _IFU and _OBJ coordinates
    inlist - list of potential files to update
    sim - run in simulation mode (i.e. don't actually make the changes).

    """

    # glob the list:
    infiles = glob.glob(inlist)
    
    # define the catalogue file.  This is only the GAMA input catalogue
    # as other catalogues have not got WCS offsets with BAD_CLASS=5
    #catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_dr3/InputCatGAMADR3.fits'
    catfile = CATFILEPATH+'/InputCatGAMADR3.fits'

    # read the catalogue file
    hdulist = fits.open(catfile)
    tab = hdulist[1].data

    # make a list of CATIDs with BAD_CLASS=5
    idx = np.where(tab['BAD_CLASS'] == 5)
    catid = tab['CATID'][idx]
    ra_obj = tab['RA_OBJ'][idx]
    dec_obj = tab['DEC_OBJ'][idx]
    ra_ifu = tab['RA_IFU'][idx]
    dec_ifu = tab['DEC_IFU'][idx]

    nobj = np.size(catid)
    print('number of objects to fix with BAD_CLASS=5:',nobj)

    # next read the updates from Sree:
    #cent_update_file = '/Users/scroom/data/sami/dr3/wcs_checks/fix_wcs/wcs_correction_new.txt'
    cent_update_file = CATFILEPATH+'/wcs_correction_new.txt'
    print(cent_update_file)
    tab_update = Table.read(cent_update_file,format='ascii')
    # rename columns to soemthing useful:
    tab_update['col1'].name = 'CATID'
    tab_update['col2'].name = 'REPFLAG'
    tab_update['col3'].name = 'DRA'
    tab_update['col4'].name = 'DDEC'
    nobj_update = np.size(tab_update['CATID'])
    
    # loop through all the CATIDs for BAD_CLASS:
    for i in range(nobj):

        catid_str = str(catid[i])
        # identify files in the list that are for this CATID:
        for infile in infiles:
            if catid_str in infile:
                print('updating ',infile)
                hdulist = fits.open(infile,mode='update')
                # loop through all HDUs in the file:
                nhdu = 0
                for hdu in hdulist:
                    hdr = hdu.header
                    if (nhdu != 0):
                        try:
                            extname = hdr['EXTNAME']
                        except KeyError:
                            continue
                    else:
                        extname = 'Primary HDU'
                    # check to see if WCS info is in the header.  Check
                    # for CRVAL2, as this means the data must be 2D, e.g.
                    # can't be a 1D spectrum.
                    if 'CRVAL2' in hdr:
                        print('Checking WCS in extension: ',extname,', CRVAL1,CRVAL2 = ',hdr['CRVAL1'],hdr['CRVAL2'])
                        # as a test, calculate the difference between the header items and the
                        # ra/dec.  As we are testing against the IFU coords these should be very close
                        # zero:
                        ra_diff = (hdr['CRVAL1'] - ra_ifu[i])*3600.0
                        dec_diff = (hdr['CRVAL2'] - dec_ifu[i])*3600.0
                        # if the difference is small, then these must coordinates as we expect
                        # we can then update them:
                        if ((abs(ra_diff) < 0.2) & (abs(dec_diff) < 0.2)):
                            # only update if not in simulation mode:
                            if (not sim):
                                print (hdr['CRVAL1'],'->',ra_obj[i],'delta=',(hdr['CRVAL1']-ra_obj[i])*3600.0)
                                print (hdr['CRVAL2'],'->',dec_obj[i],'delta=',(hdr['CRVAL2']-dec_obj[i])*3600.0)
                                hdr['CRVAL1'] = ra_obj[i]
                                hdr['CRVAL2'] = dec_obj[i]
                                print('Updated WCS in extension ',extname) 
                                
                    nhdu = nhdu+1
                    
                # close the file
                hdulist.close(output_verify='fix')

    # now loop through all the CATIDs for Sree's updates:
    for i in range(nobj_update):

        catid_str = str(tab_update['CATID'][i])
        # identify files in the list that are for this CATID:
        for infile in infiles:
            if catid_str in infile:
                print('updating ',infile)
                hdulist = fits.open(infile,mode='update')
                # loop through all HDUs in the file:
                nhdu = 0
                for hdu in hdulist:
                    print(nhdu)
                    hdr = hdu.header
                    if (nhdu != 0):
                        try:
                            extname = hdr['EXTNAME']
                        except KeyError:
                            continue
                    else:
                        extname = 'Primary HDU'
                    # check to see if WCS info is in the header.  Check
                    # for CRVAL2, as this means the data must be 2D, e.g.
                    # can't be a 1D spectrum.
                    if 'CRVAL2' in hdr:
                        print('Checking WCS in extension: ',extname,', CRVAL1,CRVAL2 = ',hdr['CRVAL1'],hdr['CRVAL2'])
                        # check if already fixed:
                        if ('WCSFIX' in hdr):
                            if (hdr['WCSFIX'] == True):
                                print('Skipping as WCSFIX is set and true')
                                continue
                        
                        # only update if not in simulation mode:
                        
                        if (not sim):
                            print (hdr['CRVAL1'],'->',hdr['CRVAL1']+tab_update['DRA'][i],' Delta = ',tab_update['DRA'][i]*3600.0)
                            print (hdr['CRVAL2'],'->',hdr['CRVAL2']+tab_update['DDEC'][i],' Delta = ',tab_update['DDEC'][i]*3600.0)
                            hdr['CRVAL1'] = hdr['CRVAL1']+tab_update['DRA'][i]
                            hdr['CRVAL2'] = hdr['CRVAL2']+tab_update['DDEC'][i]
                            hdr['WCSFIX'] = (True,'Flag indicating WCS has been adjusted')
                            print('Updated WCS in extension ',extname) 
                                
                    nhdu = nhdu+1
                    
                # close the file
                hdulist.close(output_verify='fix')


                
##########################################


def fix_binned_zeros(cubes,apspecs):

    inputs = zip(cubes,apspecs)

    pool = mp.Pool(18)
    pool.map(fix_binned_zeros_single,inputs)
    pool.close()
    pool.join()

###############################################

def fix_binned_zeros_single(inputs):

    cube,apspec = inputs[0],inputs[1]
    hc = fits.open(cube,'update')
    extc = [c[1] for c in hc.info(output=False)]


    hc[0].data[np.isfinite(hc[0].data)] = 1.0

    for ext in extc:
        if ('MASK' in ext) and ('ADAPTIVE' not in ext):
            fluxext = ext.replace('_MASK','NED_FLUX')
            varext = ext.replace('_MASK','NED_VARIANCE')

            mask = hc[ext].data
            for binid in np.unique(mask)[1:]:
                ww = np.where(mask == binid)
                #spec = np.sum(hc[0].data[:,ww[0],ww[1]],axis=1)
                #vv = np.where(np.isfinite(spec) == False)
                spec = np.nansum(hc[0].data[:,ww[0],ww[1]],axis=1)/len(ww[0])
                vv = np.where(spec <= 0.4)[0]
                for v in vv:
                    hc[fluxext].data[v,ww[0],ww[1]] = np.nan
                    hc[varext].data[v,ww[0],ww[1]] = np.nan
    try:
        ha = fits.open(apspec,'update')
    except:
        return
    exta = [a[1] for a in ha.info(output=False)]

    for ext in exta:
        if 'MASK' in ext:
            fluxext = ext.replace('_MASK','')
            varext = ext.replace('MASK','VAR')

            mask = ha[ext].data
            ww = np.where(mask == 1)
            #spec = np.sum(hc[0].data[:,ww[0],ww[1]],axis=1)
            #vv = np.where(np.isfinite(spec) == False)
            spec = np.nansum(hc[0].data[:,ww[0],ww[1]],axis=1)/len(ww[0])
            vv = np.where(spec <= 0.4)[0]
            ha[fluxext].data[vv] = np.nan
            ha[varext].data[vv] = np.nan

    hc.flush()
    hc.close()
    ha.flush()
    ha.close()

    return

#################################################

def update_cube_qc_data(cubes_in,sim=False):

    """This function looks at each cube and checks the QC data.  If it is not
    complete then all the RSS frames that make up the cube are checked to pull
    out the QC data and put it back into the cube QC extension.  If sim=True then
    run in simulation mode and do not actually change files.  This should be only
    needed for early versions of DR3 when QC data was not complete due to 
    missing throughput values."""

    # get the list of cubes.  Note that glob does not sort the files, so
    # we do it here (not that it matters directly for this code):
    cubelist = sorted(glob.glob(cubes_in))

    #number of files in list (might include red and blue):
    nfiles = np.size(cubelist)

    print('Number of cube files in input list: ',nfiles)

    # glob all the rss frames from the start to save time later:
    rsslist = sorted(glob.glob('/import/opus1/nscott/SAMI_Survey/20??_??_??-20??_??_??/reduced/1*/Y*/Y*/main/ccd_?/*sci.fits'))
    
    nrss = np.size(rsslist)
    print('Number of RSS files in list: ',nrss)

    # loop over cubes:
    nc = 0
    nbad = 0
    for cube in cubelist:
        
        # open the QC extension:
        cubename = basename(cube)

        # open cube:
        if (sim):
            hdulist = fits.open(cube)
        else:
            hdulist = fits.open(cube,mode='update')

        # get cube FWHM (not used directly):
        cubefwhm = hdulist['PRIMARY'].header['PSFFWHM']

        # get other useful info from cube primary header:
        catid = hdulist['PRIMARY'].header['NAME']
        plateid = hdulist['PRIMARY'].header['PLATEID']
        ifuprobe = hdulist['PRIMARY'].header['IFUPROBE']

        # open and get data from QC extension:
        qc_table = hdulist['QC'].data
        rssfilename = qc_table['filename']
        trans = qc_table['TRANSMIS']
        rtrans = qc_table['rel_transp']
        nframe = np.size(trans)
        
        # find the rss frames from list that match the QC info:
        for nf in range(nframe):
            good = True
            filelabel = rssfilename[nf]
            nfound = 0
            for i in range(nrss):
                if ((filelabel in rsslist[i]) and (plateid in rsslist[i])):
                    nfound = nfound + 1
                    rssmatch = rsslist[i]
            if (nfound != 1):
                print('did not find one match',nf,filelabel,nfound)
                print(filelabel,fieldlabel,plateid)
                print('no changes being made')
                nbad = nbad + 1
                good = False
            
            # get the QC transmission data from the rss frame
            rsshdu = fits.open(rssmatch)
            new_trans = rsshdu['QC'].header['TRANSMIS']
            new_rescale = rsshdu['FLUX_CALIBRATION'].header['RESCALE'] 
            new_rtrans = 1.0/new_rescale
            #print(trans[nf],new_trans,rtrans[nf],new_rtrans)
            # update the data IF a good match was found.
            if (good):
                if (new_trans > 0.0):
                    trans[nf] = new_trans

                if (new_rtrans > 0.0):
                    rtrans[nf] = new_rtrans
            else:
                print('QC values not updating as good=False')

        print('Reading QC for ',cubename, nc, ' of ',nfiles)

        if (sim):
            print('WARNING: In simulation mode, not updating cubes')
            print('TRANSMIS:',trans[0:nframe])
            print('rel_transp:',rtrans[0:nframe])
        else:
            hdulist.flush()

        hdulist.close()

        nc = nc + 1

    print('Number of rss frames not found (or not unique):',nbad)

#######################################################

def extract_qc_data(cubes_in):

    """This routine extracts QC data from cube files and writes it to a FITS 
    binary table to be used by other routines later.  This is because it is
    quicker to access the binary table than keep getting the info from the cubes."""


    # glob the input cube list.  Note that glob does NOT(!!!) sort by file name
    # like a standard ls, so could have some slighly odd ordering.  To fix that we
    # also do a sort here:
    cubelist = sorted(glob.glob(cubes_in))

    #number of files in list (might include red and blue):
    nfiles = np.size(cubelist)
    maxnframe = 20

    maxnframe_found = 0

    print('Number of cube files in input list: ',nfiles)

    # set up arrays to hold data:
    cubefwhm = np.zeros(nfiles)
    cubetexp = np.zeros(nfiles)
    frametrans = np.zeros((nfiles,maxnframe))
    framertrans = np.zeros((nfiles,maxnframe))
    framefwhm = np.zeros((nfiles,maxnframe))
    framenum = np.zeros(nfiles)

    cubenames = np.empty(nfiles,dtype='U128')
    cubecatid = np.empty(nfiles,dtype='U32')
    firstrssfile = np.empty(nfiles,dtype='U24')

    # loop over cubes:
    nf = 0
    for cube in cubelist:
        
        
        # to get the QC stats only look at the blue cube (red is the same):

        if (cube.find('red') > 0):
            continue

        cubenames[nf] = basename(cube)
        print('Reading QC for ',cubenames[nf])

        # open cube:
        hdulist = fits.open(cube)

        # get cube FWHM:
        cubefwhm[nf] = hdulist['PRIMARY'].header['PSFFWHM']

        # get the catid from the header:
        cubecatid[nf] = hdulist['PRIMARY'].header['NAME']

        # get total exp time:
        cubetexp[nf] = hdulist['PRIMARY'].header['TOTALEXP']

        # get first RSS file:
        firstrssfile[nf] = hdulist['PRIMARY'].header['RSS_FILE 1']

        # get data from QC extension:
        qc_table = hdulist['QC'].data
        trans = qc_table['TRANSMIS']
        rtrans = qc_table['rel_transp']
        fwhm = qc_table['FWHM']
        nnf = np.size(trans)

        if (nnf > maxnframe):
            print('ERROR: number of frames > maxnframe')
            return

        if (nnf > maxnframe_found):
            maxnframe_found = nnf

        # put data into arrays:
        frametrans[nf,0:nnf] = trans
        framertrans[nf,0:nnf] = rtrans
        framefwhm[nf,0:nnf] = fwhm
        framenum[nf] = nnf

        # note - some objects seem to have transmision data of -9999.
        # not sure why this is, but need to catch it  - probably don't want to reject
        # these fields.  For this reason, also keep the relative transmission data
        # so that this can be used instead.
        
        nf = nf + 1


    # because it takes time to read all the QC info from the cubes (e.g.
    # due to large cube size and zipping etc., we will now write the 
    # QC data to a binary table

    # define the columns in the output file:
    col1 = fits.Column(name='CUBENAME',format='128A',array=cubenames[0:nf])
    col2 = fits.Column(name='CATID',format='32A',array=cubecatid[0:nf])
    col3 = fits.Column(name='NFRAMES',format='E',array=framenum[0:nf])
    col4 = fits.Column(name='CUBEFWHM',format='E',array=cubefwhm[0:nf])
    col5 = fits.Column(name='CUBETEXP',format='E',array=cubetexp[0:nf])
    col6 = fits.Column(name='FRAMEFWHM',format='20E',array=framefwhm[0:nf,:])
    col7 = fits.Column(name='FRAMETRANS',format='20E',array=frametrans[0:nf,:])
    col8 = fits.Column(name='FRAMERTRANS',format='20E',array=framertrans[0:nf,:])
    col9 = fits.Column(name='FIRSTRSSFILE',format='24A',array=firstrssfile[0:nf])
    cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9])
    hdutab = fits.BinTableHDU.from_columns(cols,name='QC_DATA')

    outfile = 'sami_dr3_cube_qc.fits'
    hdutab.writeto(outfile,overwrite=True)

    print('QC data written to: ',outfile)
    print('total number of cubes: ',nf)
    print('max number of frames in a cube: ',maxnframe_found)

#####################################################################

def get_new_catids(qcfilenew,qcfileold):
    """compare two qc file lists and write out the CATIDs of any objects
    that are missing from one of them"""

    # open file and get data:
    hdulist = fits.open(qcfilenew)
    table_data = hdulist['QC_DATA'].data
    
    # put data into useful arrays:
    cubenames_new = table_data['CUBENAME']
    cubecatid_new = table_data['CATID']
    hdulist.close()

    # open file and old data:
    hdulist = fits.open(qcfileold)
    table_data = hdulist['QC_DATA'].data
    
    # put data into useful arrays:
    cubenames_old = table_data['CUBENAME']
    cubecatid_old = table_data['CATID']
    hdulist.close()
    
    n_new = np.size(cubenames_new)
    n_old = np.size(cubenames_old)
    print('number of new cubes read :',n_new)
    print('number of old cubes read :',n_old)
    n_notfound = 0
    
    # open output file:
    outfile = 'cubes_notfound.txt'
    fout = open(outfile,mode='w')

    # go through new list:
    for i in range(n_new):
        cubename = cubenames_new[i]
        #print(cubename)
        found = False
        # check if in the old list:
        if (np.isin(cubename,cubenames_old)):
            found = True
        # if not found, output cubename and catid:
        if (not found):
            n_notfound = n_notfound + 1
            print('not found: ',cubenames_new[i],cubecatid_new[i])
            outstr = '{0:s} {1:s}\n'.format(cubenames_new[i],cubecatid_new[i])
            fout.write(outstr)

    #finish up:
    print('Number of cubes not found: ',n_notfound)
    fout.close()
    return

###########################################################

def find_cat_source(catids):
    """Function to look in each of the catalogue files and see what the source of
    the cube file is.  This just cross-checks with the CATID.  There should be a 
    something present for every cube, but there are a few odd-ball cases (empty
    bundles etc) where there are not.  The flags that are set are as follows:
     -1: not in catalogue file
      1: SAMI GAMA catalogue file
      2: SAMI cluster catalogue file
      3: Extra filler targets
      4: calibrations stars GAMA
      5: calibration stars clusters

    """
    
    # number of values to test:
    nc = np.size(catids)
    icatids = np.zeros(nc,dtype=np.int64)

    # define catsource array:
    catsource = np.zeros(nc,dtype=np.int32)

    # make sure the input list are integer (sometimes string)
    for i in range(nc):
        icatids[i] = int(catids[i])

    #Read in the GAMA catalogue file:
    hdulist = fits.open(CATFILEPATH+'sami_sel_20140911_v2.0_FINALobs.fits')
    catid_gama = hdulist[1].data['CATID']
    z_spec_gama =  hdulist[1].data['z_spec']

    # get a boolean array with all the flags:
    gama_flag = np.isin(icatids,catid_gama)
    print('Number of matches for GAMA cat:',np.count_nonzero(gama_flag))

    #Read in the CLUSTER catalogue file:
    #hdulist = fits.open(CATFILEPATH+'ClustersCombined_V10_FINALobs.fits')
    # NOTE THIS IS LISTED AT CATAID IN THE OLD CATALOGUES!!!!!
    #catid_clus = hdulist[1].data['CATAID']
    #z_spec_clus =  hdulist[1].data['Z']
    # new (final) cluster catalogue format:
    hdulist = fits.open(CATFILEPATH+'InputCat_Clusters.fits')
    catid_clus = hdulist[1].data['CATID']
    z_spec_clus =  hdulist[1].data['z_spec']
    # get a boolean array with all the flags:
    clus_flag = np.isin(icatids,catid_clus)
    print('Number of matches for cluster cat:',np.count_nonzero(clus_flag))

    #Read in the filler catalogue file:
    #hdulist = fits.open(CATFILEPATH+'ObservedSecondaryFillers_notmainsurvey_short.fits')
    hdulist = fits.open(CATFILEPATH+'InputCat_Filler.fits')
    catid_fill = hdulist[1].data['CATID']
    z_spec_fill =  hdulist[1].data['z_spec']
    # get a boolean array with all the flags:
    fill_flag = np.isin(icatids,catid_fill)
    print('Number of matches for filler cat:',np.count_nonzero(fill_flag))

    #Read in the GAMA secondary standard file: 
    hdulist = fits.open(CATFILEPATH+'fstarcat_v1.0_GAMA.fits')
    catid_gama_star = hdulist[1].data['RowID']
    # get a boolean array with all the flags:
    gama_star_flag = np.isin(icatids,catid_gama_star)
    print('Number of matches for GAMA stars cat:',np.count_nonzero(gama_star_flag))

    #Read in the GAMA secondary standard file: 
    hdulist = fits.open(CATFILEPATH+'fstarcat_v1.0_clusters.fits')
    catid_clus_star = hdulist[1].data['RowID']
    # get a boolean array with all the flags:
    clus_star_flag = np.isin(icatids,catid_clus_star)
    print('Number of matches for cluster stars cat:',np.count_nonzero(clus_star_flag))

    # define an array to hold z_spec redshifts:
    z_spec = np.zeros(nc)
    z_spec[:] = np.nan

    # Now go through each cube and set the right flag:

    for i in range(nc):
        if (gama_flag[i]):
            catsource[i] = 1
            for j in range(np.size(z_spec_gama)):
                if (icatids[i] == catid_gama[j]):
                    z_spec[i] = z_spec_gama[j] 
        elif (clus_flag[i]):
            catsource[i] = 2
            for j in range(np.size(z_spec_clus)):
                if (icatids[i] == catid_clus[j]):
                    z_spec[i] = z_spec_clus[j] 
        elif (fill_flag[i]):
            catsource[i] = 3
            for j in range(np.size(z_spec_fill)):
                if (icatids[i] == int(catid_fill[j])):
                    z_spec[i] = z_spec_fill[j] 
        elif (gama_star_flag[i]):
            catsource[i] = 4
        elif (clus_star_flag[i]):
            catsource[i] = 5
        else:
            catsource[i] = -1

    return catsource,z_spec
        
##############################################################################

def find_date_firstrss(cubenames,firstrssfile):

    """From a list of cube names and list of first rss files, make a list of 
    start dates for the observations of a cube.  Returns an astropy Time object
    and the run number.  This can then be used to get mjd, e.g. t.mjd."""

    n = np.size(cubenames)

    dates = []
    runnums = []

    # loop over cube names:
    for i in range(n):
        # split by '-' first, then by '_':
        # here we are getting the info from the cube name, but this
        # is only for the start of the run.
        tmp = cubenames[i].split('-')
        tmp2 = tmp[0].split('_')
        year = tmp2[6];
        month = tmp2[7];
        day = tmp2[8];
        # now get the info from the first rss file:
        dayrss = firstrssfile[i][0:2]
        monthrss = firstrssfile[i][2:5]
        runnum =  firstrssfile[i][6:10]

        runnums.append(int(runnum))

        # note that sometimes the months at the start of the 
        # run and the actual month of the frist RSS frame will be different.
        # We should use the rss month:
        monthrss_num = monthToNum(monthrss)

        # the year is always the same, as we have not observed over new
        # year, but we could in principle!  Provide a warning if
        # the month difference is > 1:
        if (abs(int(month) - int(monthrss_num)) > 1):
            print('WARNING: Large difference between month of run and rss frame!!!')
            sys.exit()

        strdate = '-'.join([year,str(monthrss_num),dayrss])
        t = Time(strdate)

        #print(cubenames[i],firstrssfile[i],t,t.mjd)

        dates.append(t)

    # finish and return
    return dates,runnums

################################################
# simple function to return month number:

def monthToNum(shortMonth):

    return{
        'jan' : 1,
        'feb' : 2,
        'mar' : 3,
        'apr' : 4,
        'may' : 5,
        'jun' : 6,
        'jul' : 7,
        'aug' : 8,
        'sep' : 9, 
        'oct' : 10,
        'nov' : 11,
        'dec' : 12}[shortMonth]

##############################################################################

def find_date_start(cubenames):

    """From a list of cube names, make a list of start dates for the run
    in question.  Returns an astropy Time object.  This can then be used to
    get mjd, e.g. t.mjd."""

    n = np.size(cubenames)

    dates = []

    # loop over cube names:
    for i in range(n):
        # split by '-' first, then by '_':
        tmp = cubenames[i].split('-')
        tmp2 = tmp[0].split('_')
        year = tmp2[6];
        month = tmp2[7];
        day = tmp2[8];
        strdate = '-'.join([year,month,day])
        t = Time(strdate)
        dates.append(t)

    # finish and return
    return dates

##############################################################################

def get_cubeid_pub(catid,rflag):

    """From the catid and rflag generate the public cube ID.  Assumed CATID
    is an integer.
    """

    cubeid = str(catid)+'_'+rflag

    return cubeid

##############################################################################

def create_cubelist(qcfile,doplot=True,doplotall=False,qualorder=False):

    """Create the list of cubes to be ingested into DR based on the QC data in the 
    cubes.  The quality information is checked in the qcfile (that contains data from 
    the cube QC extensions).  If qualorder=True, the ordering is based on quality
    with the best being A, then B, C etc.  If qualorder=False, then the ordering is
    based on date observed and number of frames.  Those not to be included (i.e. those
    not in a catalogue, or in the ignore_list) are not output to the binary table."""


    # define a list of cube files to ignore when building the cube list.  This is 
    # generally due to major data issues, e.g. bundle plugged into wrong hole.  The
    # list is of the blue cube file name, but we assume that the red one should also be
    # ignored.  If this was a long list, we would have it in a file, but for now we
    # just place it here, as there are only 2 cubes:
    # 209613 - no flux in cube - suspected plugging in wrong hole.
    # 9011900882 - not in final cluster catalogue due to poor redshift.
    # 592621 - obs twice, 5 frames then 7 frames.  the 7 are at zero flux - plugging error.
    ignore_list = ['209613_blue_7_Y15SAR2_P002_09T108_2016_02_08-2016_02_14.fits',
                   '9011900882_blue_7_Y15SBR2_P003_A0119T040_2016_09_01-2016_09_05.fits',
                   '592621_blue_12_Y17SAR2_P003_15T119_2017_04_19-2017_05_01.fits',
                   '592621_blue_7_Y17SAR2_P003_15T119_2018_03_12-2018_03_22.fits']

    # open file and get data:
    hdulist = fits.open(qcfile)
    table_data = hdulist['QC_DATA'].data
    
    # put data into useful arrays:
    cubenames = table_data['CUBENAME']
    cubecatid = table_data['CATID']
    framenum =  table_data['NFRAMES']
    cubefwhm =  table_data['CUBEFWHM']
    cubetexp =  table_data['CUBETEXP']
    framefwhm = table_data['FRAMEFWHM']
    frametrans = table_data['FRAMETRANS']
    framertrans = table_data['FRAMERTRANS']
    firstrssfile = table_data['FIRSTRSSFILE']

    # get size of lists:
    (nc,nmf) = np.shape(framefwhm)
    print('Shape of vector cols:',nc,nmf)

    # define array to hold CUBEID:
    cubeid = np.empty(nc,dtype='U80')

    # define array to hold CUBEID for public DR3:
    cubeid_pub = np.empty(nc,dtype='U16')
    
    # define array for integer catid:
    icubecatid = np.zeros(nc,dtype = np.int64)
    for i in range(nc):
        icubecatid[i] = int(cubecatid[i])
        print(cubecatid[i],icubecatid[i])

    # generate the CUBEID from the cube name.  This should simply be the removal
    # of the _blue or _red in the name (and the .fits).
    for i in range(nc):
        cubeid[i] = gen_cubeid(cubenames[i]) 

    # get date of first rss file and run number.  Returns a list of
    # astropy Time objects and a list of integers for run numbers.
    datestart,runnums = find_date_firstrss(cubenames,firstrssfile)

    # define a relative MJD that combines the start date and the run number.
    # all we do is add 0.001*runnums to the MJD, so that they are separated
    # and correctly ordered.  Note that the actual value is NOT correct!!!:
    dates_mjd = np.zeros(np.size(datestart))
    for i in range(np.size(datestart)):
        dates_mjd[i] = datestart[i].mjd+0.001*float(runnums[i])


    # get the start date of the run from the cube names.  This returns a list
    # of astropy Time objects:
    #datestart = find_date_start(cubenames)


    # check that the cubes are in the catalogue files, and identify which catalogue file
    # they come from.  Also pull out the z_spec values:
    catsource,z_spec = find_cat_source(cubecatid)

    # set up some arrays:
    framefwhm_mean = np.zeros(nc)
    frametrans_mean = np.zeros(nc)
    framertrans_mean = np.zeros(nc)
    framefwhm_med = np.zeros(nc)
    frametrans_med = np.zeros(nc)
    framertrans_med = np.zeros(nc)
    isbest = np.full(nc,False,dtype=bool)

    qflag = np.empty(nc,dtype='U1')

    # generate some basic statistics for the QC data:
    nlow = 0
    for i in range(nc):
        fn = int(framenum[i])
        framefwhm_mean[i] = np.nanmean(framefwhm[i,0:fn])
        frametrans_mean[i] = np.nanmean(frametrans[i,0:fn])
        framertrans_mean[i] = np.nanmean(framertrans[i,0:fn])
        framefwhm_med[i] = np.nanmedian(framefwhm[i,0:fn])
        frametrans_med[i] = np.nanmedian(frametrans[i,0:fn])
        framertrans_med[i] = np.nanmedian(framertrans[i,0:fn])

        # check if any of the transmission data is below 1/3,
        if (np.min(frametrans[i,0:fn]) < 0.333):
            print('Low transmission found: ',cubenames[i])
            print(frametrans[i,0:fn],framertrans[i,0:fn])
            nlow = nlow + 1

    print('Number of cubes with at least one low transmission frame:',nlow)

    # calculate effective exposure time:
    cube_eff_exp = frametrans_mean * cubetexp 

    # plot some diagnostic plots:
    if (doplot):
        fig1 = py.figure(1)

        # plot of FWHM cube vs mean frames:
        ax1_1 = fig1.add_subplot(2,2,1)
        ax1_1.plot(framefwhm_mean,cubefwhm,'o',color='b')
        ax1_1.set(xlabel='mean frame FWHM (arcsec)',ylabel='cube FWHM (arcsec)',xlim=[0.0,4.0],ylim=[0.0,4.0])
        ax1_1.plot([0.0,10.0],[0.0,10.0],':',color='r')
        
        # plot distribution of frame numbers:
        ax1_2 = fig1.add_subplot(2,2,2)
        ax1_2.hist(framenum,bins=15,range=[0,15.0],histtype='step',color='k')
        ax1_2.set(xlabel='Number of frames',ylabel='Number')

        # plot FWHM dist:
        ax1_3 = fig1.add_subplot(2,2,3)
        ax1_3.hist(cubefwhm,bins=20,range=[0,5.0],histtype='step',color='k')
        ax1_3.set(xlabel='cube FWHM (arcsec)',ylabel='Number')

        # plot transmission dist:
        ax1_4 = fig1.add_subplot(2,2,4)
        ax1_4.hist(frametrans_mean,bins=20,range=[0.0,2.0],histtype='step',color='k')
        ax1_4.set(xlabel='mean frame transmission',ylabel='Number')

        # plot the main selection plot, FWHM vs transmission:
        fig2 = py.figure(2)
        ax2 = fig2.add_subplot(1,1,1)
        ax2.plot(cubefwhm,frametrans_mean,'o',color='b')
        ax2.set(xlabel='cube FWHM (arcsec)',ylabel='mean frame transmission',xlim=[0.5,4.0],ylim=[0.5,1.5])

    # set up arrays enable repeat assessment:
    cubefwhm_rep = np.zeros(100)
    cube_eff_exp_rep = np.zeros(100)
    cubetexp_rep = np.zeros(100)
    cube_framenum_rep = np.zeros(100,dtype='i4')
    cube_mjd_rep = np.zeros(100)
    frametrans_mean_rep = np.zeros(100)
    indx_rep = np.zeros(100,dtype='i4')
    irank = np.zeros(100,dtype='i4')
    cubename_rep = np.empty(100,dtype='U128')
    
    # define repeat flags:
    rflag = ['A','B','C','D','E','F','G','H','I','J','K','L','M']

    # set up plotting:
    if (doplotall):
        fig3 = py.figure(3)
        ax3 = fig3.add_subplot(1,1,1)

    # set up a text output file:
    outtxt = 'cube_list.txt'
    fout = open(outtxt,mode='w')

    outident = 'cube_ident_rep.txt'
    fout_ident = open(outident,mode='w')


    # next identify any repeats and assign flags to order the repeats:
    for i in range(nc):
        
        base_catid = cubecatid[i]
        # loop through the other cubes and find those that have the same
        # CATID:
        nmatch = 0
        cubefwhm_rep[nmatch] = cubefwhm[i]
        cube_eff_exp_rep[nmatch] = cube_eff_exp[i]
        frametrans_mean_rep[nmatch] = frametrans_mean[i]
        cubename_rep[nmatch] = cubenames[i]
        cubetexp_rep[nmatch] = cubetexp[i]
        cube_framenum_rep[nmatch] = framenum[i]
        cube_mjd_rep[nmatch] = dates_mjd[i]
        indx_rep[nmatch] = i
        
#        for j in range(i+1,nc):
        for j in range(1,nc):
            # skip if identical:
            if (i == j):
                continue
            # if this object already has a  flag, then skip.
            # is has already been checked:
            #if (qflag[j].isalpha()):
            #    print('Skipping... ',qflag[j])
            #    continue

            if (base_catid == cubecatid[j]):
                print('match found')
                nmatch = nmatch + 1
                cubefwhm_rep[nmatch] = cubefwhm[j]
                cube_eff_exp_rep[nmatch] = cube_eff_exp[j]
                frametrans_mean_rep[nmatch] = frametrans_mean[j]
                cubename_rep[nmatch] = cubenames[j]
                cubetexp_rep[nmatch] = cubetexp[j]
                cube_framenum_rep[nmatch] = framenum[j]
                cube_mjd_rep[nmatch] = dates_mjd[j]
                indx_rep[nmatch] = j

        # if no match is found, set the flag to 'A' and continue:
        if (nmatch == 0 and (not qflag[i].isalpha())):
            qflag[i] = 'A'
            isbest[i] = True

        # if we find a match, then we need to figure out which 
        # of the cubes is the best one.  Note that if the cubes 
        # are star cubes, we will not rank them.  Instead they
        # are just ranked by date of observation.
        if (nmatch > 0):

            # if we are plotting each individual matched set, do it here:
            if (doplotall):
                ax3.cla()
                ax3.plot(cubefwhm,cube_eff_exp,'o',color='k')
                ax3.set(ylim=[0.0,27000.0])

            nmm = nmatch+1

            # assign the ranking for frames.
            #
            # here we do a basic ranking that is just based on exposure time
            # and seeing.   What we will do is pick the longest exposure time
            # unless seeing is a bit worse (say 10%):
            #
            if (qualorder):
                # sort by data quality:
                rep_exp_sindex = sort_rep_obs(cubetexp_rep[0:nmm],cubefwhm_rep[0:nmm])
                ibest = rep_exp_sindex[0]
            else:
                # sort by data quality, but also return the listing for the quality
                # sort, so that we can identify the best cube:
                rep_exp_sindex = sort_rep_obs_date(cube_framenum_rep[0:nmm],cube_mjd_rep[0:nmm])
                rep_exp_sindex_qual = sort_rep_obs(cubetexp_rep[0:nmm],cubefwhm_rep[0:nmm])

                # check for identical MJD and framenum:
                for j in range(nmm):
                    for jj in range(j+1,nmm):
                        if ((cube_framenum_rep[j] == cube_framenum_rep[jj]) & (cube_mjd_rep[j] == cube_mjd_rep[jj])):
                            
                            sout = cubename_rep[j].rstrip()+' '+cubename_rep[jj].rstrip()+'\n'
                            fout_ident.write(sout)
                            print(cubename_rep[j].rstrip())
                            print(cubename_rep[jj].rstrip())

                # use the quality ordering to figure out which is the best;
                ibest = rep_exp_sindex_qual[0]

            # if ibest is zero (i.e. the main object in the list is the best), 
            # then set isbest to true:
            if (ibest == 0):
                isbest[i] = True

            # assign flags:
            for j in range(nmm):
                jj = rep_exp_sindex[j]
                qflag[indx_rep[jj]] = rflag[j]

            print('ibest: ',ibest)
            for j in range(nmm):
                # assign flags (randomly for now):                
                #qflag[indx_rep[j]] = rflag[j]
                print(cubename_rep[j].rstrip(),cubefwhm_rep[j],cubetexp_rep[j],indx_rep[j],qflag[indx_rep[j]])
            if (doplotall):
                ax3.plot(cubefwhm_rep[0:nmm],cube_eff_exp_rep[0:nmm],'o')

            if (doplotall):
                py.draw()
                yn = input('Continue (y/n)?')
                    
        outstr = '{0:s} {1:s} {2:s} {3:d}\n'.format(cubenames[i],cubecatid[i],qflag[i],catsource[i])
        fout.write(outstr)


    fout.close()
    fout_ident.close()

    # to make things a little more tidy, sort the list:
    # actually, don;t sort the list, as this means its in a different 
    # order to the QC file, which is a bit of a pain at times...
    #sidx = np.lexsort((qflag[0:nc],cubecatid[0:nc]))

    # check against the ignore list, and set catsource to < 0 for objects found 
    # in that list:
    print('\n Cubes in ignore list:')
    for i in range(nc):
        for ignore_file in ignore_list:
            if (cubenames[i] == ignore_file):
                catsource[i] = -10
                print(cubenames[i])

    # define the public CUBEID:
    for i in range(nc):
        cubeid_pub[i] = get_cubeid_pub(icubecatid[i],qflag[i])

    # get the indicies for all rows where there is a match to a catalogue.
    # only these will be output:
    idx = np.where(catsource > 0)

    # also write a FITS binary table with the list.  This should be a bit more
    # robust:
    col0 = fits.Column(name='CUBEID',format='80A',array=cubeid[idx])
    col00 = fits.Column(name='CUBEIDPUB',format='16A',array=cubeid_pub[idx])
    col1 = fits.Column(name='CUBENAME',format='80A',array=cubenames[idx])
    # note this seems to work writing at 64 bit interger, but fv fits viewer does not seem to 
    # like it.
    col2 = fits.Column(name='CATID',format='K',array=icubecatid[idx])
    # QFLAG renamed to RFLAG, as no longer quality, but based on time. 
    # now just a repeat flag.
    col3 = fits.Column(name='RFLAG',format='1A',array=qflag[idx])
    # quality data:
    col4 = fits.Column(name='CUBEFWHM',format='E',array=cubefwhm[idx],unit='arcsec')
    col5 = fits.Column(name='CUBETEXP',format='E',array=cubetexp[idx],unit='s')
    col6 = fits.Column(name='MEANTRANS',format='E',array=frametrans_mean[idx])
    col7 = fits.Column(name='ISBEST',format='L',array=isbest[idx])
    col8 = fits.Column(name='CATSOURCE',format='I',array=catsource[idx])
    col9 = fits.Column(name='Z_SPEC',format='E',array=z_spec[idx])
    cols = fits.ColDefs([col0,col00,col1,col2,col3,col4,col5,col6,col7,col8,col9])
    hdutab = fits.BinTableHDU.from_columns(cols,name='DR_CUBES')

    outfits = 'cube_list.fits'
    hdutab.writeto(outfits,overwrite=True)

    hdulist.close()

    # finally, list the objects that are not in catalogues:
    print('Objects that are not in catalogues:')
    nnotinc = 0
    nignore = 0
    for i in range(nc):
        if (catsource[i] == -1):
            print(cubecatid[i],cubenames[i],'NOTINCAT')
            nnotinc = nnotinc + 1
        if (catsource[i] == -10):
            print(cubecatid[i],cubenames[i],'IGNORE')
            nignore = nignore + 1

    print('Number of cubes input:',nc)
    print('Number of rows output:',np.size(cubecatid[idx]))
    print('Number of cubes not included (not found in cats):',nnotinc)
    print('Number of cubes ignored (in ignore list):',nignore)


##########################################################################


def sort_rep_obs(texp,fwhm,dfwhm=0.1):

    # function to sort repeat observations by quality

    ndat = np.size(texp)

    # first get indices for sort based on cubetexp as the primary
    # reference, but then also fwhm as a secondary.  Note we 
    # sort against -1*cubetexp as we want to have the longest
    # at the top.  For lexsort() the last col in the list is
    # the primary thing to sort by:
    rep_sindex = np.lexsort((fwhm,-1.0*texp))
    print('list after first sort:',rep_sindex)

    # now that we have an approx sorting, we need to go through
    # them all and change if the fwhm is much worse for longer
    # exposures.  Note that we go through the sorting several
    # times as we are only sorting adjacent pairs on each 
    # pass
    for i in range(ndat):
        si1 = rep_sindex[i]

        # loop through all the other frames to see if any
        # have significantly better seeing:
        deltabest = -999.9
        for j in range(i+1,ndat):
            si2 = rep_sindex[j]
            delta = fwhm[si1] - fwhm[si2]
            print(i,si1,si2,fwhm[si1],fwhm[si2],delta)
                # if delta is > some value, swap indices;
            if ((delta > deltabest) and (delta > dfwhm)):
                jbest = j
                deltabest = delta
        if (deltabest > 0):
            # if we find a better case, then we move that one
            # to the currently best location and shift all others
            # down.  Note delete() and insert() do not happen in place
            # so have to reassign them.
            i_tmp = rep_sindex[jbest]
            rep_sindex = np.delete(rep_sindex,jbest)
            rep_sindex = np.insert(rep_sindex,i,i_tmp)

    print('list after second  sort:',rep_sindex)
    return rep_sindex

#############################################################

def sort_rep_obs_date(framenum,mjd,dfwhm=0.1):

    # function to sort repeat observations by quality

    ndat = np.size(framenum)

    # first get indices for sort based on MJD of run start
    # as the primary refernce, but then frame number as the
    # second.  Note we sort against -1*framenum as we want to
    # have the largest at the top.  For lexsort() the last col 
    # in the list is the primary thing to sort by.
    #
    # Note that is a few cases the MJD of the start of the run 
    # and the frame number could be the same.  This is only likely
    # in the case of calibration stars, but we should try to catch
    # these cases
    rep_sindex = np.lexsort((-1*framenum,mjd))
    print('list after first sort by MJD and frame number:',rep_sindex)

    return rep_sindex

def get_product_name(filename):

    """From the filename of a data product, get the product name."""

    # first remove leading object name and repflag (e.g. A etc)
    tmp1 = filename[filename.index('_')+3:]

    # next remove trailing .gz if needed:
    tmp2 = re.sub(r'.gz','',tmp1)

    # lastly, remove the .fits bit:
    name = re.sub(r'.fits','',tmp2)

    return name

def get_source_name(filename):
    
    """Get the object name from the file name, assuming it is at the start of the 
    filename."""

    name = filename.split('_')[0]

    return name

def get_rep_flag(filename):

    """Get the repeat flag, A, B, C, etc, from a file name.  Assumes this is after
    the CATID, e.g. 1245_A..."""

    loc = filename.index('_')
    rflag = filename[loc+1:loc+2]

    return rflag

def check_product_meta(filename,root_directory):

    # Check if a data product has an existing entry in the product_meta.txt file

    tab = Table.read(root_directory+'metadata/sami/dr3/ifs/sami_dr3_product_meta.txt',format='ascii')
    product_names = tab['name'].data

    # if needed, remove gzip in filename:
    tmp1 = re.sub(r'.gz','',filename)
    # remove catid and repeat flag (first two elements in file name):
    tmp2 = tmp1[tmp1.index('_')+3:]
    # remove '.fits' or equivalent:
    name = re.sub(r'.fits','',tmp2)

    if name in product_names:
        return True
    else:
        return False

def create_basic_metadata_files(root_directory):

    # Create the simple (i.e. independent of what data is actually present)
    # metadata files. Complicated things like column_meta and product_meta
    # are made elsewhere. For complicated things, just add the header row. 

    if verify_directory_structure(root_directory):
        

        # create the sami_survey_meta.txt file.  This is just copied from elsewhere:
        shutil.copy2(os.path.join(CATFILEPATH,'sami_survey_meta.txt'),os.path.join(root_directory,'metadata/sami/sami_survey_meta.txt'))
        
        #with open(root_directory+'metadata/sami/sami_survey_meta.txt','w') as product_file:
        #    product_file.write('name|pretty_name|title|description|pi|contact|website\n')
        #    product_file.write('sami| SAMI|SAMI Galaxy Survey|The SAMI Galaxy Survey '
        #                       'is a project to observe over 3000 local galaxies using integral '
        #                       'field spectroscopy|Scott Croom|Scott Croom '
        #                       '<scott.croom@sydney.edu.au>|http://sami-survey.org/\n')

        # create the sami_dr3_data_release_meta.txt
        with open(root_directory+'metadata/sami/dr3/sami_dr3_data_release_meta.txt','w') as product_file:
            product_file.write('name|pretty_name|version|data_release_number|contact\n')
            product_file.write('dr3|Data Release 3|0.12|3|Scott Croom <scott.croom@sydney.edu.au>\n')
        
        # Create ifs product_meta.txt
        with open(root_directory+'metadata/sami/dr3/ifs/sami_dr3_product_meta.txt','w') as product_file:
            product_file.write('facility_name|name|description|documentation|group_name|version|contact\n')

        # Create ifs group_meta.txt
        ifs_groups = dict()
        ifs_groups['cube'] = {'pretty_name':'Cubes'}
        ifs_groups['spectra'] = {'pretty_name':'Spectra'}
        ifs_groups['stelkin'] = {'pretty_name':'Stellar Kinematics'}
        ifs_groups['lzifu-flux'] = {'pretty_name':'Emission Line Fluxes'}
        ifs_groups['lzifu-kin'] = {'pretty_name':'Ionised Gas Kinematics'}
        ifs_groups['sfr'] = {'pretty_name':'Star Formation Rates (and associated products)'}
        with open(root_directory+'metadata/sami/dr3/ifs/sami_dr3_group_meta.txt','w') as ifsgroup_file:
            ifsgroup_file.write('name|pretty_name\n')
            for group in ifs_groups:
                ifsgroup_file.write('{}|{}\n'.format(group,ifs_groups[group]['pretty_name']))
        

        # Create facility_meta.txt
        with open(root_directory+'metadata/sami/dr3/ifs/sami_dr3_facility_meta.txt','w') as facility_file:
            facility_file.write('name|pretty_name|description|documentation\n')
            facility_file.write('sami|SAMI|SAMI is a fibre-based multi-object integral field optical spectrograph|sami_instrument.html\n')

        shutil.copy2(os.path.join(CATFILEPATH,'sami_instrument.html'),os.path.join(root_directory,'metadata/sami/dr3/ifs/docs/sami_instrument.html'))
        
        # create coordinate metadata file:
        with open(root_directory+'metadata/sami/dr3/catalogues/sami_dr3_coordinate_meta.txt','w') as coord_file:
            coord_file.write('table_name|source_name_col|long_col|lat_col|long_format|lat_format|frame|equinox\n')
            coord_file.write('InputCatGAMADR3|CATID|RA_OBJ|DEC_OBJ|deg|deg|fk5|J2000\n')
            coord_file.write('InputCatClustersDR3|CATID|RA_OBJ|DEC_OBJ|deg|deg|fk5|J2000\n')
            coord_file.write('InputCatFiller|CATID|RA_OBJ|DEC_OBJ|deg|deg|fk5|J2000\n')
            coord_file.write('FstarCatGAMA|CATID|RA|Dec|deg|deg|fk5|J2000\n')
            coord_file.write('FstarCatClusters|CATID|RA|Dec|deg|deg|fk5|J2000\n')
        
        # create SQL for SOV metadata file: 
        with open(root_directory+'metadata/sami/dr3/catalogues/sami_dr3_sql_meta.txt','w') as sql_file:
            dr = 'sami_dr3'
            sql_file.write('table_name|sql\n')
            sql_file.write('InputCatGAMADR3|“SELECT * FROM '+dr+'.InputCatGAMADR3 WHERE CATID = \'{objid}\'”\n')
            sql_file.write('InputCatClustersDR3|“SELECT * FROM '+dr+'.InputCatClustersDR3 WHERE CATID = \'{objid}\'”\n')
            sql_file.write('InputCatFiller|“SELECT * FROM '+dr+'.InputCatFiller WHERE CATID = \'{objid}\'”\n')
            sql_file.write('FstarCatGAMA|“SELECT * FROM '+dr+'.FstarCatGAMA WHERE CATID = \'{objid}\'”\n')
            sql_file.write('FstarCatClusters|“SELECT * FROM '+dr+'.FstarCatClusters WHERE CATID = \'{objid}\'”\n')
            sql_file.write('CubeObs|“SELECT * FROM '+dr+'.CubeObs WHERE CATID = \'{objid}\'”\n')
            sql_file.write('DensityCatDR3|“SELECT * FROM '+dr+'.DensityCatDR3 WHERE CATID = \'{objid}\'”\n')
            sql_file.write('IndexAperturesDR3|“SELECT * FROM '+dr+'.IndexAperturesDR3 WHERE CATID = \'{objid}\'”\n')
            sql_file.write('SSPAperturesDR3|“SELECT * FROM '+dr+'.SSPAperturesDR3 WHERE CATID = \'{objid}\'”\n')
            sql_file.write('MGEPhotomUnregDR3|“SELECT * FROM '+dr+'.MGEPhotomUnregDR3 WHERE CATID = \'{objid}\'”\n')
            sql_file.write('samiDR3gaskinPA|“SELECT * FROM '+dr+'.samiDR3gaskinPA WHERE CATID = \'{objid}\'”\n')
            sql_file.write('samiDR3Stelkin|“SELECT * FROM '+dr+'.samiDR3Stelkin WHERE CATID = \'{objid}\'”\n')
            sql_file.write('VisualMorphologyDR3|“SELECT * FROM '+dr+'.VisualMorphologyDR3 WHERE CATID = \'{objid}\'”\n')
            sql_file.write('EmissionLine1compDR3|“SELECT * FROM '+dr+'.EmissionLine1compDR3 WHERE CATID = \'{objid}\'”\n')


        # Create catalogue group_meta.txt
        # *** WHAT IS OTHER??? WHAT DO THESE RELATE TO?
        cata_groups = dict()
        cata_groups['sami'] = {'pretty_name':'SAMI',
                               'desc':'This group contains catalogue data derived from SAMI data',
                               'doc':'',
                               'contact':'Scott Croom <scott.croom@sydney.edu.au>',
                               'date':DRDATE,
                               'version':REDVER}
        cata_groups['other'] =  {'pretty_name':'Other',
                               'desc':'This group contains catalogue data derived from non-SAMI data',
                               'doc':'',
                               'contact':'Julia Bryant <julia.bryant@sydney.edu.au>',
                               'date':DRDATE,
                               'version':'v3'}
        with open(root_directory+'metadata/sami/dr3/catalogues/sami_dr3_group_meta.txt','w') as catagroup_file:
            catagroup_file.write('name|pretty_name|description|documentation|contact|date|version\n')
            for group in cata_groups:
                catagroup_file.write('{}|{}|{}|{}|{}|{}|{}\n'.format(group,
                                                                     cata_groups[group]['pretty_name'],
                                                                     cata_groups[group]['desc'],
                                                                     cata_groups[group]['doc'],
                                                                     cata_groups[group]['contact'],
                                                                     cata_groups[group]['date'],
                                                                     cata_groups[group]['version']))

            
def create_coordinate_meta_files(root_directory,input_cat,listfile_path):

    """This is no longer needed in DR3!!!  Now just need a metafile 
    pointing to the coordinate file"""

    # Take the SAMI main sample catalogue for all release galaxies and
    # extract the info to make an astro_objects_meta.txt file

    tab = Table.read(input_cat,format='ascii') #Should be a fits file here
    # this assumed that the CATID list is ascii:
    #tab_ids = Table.read(listfile_path,format='ascii.commented_header')
    #ids = tab_ids['CATID'].data
    # instead we can use a FITS file:
    hdulist = fits.open(listfile_path)
    table_data = hdulist['DR_CUBES'].data
    ids = table_data['CATID']
    # and find unique ids:
    ids_unique = np.unique(ids)

    n_unique = np.size(ids_unique)
    n_notfound = 0

    with open(root_directory+'metadata/sami/dr3/catalogues/sami_dr3_coordinate_meta.txt','w') as coords_file:
        coords_file.write('table_name|source_name_col|long_col|lat_col|long_format|lat_format|frame|equinox\n')
        coords_file.write('DR2Sample|CATID|RA|Dec|deg|deg|fk5|j2000\n')

    with open(root_directory+'metadata/sami/dr3/astro_objects/sami_dr3_astro_objects_meta.txt','w') as astro_objects_file:
        astro_objects_file.write('source_name|ra|dec|frame|equinox\n')
        for catid in ids_unique:
            # need to convert to int as catid in list is a string:
            ww = np.where(tab['CATID'] == int(catid))[0]
            try:
                ra = tab['RA'][ww].data[0]
                dec = tab['Dec'][ww].data[0]
                astro_objects_file.write('{}|{}|{}|fk5|j2000\n'.format(catid,ra,dec))
            except:              
                print(str(catid)+' not found in '+input_cat)
                n_notfound = n_notfound + 1

    print('number of unique CATIDs in list: ',n_unique)
    print('number of CATIDs not found in catalogue files: ',n_notfound)

def create_product_ingestion(root_directory):

    """Write the sami_dr3_product_ingestion.txt metadata file
    that contains a list of all data files to ingest."""

    # read in the CubeObs.fits file to get the is_best flag:
    hdulist = fits.open(os.path.join(CATFILEPATH,'CubeObs.fits'))
    table_data = hdulist[1].data
    cubeidpub = table_data['CUBEIDPUB']
    catid = table_data['CATID']
    isbest = table_data['ISBEST']
    ncube = np.size(cubeidpub) 

    # create the file and write the header:
    with open(root_directory+'metadata/sami/dr3/ifs/sami_dr3_product_ingestion.txt','w') as ingestion_file:
        ingestion_file.write('facility_name|data_product_name|file_name|rel_file_path|source_name|is_best\n')
    
        # now look through all the data folders and get a list of all files:
        all_data_files = sorted(glob.glob(root_directory+'/data/sami/dr3/ifs/[0-9]*/[0-9]*.fits*'))

        # loop through the files and write them all to the ingestion file:
        for data_file in all_data_files:
            # get the values needed:
            file_name = basename(data_file)
            product_name = get_product_name(file_name)
            source_name = get_source_name(file_name)
            rflag = get_rep_flag(file_name)
            cubeid = get_cubeid_pub(source_name,rflag)
            file_path = os.path.join(source_name,file_name)
            #print(data_file,product_name,source_name,cubeid)
            # get the is_best flag by matching to the cube ID:
            found = False
            for i in range(ncube):
                if (cubeidpub[i] == cubeid):
                    is_best = isbest[i]
                    found = True
            if (not found):
                print('ERROR: cubeid not found ',cubeid)
            # write to the metadata file:
            ingestion_file.write('sami|{}|{}|{}|{}|{}\n'.format(product_name,file_name,file_path,source_name,is_best))


def create_directory_structure(root_directory):

    # Create the required Data Central directory structure.

    if not verify_directory_structure(root_directory):
        if not os.path.isdir(root_directory):
            os.mkdir(root_directory)
        if not os.path.isdir(root_directory+'data'):
            os.mkdir(root_directory+'data')
        if not os.path.isdir(root_directory+'data/sami'):
            os.mkdir(root_directory+'data/sami')
        if not os.path.isdir(root_directory+'data/sami/dr3'):
            os.mkdir(root_directory+'data/sami/dr3')
        if not os.path.isdir(root_directory+'data/sami/dr3/ifs'):
            os.mkdir(root_directory+'data/sami/dr3/ifs')
        if not os.path.isdir(root_directory+'data/sami/dr3/catalogues'):
            os.mkdir(root_directory+'data/sami/dr3/catalogues')
        if not os.path.isdir(root_directory+'metadata'):
            os.mkdir(root_directory+'metadata')
        if not os.path.isdir(root_directory+'metadata/sami'):
            os.mkdir(root_directory+'metadata/sami')
        if not os.path.isdir(root_directory+'metadata/sami/dr3'):
            os.mkdir(root_directory+'metadata/sami/dr3')
        #if not os.path.isdir(root_directory+'metadata/sami/dr3/astro_objects'):
        #    os.mkdir(root_directory+'metadata/sami/dr3/astro_objects')
        if not os.path.isdir(root_directory+'metadata/sami/dr3/catalogues'):
            os.mkdir(root_directory+'metadata/sami/dr3/catalogues')
        if not os.path.isdir(root_directory+'metadata/sami/dr3/catalogues/docs'):
            os.mkdir(root_directory+'metadata/sami/dr3/catalogues/docs')
        if not os.path.isdir(root_directory+'metadata/sami/dr3/ifs'):
            os.mkdir(root_directory+'metadata/sami/dr3/ifs')
        if not os.path.isdir(root_directory+'metadata/sami/dr3/ifs/docs'):
            os.mkdir(root_directory+'metadata/sami/dr3/ifs/docs')

        print('Ingestion directory structure created in: '+root_directory)

def verify_directory_structure(root_directory):

    # Ensure that all required Data Central directories exist.
    # Warn if not.

    ok = True

    if not os.path.isdir(root_directory+'data'):
        print('Missing directory: '+root_directory+'data')
        ok = False
    if not os.path.isdir(root_directory+'data/sami/dr3'):
        print('Missing directory: '+root_directory+'data/sami/dr3')
        ok = False
    #if not os.path.isdir(root_directory+'metadata/sami/dr3/astro_objects'):
    #    print('Missing directory: '+root_directory+'metadata/sami/dr3/astro_objects')
    #    ok = False
    if not os.path.isdir(root_directory+'data/sami/dr3/catalogues'):
         print('Missing directory: '+root_directory+'data/sami/dr3/catalogues')
         ok = False
    if not os.path.isdir(root_directory+'data/sami/dr3/ifs'):
         print('Missing directory: '+root_directory+'data/sami/dr3/ifs')
         ok = False
    if not os.path.isdir(root_directory+'metadata/sami/dr3/catalogues'):
         print('Missing directory: '+root_directory+'metadata/sami/dr3/catalogues')
         ok = False
    if not os.path.isdir(root_directory+'metadata/sami/dr3/catalogues/docs'):
         print('Missing directory: '+root_directory+'metadata/sami/dr3/catalogues/docs')
         ok = False
    if not os.path.isdir(root_directory+'metadata/sami/dr3/ifs'):
         print('Missing directory: '+root_directory+'metadata/sami/dr3/ifs')
         ok = False
    if not os.path.isdir(root_directory+'metadata/sami/dr3/ifs/docs'):
         print('Missing directory: '+root_directory+'metadata/sami/dr3/ifs/docs')
         ok = False

    if ok == True:
        print('')
        print('Directory structure successfully verified')
    else:
        print('')
        print('Directory structure failed verification')
        print('If you get this message on the first run of create_directory_structure()')
        print('then rerun and you should no longer get a verification fail')

    return ok

def verify_source_directory(catid,root_directory,create=True):

    # Check that the directory exists for the current source ID.
    # If not create (default) or raise exception

    if not os.path.isdir(root_directory+'data/sami/dr3/ifs/'+catid):
        if create:
            print('Source directory for '+str(catid)+' missing')
            print('Creating...')
            os.mkdir(root_directory+'data/sami/dr3/ifs/'+catid)
        else:
            raise Exception( 'Source directory for '+str(catid)+' missing')

    return root_directory+'data/sami/dr3/ifs/'+catid+'/'

def reformat_cube(catid,cube_path,qf,source_directory,root_directory,sim=False,check_only=True,zspec=np.nan):

    # Taking a SAMI cube filepath as input, extract the extensions
    # into several .fits files for DR3 ingestion

    if (sim):
        return

    found = False
    # check to see if the cube file exists:
    if not os.path.exists(cube_path):
        print(cube_path+' does not exist')

        # if the file name does not exist, check to see if it is
        # already a .gz file.  If it is, look for the unzipped version,
        # if not, then look for the zipped version:
        if '.gz' not in cube_path:
            print('Cube name in cube_path is not gzipped')
            cube_path_gz = cube_path+'.gz'
            if (os.path.exists(cube_path_gz)):
                print('Gzipped cube found')
                cube_path = cube_path_gz
                found = True
        else:
            print('Cubename in cube_path is gzipped')
            cube_path_nogz = cube_path.replace('.gz','')
            if (os.path.exists(cube_path_nogz)):
                print('non-gzipped cube found')
                cube_path = cube_path_nogz
                found = True
    else:
        found = True

    # do not continue if the file is not found:
    if not found:
        return

    # if the file is found, then we continue...
    if 'blue' in cube_path:
        colour = 'blue'
    else:
        colour = 'red'

    # get the public cube ID:
    pubcubeid = get_cubeid_pub(catid,qf)

    # find the output file name for one version to check if files
    # exist:
    ofile = source_directory+'/'+pubcubeid+'_cube_'+colour+'.fits.gz'
    if os.path.exists(ofile):
        #Assume all cubes exist if one exists - semi-risky
        print('Cubes for '+catid+' already exist. Skipping ',ofile)
        return
    # if check_only is true, then we output a list of files that are NOT present:
    else:
        if (check_only):
            # write to log file:
            logging.info('file not found: '+ofile)
            print('file not found: ',ofile)
            return

    # Set some product_meta info
    dp_type = '3d_cube'
    version = REDVER
    contact = 'Nic Scott <nicholas.scott@sydney.edu.au>'
    group = 'cube'
    doc_file = 'binned_cube.html'

    hdu_input = fits.open(cube_path)

    hdu_default = fits.HDUList([hdu_input['PRIMARY'],
                                hdu_input['VARIANCE'],
                                hdu_input['WEIGHT'],
                                hdu_input['COVAR'],
                                hdu_input['QC'],
                                hdu_input['DUST']])
    # add a new keyword, the original cube file name:
    hdu_default['PRIMARY'].header = fix_cube_header(hdu_default['PRIMARY'].header,cube_path,zspec)

    # output file name for new cube:
    filename = source_directory+'/'+pubcubeid+'_cube_'+colour+'.fits'
    hdu_default.writeto(filename)
    # this gzips the new file:
    #with open(filename,'rb') as f_in:
    #    with gzip.open(filename+'.gz','wb') as f_out:
    #        f_out.writelines(f_in)
    #os.remove(filename)
    # alternative just calling native gzip:
    subprocess.call(["gzip",filename])
    filename = filename.split('/')[-1]
    if not check_product_meta(filename+'.gz',root_directory):
        cube_type = ' default (unbinned) '
        description = 'Fully reduced and flux calibrated{}SAMI cube ({})'.format(cube_type,colour)
        product_meta_dict = {'dp_type':dp_type,
                             'version':version,
                             'contact':contact,
                             'group_name':group,
                             'desc':description,
                             'doc_file':'cube.html'}
        update_product_meta(filename+'.gz',product_meta_dict,root_directory)

    # do adaptively binned cube:
    new_primary = fits.PrimaryHDU(hdu_input['BINNED_FLUX_ADAPTIVE'].data)
    new_primary.header = hdu_input[0].header
    hdu_adaptive = fits.HDUList([new_primary,
                                 hdu_input['BINNED_VARIANCE_ADAPTIVE'],
                                 hdu_input['BIN_MASK_ADAPTIVE'],
                                 hdu_input['QC'],
                                 hdu_input['DUST']])
    hdu_adaptive['BINNED_VARIANCE_ADAPTIVE'].header['EXTNAME'] = 'VARIANCE'
    hdu_adaptive['BIN_MASK_ADAPTIVE'].header['EXTNAME'] = 'BIN_MASK'
    # add a new keyword, the original cube file name:
    hdu_default['PRIMARY'].header = fix_cube_header(hdu_default['PRIMARY'].header,cube_path,zspec)

    # file name for adaptively binned cube:
    filename = source_directory+'/'+pubcubeid+'_adaptive_'+colour+'.fits'
    hdu_adaptive.writeto(filename)
    #with open(filename,'rb') as f_in:
    #    with gzip.open(filename+'.gz','wb') as f_out:
    #        f_out.writelines(f_in)
    #os.remove(filename)
    subprocess.call(["gzip",filename])
    filename = filename.split('/')[-1]
    if not check_product_meta(filename+'.gz',root_directory):
        cube_type = ' adaptive binned '
        description = 'Fully reduced and flux calibrated{}SAMI cube ({})'.format(cube_type,colour)
        product_meta_dict = {'dp_type':dp_type,
                             'version':version,
                             'contact':contact,
                             'group_name':group,
                             'desc':description,
                             'doc_file':doc_file}
        update_product_meta(filename+'.gz',product_meta_dict,root_directory)

    # do annular binned cube:
    new_primary = fits.PrimaryHDU(hdu_input['BINNED_FLUX_ANNULAR'].data)
    new_primary.header = hdu_input[0].header
    hdu_annular = fits.HDUList([new_primary,
                                 hdu_input['BINNED_VARIANCE_ANNULAR'],
                                 hdu_input['BIN_MASK_ANNULAR'],
                                 hdu_input['QC'],
                                 hdu_input['DUST']])
    hdu_annular['BINNED_VARIANCE_ANNULAR'].header['EXTNAME'] = 'VARIANCE'
    hdu_annular['BIN_MASK_ANNULAR'].header['EXTNAME'] = 'BIN_MASK'
    # add a new keyword, the original cube file name:
    hdu_default['PRIMARY'].header = fix_cube_header(hdu_default['PRIMARY'].header,cube_path,zspec)

    # filename for annular binned cube:
    filename = source_directory+'/'+pubcubeid+'_annular_'+colour+'.fits'
    hdu_annular.writeto(filename)
    #with open(filename,'rb') as f_in:
    #    with gzip.open(filename+'.gz','wb') as f_out:
    #        f_out.writelines(f_in)
    #os.remove(filename)
    subprocess.call(["gzip",filename])
    filename = filename.split('/')[-1]
    if not check_product_meta(filename+'.gz',root_directory):
        cube_type = ' annular binned '
        description = 'Fully reduced and flux calibrated{}SAMI cube ({})'.format(cube_type,colour)
        product_meta_dict = {'dp_type':dp_type,
                             'version':version,
                             'contact':contact,
                             'group_name':group,
                             'desc':description,
                             'doc_file':doc_file}
        update_product_meta(filename+'.gz',product_meta_dict,root_directory)

    # do sectors binned cube:
    new_primary = fits.PrimaryHDU(hdu_input['BINNED_FLUX_SECTORS'].data)
    new_primary.header = hdu_input[0].header
    hdu_sectors = fits.HDUList([new_primary,
                                 hdu_input['BINNED_VARIANCE_SECTORS'],
                                 hdu_input['BIN_MASK_SECTORS'],
                                 hdu_input['QC'],
                                 hdu_input['DUST']])
    hdu_sectors['BINNED_VARIANCE_SECTORS'].header['EXTNAME'] = 'VARIANCE'
    hdu_sectors['BIN_MASK_SECTORS'].header['EXTNAME'] = 'BIN_MASK'
    # add a new keyword, the original cube file name:
    hdu_default['PRIMARY'].header = fix_cube_header(hdu_default['PRIMARY'].header,cube_path,zspec)

    # filename for sectors binned cube:
    filename = source_directory+'/'+pubcubeid+'_sectors_'+colour+'.fits'
    hdu_sectors.writeto(filename)
    #with open(filename,'rb') as f_in:
    #    with gzip.open(filename+'.gz','wb') as f_out:
    #        f_out.writelines(f_in)
    #os.remove(filename)
    subprocess.call(["gzip",filename])
    filename = filename.split('/')[-1]
    if not check_product_meta(filename+'.gz',root_directory):
        cube_type = ' sectors binned '
        description = 'Fully reduced and flux calibrated{}SAMI cube ({})'.format(cube_type,colour)
        product_meta_dict = {'dp_type':dp_type,
                             'version':version,
                             'contact':contact,
                             'group_name':group,
                             'desc':description,
                             'doc_file':doc_file}
        update_product_meta(filename+'.gz',product_meta_dict,root_directory)

    return

############################################################

def append_aperture_header(header,hdr):

    # list of keywords to copy across to aperture spectra
    keylist = ['BUNIT','STDNAME','IFUPROBE','PSFFWHM','PSFALPHA','PSFBETA','DROPFACT','RO_GAIN','RO_NOISE','GRATID','NAME','CATARA','CATADEC','RADESYS','LONPOLE','LATPOLE','EQUINOX','ORIGIN','TELESCOP','ALT_OBS','LAT_OBS','LONG_OBS','INSTRUME','SPECTID','GRATTILT','GRATLPMM','ORDER','DICHROIC','TOPEND','AXIS','WCS_SRC','TOTALEXP','PLATEID','LABEL','CBINGMET','HGCUBING','HGAPER']


    # loop through keywords, remembering to also pass the comments:
    for key in keylist:
        try:
            header[key] = (hdr[key],hdr.comments[key])
        except:
            pass

    header.insert(0,('SIMPLE',hdr['SIMPLE']))

    # remove keywords not required:
    header.remove('XTENSION')
    header.remove('PCOUNT')
    header.remove('GCOUNT')
    header.remove('EXTNAME')

    # move the WCSAXES header to before the other WCS items.  This
    # to to conform to FITS standard, but does not seem to make much 
    # difference
    header.set('WCSAXES',before='CRVAL1')

    return header

###############################################################
def fix_cube_header(header,cube_path,zspec):

    # list of keywords to remove:
    # these are mostly version numbers for AAT control tasks.  This is because sometimes they 
    # are not present as the data can have different values for the same cube (e.g. frames
    # from different runs).  Remove to make headers consistent.
    keylist_rem = ['DCT_DATE','DCT_VER','RCT_VER','RCT_DATE','TDFCTVER','TDFCTDAT']

    new_header = header

    # loop through keywords to delete the ones we requested:
    for key in keylist_rem:
        try:
            new_header.remove(key)
        except:
            pass
    
    # now add zspec if not NaN:
    if (np.isfinite(zspec)):
        new_header['Z_SPEC'] = (zspec,'Heliocentric redshift from input catalogue')

    # add the original cube file to the header.  Can't add comment as there is not enough room!
    new_header['ORIGFILE'] = basename(cube_path)


    return new_header

####################################################

def reformat_aperture_spectrum(catid,aperture_path,qf,source_directory,root_directory,sim=False,check_only=True,zspec=np.nan):

    # Taking a SAMI aperture spectrum filepath as input, extract
    # the extensions into several .fits files for DR3 ingestion

    if not os.path.exists(aperture_path):
        print(aperture_path+' does not exist')
    else:
        if 'blue' in aperture_path:
            colour = 'blue'
        else:
            colour = 'red'

        # get the public cube ID:
        pubcubeid = get_cubeid_pub(catid,qf)

        ofile = source_directory+'/'+pubcubeid+'_spectrum_3-kpc_'+colour+'.fits'
        if os.path.exists(ofile):
            #Assume all aperture spectra exist if one exists - semi-risky
            print('Aperture spectra for '+catid+' already exist. Skipping')
            return
        # if check_only is true, then we output a list of files that are NOT present:
        else:
            if (check_only):
                print('file not found: ',ofile)
                return


        # Product meta settings
        dp_type = '1d_spectra'
        version = REDVER
        contact = 'Nic Scott <nicholas.scott@sydney.edu.au>'
        doc_file = 'aperture_spec.html'
        group = 'spectra'

        hdu_input = fits.open(aperture_path)

        ############# 3KPC #############

        try: 
            new_primary = fits.PrimaryHDU(hdu_input['3KPC_ROUND'].data)
            new_primary.header = hdu_input['3KPC_ROUND'].header
            new_primary.header = append_aperture_header(new_primary.header,
                                                    hdu_input['PRIMARY'].header)
            new_primary.header.remove('AREACORR')
            if (np.isfinite(zspec)):
                new_primary.header['Z_SPEC'] = (zspec,'Heliocentric redshift from input catalogue')
            new_primary.header['ORIGFILE'] = basename(aperture_path)
            hdu_3kpc = fits.HDUList([new_primary,
                                     hdu_input['3KPC_ROUND_VAR'],
                                     hdu_input['3KPC_ROUND_MASK']])
            hdu_3kpc[1].header['EXTNAME'] = 'VARIANCE'
            hdu_3kpc[2].header['EXTNAME'] = 'BIN_MASK'
            filename = pubcubeid+'_spectrum_3-kpc_'+colour+'.fits'
            hdu_3kpc.writeto(os.path.join(source_directory,filename))
            if not check_product_meta(filename,root_directory):
                spec_type = ' round 3 kpc diameter'
                description = 'Flux calibrated spectrum extracted from a{} aperture ({})'.format(spec_type,colour)
                product_meta_dict = {'dp_type':dp_type,
                                     'version':version,
                                     'contact':contact,
                                     'group_name':group,
                                     'desc':description,
                                     'doc_file':doc_file}
                update_product_meta(filename,product_meta_dict,root_directory)

        except KeyError:
            print('3kpc extension not found for',aperture_path)

        ############# RE #############

        # check if RE aperture is actually present:
        try: 
            new_primary = fits.PrimaryHDU(hdu_input['RE'].data)
            new_primary.header = hdu_input['RE'].header
            new_primary.header = append_aperture_header(new_primary.header,
                                                    hdu_input['PRIMARY'].header)
            new_primary.header.remove('AREACORR')
            if (np.isfinite(zspec)):
                new_primary.header['Z_SPEC'] = (zspec,'Heliocentric redshift from input catalogue')
            new_primary.header['ORIGFILE'] = basename(aperture_path)
            hdu_re = fits.HDUList([new_primary,
                                     hdu_input['RE_VAR'],
                                     hdu_input['RE_MASK']])
            hdu_re[1].header['EXTNAME'] = 'VARIANCE'
            hdu_re[2].header['EXTNAME'] = 'BIN_MASK'
            filename = pubcubeid+'_spectrum_re_'+colour+'.fits'
            hdu_re.writeto(os.path.join(source_directory,filename))
            if not check_product_meta(filename,root_directory):
                spec_type = 'n elliptical Re radius'
                description = 'Flux calibrated spectrum extracted from a{} aperture ({})'.format(spec_type,colour)
                product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file}
                update_product_meta(filename,product_meta_dict,root_directory)

        except KeyError:
            print('RE extension not found for',aperture_path)

        ############# RE_MGE #############

        # check if RE_MGE aperture is actually present:
        try: 
            new_primary = fits.PrimaryHDU(hdu_input['RE_MGE'].data)
            new_primary.header = hdu_input['RE_MGE'].header
            new_primary.header = append_aperture_header(new_primary.header,
                                                    hdu_input['PRIMARY'].header)
            new_primary.header.remove('AREACORR')
            if (np.isfinite(zspec)):
                new_primary.header['Z_SPEC'] = (zspec,'Heliocentric redshift from input catalogue')
            new_primary.header['ORIGFILE'] = basename(aperture_path)
            hdu_re = fits.HDUList([new_primary,
                                     hdu_input['RE_MGE_VAR'],
                                     hdu_input['RE_MGE_MASK']])
            hdu_re[1].header['EXTNAME'] = 'VARIANCE'
            hdu_re[2].header['EXTNAME'] = 'BIN_MASK'
            filename = pubcubeid+'_spectrum_remge_'+colour+'.fits'
            hdu_re.writeto(os.path.join(source_directory,filename))
            if not check_product_meta(filename,root_directory):
                spec_type = 'n elliptical Re radius'
                description = 'Flux calibrated spectrum extracted from a{} aperture ({})'.format(spec_type,colour)
                product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file}
                update_product_meta(filename,product_meta_dict,root_directory)

        except KeyError:
            print('RE_MGE extension not found for',aperture_path)

        ############# 1-4 arcsec #############

        new_primary = fits.PrimaryHDU(hdu_input['1.4_ARCSECOND'].data)
        new_primary.header = hdu_input['1.4_ARCSECOND'].header
        new_primary.header = append_aperture_header(new_primary.header,
                                                    hdu_input['PRIMARY'].header)
        new_primary.header.remove('AREACORR')
        if (np.isfinite(zspec)):
            new_primary.header['Z_SPEC'] = (zspec,'Heliocentric redshift from input catalogue')
        new_primary.header['ORIGFILE'] = basename(aperture_path)
        hdu_14arcsec = fits.HDUList([new_primary,
                                     hdu_input['1.4_ARCSECOND_VAR'],
                                     hdu_input['1.4_ARCSECOND_MASK']])
        hdu_14arcsec[1].header['EXTNAME'] = 'VARIANCE'
        hdu_14arcsec[2].header['EXTNAME'] = 'BIN_MASK'
        filename = pubcubeid+'_spectrum_1-4-arcsec_'+colour+'.fits'
        hdu_14arcsec.writeto(os.path.join(source_directory,filename))
        if not check_product_meta(filename,root_directory):
            spec_type = ' round 1.4 arcsecond diameter'
            description = 'Flux calibrated spectrum extracted from a{} aperture ({})'.format(spec_type,colour)
            product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file}
            update_product_meta(filename,product_meta_dict,root_directory)

        ############# 2 arcsec #############

        new_primary = fits.PrimaryHDU(hdu_input['2_ARCSECOND'].data)
        new_primary.header = hdu_input['2_ARCSECOND'].header
        new_primary.header = append_aperture_header(new_primary.header,
                                                    hdu_input['PRIMARY'].header)
        new_primary.header.remove('AREACORR')
        if (np.isfinite(zspec)):
            new_primary.header['Z_SPEC'] = (zspec,'Heliocentric redshift from input catalogue')
        new_primary.header['ORIGFILE'] = basename(aperture_path)
        hdu_2arcsec = fits.HDUList([new_primary,
                                     hdu_input['2_ARCSECOND_VAR'],
                                     hdu_input['2_ARCSECOND_MASK']])
        hdu_2arcsec[1].header['EXTNAME'] = 'VARIANCE'
        hdu_2arcsec[2].header['EXTNAME'] = 'BIN_MASK'
        filename = pubcubeid+'_spectrum_2-arcsec_'+colour+'.fits'
        hdu_2arcsec.writeto(os.path.join(source_directory,filename))
        if not check_product_meta(filename,root_directory):
            spec_type = ' round 2 arcsecond diameter'
            description = 'Flux calibrated spectrum extracted from a{} aperture ({})'.format(spec_type,colour)
            product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file}
            update_product_meta(filename,product_meta_dict,root_directory)

        ############# 3 arcsec #############

        new_primary = fits.PrimaryHDU(hdu_input['3_ARCSECOND'].data)
        new_primary.header = hdu_input['3_ARCSECOND'].header
        new_primary.header = append_aperture_header(new_primary.header,
                                                    hdu_input['PRIMARY'].header)
        new_primary.header.remove('AREACORR')
        if (np.isfinite(zspec)):
            new_primary.header['Z_SPEC'] = (zspec,'Heliocentric redshift from input catalogue')
        new_primary.header['ORIGFILE'] = basename(aperture_path)
        hdu_3arcsec = fits.HDUList([new_primary,
                                     hdu_input['3_ARCSECOND_VAR'],
                                     hdu_input['3_ARCSECOND_MASK']])
        hdu_3arcsec[1].header['EXTNAME'] = 'VARIANCE'
        hdu_3arcsec[2].header['EXTNAME'] = 'BIN_MASK'

        filename = pubcubeid+'_spectrum_3-arcsec_'+colour+'.fits'
        hdu_3arcsec.writeto(os.path.join(source_directory,filename))
        if not check_product_meta(filename,root_directory):
            spec_type = ' round 3 arcsecond diameter'
            description = 'Flux calibrated spectrum extracted from a{} aperture ({})'.format(spec_type,colour)
            product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file}
            update_product_meta(filename,product_meta_dict,root_directory)

        ############# 4 arcsec #############

        new_primary = fits.PrimaryHDU(hdu_input['4_ARCSECOND'].data)
        new_primary.header = hdu_input['4_ARCSECOND'].header
        new_primary.header = append_aperture_header(new_primary.header,
                                                    hdu_input['PRIMARY'].header)
        new_primary.header.remove('AREACORR')
        if (np.isfinite(zspec)):
            new_primary.header['Z_SPEC'] = (zspec,'Heliocentric redshift from input catalogue')
        new_primary.header['ORIGFILE'] = basename(aperture_path)
        hdu_4arcsec = fits.HDUList([new_primary,
                                     hdu_input['4_ARCSECOND_VAR'],
                                     hdu_input['4_ARCSECOND_MASK']])
        hdu_4arcsec[1].header['EXTNAME'] = 'VARIANCE'
        hdu_4arcsec[2].header['EXTNAME'] = 'BIN_MASK'

        filename = pubcubeid+'_spectrum_4-arcsec_'+colour+'.fits'
        hdu_4arcsec.writeto(os.path.join(source_directory,filename))
        if not check_product_meta(filename,root_directory):
            spec_type = ' round 4 arcsecond diameter'
            description = 'Flux calibrated spectrum extracted from a{} aperture ({})'.format(spec_type,colour)
            product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file}
            update_product_meta(filename,product_meta_dict,root_directory)

    return

def reformat_stelkin(catid,stelkin_path,qf,source_directory,root_directory):

    # Taking a SAMI stellar kinematics data (in format generated by Jesse)
    # and split the extensions into several .fits files for DR3 ingestion

    # extra keywords to copy from extensions to primary:
    extra_kwd = ['UNIT','DP_NAME','DESCRIPT']


    # first check to see if the input file exists.  If it does not, then
    # don't do anything (basically, skips to the end):
    if not os.path.exists(stelkin_path):
        print(stelkin_path+' does not exist')
    else:
        # get the public cube ID:
        pubcubeid = get_cubeid_pub(catid,qf)

        # here we do have an input file, so start to do something with it:
        ofile = source_directory+'/'+pubcubeid+'_stellar-velocity_default_two-moment.fits'
        if os.path.exists(ofile):
            #Assume all stelkin maps exist if one exists, so we just check for one here.
            print('Stellar kinematic maps for '+catid+' already exist. Skipping')
            return

        # Product meta settings
        dp_type = '2d_map'
        version = REDVER
        contact = 'Jesse van de Sande <jesse.vandesande@sydney.edu.au>'
        doc_files = [['stellar-velocity_default_two-moment.html',
                      'stellar-velocity-dispersion_default_two-moment.html'],
                     ['stellar-velocity_adaptive_two-moment.html',
                      'stellar-velocity-dispersion_adaptive_two-moment.html'],
                     ['stellar-velocity_annular_two-moment.html',
                      'stellar-velocity-dispersion_annular_two-moment.html'],
                     ['stellar-velocity_sectors_two-moment.html',
                      'stellar-velocity-dispersion_sectors_two-moment.html']]
        group = 'stelkin'

        # open the input file (to read only):
        hdu_input = fits.open(stelkin_path)
        hdr_input = hdu_input[0].header
        # identify keywords than need to be removed from main header.
        keywords_to_remove = ['SIMPLE','BITPIX','NAXIS','AP_SIG','AP_SERR', 'AP_VEl',
                              'AP_VERR','APER','VERSION','SAMI_VER','DATE','A_POLY']
        for keyword_to_remove in keywords_to_remove:
            hdr_input.remove(keyword_to_remove)
        # define the different bin names (strings for file names) and extensions corresponding to those:
        bin_names = ['default','adaptive','annular','sectors']
        bin_fits = ['','_BINNED_ADAPTIVE','_BINNED_ANNULAR','_BINNED_SECTORS']
        # loop over the different binning schemes:
        for bi in range(len(bin_names)):
            hdu_input = fits.open(stelkin_path)
            bin_name = bin_names[bi]
            bin_fit = bin_fits[bi]
            doc_file = ['stellar-velocity.html','stellar-velocity-dispersion.html']#doc_files[bi]
            if bi == 0:
                binned = ''
            else:
                binned = ' binned'

            new_primary = fits.PrimaryHDU(hdu_input['vel'+bin_fit].data)
            for card in hdr_input.cards:
                new_primary.header.append(card)

            # append extra header items from extension:
            new_primary.header = copy_kwd(extra_kwd,hdu_input['vel'+bin_fit],new_primary)

            qc = hdu_input['qc'+bin_fit].data
            if qc.size > 5:
                sn = qc[:,:,3]
            hdu_sn = fits.ImageHDU(sn)
            hdu_sn.header = hdu_input['qc'+bin_fit].header.copy()
            hdu_sn.header['DESCRIPT'] = 'S/N per pix, ave over non-clipped wavelength pix'
            hdu_sn.header['EXTNAME'] = 'SNR'
            try:
                for i in range(8):
                    hdu_sn.header.remove('SLICE_'+str(i))
            except:
                pass
            hdu_stelkin = fits.HDUList([new_primary,
                                        hdu_input['vel_err'+bin_fit].copy(),
                                        hdu_input['flux'+bin_fit].copy(),
                                        hdu_input['flux_err'+bin_fit].copy(),
                                        hdu_sn])
            hdu_stelkin['VEL_ERR'+bin_fit].header['EXTNAME'] = 'VEL_ERR'
            hdu_stelkin['FLUX'+bin_fit].header['EXTNAME'] = 'FLUX'
            hdu_stelkin['FLUX_ERR'+bin_fit].header['EXTNAME'] = 'FLUX_ERR'

            filename = pubcubeid+'_stellar-velocity_'+bin_name+'_two-moment.fits'
            hdu_stelkin.writeto(os.path.join(source_directory,filename))
            if not check_product_meta(filename,root_directory):
                description = 'Stellar velocity map (two moment) from {}{} cube'.format(bin_name,binned)
                product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file[0]}
                update_product_meta(filename,product_meta_dict,root_directory)

            new_primary = fits.PrimaryHDU(hdu_input['sig'+bin_fit].data)
            for card in hdr_input.cards:
                new_primary.header.append(card)

            # append extra header items from extension:
            new_primary.header = copy_kwd(extra_kwd,hdu_input['sig'+bin_fit],new_primary)

            qc = hdu_input['qc'+bin_fit].data
            if qc.size > 5:
                sn = qc[:,:,3]
            hdu_sn = fits.ImageHDU(sn)
            hdu_sn.header = hdu_input['qc'+bin_fit].header.copy()
            hdu_sn.header['DESCRIPT'] = 'S/N per pix, ave over non-clipped wavelength pix'
            hdu_sn.header['EXTNAME'] = 'SNR'
            try:
                for i in range(8):
                    hdu_sn.header.remove('SLICE_'+str(i))
            except:
                pass
            hdu_stelkin = fits.HDUList([new_primary,
                                        hdu_input['sig_err'+bin_fit].copy(),
                                        hdu_input['flux'+bin_fit].copy(),
                                        hdu_input['flux_err'+bin_fit].copy(),
                                        hdu_sn])
            hdu_stelkin['SIG_ERR'+bin_fit].header['EXTNAME'] = 'SIG_ERR'
            hdu_stelkin['FLUX'+bin_fit].header['EXTNAME'] = 'FLUX'
            hdu_stelkin['FLUX_ERR'+bin_fit].header['EXTNAME'] = 'FLUX_ERR'

            filename = pubcubeid+'_stellar-velocity-dispersion_'+bin_name+'_two-moment.fits'
            hdu_stelkin.writeto(os.path.join(source_directory,filename))
            if not check_product_meta(filename,root_directory):
                description = 'Stellar velocity dispersion map (two moment) from {}{} cube'.format(bin_name,binned)
                product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file[1]}
                update_product_meta(filename,product_meta_dict,root_directory)
            


def reformat_stelkin4(catid,stelkin_path,qf,source_directory,root_directory):

    # Taking a SAMI stellar kinematics data (in format generated by Jesse)
    # and split the extensions into several .fits files for DR3 ingestion
    # this is a modified version that does the 4 moment files:

    # extra keywords to copy from extensions to primary:
    extra_kwd = ['UNIT','DP_NAME','DESCRIPT']


    # first check to see if the input file exists.  If it does not, then
    # don't do anything (basically, skips to the end):
    if not os.path.exists(stelkin_path):
        print(stelkin_path+' does not exist')
    else:
        # get the public cube ID:
        pubcubeid = get_cubeid_pub(catid,qf)

        # here we do have an input file, so start to do something with it:
        ofile = source_directory+'/'+pubcubeid+'_stellar-velocity_default_four-moment.fits'
        if os.path.exists(ofile):
            #Assume all stelkin maps exist if one exists, so we just check for one here.
            print('Stellar kinematic maps for '+catid+' already exist. Skipping')
            return

        # Product meta settings
        dp_type = '2d_map'
        version = REDVER
        contact = 'Jesse van de Sande <jesse.vandesande@sydney.edu.au>'
        doc_files = [['stellar-velocity_default_four-moment.html',
                      'stellar-velocity-dispersion_default_four-moment.html'],
                     ['stellar-velocity_adaptive_four-moment.html',
                      'stellar-velocity-dispersion_adaptive_four-moment.html'],
                     ['stellar-velocity_annular_four-moment.html',
                      'stellar-velocity-dispersion_annular_four-moment.html'],
                     ['stellar-velocity_sectors_four-moment.html',
                      'stellar-velocity-dispersion_sectors_four-moment.html']]
        group = 'stelkin'

        # open the input file (to read only):
        hdu_input = fits.open(stelkin_path)
        hdr_input = hdu_input[0].header
        # identify keywords than need to be removed from main header.
        keywords_to_remove = ['SIMPLE','BITPIX','NAXIS','AP_SIG','AP_SERR', 'AP_VEl',
                              'AP_VERR','APER','VERSION','SAMI_VER','DATE','A_POLY']
        for keyword_to_remove in keywords_to_remove:
            hdr_input.remove(keyword_to_remove)
        # define the different bin names (strings for file names) and extensions corresponding to those.
        # for M4 there is also ahigh S/N (HISN) binning that is S/N=20.
        bin_names = ['default','adaptive','annular','sectors','adaptive-hisn']
        bin_fits = ['','_BINNED_ADAPTIVE','_BINNED_ANNULAR','_BINNED_SECTORS','_BINNED_ADAPTIVE_HISN']
        # loop over the different binning schemes:
        for bi in range(len(bin_names)):
            hdu_input = fits.open(stelkin_path)
            bin_name = bin_names[bi]
            bin_fit = bin_fits[bi]
            doc_file = ['stellar-velocity.html','stellar-velocity-dispersion.html']#doc_files[bi]
            if bi == 0:
                binned = ''
            else:
                binned = ' binned'

            # vel:
            # for the first map (vel), we will check to make sure that the correct extension is
            # actually present.  This is particularly the case for the HISN adaptive bins that
            # only seem to be present is some cases.
            try:
                new_primary = fits.PrimaryHDU(hdu_input['vel'+bin_fit].data)
            except KeyError:
                # if a KeyError is found looking for the extension, log the problem
                # and continue with the next bin.
                logging.warning(basename(stelkin_path)+' extension not found: vel'+bin_fit)
                print('WARNING: extension not found: vel'+bin_fit)
                continue
                
            for card in hdr_input.cards:
                new_primary.header.append(card)

            # append extra header items from extension:
            new_primary.header = copy_kwd(extra_kwd,hdu_input['vel'+bin_fit],new_primary)


            qc = hdu_input['qc'+bin_fit].data
            if qc.size > 5:
                sn = qc[:,:,3]
            hdu_sn = fits.ImageHDU(sn)
            hdu_sn.header = hdu_input['qc'+bin_fit].header.copy()
            hdu_sn.header['DESCRIPT'] = 'S/N per pix, ave over non-clipped wavelength pix'
            hdu_sn.header['EXTNAME'] = 'SNR'
            try:
                for i in range(10):
                    hdu_sn.header.remove('SLICE_'+str(i))
            except:
                pass
            hdu_stelkin = fits.HDUList([new_primary,
                                        hdu_input['vel_err'+bin_fit].copy(),
                                        hdu_input['flux'+bin_fit].copy(),
                                        hdu_input['flux_err'+bin_fit].copy(),
                                        hdu_sn])
            hdu_stelkin['VEL_ERR'+bin_fit].header['EXTNAME'] = 'VEL_ERR'
            hdu_stelkin['FLUX'+bin_fit].header['EXTNAME'] = 'FLUX'
            hdu_stelkin['FLUX_ERR'+bin_fit].header['EXTNAME'] = 'FLUX_ERR'

            filename = pubcubeid+'_stellar-velocity_'+bin_name+'_four-moment.fits'
            hdu_stelkin.writeto(os.path.join(source_directory,filename))
            if not check_product_meta(filename,root_directory):
                description = 'Stellar velocity map (four moment) from {}{} cube'.format(bin_name,binned)
                product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file[0]}
                update_product_meta(filename,product_meta_dict,root_directory)

            # sig:
            new_primary = fits.PrimaryHDU(hdu_input['sig'+bin_fit].data)
            for card in hdr_input.cards:
                new_primary.header.append(card)

            # append extra header items from extension:
            new_primary.header = copy_kwd(extra_kwd,hdu_input['sig'+bin_fit],new_primary)

            qc = hdu_input['qc'+bin_fit].data
            if qc.size > 5:
                sn = qc[:,:,3]
            hdu_sn = fits.ImageHDU(sn)
            hdu_sn.header = hdu_input['qc'+bin_fit].header.copy()
            hdu_sn.header['DESCRIPT'] = 'S/N per pix, ave over non-clipped wavelength pix'
            hdu_sn.header['EXTNAME'] = 'SNR'
            try:
                for i in range(10):
                    hdu_sn.header.remove('SLICE_'+str(i))
            except:
                pass
            hdu_stelkin = fits.HDUList([new_primary,
                                        hdu_input['sig_err'+bin_fit].copy(),
                                        hdu_input['flux'+bin_fit].copy(),
                                        hdu_input['flux_err'+bin_fit].copy(),
                                        hdu_sn])
            hdu_stelkin['SIG_ERR'+bin_fit].header['EXTNAME'] = 'SIG_ERR'
            hdu_stelkin['FLUX'+bin_fit].header['EXTNAME'] = 'FLUX'
            hdu_stelkin['FLUX_ERR'+bin_fit].header['EXTNAME'] = 'FLUX_ERR'

            filename = pubcubeid+'_stellar-velocity-dispersion_'+bin_name+'_four-moment.fits'
            hdu_stelkin.writeto(os.path.join(source_directory,filename))
            if not check_product_meta(filename,root_directory):
                description = 'Stellar velocity dispersion map (four moment) from {}{} cube'.format(bin_name,binned)
                product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file[1]}
                update_product_meta(filename,product_meta_dict,root_directory)

            # H3:
            new_primary = fits.PrimaryHDU(hdu_input['h3'+bin_fit].data)
            for card in hdr_input.cards:
                new_primary.header.append(card)

            # append extra header items from extension:
            new_primary.header = copy_kwd(extra_kwd,hdu_input['h3'+bin_fit],new_primary)

            qc = hdu_input['qc'+bin_fit].data
            if qc.size > 5:
                sn = qc[:,:,3]
            hdu_sn = fits.ImageHDU(sn)
            hdu_sn.header = hdu_input['qc'+bin_fit].header.copy()
            hdu_sn.header['DESCRIPT'] = 'S/N per pix, ave over non-clipped wavelength pix'
            hdu_sn.header['EXTNAME'] = 'SNR'
            try:
                for i in range(10):
                    hdu_sn.header.remove('SLICE_'+str(i))
            except:
                pass
            hdu_stelkin = fits.HDUList([new_primary,
                                        hdu_input['h3_err'+bin_fit].copy(),
                                        hdu_input['flux'+bin_fit].copy(),
                                        hdu_input['flux_err'+bin_fit].copy(),
                                        hdu_sn])
            hdu_stelkin['H3_ERR'+bin_fit].header['EXTNAME'] = 'H3_ERR'
            hdu_stelkin['FLUX'+bin_fit].header['EXTNAME'] = 'FLUX'
            hdu_stelkin['FLUX_ERR'+bin_fit].header['EXTNAME'] = 'FLUX_ERR'

            filename = pubcubeid+'_stellar-velocity-h3_'+bin_name+'_four-moment.fits'
            hdu_stelkin.writeto(os.path.join(source_directory,filename))
            if not check_product_meta(filename,root_directory):
                description = 'Stellar velocity h3 map (four moment) from {}{} cube'.format(bin_name,binned)
                product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file[1]}
                update_product_meta(filename,product_meta_dict,root_directory)
            
            # H4:
            new_primary = fits.PrimaryHDU(hdu_input['h4'+bin_fit].data)
            for card in hdr_input.cards:
                new_primary.header.append(card)

            # append extra header items from extension:
            new_primary.header = copy_kwd(extra_kwd,hdu_input['h4'+bin_fit],new_primary)

            qc = hdu_input['qc'+bin_fit].data
            if qc.size > 5:
                sn = qc[:,:,3]
            hdu_sn = fits.ImageHDU(sn)
            hdu_sn.header = hdu_input['qc'+bin_fit].header.copy()
            hdu_sn.header['DESCRIPT'] = 'S/N per pix, ave over non-clipped wavelength pix'
            hdu_sn.header['EXTNAME'] = 'SNR'
            try:
                for i in range(10):
                    hdu_sn.header.remove('SLICE_'+str(i))
            except:
                pass
            hdu_stelkin = fits.HDUList([new_primary,
                                        hdu_input['h4_err'+bin_fit].copy(),
                                        hdu_input['flux'+bin_fit].copy(),
                                        hdu_input['flux_err'+bin_fit].copy(),
                                        hdu_sn])
            hdu_stelkin['H4_ERR'+bin_fit].header['EXTNAME'] = 'H4_ERR'
            hdu_stelkin['FLUX'+bin_fit].header['EXTNAME'] = 'FLUX'
            hdu_stelkin['FLUX_ERR'+bin_fit].header['EXTNAME'] = 'FLUX_ERR'

            filename = pubcubeid+'_stellar-velocity-h4_'+bin_name+'_four-moment.fits'
            hdu_stelkin.writeto(os.path.join(source_directory,filename))
            if not check_product_meta(filename,root_directory):
                description = 'Stellar velocity h4 map (four moment) from {}{} cube'.format(bin_name,binned)
                product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_file[1]}
                update_product_meta(filename,product_meta_dict,root_directory)
            


def reformat_lzifu(catid,cname,qf,source_directory,root_directory,
                   component='1',cube_label='default'):

    # generate the file name:
    filename = cname.replace('.fits','')
    filename = filename.replace('_blue','')
    filename = filename+'_'+component+'_comp.fits.gz'
    lzifu_path = os.path.join(LZIFUPATH,'lzifu_'+cube_label+'_products',filename)
    print(lzifu_path)

    # check that the file exists:
    if not os.path.exists(lzifu_path):
        print(lzifu_path,' does not exist')
        return
    
    if component == 'recom':
        n_comp = 3
        comp_label = 'Recommended'
    else:
        n_comp = int(component)
        comp_label = component

    # get the public cube ID:
    pubcubeid = get_cubeid_pub(catid,qf)
    ofile = source_directory+'/'+pubcubeid+'_gas-velocity_'+cube_label+'_'+component+'-comp.fits'

    if os.path.exists(ofile):
        #Assume all lzifu maps exist if one exists - semi-risky
        print('LZIFU '+cube_label+' '+component+'-comp maps for '+catid+' already exist. Skipping')
        return

    hdu_input = fits.open(lzifu_path)

    # Product meta settings
    dp_type = '2d_map'
    version = REDVER
    contact = 'Brent Groves <brent.groves@uwa.edu.au>'
    doc_files = ['gas_kinematics.html','emission_line_flux.html']
    group = 'lzifu-kin'

    ############# Velocity and Velocity Dispersion #############

    labels=['gas-velocity','gas-vdisp']
    fits_labels = ['V','VDISP']
    desc_labels = ['velocity','velocity dispersion']

    if cube_label == 'default':
        binned = ''
    else:
        binned = ' binned'
        
    for i in range(len(labels)):
        new_primary = fits.PrimaryHDU(hdu_input[fits_labels[i]].data[1:n_comp+1,:,:])
        ww = np.where(np.array(list(hdu_input[fits_labels[i]].header.keys())) == 'CRPIX1')[0][0]
        for card in hdu_input[fits_labels[i]].header.cards[ww:]:
            new_primary.header.append(card)
        new_primary.header['NAXIS3'] = n_comp
        add_keyword_comments_wcs(new_primary.header)
        try:
            new_primary.header.remove('EXTNAME')
        except:
            pass
        new_primary.header['BUNIT'] = ('km/s','units')
        hdu_input[fits_labels[i]+'_ERR'].header['NAXIS3'] = n_comp
        hdu_input[fits_labels[i]+'_ERR'].header['BUNIT'] = ('km/s','units')
        #hdu_input[fits_labels[i]+'_ERR'].header['EXTNAME'] = labels[i]+'_err'
            
        hdu_err = fits.ImageHDU(hdu_input[fits_labels[i]+'_ERR'].data[1:n_comp+1,:,:])
        hdu_err.header = hdu_input[fits_labels[i]+'_ERR'].header.copy()
        add_keyword_comments_wcs(hdu_err.header)
        hdu_lzifu = fits.HDUList([new_primary,hdu_err ])

        filename = pubcubeid+'_'+labels[i]+'_'+cube_label+'_'+component+'-comp.fits'
        hdu_lzifu.writeto(os.path.join(source_directory,filename),output_verify='fix')
        if not check_product_meta(filename,root_directory):
            description = '{}-component ionised gas {} map from {}{} cube'.format(comp_label,desc_labels[i],cube_label,binned)
            product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_files[0]}
            update_product_meta(filename,product_meta_dict,root_directory)

        ############# OII doublet #############

    group = 'lzifu-flux'
    labels = 'OII3728'
    fits_labels = ['OII3726','OII3729']
    new_primary = fits.PrimaryHDU(hdu_input[fits_labels[0]].data[0,:,:]+hdu_input[fits_labels[1]].data[0,:,:])
    ww = np.where(np.array(list(hdu_input[fits_labels[0]].header.keys())) == 'CRPIX1')[0][0]
    for card in hdu_input[fits_labels[0]].header.cards[ww:]:
        new_primary.header.append(card)
    new_primary.header['BUNIT'] = ('10**(-16) erg/s/cm**2/angstrom/pixel','units')
    try:
        new_primary.header.remove('EXTNAME')
    except:
        pass
    add_keyword_comments_wcs(new_primary.header)

    hdu_err = fits.ImageHDU(hdu_input[fits_labels[0]+'_ERR'].data[0,:,:] + hdu_input[fits_labels[1]+'_ERR'].data[0,:,:])
    ww = np.where(np.array(list(hdu_input[fits_labels[0]+'_ERR'].header.keys())) == 'CRPIX1')[0][0]
    for card in hdu_input[fits_labels[0]+'_ERR'].header.cards[ww:]:
        hdu_err.header.append(card)
    hdu_err.header['BUNIT'] = ('10**(-16) erg/s/cm**2/angstrom/pixel','units')
    add_keyword_comments_wcs(hdu_err.header)

    hdu_lzifu = fits.HDUList([new_primary,hdu_err])


    filename = pubcubeid+'_'+labels+'_'+cube_label+'_'+component+'-comp.fits'
    hdu_lzifu.writeto(os.path.join(source_directory,filename),output_verify='fix')
    if not check_product_meta(filename,root_directory):
        description = '{}-component OII3728 flux map from {}{} cube'.format(comp_label,cube_label,binned)
        product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_files[1]}
        update_product_meta(filename,product_meta_dict,root_directory)

    ############# Halpha #############

    labels = 'Halpha'
    fits_label = 'HALPHA'
    new_primary = fits.PrimaryHDU(hdu_input[fits_label].data[0:n_comp+1,:,:])
    ww = np.where(np.array(list(hdu_input[fits_label].header.keys())) == 'CRPIX1')[0][0]
    for card in hdu_input[fits_label].header.cards[ww:]:
        new_primary.header.append(card)
    new_primary.header['BUNIT'] = ('10**(-16) erg/s/cm**2/angstrom/pixel','units')
    try:
        new_primary.header.remove('EXTNAME')
    except:
        pass
    add_keyword_comments_wcs(new_primary.header)
    hdu_input[fits_label+'_ERR'].header['BUNIT'] = ('10**(-16) erg/s/cm**2/angstrom/pixel','units')
    hdu_input[fits_label+'_ERR'].header['NAXIS3'] = n_comp+1
    hdu_input[fits_label+'_ERR'].header.add_history('Halpha has {} components'.format(n_comp+1))
    hdu_input[fits_label+'_ERR'].header.add_history('These represent the total (0) and {} velocity components'.format(n_comp))
    new_primary.header.add_history('Halpha has {} components'.format(n_comp+1))
    new_primary.header.add_history('These represent the total (0) and {} velocity components'.format(n_comp))
            
    hdu_err = hdu_input[fits_label+'_ERR']
    hdu_err.data = hdu_err.data[0:n_comp+1,:,:]
    add_keyword_comments_wcs(hdu_err.header)

    hdu_lzifu = fits.HDUList([new_primary,hdu_err])


    filename = pubcubeid+'_'+labels+'_'+cube_label+'_'+component+'-comp.fits'
    hdu_lzifu.writeto(os.path.join(source_directory,filename),output_verify='fix')
    if not check_product_meta(filename,root_directory):
        description = '{}-component Halpha flux map from {}{} cube'.format(comp_label,cube_label,binned)
        product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_files[1]}
        update_product_meta(filename,product_meta_dict,root_directory)

    ############# Other lines #############

    labels = ['Hbeta','OIII5007','OI6300','NII6583','SII6716','SII6731']
    fits_labels = ['HBETA','OIII5007','OI6300','NII6583','SII6716','SII6731']
    ids = [15,17,19,23,25,27]
    for i in range(len(labels)):
        # use try here to find the data, this allows us to catch the
        # exception if there is not the expected extensions:
        try:
            new_primary = fits.PrimaryHDU(hdu_input[fits_labels[i]].data[0,:,:])
        except KeyError:
            # in this case the extension we are looking for does not exist.  This should
            # not happen, but can.  We need to make sure this does not cause a crash, but
            # also record the file that has the problem.
            print('Warning: '+lzifu_path+' cannot find extension '+fits_labels[i])
            logging.warning(lzifu_path+' cannot find extension '+fits_labels[i])
            continue
            
        ww = np.where(np.array(list(hdu_input[fits_labels[i]].header.keys())) == 'CRPIX1')[0][0]
        for card in hdu_input[fits_labels[i]].header.cards[ww:]:
            new_primary.header.append(card)
        new_primary.header['BUNIT'] = ('10**(-16) erg/s/cm**2/angstrom/pixel','units')
        try:
            new_primary.header.remove('EXTNAME')
        except:
            pass
        add_keyword_comments_wcs(new_primary.header)

        hdu_err = fits.ImageHDU(hdu_input[fits_labels[i]+'_ERR'].data[0,:,:])
        ww = np.where(np.array(list(hdu_input[fits_labels[i]+'_ERR'].header.keys())) == 'CRPIX1')[0][0]
        for card in hdu_input[fits_labels[i]+'_ERR'].header.cards[ww:]:
            hdu_err.header.append(card)
        hdu_err.header['BUNIT'] = ('10**(-16) erg/s/cm**2/angstrom/pixel','units')
        try:
            hdu_err.header.remove('PCOUNT')
            hdu_err.header.remove('GCOUNT')
        except:
            pass

        add_keyword_comments_wcs(hdu_err.header)
        
        hdu_lzifu = fits.HDUList([new_primary,hdu_err])


        filename = pubcubeid+'_'+labels[i]+'_'+cube_label+'_'+component+'-comp.fits'
        hdu_lzifu.writeto(os.path.join(source_directory,filename),output_verify='fix')
        if not check_product_meta(filename,root_directory):
            description = '{}-component {} flux map from {}{} cube'.format(comp_label,labels[i],cube_label,binned)
            product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_files[1]}
            update_product_meta(filename,product_meta_dict,root_directory)



def reformat_sfr(catid,cname,qf,source_directory,root_directory,
                   component='1',cube_label='default'):

    # generate the file names for the 3 maps SFRmap SFMask, ExtinctCorr
    filename_root = cname.replace('.fits','')
    filename_root = filename_root.replace('_blue','')
    # SFRmap:
    filename = filename_root+'_SFR_'+component+'_comp.fits'
    sfr_path = os.path.join(LZIFUPATH,'lzifu_'+cube_label+'_products',filename)
    #sfr_path = os.path.join(LZIFUPATH,'lzifu_VAP','lzifu_'+cube_label+'_SFRmap',filename)
    # SFMask:
    filename = filename_root+'_SFMask_'+component+'_comp.fits'
    sfmask_path = os.path.join(LZIFUPATH,'lzifu_'+cube_label+'_products',filename)
    #sfmask_path = os.path.join(LZIFUPATH,'lzifu_VAP','lzifu_'+cube_label+'_SFMasks',filename)
    # ExtinctCorr:
    filename = filename_root+'_extinction_'+component+'_comp.fits'
    ext_path = os.path.join(LZIFUPATH,'lzifu_'+cube_label+'_products',filename)
    #ext_path = os.path.join(LZIFUPATH,'lzifu_VAP','lzifu_'+cube_label+'_ExtinctCorr',filename)
    
    # check if the file exists:
    if not os.path.exists(sfr_path):
        print(sfr_path,' does not exist')
        return

    if not os.path.exists(sfmask_path):
        print(sfmask_path,' does not exist')
        return

    if not os.path.exists(ext_path):
        print(ext_path,' does not exist')
        return

    if component == 'recom':
        n_comp = 3
        comp_label = 'Recommended'
    else:
        n_comp = int(component)
        comp_label = component

    if cube_label == 'default':
        binned = ''
    else:
        binned = ' binned'

    # get the public cube ID:
    pubcubeid = get_cubeid_pub(catid,qf)
    ofile = source_directory+'/'+pubcubeid+'_sfr_'+cube_label+'_'+component+'-comp.fits'

    if os.path.exists(ofile):
        #Assume all lzifu maps exist if one exists - semi-risky
        print('SFR '+cube_label+' '+component+'-comp maps for '+catid+' already exist. Skipping')
        return

    # Product meta settings
    dp_type = '2d_map'
    version = REDVER
    contact = 'Brent Groves <brent.groves@uwa.edu.au>'
    doc_files = ['sfr.html','sfr_dens.html','sfr_mask.html','extinction.html']
    group = 'sfr'

    #Extinction maps
    hdu_input = fits.open(ext_path)
    new_primary = fits.PrimaryHDU(hdu_input['EXTINCT_CORR'].data)
    ww = np.where(np.array(list(hdu_input['EXTINCT_CORR'].header.keys())) == 'CATID')[0][0]
    for card in hdu_input['EXTINCT_CORR'].header.cards[ww:]:
        new_primary.header.append(card)
    new_primary.header['SAMI_VER'] = REDVER
    add_keyword_comments_lzifu(new_primary.header)
    try:
        new_primary.header.remove('EXTNAME')
    except:
        pass

    hdu_input['EXTINCT_CORR_ERR'].header['SAMI_VER'] = REDVER
    add_keyword_comments_lzifu(hdu_input['EXTINCT_CORR_ERR'].header)

    hdu_sfr = fits.HDUList([new_primary,hdu_input['EXTINCT_CORR_ERR']])

    filename = pubcubeid+'_extinct-corr_'+cube_label+'_'+component+'-comp.fits'
    hdu_sfr.writeto(os.path.join(source_directory,filename),output_verify='fix')
    if not check_product_meta(filename,root_directory):
        description = '{}-component extinction correction map from {}{} cube'.format(comp_label,cube_label,binned)
        product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_files[3]}
        update_product_meta(filename,product_meta_dict,root_directory)

    #SFR Mask maps
    # this is just a copy, as the file is simple, however we will reopen it to tidy up headers:
    filename = pubcubeid+'_sfr-mask_'+cube_label+'_'+component+'-comp.fits'
    outfile = os.path.join(source_directory,filename)
    shutil.copy2(sfmask_path,outfile)
    hdulist_sfm = fits.open(outfile,'update')
    try:
        hdulist_sfm[0].header.remove('EXTNAME')
    except:
        pass
    add_keyword_comments_lzifu(hdulist_sfm[0].header)
    hdulist_sfm.flush()
    hdulist_sfm.close()
    if not check_product_meta(filename,root_directory):
        description = '{}-component star formation mask map from {}{} cube'.format(comp_label,cube_label,binned)
        product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_files[2]}
        update_product_meta(filename,product_meta_dict,root_directory)
        

    #SFR and SFRDens maps
    hdu_input = fits.open(sfr_path)
    new_primary = fits.PrimaryHDU(hdu_input['SFR'].data[0:n_comp+1,:,:])
    ww = np.where(np.array(list(hdu_input['SFR'].header.keys())) == 'CATID')[0][0]
    for card in hdu_input['SFR'].header.cards[ww:]:
        new_primary.header.append(card)
    try:
        new_primary.header.remove('EXTNAME')
    except:
        pass

    new_primary.header['SAMI_VER'] = REDVER
    add_keyword_comments_lzifu(new_primary.header)

    hdu_err = fits.ImageHDU(hdu_input['SFR_ERR'].data[0:n_comp+1,:,:])
    ww = np.where(np.array(list(hdu_input['SFR_ERR'].header.keys())) == 'CATID')[0][0]
    for card in hdu_input['SFR_ERR'].header.cards[ww:]:
        hdu_err.header.append(card)
    hdu_err.header['SAMI_VER'] = REDVER
    add_keyword_comments_lzifu(hdu_err.header)
    hdu_err.header.add_history('SFR has {} components'.format(n_comp+1))
    hdu_err.header.add_history('These represent the total (0) and {} velocity components'.format(n_comp))
    new_primary.header.add_history('SFR has {} components'.format(n_comp+1))
    new_primary.header.add_history('These represent the total (0) and {} velocity components'.format(n_comp))

    hdu_sfr = fits.HDUList([new_primary,hdu_err])

    filename = pubcubeid+'_sfr_'+cube_label+'_'+component+'-comp.fits'
    hdu_sfr.writeto(os.path.join(source_directory,filename),output_verify='fix')
    if not check_product_meta(filename,root_directory):
        description = '{}-component star formation rate map from {}{} cube'.format(comp_label,cube_label,binned)
        product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_files[0]}
        update_product_meta(filename,product_meta_dict,root_directory)

    new_primary = fits.PrimaryHDU(hdu_input['SFRSurfDensity'].data[0:n_comp+1,:,:])
    ww = np.where(np.array(list(hdu_input['SFRSurfDensity'].header.keys())) == 'CATID')[0][0]
    for card in hdu_input['SFRSurfDensity'].header.cards[ww:]:
        new_primary.header.append(card)
    try:
        new_primary.header.remove('EXTNAME')
    except:
        pass

    new_primary.header['SAMI_VER'] = REDVER
    add_keyword_comments_lzifu(new_primary.header)
    # *** why data added here???  And in different format
    #new_primary.header['DATE'] = 'Aug 29 2018'
    hdu_err = fits.ImageHDU(hdu_input['SFRSurfDensity_ERR'].data[0:n_comp+1,:,:])
    ww = np.where(np.array(list(hdu_input['SFRSurfDensity_ERR'].header.keys())) == 'CATID')[0][0]
    for card in hdu_input['SFRSurfDensity_ERR'].header.cards[ww:]:
        hdu_err.header.append(card)
    hdu_err.header['SAMI_VER'] = REDVER
    add_keyword_comments_lzifu(hdu_err.header)
    #hdu_err.header['DATE'] = 'Aug 29 2018'
    hdu_err.header.add_history('SFRSurfDensity has {} components'.format(n_comp+1))
    hdu_err.header.add_history('These represent the total (0) and {} velocity components'.format(n_comp))
    new_primary.header.add_history('SFRSurfDensity has {} components'.format(n_comp+1))
    new_primary.header.add_history('These represent the total (0) and {} velocity components'.format(n_comp))

    hdu_sfr = fits.HDUList([new_primary,hdu_err])

    filename = pubcubeid+'_sfr-dens_'+cube_label+'_'+component+'-comp.fits'
    hdu_sfr.writeto(os.path.join(source_directory,filename),output_verify='fix')
    if not check_product_meta(filename,root_directory):
        description = '{}-component star formation rate surface density map from {}{} cube'.format(comp_label,cube_label,binned)
        product_meta_dict = {'dp_type':dp_type,
                                 'version':version,
                                 'contact':contact,
                                 'group_name':group,
                                 'desc':description,
                                 'doc_file':doc_files[0]}
        update_product_meta(filename,product_meta_dict,root_directory)
        

def reformat_one(root_directory,catid,cname,qf,cube_folder,
                 aperture_folder,stelkin_folder,stelkin_folder4,
                 lzifu_folder_1,lzifu_folder_recom,
                 lzifu_folder_binned,dataflag=0,sim=False,
                 check_only=True,zspec=np.nan):

    # Extract "all" data products for one CATID into DR2
    # ingestion-ready format

    # optional dataflag sets which data for the object we will
    # actually ingest.  The possibilities are:
    #
    # 0 - all data
    # 1 - cubes only 
    # 2 - aperture spectra only
    # 3 - stellar kinematics only
    # 4 - lzifu only


    source_directory = verify_source_directory(catid,root_directory)

    cube_path = cube_folder+'/'+cname

    # do the cubes first:
    if ((dataflag == 0) or (dataflag == 1)): 

        # first do blue cube:
        reformat_cube(catid,cube_path,qf,source_directory,root_directory,
                      sim=sim,check_only=check_only,zspec=zspec)
        # then do red cube:
        rcube_path = cube_path.replace('blue','red')
        reformat_cube(catid,rcube_path,qf,source_directory,root_directory,
                      sim=sim,check_only=check_only,zspec=zspec)

    # next do the aperture spectra:
    if ((dataflag == 0) or (dataflag == 2)): 

        # based on the cube path, derive the apspec path:
        #comp = cname.split('_')
        #comp.insert(6,'apspec')
        #aperture_file = '_'.join(comp)
        aperture_file = cname.replace('.fits','_apspec.fits')
        aperture_path = aperture_folder+'/'+aperture_file
        reformat_aperture_spectrum(catid,aperture_path,qf,source_directory,root_directory,
                                   sim=sim,check_only=check_only,zspec=zspec)
        raperture_path = aperture_path.replace('blue','red')
        reformat_aperture_spectrum(catid,raperture_path,qf,source_directory,root_directory,
                                   sim=sim,check_only=check_only,zspec=zspec)


    # next the stellar kinematics:
    if ((dataflag == 0) or (dataflag == 3)): 
        # based on the cube path, derive the stelkin path:
        comp = cname.split('_')
        comp[1] = 'blue_red'
        stelkin_file = '_'.join(comp)
        stelkin_file = stelkin_file.replace('.fits','_kinematicsM2.fits')
        print('Copying M2 stellar kinematics for: '+catid)
        stelkin_path = stelkin_folder+'/'+stelkin_file
        reformat_stelkin(catid,stelkin_path,qf,source_directory,root_directory)

        # do the M4 stellar kinematics:
        stelkin_file4 = stelkin_file.replace('M2.fits','M4.fits')
        print('Copying M4 stellar kinematics for: '+catid)
        stelkin_path = stelkin_folder4+'/'+stelkin_file4
        reformat_stelkin4(catid,stelkin_path,qf,source_directory,root_directory)

    if ((dataflag == 0) or (dataflag == 4)): 
        print('Copying LZIFU outputs for: '+catid)
        # 1-comp unbinned
        reformat_lzifu(catid,cname,qf,source_directory,root_directory,
                       component='1',cube_label='default')
        # recom-comp unbinned
        reformat_lzifu(catid,cname,qf,source_directory,root_directory,
                       component='recom',cube_label='default')
        # 1-comp sectors
        reformat_lzifu(catid,cname,qf,source_directory,root_directory,
                       component='1',cube_label='sectors')
        # recom-comp sectors
        reformat_lzifu(catid,cname,qf,source_directory,root_directory,
                       component='recom',cube_label='sectors')
        # 2-comp annular
        reformat_lzifu(catid,cname,qf,source_directory,root_directory,
                       component='2',cube_label='annular')
        # 1-comp adaptive
        reformat_lzifu(catid,cname,qf,source_directory,root_directory,
                       component='1',cube_label='adaptive')
        # recom-comp adaptive
        reformat_lzifu(catid,cname,qf,source_directory,root_directory,
                       component='recom',cube_label='adaptive')

    # now doing SFR etc
        print('Copying SFR products for: '+catid)
        reformat_sfr(catid,cname,qf,source_directory,root_directory,
                     component='1',cube_label='default')
        reformat_sfr(catid,cname,qf,source_directory,root_directory,
                     component='recom',cube_label='default')

    # 
    # COMMENT BACK IN WHEN BINNED SFR PRODUCTS BECOME AVAILABLE

    #cube_labels = ['annular','sectors','sectors','adaptive','adaptive']
    #components = ['2','1','recom','1','recom']
    #path_labels = ['annular_2_comp','sectors_1_comp','sectors_recom','adaptive_1_comp','adaptive_recom']
    #for cube_label,component,path_label in zip(cube_labels,components,path_labels):
    #    sfr_path = glob.glob(lzifu_folder_binned+'products_'+path_label+'/'+catid+'*_'+component+'_comp.fits')
    #    reformat_sfr(catid,sfr_path,source_directory,root_directory,
    #                   component=component,cube_label=cube_label)

        # adaptive:
        reformat_sfr(catid,cname,qf,source_directory,root_directory,
                     component='1',cube_label='adaptive') 
        reformat_sfr(catid,cname,qf,source_directory,root_directory,
                     component='recom',cube_label='adaptive')
        # sectors:
        reformat_sfr(catid,cname,qf,source_directory,root_directory,
                     component='1',cube_label='sectors') 
        reformat_sfr(catid,cname,qf,source_directory,root_directory,
                     component='recom',cube_label='sectors')
        # Annular (only 2 comp):
        reformat_sfr(catid,cname,qf,source_directory,root_directory,
                     component='2',cube_label='annular') 

    

def reformat_many(root_directory,listfile_path,cube_folder,
                  aperture_folder,stelkin_folder,stelkin_folder4,lzifu_folder_1,
                  lzifu_folder_recom,lzifu_folder_binned,
                  split=[0.0,1.0],testrun=False,dataflag=0,sim=False,
                  check_only=True): 

    # optional dataflag = 0 for all, or 1 for just cubes etc...

    # set up timing for this step as it take a long time!
    start_time = datetime.now()

    # set up logging for this routine:
    logging.basicConfig(filename='reformat_many.log',level=logging.INFO,
                        format='%(levelname)s:%(message)s')
    
    # Extract all data products from a list of CATIDs into
    # DR2 ingestion-ready format

    # this assumes the list file in ascii:
    #tab = Table.read(listfile_path,format='ascii.commented_header')

    # newer version assumes it is a fits binary table
    hdulist = fits.open(listfile_path)
    table_data = hdulist['DR_CUBES'].data
    ids = table_data['CATID']
    cubename = table_data['CUBENAME']
    #qflag = table_data['QFLAG']
    #renamed to RFLAG (R for repeats)
    qflag = table_data['RFLAG']
    catsource = table_data['CATSOURCE']
    z_spec = table_data['Z_SPEC']


    # change behaviour depending on whether a test run.  If a test run, only do a 
    # few objects:
    #if (testrun):
        #ids = [22887,56140,570206,69740]
        #ids = [22887]
        # cluster gal with 4(!) different cubes:
        #ids = [999403800283]
        # gama gal with 3 diff cubes:
        #ids = [79810]
        # gama gal with 2 diff cubes:
        #ids = [77452]

    #else:
        #ids = tab['CATID'].data

    #n = len(ids)
    n=len(cubename)

    if (testrun):
        n = 20
        n1 = 200
        n2 = 220
    else:
        # I think it allows us to split up the run into 
        # chunks based on passing the split array.
        #ids = ids[np.int(np.floor(n*split[0])):np.int(np.ceil(n*split[1]))]
        # as we have multiple arrays to handle now we will do this a little
        # differently:
        n1 = np.int(np.floor(n*split[0]))
        n2 = np.int(np.ceil(n*split[1]))

    print('Number of cubes to ingest: ',n)
    
    #for catid in ids:
    #    catid = str(catid)
    for i in range(n1,n2):
        catid = str(ids[i])
        cname = cubename[i]
        qf = qflag[i]
        # check if this object has a source catalogue.  If this parameter is
        # -1, it does not.  All catalogue should have this set to 1 or more.
        # these chould now be removed from the cube list, but checked here
        # as well.
        # if it is set to zero, there has been an error somewhere
        if (catsource[i] < 1):
            print(cname,catid,' NOT in catalogues',catsource[i])
            logging.warning(cname+' '+catid+' not in catalogues')
            continue

        print('Reformatting data for ',catid,cname,i,' of ',n1,' to ',n2)
        reformat_one(root_directory,catid,cname,qf,cube_folder,aperture_folder,stelkin_folder,
                     stelkin_folder4,lzifu_folder_1,lzifu_folder_recom,lzifu_folder_binned,dataflag=dataflag,
                     sim=sim,check_only=check_only,zspec=z_spec[i])

    # check the timing:
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print('Time ellapsed running reformat_many(): ',elapsed_time)


def add_table(root_directory,cat_name,table_info,group=''):
    
    # Add a table, copying it into the correct folder, updating the table_meta
    # and column_meta files and importing a doc file if requested

    # define paths to files:
    tab_path = CATFILEPATH
    table_file = os.path.join(tab_path,cat_name+'.fits')
    column_file = os.path.join(tab_path,cat_name+'.col')
    doc_file_in = os.path.join(tab_path,cat_name+'.html')
    doc_file_out = cat_name+'.html'

    verify_directory_structure(root_directory)
    
    if not os.path.exists(table_file):
        print('{} does not exist'.format(table_file))
        return

    print(table_info)
    with open(root_directory+'metadata/sami/dr3/catalogues/sami_dr3_table_meta.txt','a') as table_meta_file:
        table_meta_file.write('{}|{}|{}|{}|{}|{}|{}|{}\n'.format(table_info['name'],table_info['description'],
                                                               doc_file_out,group,table_file.split('/')[-1],table_info['contact'],
                                                               table_info['date'],table_info['version']))

    with open(root_directory+'metadata/sami/dr3/catalogues/sami_dr3_column_meta.txt','a') as column_meta_file:
        tab_col = Table.read(column_file,format='ascii')
        for i in range(len(tab_col)):
            column_meta_file.write('{}|{}|{}|{}|{}|{}\n'.format(tab_col['name'][i],table_info['name'],
                                                                    tab_col['description'][i],tab_col['ucd'][i],
                                                                    tab_col['unit'][i],tab_col['data_type'][i]))

    shutil.copy2(table_file,root_directory+'data/sami/dr3/catalogues/'+table_file.split('/')[-1])
    if not doc_file_in == '':
        shutil.copy2(doc_file_in,root_directory+'metadata/sami/dr3/catalogues/docs/'+doc_file_out)

    
    print('Added catalogue:',table_file)

        
def add_all_tables(root_directory):

    """Ingest all the catalogues and set up metadata files for catalogues."""

    # set upthe headers for the table files.  This has the effect of re-initializing them
    # and removing old versions.

    # Create catalogue table_meta.txt
    with open(root_directory+'metadata/sami/dr3/catalogues/sami_dr3_table_meta.txt','w') as catatable_file:
        catatable_file.write('name|description|documentation|group_name|filename|contact|date|version\n')

    # Create catalogue column_meta.txt
    with open(root_directory+'metadata/sami/dr3/catalogues/sami_dr3_column_meta.txt','w') as catacolumn_file:
        catacolumn_file.write('name|table_name|description|ucd|unit|data_type\n')

    
    ver = 'DR3'

    # InputCat_GAMA:
    cat_name = 'InputCatGAMADR3'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Scott Croom <scott.croom@sydney.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'SAMI input catalogue for GAMA regions'
    group = 'other'
    add_table(root_directory,cat_name,table_info,group=group)

    # InputCat_Clusters:
    cat_name = 'InputCatClustersDR3'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Matt Owers <matt.owers@mq.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'SAMI input catalogue for cluster regions'
    group = 'other'
    add_table(root_directory,cat_name,table_info,group=group)

    # InputCat_Filler:
    cat_name = 'InputCatFiller'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Scott Croom <scott.croom@sydney.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'SAMI input catalogue for filler objects'
    group = 'other'
    add_table(root_directory,cat_name,table_info,group=group)

    # Fstars for GAMA
    cat_name = 'FstarCatGAMA'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Scott Croom <scott.croom@sydney.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'SAMI calibration stars in GAMA regions'
    group = 'other'
    add_table(root_directory,cat_name,table_info,group=group)

    # Fstars for Clusters
    cat_name = 'FstarCatClusters'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Matt Owers <matt.owers@mq.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'SAMI calibration stars in cluster regions'
    group = 'other'
    add_table(root_directory,cat_name,table_info,group=group)

    # cube quality cat
    cat_name = 'CubeObs'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Scott Croom <scott.croom@sydney.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'SAMI cube observations, quality and flagging catalogue'
    group = 'sami'
    add_table(root_directory,cat_name,table_info,group=group)

    # local density estimates:
    cat_name = 'DensityCatDR3'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Matt Owers <matt.owers@mq.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = '5th nearest neighbour surface density estimates'
    group = 'other'
    add_table(root_directory,cat_name,table_info,group=group)

    # Lick index measurements:
    cat_name = 'IndexAperturesDR3'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Nic Scott <nicholas.scott@sydney.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'Lick index measurements from aperture spectra'
    group = 'sami'
    add_table(root_directory,cat_name,table_info,group=group)

    # SSP aperture measurements:
    cat_name = 'SSPAperturesDR3'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Nic Scott <nicholas.scott@sydney.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'Single stellar population equivalent age, [Z/H] and [alpha/Fe] measured from aperture spectra'
    group = 'sami'
    add_table(root_directory,cat_name,table_info,group=group)

    # MGE measurements:
    cat_name = 'MGEPhotomUnregDR3'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Francesco  D\'Eugenio <francesco.deugenio@gmail.com>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'Results from Multi Gaussian Expansion fitting of imaging data'
    group = 'other'
    add_table(root_directory,cat_name,table_info,group=group)

    # Gas kinematic PA:
    cat_name = 'samiDR3gaskinPA'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Jesse van de Sande <jesse.vandesande@sydney.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'Gas Kinematic PA Catalogue'
    group = 'sami'
    add_table(root_directory,cat_name,table_info,group=group)

    # Stellar kinematics:
    cat_name = 'samiDR3Stelkin'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Jesse van de Sande <jesse.vandesande@sydney.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'Stellar kinematic measurements catalogue'
    group = 'sami'
    add_table(root_directory,cat_name,table_info,group=group)

    # Morphology cat:
    cat_name = 'VisualMorphologyDR3'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Luca Cortese <luca.cortese@uwa.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'SDSS/VST-based Visual morphological classification'
    group = 'other'
    add_table(root_directory,cat_name,table_info,group=group)

    # LZIFU 1-comp cat:
    cat_name = 'EmissionLine1compDR3'
    table_info = dict()
    table_info['name'] = cat_name
    table_info['contact'] = 'Brent Groves <brent.groves@uwa.edu.au>'
    table_info['version'] = ver
    table_info['date'] = DRDATE
    table_info['description'] = 'Aperture 1-component emission line fluxes, ionised gas kinematics, and star formation rates'
    group = 'sami'
    add_table(root_directory,cat_name,table_info,group=group)

    # LZIFU recommended-comp cat.  Will not be made public!
    #cat_name = 'EmissionLineRecomcompDR3'
    #table_info = dict()
    #table_info['name'] = cat_name
    #table_info['contact'] = 'Brent Groves <brent.groves@uwa.edu.au>'
    #table_info['version'] = ver
    #table_info['date'] = DRDATE
    #table_info['description'] = 'Aperture recommended-component emission line fluxes, ionised gas kinematics, and star formation rates'
    #group = 'sami'
    #add_table(root_directory,cat_name,table_info,group=group)

    return

def update_product_meta(file_pattern,product_meta_dict,root_directory):

    with open(root_directory+'metadata/sami/dr3/ifs/sami_dr3_product_meta.txt','a') as product_file:
        # old DR2 version:
#        name = file_pattern.split('.')[0]
        # if needed, remove gzip in filename:
        tmp1 = re.sub(r'.gz','',file_pattern)
        # remove catid and repeat flag (first two elements in file name), the format is
        # 123456_A_...:
        tmp2 = tmp1[tmp1.index('_')+3:]
        # remove '_A.fits' or equivalent:
        name = re.sub(r'.fits','',tmp2)

        
        file_pattern = '{source_name}_'+file_pattern
        product_file.write('sami|{}|{}|{}|{}|{}|{}\n'.format(name,
                                                                   product_meta_dict['desc'],
                                                                   product_meta_dict['doc_file'],
                                                                   product_meta_dict['group_name'],
                                                                   product_meta_dict['version'],
                                                                   product_meta_dict['contact']))
        print('Written '+name+' to product_meta.txt')

def clean_ingestion_catids(root_directory,catidlistfile,sim=True):

    """Script to remove specific CATIDs from the ingestion.  The 
    input file catidlistfile should be a text file with only cubenames and catids
    as two cols.  First col is cubenames (not actually used) and the second col
    is the CATIDs.  This is the format from get_new_catids()."""

    # read text file columns using numpy:
    cubes, catids = np.genfromtxt(catidlistfile,usecols=(0,1),unpack=True,dtype='str')

    # get the unique CATIDs:
    unique_catids = np.unique(catids)

    print('Number of CATIDs read:',np.size(catids))
    print('Number of unique CATIDs read:',np.size(unique_catids))

    # for each CATID, find the folder and all the files in it:
    n_rem = 0
    for catid in unique_catids:
        print('Searching for files in with CATID ',catid)
        
        # derive the correct path:
        path = os.path.join(root_directory,'data',catid)

        # search for all the files in this path:
        files = os.listdir(path)

        # go through each file and remove:
        for fname in files:
            
            # get the full file name:
            fullname = os.path.join(path,fname)
            
            # remove if not in simulation mode:
            if (sim):
                print(fullname,' not removed, as in simulation mode')
            else:
                os.remove(fullname)
                n_rem = n_rem + 1

    print('Number of files removed:',n_rem)
    if (sim):
        print('In simulation mode, no actual action taken')

    return

def compare_cube_wcs():
    """Read in catalogues and get object ra,dec and then compare this to the
    WCS CRVAL1 and CRVAL2.  Objects that have large differences between these
    will need to be flagged."""

    # get list file:
    # get listfile.  Here we will use the global variable pointing to the file
    # so that we also check that this file is correct:
    listfile_path = LISTFILE

    # first read in cube list file, that forms the basis for all the ingestion
    # data.  This file is a list of cubes and all the derived data products are
    # generated based on these.  In most cases there is one of every data product
    # for each cube.
    print(listfile_path)
    hdulist = fits.open(listfile_path)
    table_data = hdulist['DR_CUBES'].data

    # get all relevant columns from the cube_list.fits file:
    cubeid = table_data['CUBEID']
    cubeidpub = table_data['CUBEIDPUB']
    cubename = table_data['CUBENAME']
    catid = table_data['CATID']
    qflag = table_data['RFLAG']
    cubefwhm = table_data['CUBEFWHM']
    cubetexp = table_data['CUBETEXP']
    meantrans = table_data['MEANTRANS']
    isbest = table_data['ISBEST']
    catsource = table_data['CATSOURCE']
    z_spec = table_data['Z_SPEC']

    hdulist.close()

    n_cubes = np.size(cubename)
    print('Number of cubes in list file:',n_cubes)

    # find the number of galaxy cubes, based on catsource:
    n_gal1 = np.count_nonzero(catsource == 1)  # GAMA sample 
    n_gal2 = np.count_nonzero(catsource == 2)  # Cluster sample
    n_gal3 = np.count_nonzero(catsource == 3)  # filler sample

    n_gal = n_gal1+n_gal2+n_gal3

    print('Number of galaxy cubes:',n_gal,n_gal1,n_gal2,n_gal3)

    # get cat files with coordinates:
    # GAMA:
    catfile = CATFILEPATH+'/InputCatGAMADR3.fits'
    hdulist = fits.open(catfile)
    tab_gama = hdulist[1].data
    hdulist.close()
    ngama = np.size(tab_gama['CATID'])
    # get cat files with coordinates:
    # Clusters:
    catfile = CATFILEPATH+'/InputCatClustersDR3.fits'
    hdulist = fits.open(catfile)
    tab_clus = hdulist[1].data
    nclus = np.size(tab_clus['CATID'])
    hdulist.close()
    # get cat files with coordinates:
    # filler:
    catfile = CATFILEPATH+'/InputCatFiller.fits'
    hdulist = fits.open(catfile)
    tab_fill = hdulist[1].data
    nfill = np.size(tab_fill['CATID'])
    hdulist.close()

    # loop through catalogues and get RA/DEC:
    ra_cat = np.zeros(n_cubes)
    dec_cat = np.zeros(n_cubes)
    crval1 = np.zeros(n_cubes)
    crval2 = np.zeros(n_cubes)

    for i in range(n_cubes):
        if (catsource[i] == 1):
            for j in range(ngama):
                if (tab_gama['CATID'][j] == table_data['CATID'][i]):
                    print(i,table_data['CATID'][i],tab_gama['RA_OBJ'][j])
                    ra_cat[i] = tab_gama['RA_OBJ'][j]
                    dec_cat[i] = tab_gama['DEC_OBJ'][j]
                    break
        if (catsource[i] == 2):
            for j in range(nclus):
                if (tab_clus['CATID'][j] == table_data['CATID'][i]):
                    print(i,table_data['CATID'][i],tab_clus['RA_OBJ'][j])
                    ra_cat[i] = tab_clus['RA_OBJ'][j]
                    dec_cat[i] = tab_clus['DEC_OBJ'][j]
                    break
        if (catsource[i] == 3):
            for j in range(nfill):
                if (tab_fill['CATID'][j] == table_data['CATID'][i]):
                    print(i,table_data['CATID'][i],tab_fill['RA_OBJ'][j])
                    ra_cat[i] = tab_fill['RA_OBJ'][j]
                    dec_cat[i] = tab_fill['DEC_OBJ'][j]
                    break

        # go to cube and pull out the WCS info:        
        cubefile = os.path.join(CUBEPATH,table_data['CUBENAME'][i])
        hdr = fits.getheader(cubefile)
        crval1[i] = hdr['CRVAL1']
        crval2[i] = hdr['CRVAL2']
        print(cubefile,crval1[i],crval2[i])

    # define output table:
    col1 = fits.Column(name='CUBEID',format='80A',array=cubeid)
    col2 = fits.Column(name='CUBENAME',format='80A',array=cubename)
    col3 = fits.Column(name='CUBEIDPUB',format='16A',array=cubeidpub)
    col4 = fits.Column(name='CATID',format='K',array=catid)
    col5 = fits.Column(name='CATSOURCE',format='I',array=catsource)
    col6 = fits.Column(name='RA_CAT',format='D',array=ra_cat)
    col7 = fits.Column(name='DEC_CAT',format='D',array=dec_cat)
    col8 = fits.Column(name='CRVAL1',format='D',array=crval1)
    col9 = fits.Column(name='CRVAL2',format='D',array=crval2)
    cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9])
    hdutab = fits.BinTableHDU.from_columns(cols,name='CUBEOBS')

    outfits = 'wcs_comp.fits'
    hdutab.writeto(outfits,overwrite=True)
    return

def compare_cube_wcs_plot(infile,lim=1.0):
    """Plot output from compare_cube_wcs()"""

    #read in data: 
    hdulist = fits.open(infile)
    tab = hdulist[1].data

    ra_diff = (tab['RA_CAT'] - tab['CRVAL1'])*np.cos(np.radians(tab['DEC_CAT']))*3600.0
    dec_diff = (tab['DEC_CAT'] - tab['CRVAL2'])*3600.0

    n = np.size(tab['CATID'])
    nd = 0
    for i in range(n):
        diff = np.sqrt(ra_diff[i]**2 + dec_diff[i]**2)
        if ((diff > lim) and (diff < 30.0)):
            print(tab['CUBEIDPUB'][i],ra_diff[i],dec_diff[i],diff)
            nd = nd + 1

    print(nd)
    # plot:
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(ra_diff,dec_diff,'.')
    ax1.set(xlim=[-2.0,2.0],ylim=[-2.0,2.0])
    
    
    return

def verify_ingestion_data(root_directory,dataflag=0,verbose=True):
    """Verify the existence of all data files, and flag any missing files.  The
    flags are written to the main cube quality fits binary table file - CubeObs.fits.  
    This file will contain all the flags for cubes and related data and is newly 
    generated using this script, based on previous tables.
    Useage:
    > sami_dr3_ingestion.verify_ingestion_data('dr3_ingestion/') 


    """

    # get listfile.  Here we will use the global variable pointing to the file
    # so that we also check that this file is correct:
    listfile_path = LISTFILE

    # first read in cube list file, that forms the basis for all the ingestion
    # data.  This file is a list of cubes and all the derived data products are
    # generated based on these.  In most cases there is one of every data product
    # for each cube.
    print(listfile_path)
    hdulist = fits.open(listfile_path)
    table_data = hdulist['DR_CUBES'].data

    # get all relevant columns from the cube_list.fits file:
    cubeid = table_data['CUBEID']
    cubeidpub = table_data['CUBEIDPUB']
    cubename = table_data['CUBENAME']
    catid = table_data['CATID']
    qflag = table_data['RFLAG']
    cubefwhm = table_data['CUBEFWHM']
    cubetexp = table_data['CUBETEXP']
    meantrans = table_data['MEANTRANS']
    isbest = table_data['ISBEST']
    catsource = table_data['CATSOURCE']
    z_spec = table_data['Z_SPEC']

    hdulist.close()

    n_cubes = np.size(cubename)
    print('Number of cubes in list file:',n_cubes)

    # find the number of galaxy cubes, based on catsource:
    n_gal1 = np.count_nonzero(catsource == 1)  # GAMA sample 
    n_gal2 = np.count_nonzero(catsource == 2)  # Cluster sample
    n_gal3 = np.count_nonzero(catsource == 3)  # filler sample

    n_gal = n_gal1+n_gal2+n_gal3

    print('Number of galaxy cubes:',n_gal,n_gal1,n_gal2,n_gal3)

    # get the number of unique galaxies with cubes:
    idx = np.where((catsource == 1) | (catsource == 2) | (catsource == 3))
    n_unique_gal = np.size(np.unique(catid[idx]))
    print('Number of unique galaxies: ',n_unique_gal)

    # find the number of star cubes based on catsouce:
    n_star1 = np.count_nonzero(catsource == 4)  # GAMA stars
    n_star2 = np.count_nonzero(catsource == 5)  # Cluster stars

    n_star = n_star1 + n_star2

    print('Number of star cubes:',n_star,n_star1,n_star2)

    # get the number of unique galaxies with cubes:
    idx = np.where((catsource == 4) | (catsource == 5))
    n_unique_star = np.size(np.unique(catid[idx]))
    print('Number of unique stars: ',n_unique_star)

    # next read the cube QC file:
    qcfile_path = QCFILE
    hdulist = fits.open(qcfile_path)
    table_data = hdulist['QC_DATA'].data
    qc_catid = table_data['CATID']
    qc_nframes = table_data['NFRAMES']
    qc_cubename = table_data['CUBENAME']
    qc_cubefwhm = table_data['CUBEFWHM']
    qc_cubetexp = table_data['CUBETEXP']
    qc_framefwhm = table_data['FRAMEFWHM']
    qc_frametrans = table_data['FRAMETRANS']
    qc_framertrans = table_data['FRAMERTRANS']
    hdulist.close()

    n_qc = np.size(qc_cubename)
    print('Number of cubes in QC file:',n_qc)

    # check that the files are in the same order in the QC and cubelist files:
    #for i in range(n_qc):
    #    if (qc_cubename[i] == cubename[i]):
    #        pass
    #    else:
    #        print('ERROR: Cube names do not match:',i,qc_cubename[i],cubename[i])

    # set up arrays to hold data flags.  These are the cols that are not already 
    # present in the cube QC data above (which is mostly FWHM and transmission).
    # main point here is to flag problems and NOT reproduce values that are in 
    # the main input catalogues.
    # Default should be that 0 is a good flag, and 1 is bad, so we will try to keep
    # this convension.
    #
    # flags for object types (not primary, secondary etc as these are in the 
    # main cat:
    #
    # is this a cube of a secondary std:
    warn_star = np.zeros(n_cubes)
    # is this a cube of a filler target (i.e. not from main cats).
    warn_fill = np.zeros(n_cubes)

    # flags for aperture spectra:
    # 3-kpc aperture missing:
    warn_akpc = np.zeros(n_cubes)
    # sersic re missing:
    warn_are = np.zeros(n_cubes)
    # MGE re missing:
    warn_amge = np.zeros(n_cubes)

    # flags for stellar kinematics not present, 2-moment and 4-moment:
    warn_sk2m = np.zeros(n_cubes)
    warn_sk4m = np.zeros(n_cubes)
    warn_sk4mhsn = np.zeros(n_cubes)

    #
    # various QC related flags that are identified in QC:
    # bad fluxcal (e.g. shape of spectrum wrong)
    warn_fcal = np.zeros(n_cubes)
    # bad fluxcal with aparent step between red and blue:
    warn_fcrb = np.zeros(n_cubes)
    # bad WCS centre, galaxy clearly not centred at centre of cube
    warn_wcs = np.zeros(n_cubes)
    # problem due to obvious multiple sources in the field
    warn_mult = np.zeros(n_cubes)
    # problem due to redshift different from catalogue redshift
    warn_z = np.zeros(n_cubes)
    # sky subtraction problems:
    warn_skyb = np.zeros(n_cubes)
    warn_skyr = np.zeros(n_cubes)
    # stellar kin warning flag (i.e. some problem with kinematics):
    warn_sker = np.zeros(n_cubes)
    # lambda_r warning when discrepant between sersic and MGE
    warn_lamr = np.zeros(n_cubes)
    # warning for possible sky contamination of emline fits:
    warn_skem = np.zeros(n_cubes)
    # warning of missing fits/files for emline fits:
    warn_emft = np.zeros(n_cubes)

    # set warning flags based on catalogue properties:
    for i in range(n_cubes):
        if (catsource[i] == 3):
            warn_fill[i] = 1
        if ((catsource[i] == 4) or (catsource[i] ==5)):
            warn_star[i] = 1


    # types of cube, apertures, etc:
    colours = ['blue','red']
    cube_types = ['cube','adaptive','annular','sectors']
    cube_types_n = np.zeros((4,2),dtype='i4')
    aper_types = ['3-kpc','re','remge','1-4-arcsec','2-arcsec','3-arcsec','4-arcsec']
    aper_types_n = np.zeros((7,2),dtype='i4')
    # stellar kin types (2 moment):
    stelkin_types = ['default','adaptive','annular','sectors']
    stelkinv_types = ['stellar-velocity','stellar-velocity-dispersion']
    stelkin_types_n = np.zeros((4,2),dtype='i4')
    # stellar kin types (4 moment):
    stelkin4_types = ['default','adaptive','annular','sectors','adaptive-hisn']
    stelkin4v_types = ['stellar-velocity','stellar-velocity-dispersion','stellar-velocity-h3','stellar-velocity-h4']
    stelkin4_types_n = np.zeros((5,4),dtype='i4')
    # gas types:
    lzifu_types = ['extinct-corr','gas-vdisp','gas-velocity','Halpha','Hbeta','NII6583','OI6300','OII3728','OIII5007','sfr','sfr-dens','sfr-mask','SII6716','SII6731']
    lzifubin_types = ['adaptive_1-comp','adaptive_recom-comp','annular_2-comp','default_1-comp','default_recom-comp','sectors_1-comp','sectors_recom-comp']
    lzifu_types_n = np.zeros((14,7),dtype='i4')

    

    # we want to be able to list the files that are missing for different aperture types
    # and different catsources (as we expect differences to vary).
    nap = len(aper_types)

    aper_missing_gal = np.empty((5,nap,n_cubes),dtype='U100')


    # for each cube (only the blue cubes names are in the list) generate all the 
    # expected file names:
    ncheck = 0
    nmissing = 0
    for i in range(n_cubes):
        
        # generate the path for this catid.  Updated for new
        # ADC file structure:
        path = os.path.join(root_directory,'data/sami/dr3/ifs',str(catid[i]))

        if ((dataflag == 0) or (dataflag == 1)): 
            # generate all the cube file names
            it = 0
            for ctype in cube_types:
                ic = 0
                for col in colours:
                    cubefile = gen_cube_filename(catid[i],qflag[i],ctype,col,gzip=True)

                    # check if the files exist:
                    fname = os.path.join(path,cubefile)
                    ncheck = ncheck + 1
                
                    if not os.path.exists(fname):
                        nmissing = nmissing + 1
                        if (verbose):
                            print('File not found:',fname)
                    else:
                        cube_types_n[it,ic] =  cube_types_n[it,ic] + 1
                    
                    ic = ic + 1
                it = it + 1

        if ((dataflag == 0) or (dataflag == 2)): 
            # generate all aperture spectrum file names and test for their presence
            # First we calculate how many cubes might have aperture spectra.  This is
            # only the galaxies, not calibration stars.  Therefore use the catsource
            # flag to get the expected ones.
            it = 0
            # only test on galaxies.  If a star, skip, but set the 
            # aperture flags to warn to be consistent:
            if ((catsource[i] < 1) or (catsource[i] > 3)):
                warn_akpc[i] = 1
                warn_are[i] = 1
                warn_amge[i] = 1
                continue

            iap = 0
            for atype in aper_types:
                ic = 0
                for col in colours:
                    aperfile = gen_aper_filename(catid[i],qflag[i],atype,col,gzip=False)

                    # check if the files exist:
                    fname = os.path.join(path,aperfile)
                    ncheck = ncheck + 1
                
                    if not os.path.exists(fname):
                        nmissing = nmissing + 1
                        aper_missing_gal[catsource[i]-1,iap,i] = fname
                        if (verbose):
                            print('File not found:',fname,catsource[i])

                        # write flags based on missing data:
                        if (atype == '3-kpc'):
                            warn_akpc[i] = 1
                        if (atype == 're'):
                            warn_are[i] = 1
                        if (atype == 'remge'):
                            warn_amge[i] = 1
  
                    else:
                        aper_types_n[it,ic] =  aper_types_n[it,ic] + 1
                    
                    ic = ic + 1
                it = it + 1
                iap = iap + 1

        # now check all the stellar kinematic files:
        if ((dataflag == 0) or (dataflag == 3)): 

            # only do the checks for galaxies, not stars, but
            # set the warning flags for stars:
            if ((catsource[i] < 1) or (catsource[i] > 3)):
                warn_sk2m[i] = 1
                warn_sk2m[i] = 1
                continue

            # first do this for the 2 moment
            it = 0
            for stype in stelkin_types:
                ic = 0
                for svtype in stelkinv_types:
                    stelkinfile = gen_stelkin_filename(catid[i],qflag[i],stype,svtype,mom=2,gzip=False)

                    # check if the files exist:
                    fname = os.path.join(path,stelkinfile)
                    ncheck = ncheck + 1
                
                    if not os.path.exists(fname):
                        nmissing = nmissing + 1
                        # set the flag:
                        warn_sk2m[i] = 1
                        if (verbose):
                            print('File not found:',fname)
                    else:
                        stelkin_types_n[it,ic] =  stelkin_types_n[it,ic] + 1
                    
                    ic = ic + 1
                it = it + 1

            # now do it for the 4 moment files:
            it = 0
            for stype in stelkin4_types:
                ic = 0
                for svtype in stelkin4v_types:
                    stelkinfile = gen_stelkin_filename(catid[i],qflag[i],stype,svtype,mom=4,gzip=False)

                    # check if the files exist:
                    fname = os.path.join(path,stelkinfile)
                    ncheck = ncheck + 1
                
                    if not os.path.exists(fname):
                        nmissing = nmissing + 1
                        # set the flag.  One for regular data and one for 
                        # highsn adaptive binning:
                        if (stype == 'adaptive-hisn'):
                            warn_sk4mhsn[i] = 1
                        else:
                            warn_sk4m[i] = 1

                        if (verbose):
                            print('File not found:',fname)
                    else:
                        stelkin4_types_n[it,ic] =  stelkin4_types_n[it,ic] + 1
                    
                    ic = ic + 1
                it = it + 1


        # now check all the LZIFU files:
        if ((dataflag == 0) or (dataflag == 4)): 

            # only do the checks for galaxies, not stars, but
            # set the warning flags for stars:
            if ((catsource[i] < 1) or (catsource[i] > 3)):
                warn_emft[i] = 1
                warn_emft[i] = 1
                continue

            # first do this for the 2 moment
            it = 0
            for ltype in lzifu_types:
                ic = 0
                for lbtype in lzifubin_types:
                    lzifufile = gen_lzifu_filename(catid[i],qflag[i],ltype,lbtype,gzip=False)

                    # check if the files exist:
                    fname = os.path.join(path,lzifufile)
                    ncheck = ncheck + 1
                
                    if not os.path.exists(fname):
                        nmissing = nmissing + 1
                        # set the flag:
                        warn_emft[i] = 1
                        if (verbose):
                            print('File not found:',fname)
                    else:
                        lzifu_types_n[it,ic] =  lzifu_types_n[it,ic] + 1
                    
                    ic = ic + 1
                it = it + 1


    # list the files that are missing for difference catsources:

    for j in range(3):
        print('Check of missing apertures for cat source:',j+1)

        for i in range(nap):
            nmissing = np.count_nonzero(aper_missing_gal[j,i,:])
            print(aper_types[i],'Number missing: ',nmissing)

            # get index for missing ones to get file names:
            for k in range(n_cubes):
                if (aper_missing_gal[j,i,k]):
                    if (verbose):
                        print(aper_missing_gal[j,i,k])


    print('Number of original cube files:',n_cubes)
    print('Number of original galaxy cube files:',n_gal)
    print('Number of original galaxy cube files (GAMA):',n_gal1)
    print('Number of original galaxy cube files (cluster):',n_gal2)
    print('Number of original galaxy cube files (filler):',n_gal3)
    print('Number of files checked:',ncheck)
    print('Number of files missing:',nmissing)

    # check cube numbers per cube type:
    ntype = len(cube_types)
    print('\n Cube type    N_blue N_red')
    for i in range(ntype):
        print('{0:15s} {1:4d} {2:4d}'.format(cube_types[i],cube_types_n[i,0],cube_types_n[i,1]))

    # check aperture spectra numbers per cube type:
    ntype = len(aper_types)
    print('\n Aper type    N_blue N_red')
    for i in range(ntype):
        print('{0:15s} {1:4d} {2:4d}'.format(aper_types[i],aper_types_n[i,0],aper_types_n[i,1]))

    # check stel kin file numbers per cube type (2 moment):
    ntype = len(stelkin_types)
    print('\n 2-moment stellar kin')
    print('stelkin type   N_v N_sig')
    for i in range(ntype):
        print('{0:15s} {1:4d} {2:4d}'.format(stelkin_types[i],stelkin_types_n[i,0],stelkin_types_n[i,1]))

    # check stel kin file numbers per cube type (4 moment):
    ntype = len(stelkin4_types)
    print('\n 4-moment stellar kin')
    print('stelkin type   N_v N_sig N_h3 N_h4')
    for i in range(ntype):
        print('{0:15s} {1:4d} {2:4d} {3:4d} {4:4d}'.format(stelkin4_types[i],stelkin4_types_n[i,0],stelkin4_types_n[i,1],stelkin4_types_n[i,2],stelkin4_types_n[i,3]))

    # check LZIFU numbers per cube/comp type:
    ntype = len(lzifu_types)
    print('\n')    
    print('\n LZIFU')
    print('LZIFU type  N_ad1 N_adr N_an2 N_de1 N_der N_se1 N_ser')
    for i in range(ntype):
        print('{0:15s} {1:4d} {2:4d} {3:4d} {4:4d}  {5:4d} {6:4d} {7:4d}'.format(lzifu_types[i],lzifu_types_n[i,0],lzifu_types_n[i,1],lzifu_types_n[i,2],lzifu_types_n[i,3],lzifu_types_n[i,4],lzifu_types_n[i,5],lzifu_types_n[i,6]))



    # read in the list of objects flagged in QC.  This is a text file with
    # cubename (public and private) and catid and an integer stating what the problem is.
    # these values are:
    # 1 - general fluxcal problem (e.g. shape)
    # 2 - red-blue offset fluxcal problem.
    # 3 - wcs problem, e.g. bad centering, 31 or 32
    # 4 - multiple sources in cube
    # 5 - catastrophic redshift errors
    # 6 - blue sky sub
    # 7 - red sky sub
    # 8 - lambda_r warning - sersic/MGE differences
    # 9 - possible skyline in em fitting window
    qc_dtype = np.dtype([('QC_CATID','i8'),('QC_REP','U1'),('QC_FLAG','i8')])
    qc_data = np.loadtxt(CATFILEPATH+'/qc_check_flags.txt',dtype=qc_dtype,usecols=[0,1,2],unpack=True)
    ncol,nqc = np.shape(qc_data)
    
    # loop through all cubes and adjust flags for those with QC issues:
    nn_qc = 0
    for i in range(n_cubes):
        
        for j in range(nqc):
            
            qc_cubeidpub = get_cubeid_pub(qc_data[0][j],qc_data[1][j])
            if (qc_cubeidpub == cubeidpub[i]):
                nn_qc = nn_qc + 1
                qc_flag = qc_data[0][j]
                if (qc_flag == 1):
                    warn_fcal[i] = 1
                if (qc_flag == 2):
                    warn_fcrb[i] = 1
                if (qc_flag == 31):
                    warn_wcs[i] = 1
                if (qc_flag == 32):
                    warn_wcs[i] = 2
                if (qc_flag == 4):
                    warn_mult[i] = 1
                if (qc_flag == 5):
                    warn_z[i] = 1
                if (qc_flag == 6):
                    warn_skyb[i] = 1
                if (qc_flag == 7):
                    warn_skyr[i] = 1
                if (qc_flag == 8):
                    warn_lamr[i] = 1
                if (qc_flag == 9):
                    warn_skem[i] = 1

    print('Number of cubes flagged from QC:',nn_qc)

    # next read in Jesse's stellar kinematics list:
    jvds_dtype = np.dtype([('JVDS_CATID','i8'),('JVDS_FLAG','i8'),('JVDS_CUBE','U128')])
    jvds_data = np.loadtxt(CATFILEPATH+'/jvds_flag_list_smc_v070820.cat',dtype=jvds_dtype,usecols=[0,1,2],unpack=True)
    ncol,njvds = np.shape(jvds_data)

    print(njvds)


    # loop through all cubes and adjust flags for those with stellar kinematics issues:
    nn_jvds = 0
    for i in range(n_cubes):
        found = False
        isbest_tmp = isbest[i]
        for j in range(njvds):
            
            jvds_cubename = jvds_data[2][j].replace('.gz','')
            jvds_catid = jvds_data[0][j]

            if (jvds_cubename == cubename[i]):
                nn_jvds = nn_jvds + 1
                qc_flag = jvds_data[1][j]
                found = True
                # Need to actually set flags for JVDS file...
                if ((qc_flag == 1) or (qc_flag == 2)):
                    warn_sker[i] = 1
                if (qc_flag == 3):
                    warn_mult[i] = 2
                if (qc_flag == 4):
                    warn_mult[i] = 1
                if ((qc_flag == 7) or (qc_flag == 8)):
                    warn_z[i] = 1
                # Need to reset isbest flag, based on Jesse's list.
                # do this here.
                if (jvds_catid < 1000000000000):
                    isbest[i] = True
                else:
                    isbest[i] = False

        if (not found):
            # only print warning for actual galaxies, not stars:
            if (warn_star[i] == 0):
                print('No match found for galaxy in JVDS list:',cubename[i],i)
            # if it is not found, then we use the isbest flag
            # that was previously set:
            isbest[i] = isbest_tmp
            

    print('Number of cubes flagged from JVDS QC:',nn_jvds)


    # finally, define columns for the new cube_obs table:
    col0 = fits.Column(name='CUBEIDPUB',format='16A',array=cubeidpub)
    col1 = fits.Column(name='CUBEID',format='80A',array=cubeid)
    col2 = fits.Column(name='CUBENAME',format='80A',array=cubename)
    col3 = fits.Column(name='CATID',format='K',array=catid)
    col4 = fits.Column(name='CUBEFWHM',format='E',array=cubefwhm,unit='arcsec')
    col5 = fits.Column(name='CUBETEXP',format='E',array=cubetexp,unit='s')
    col6 = fits.Column(name='MEANTRANS',format='E',array=meantrans)
    col7 = fits.Column(name='ISBEST',format='L',array=isbest)
    col8 = fits.Column(name='CATSOURCE',format='I',array=catsource)
    col9 = fits.Column(name='WARNSTAR',format='I',array=warn_star)
    col10 = fits.Column(name='WARNFILL',format='I',array=warn_fill)
    col11 = fits.Column(name='WARNZ',format='I',array=warn_z)
    col12 = fits.Column(name='WARNMULT',format='I',array=warn_mult)
    col13 = fits.Column(name='WARNAKPC',format='I',array=warn_akpc)
    col14 = fits.Column(name='WARNARE',format='I',array=warn_are)
    col15 = fits.Column(name='WARNAMGE',format='I',array=warn_amge)
    col16 = fits.Column(name='WARNSK2M',format='I',array=warn_sk2m)
    col17 = fits.Column(name='WARNSK4M',format='I',array=warn_sk4m)
    col18 = fits.Column(name='WARNSK4MHSN',format='I',array=warn_sk4m)
    col19 = fits.Column(name='WARNFCAL',format='I',array=warn_fcal)
    col20 = fits.Column(name='WARNFCBR',format='I',array=warn_fcrb)
    col21 = fits.Column(name='WARNSKYB',format='I',array=warn_skyb)
    col22 = fits.Column(name='WARNSKYR',format='I',array=warn_skyr)
    col23 = fits.Column(name='WARNSKER',format='I',array=warn_sker)
    col24 = fits.Column(name='WARNWCS',format='I',array=warn_wcs)
    col25 = fits.Column(name='WARNRE',format='I',array=warn_lamr)
    col26 = fits.Column(name='WARNSKEM',format='I',array=warn_skem)
    col27 = fits.Column(name='WARNEMFT',format='I',array=warn_emft)

    cols = fits.ColDefs([col0,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27])

    hdutab = fits.BinTableHDU.from_columns(cols,name='CUBEOBS')

    # add header keywords:
    hdr = hdutab.header

    # get the time and date now for the header;
    t = datetime.now()
    
    hdr['DATE'] = (t.strftime('%d-%m-%Y %H:%M:%S'),'Date and time DMU was generated')
    hdr['AUTHOR']  = ('Scott Croom','Catalogue author')
    hdr['CONTACT'] = ('scott.croom@sydney.edu.au','Contact for author')
    hdr['DESCRIPT']= ('SAMI observed cubes and quality flags','Short description')
    hdr['DMU_NAME']= ('Cube observations','DMU Name')
      
    outfits = os.path.join(CATFILEPATH,'CubeObs.fits')
    hdutab.writeto(outfits,overwrite=True)

    return


def gen_cubeid(cubename):
    """Generate the individual cube ID for this cube, based on
    a file name being given (typically the file name of the 
    blue cube"""

    cubeid = cubename.replace('.gz','')
    cubeid = cubeid.replace('.fits','')
    cubeid = cubeid.replace('_blue','')
    cubeid = cubeid.replace('_red','')
    
    return cubeid


def gen_cube_filename(catid,qflag,cubetype,colour,gzip=False):
    """function to generate the correct ingestion DR3 cube name, based on the standard
    internal cubenames and various other flags, such as quality and colour.  If gzip
    is set to true, assume that the file name has .gz as the end of it."""

    filename = str(catid)+'_'+qflag+'_'+cubetype+'_'+colour+'.fits'
    if (gzip):
        filename = filename+'.gz'

    return filename

def gen_aper_filename(catid,qflag,apertype,colour,gzip=False):
    """function to generate the correct ingestion DR3 cube name, based on the standard
    internal cubenames and various other flags, such as quality and colour.  If gzip
    is set to true, assume that the file name has .gz as the end of it."""

    filename = str(catid)+'_'+qflag+'_spectrum_'+apertype+'_'+colour+'.fits'
    if (gzip):
        filename = filename+'.gz'

    return filename

def gen_stelkin_filename(catid,qflag,sktype,skvtype,mom=2,gzip=False):
    """function to generate the correct ingestion DR3 cube name, based on the standard
    internal cubenames and various other flags, such as quality and colour.  If gzip
    is set to true, assume that the file name has .gz as the end of it."""

    if (mom == 2):
        filename = str(catid)+'_'+qflag+'_'+skvtype+'_'+sktype+'_two-moment.fits'
    elif (mom == 4):
        filename = str(catid)+'_'+qflag+'_'+skvtype+'_'+sktype+'_four-moment.fits'
    else:
        print('ERROR: mom value incorrect')

    if (gzip):
        filename = filename+'.gz'

    return filename

def gen_lzifu_filename(catid,qflag,ltype,lbtype,gzip=False):
    """function to generate the correct ingestion DR3 LZIFU name, based on the standard
    internal cubenames and various other flags.  If gzip
    is set to true, assume that the file name has .gz as the end of it."""

    #filename = str(catid)+'_'+ltype+'_'+lbtype+'_'+qflag+'.fits'
    filename = str(catid)+'_'+qflag+'_'+ltype+'_'+lbtype+'.fits'

    if (gzip):
        filename = filename+'.gz'

    return filename


def ingest_all_data(root_directory,split=[0.0,1.0],dataflag=0,testrun=False,check_only=False):

    # NB These hard-coded file paths assume that this code is being run under the 
    # SAMI account on bill/milo/opus

    listfile_path = LISTFILE
    catafile_path = CATFILE
    cube_directory = CUBEPATH
    aperture_directory = APERPATH
    stelkin_directory = STELKINPATH
    stelkin_directory4 = STELKINPATH4
    lzifu_1comp_directory = LZIFU1COMPPATH
    lzifu_recom_directory = LZIFURECOMPATH
    lzifu_binned_directory = LZIFUBINNEDPATH

    reformat_many(root_directory,listfile_path,cube_directory,aperture_directory,
                  stelkin_directory,stelkin_directory4,lzifu_1comp_directory,lzifu_recom_directory,
                  lzifu_binned_directory,split=split,testrun=testrun,
                  dataflag=dataflag,check_only=check_only)


def create_ingestion_ready_dr2_from_scratch(root_directory,split=[0.0,1.0],dataflag=0,testrun=False):

    if not os.path.isdir(root_directory):
        os.mkdir(root_directory)

    # NB These hard-coded file paths assume that this code is being run under the 
    # SAMI account on bill/milo/opus

    listfile_path = LISTFILE
    catafile_path = CATFILE
    cube_directory = CUBEPATH
    aperture_directory = APERPATH
    stelkin_directory = STELKINPATH
    lzifu_1comp_directory = LZIFU1COMPPATH
    lzifu_recom_directory = LZIFURECOMPATH
    lzifu_binned_directory = LZIFUBINNEDPATH
    

    create_directory_structure(root_directory)
    create_basic_metadata_files(root_directory)
    create_coordinate_meta_files(root_directory,catafile_path,listfile_path)

    reformat_many(root_directory,listfile_path,cube_directory,aperture_directory,
                  stelkin_directory,stelkin_directory4,lzifu_1comp_directory,lzifu_recom_directory,
                  lzifu_binned_directory,split=split,testrun=testrun,dataflag=dataflag)

    add_all_tables(root_directory)

    ifs_meta_files = glob.glob('/import/bill1/sami/dr3_storage/docs/ifs/*.html')
    for f in ifs_meta_files:
        fname = f.split('/')[-1]
        shutil.copy(f,root_directory+'metadata/sami/dr3/ifs/docs/'+fname)

    shutil.copy('/import/bill1/sami/dr3_storage/docs/ifs/sami_dr3_dp_meta.txt',root_directory+'metadataxs/ifs/sami_dr3_dp_meta.txt')

#####################################

def copy_kwd(kwds,hdu1,hdu2):

    """Copy keywords in list from HDU1 to HDU2"""

    hdr1 = hdu1.header
    hdr2 = hdu2.header

    for kwd in kwds:
        if (kwd in hdr1):
            hdr2.append((kwd,hdr1[kwd],hdr1.comments[kwd]))
        else:
            print('WARNING:',kwd,' not found in header')

    return hdr2


#####################################

def add_keyword_comments_lzifu(hdr):

    """Function to add comments to LZIFU public format data
    as some of the comments are missing in the default versions."""

    hdr.comments['CATID'] = 'Object ID'
    hdr.comments['PRODUCT'] = 'Data product'
    hdr.comments['VERSION'] = 'LZIFU run version'
    hdr.comments['SAMI_VER'] = 'SAMI pipeline reduction version'
    hdr.comments['DATE'] = 'Date generated'
    hdr.comments['AUTHOR'] = 'Author of data products'
    hdr.comments['DESCRIPT'] = 'Short description'

    return

#######################################

def add_keyword_comments_wcs(hdr):

    """Add missing comments to WCS FITS keywords."""

    hdr.comments['CRPIX1'] = 'Pixel coordinate of reference point'
    hdr.comments['CRPIX2'] = 'Pixel coordinate of reference point'
    hdr.comments['CDELT1'] = '[deg] Coordinate increment at reference point'
    hdr.comments['CDELT2'] = '[deg] Coordinate increment at reference point'
    hdr.comments['CRVAL1'] = '[deg] Coordinate value at reference point'
    hdr.comments['CRVAL2'] = '[deg] Coordinate value at reference point'
    hdr.comments['CTYPE1'] = 'Right ascension, gnomonic projection'
    hdr.comments['CTYPE2'] = 'Declination, gnomonic projection'
    hdr.comments['CUNIT1'] = 'Units of coordinate increment and value'
    hdr.comments['CUNIT2'] = 'Units of coordinate increment and value'

    return

#####################################

def add_keyword_comments(infiles):
    """Function to add missing FITS keyword comments to SAMI data.  This
    is most easily run on the original cubes, rather on DR3 products, so we
    do not have to repet the process."""

    # glob the list.  ALso sort, so they are in order (not that this really
    # matters for this routine):
    inlist = sorted(glob.glob(infiles))

    num = np.size(inlist)

    n=0
    for infile in inlist:

        # open the file:
        hdrlist = fits.open(infile,'update')

        # get the primary header:
        hdr = hdrlist[0].header

        # now add a comment for the following keywords:
        # PSF related:
        hdr.comments['PSFFWHM'] = 'FWHM of Moffat fit to PSF (arcsec)'
        hdr.comments['PSFALPHA'] = 'Alpha parameter of Moffat fit to PSF'
        hdr.comments['PSFBETA'] = 'Beta parameter of Moffat fit to PSF'

        # coordinate related:
        hdr.comments['CATARA'] = 'Catalogue nominal RA_IFU coordinate (deg)'
        hdr.comments['CATADEC'] = 'Catalogue nominal DEC_IFU coordinate (deg)'
    
        hdrlist.flush()
        hdrlist.close()

        n = n + 1
        print('Updated header comments for: ',infile,'   ',n,' of ',num)

    
################################################
