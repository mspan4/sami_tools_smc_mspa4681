###############################################################
# various tools to manage and analyse SAMI catalogues

import shutil
import numpy as np
import pylab as py

from datetime import datetime

from astropy.io import ascii,fits
from astropy.table import Column
from astropy import table
from astropy.table import Table

##############################################################
#
def convert_names(infile,cubeobsfile='/Users/scroom/data/sami/dr3/cats/all_cats_dr3/CubeObs.fits'):
    """convert from one set of names to another, using the CubeObs.fits file
    as a reference"""

    # read in the cubeobs file:
    tab_cubeobs = Table.read(cubeobsfile)
    print(tab_cubeobs)
    
    # read in the text file with names:
    name_data =ascii.read(infile)
    print(name_data)

    tab_merge = table.join(name_data,tab_cubeobs, keys='CUBEID',join_type='left')

    print(tab_merge)
    print(tab_merge['CUBEIDPUB'])

    tab_merge.remove_columns(['CUBEID','CUBENAME','CATID','CUBEFWHM','CUBETEXP','MEANTRANS','ISBEST','CATSOURCE','WARNSTAR','WARNFILL','WARNZ','WARNMULT','WARNAKPC','WARNARE','WARNAMGE','WARNSK2M','WARNSK4M','WARNSK4MHSN','WARNFCAL','WARNFCBR','WARNSKYB','WARNSKYR','WARNSKER','WARNWCS','WARNLAMR'])
    tab_merge.write('names.dat', overwrite=True,format='ascii')

    
    
    


##############################################################
def fix_pa_inputgama(infile,apfile='/Users/scroom/data/sami/gama/ApMatchedCatv03.fits'):
    """Check and fix PAs from GAMA input cat, as these come from
    the Sersic photomtry catalogue and are not proper PA on the sky"""

    # read in main cat:
    tab_gama = Table.read(infile)
    #hdulist = fits.open(infile)
    #tab_gama = hdulist[1].data
    #catid_gama = tab_gama['CATID']
    #pa_gama = tab_gama['PA']

    # read in GAMA aperture phot cat:
    tab_ap = Table.read(apfile)
    #hdulist_ap = fits.open(apfile)
    #tab_ap = hdulist_ap[1].data
    #catid_ap = tab_ap['CATAID']
    # rename column for CATAID to CATID...
    tab_ap.rename_column('CATAID', 'CATID')
    
    # do a join:
    tab_new = table.join(tab_gama,tab_ap,keys='CATID',join_type='left')
    tab_new.info()

    # calculate the PA on the sky:
    tab_new['PA_NEW'] = (tab_new['PA'] + (tab_new['THETA_J2000'] - tab_new['THETA_IMAGE'])) % 180
    nnew = np.size(tab_new['PA_NEW'])
    
    # note that the new table made with the join will not have the same row order
    # this is a frustrating feature of the join function.  As a result we will need to
    # rematch these up.  Easiest way is to read in FITS file again and then just go through cols.
    # first copy to a new file, so we are not changing the file in place:
    outfile = 'pa_fix.fits'
    shutil.copyfile(infile,outfile)
    hdulist = fits.open(outfile,mode='update')
    tab_out = hdulist[1].data
    catid_out = tab_out['CATID']
    nout = np.size(catid_out)

    nfix = 0
    for i in range(nout):

        catid_i = catid_out[i]

        for j in range(nnew):

            catid_j = tab_new['CATID'][j]

            if (catid_i == catid_j):
                tab_out['PA'][i] = tab_new['PA_NEW'][j]
                nfix = nfix +1
                break
            
    print(nnew,nout,nfix)
    hdulist.flush()
    hdulist.close()
    
    # compare PAs:
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(2,2,1)
    ax1.plot(tab_new['PA'],tab_new['THETA_IMAGE'],'.')
    
    ax2 = fig1.add_subplot(2,2,2)
    ax2.plot(tab_new['THETA_J2000'],tab_new['THETA_IMAGE'],'.')

    ax3 = fig1.add_subplot(2,2,3)
    ax3.plot(tab_new['THETA_J2000'],tab_new['PA_NEW'],'.')

    ax4 = fig1.add_subplot(2,2,4)
    ax4.hist(tab_new['PA_NEW'],36)


###############################################################

def calc_cat_comp(infile):
    """For a given input catalogue file, calculate the completeness
    of the sample in terms of what fraction of objects where observed."""

    hdulist = fits.open(infile)
    tab_gama = hdulist[1].data
    catid_gama = tab_gama['CATID']
    pri_gama = tab_gama['SURV_SAMI']
    badflag_gama = tab_gama['BAD_CLASS']
    ra_gama = tab_gama['RA_OBJ']        
    ngama = np.size(catid_gama)
    obsflag_gama = np.zeros(ngama)
    repflag_gama = np.zeros(ngama)
    
    # read in CubeObs file:
    hdulist = fits.open('/Users/scroom/data/sami/dr3/cats/CubeObs/CubeObs.fits')
    tab_obs = hdulist[1].data
    catid_obs = tab_obs['CATID']
    cubeidpub = tab_obs['CUBEIDPUB']
    cubeid = tab_obs['CUBEID']
    ncube = np.size(catid_obs)
    
    nfound = 0
    nfound_unique = 0
    n_a = 0
    n_b = 0
    n_c = 0
    n_d = 0
    n_e = 0
    n_f = 0
    n_diff_plate = 0
    n_same_plate = 0
    for i in range(ngama):
        cubeid_list=[]
        for j in range(ncube):
            catid = catid_obs[j]
            if (catid == catid_gama[i]):
                cubeid_list.append(cubeid[j])
                nfound = nfound + 1
                obsflag_gama[i] = 1
                repflag_gama[i] = repflag_gama[i] + 1
                # count repeats based on CUBEIDPUB:
                if ('A' in cubeidpub[j]):
                    n_a = n_a + 1
                if ('B' in cubeidpub[j]):
                    n_b = n_b + 1
                if ('C' in cubeidpub[j]):
                    n_c = n_c + 1
                if ('D' in cubeidpub[j]):
                    n_d = n_d + 1
                if ('E' in cubeidpub[j]):
                    n_e = n_e + 1
                if ('F' in cubeidpub[j]):
                    n_f = n_f + 1
        # ouput list of repeats:
        if (repflag_gama[i] > 1):
            # check if the repeats are from the same plate:
            plate_list = []
            print(cubeid_list)
            for cid in cubeid_list:
                plate=cid.split('_')[4]
                plate_list.append(plate)
                print(plate)
            nplate_u = np.size(np.unique(plate_list))
            nplate = np.size(plate_list)
            print(nplate,nplate_u)
            if (nplate_u > 1):
                n_diff_plate = n_diff_plate +1
            if (nplate > nplate_u):
                n_same_plate = n_same_plate +1
                

        if (obsflag_gama[i] == 1):
            nfound_unique = nfound_unique + 1

    print('Number of GAMA targets:',ngama)
    print('Number of GAMA input gals found in obs cat:',nfound)
    print('Number of unique GAMA input gals found in obs cat:',nfound_unique)

    print('Number of galaxies with 2 or more cubes made from diff plates:',n_diff_plate)
    print('Number of galaxies with 2 or more cubes made from same plates:',n_same_plate)
    
    print('Number of A obs in GAMA:',n_a)
    print('Number of B obs in GAMA:',n_b)
    print('Number of C obs in GAMA:',n_c)
    print('Number of D obs in GAMA:',n_d)
    print('Number of E obs in GAMA:',n_e)
    print('Number of F obs in GAMA:',n_f)

    # use repflag to count thr number of repeats:
    for i in range(10):
        print('Number of galaxies with ',i,' cubes:',np.size(np.where(repflag_gama == i)))
    
    # calculate numbers for different selections:
    ra1 = [0.0,129.0,174.0,211.5]
    ra2 = [360.0,141.0,186.0,223.5]
    
    # primary sample:
    for i in range(4):
        print('RA range:',ra1[i],ra2[i])
        idx = np.where((ra_gama > ra1[i]) & (ra_gama < ra2[i]))
        n_prim = np.size(np.where(pri_gama[idx] == 8))
        n_prim_good = np.size(np.where((pri_gama[idx] == 8) & ((badflag_gama[idx] == 0) | (badflag_gama[idx] == 5))))
        n_prim_good_obs = np.size(np.where((pri_gama[idx] == 8) & (obsflag_gama[idx] == 1) & ((badflag_gama[idx] == 0) | (badflag_gama[idx] == 5))))
        print('Number of galaxies in primary sample:',n_prim)
        print('Number of galaxies in primary sample that are good:',n_prim_good,n_prim_good/n_prim)
        print('Number of galaxies in primary sample that are good abd obs:',n_prim_good_obs,n_prim_good_obs/n_prim_good)

        # secondaries:
        n_sec = np.size(np.where(pri_gama[idx] < 8))
        n_sec_good = np.size(np.where((pri_gama[idx] < 8) & ((badflag_gama[idx] == 0) | (badflag_gama[idx] == 5))))
        n_sec_good_obs = np.size(np.where((pri_gama[idx] < 8) & (obsflag_gama[idx] == 1) & ((badflag_gama[idx] == 0) | (badflag_gama[idx] == 5))))
        print('Number of galaxies in secondary sample:',n_sec)
        print('Number of galaxies in secondary sample that are good:',n_sec_good,n_sec_good/n_sec)
        print('Number of galaxies in secondary sample that are good and obs:',n_sec_good_obs,n_sec_good_obs/n_sec_good)

#########################################################
        
def calc_cat_comp_filler(infile):
    """For a given input catalogue file, calculate the completeness
    of the sample in terms of what fraction of objects where observed.
    this version works on the filler catalogues only"""

    hdulist = fits.open(infile)
    tab_gama = hdulist[1].data
    catid_gama = tab_gama['CATID']
    fillflag_gama = tab_gama['FILLFLAG']
    ra_gama = tab_gama['RA_OBJ']        
    ngama = np.size(catid_gama)
    obsflag_gama = np.zeros(ngama)

    # read in CubeObs file:
    hdulist = fits.open('/Users/scroom/data/sami/dr3/cats/CubeObs/CubeObs.fits')
    tab_obs = hdulist[1].data
    catid_obs = tab_obs['CATID']

    nfound = 0
    nfound_unique = 0
    for i in range(ngama):
        for catid in catid_obs:
            if (catid == catid_gama[i]):
                nfound = nfound + 1
                obsflag_gama[i] = 1

        if (obsflag_gama[i] == 1):
            nfound_unique = nfound_unique + 1
            
    print('Number of Filler targets:',ngama)
    print('Number of Filler input gals found in obs cat:',nfound)
    print('Number of unique Filler input gals found in obs cat:',nfound_unique)

    # calculate numbers for different selections:
    ra1 = [0.0,129.0,174.0,211.5]
    ra2 = [360.0,141.0,186.0,223.5]

    #Luca_AA_fillers_format => BAD_CLASS= 20
    #Luca_Pairs_outsideSAMI_format => BAD_CLASS= 30
    #Luca_Pairs_withSAMIprimary_format => BAD_CLASS= 40
    #Ned_sami_additional_targets_format => BAD_CLASS= 50
    #jvds_primary_fillers_stellar_kinematics_format => BAD_CLASS= 60
    #jvds_secondary_fillers_stellar_kinematics_format => BAD_CLASS= 70
    #sami_seeing_format => BAD_CLASS= 80 (Nic)

    fill_list = [20,30,40,50,90]

    # primary sample:
    for ifill in fill_list:
        print('Fill flag:',ifill)
        idx = np.where((fillflag_gama == ifill))
        n_fill = np.size(fillflag_gama[idx])
        n_fill_obs = np.size(np.where(obsflag_gama[idx] == 1))
        print('Number of galaxies in filler sample:',n_fill)
        print('Number of galaxies in filler sample that are obs:',n_fill_obs,n_fill_obs/n_fill)

    

###############################################################
def make_short_filler(infile):
    """Take a file list from Julia of all the SAMI filler targets
    and build a short catalogue of just basic quantities that is
    uniform for all the filler objects.
    
    usage:
    > sami_tools_smc.cat_tools.make_short_filler('/Users/scroom/data/sami/dr3/cats/all_inputcats/Secondary_fillers_AAfillers_PairswithSAMIprimary_jvdsBOTH_seeing108worst_Ned_PairsoutsideSAMI.txt_updatedFINALMay2018.txt') 

    """

    # read the text file that contains offsets:
    filler_data =ascii.read(infile)

    print(filler_data)

    # get the required data from the columns:
    ra = filler_data['col2'].data
    dec =  filler_data['col3'].data
    z_spec = filler_data['col7'].data
    z_tonry = filler_data['col6'].data
    catid =  filler_data['col18'].data
    flag = filler_data['col21'].data

    nfill = np.size(catid)
    print('Number of fillers found in cat: ',nfill)
    
    # define list of galaxies with cubes that are not in the final cluster catalogue:
    # this object not released, as poor redshift: 9011900882
    #
    # object not in final cluster cat:
    notinc_clus_catid = [9008500175,
                        9011900117,
                        9091700047,
                        9091700111,
                        9091700141,
                        9091700200,
                        9091700225,
                        9091700404,
                        9091700469,
                        9091701020,
                        9091701222,
                        9091701454,
                        9239900038,
                        9239900040,
                        9239900126,
                        9239900161,
                        9239901360,
                        9239901557,
                        9388000278,
                        9388001101,
                        9388001133]
    
    # read in full cluster cat:
    hdulist = fits.open('/Users/scroom/data/sami/dr3/cats/all_inputcats/ClustersCombined_V10_FINALobs.fits')
    tab_full = hdulist[1].data
    catid_full = tab_full['CATAID']
    ra_full = tab_full['R.A.']
    dec_full = tab_full['decl.']
    z_full = tab_full['Z']
    n_full = np.size(z_full)
    n_notinc =  np.size(notinc_clus_catid)
    notinc = np.zeros(n_full,dtype=np.int32)
    flag_full = np.full(n_full,90,dtype=np.int32)
    hdulist.close()
    
    for i in range(n_full):

        for j in range(n_notinc):
            if (catid_full[i] == notinc_clus_catid[j]):
                notinc[i] = 1

    idx_ni = np.where(notinc == 1)

    print(catid_full[idx_ni])
    
    # in some cases z_tonry is not set.  They are given large numbers, but we
    # can convert to Nan:
    z_tonry[z_tonry> 100.0]=np.nan
    
    # from Julia, the flags are:
    # 20 => galaxies with ALFALFA data and close to SAMI selection
    # 30 => GAMA pairs outside the original SAMI catalogue
    # 40 => Galaxies from GAMA that are the pair of a SAMI primary target
    # 50 => Ned’s additional targets
    # Classes 60,70,80 are galaxies from the SAMI catalogue previously observed
    # with borderline S/N or seeing, that benefitted from re-observation.

    # read in the main GAMA cat to check that there are no objects in that
    # are in both:
    hdulist = fits.open('/Users/scroom/data/sami/dr3/cats/InputCat_GAMA/InputCatGAMA.fits')
    tab_gama = hdulist[1].data
    catid_gama = tab_gama['CATID']
    ngama = np.size(catid_gama)
    
    for i in range(nfill):
        if (flag[i]<55):
            for j in range (ngama):
                if (catid[i] == catid_gama[j]):
                    # if found, set flag to large number, so ignored:
                    print('match found between filler and GAMA cat:',catid[i])
                    flag[i] = 80

    # next do a self check to see that there are no repeats in the list:
    for i in range(nfill):
        if (flag[i]<55):
            for j in range (i+1,nfill):
                if (catid[i] == catid[j]):
                    # if found, set flag to large number, so ignored:
                    print('self-match found in filler cat:',catid[i],flag[i],flag[j])
                    flag[j] = 80

                    
    # will ignore 60, 70, 80 as they are already in the main cat:
    idx = np.where(flag<55)

    col1 = fits.Column(name='CATID',format='K',unit='',array=np.concatenate((catid[idx],catid_full[idx_ni])))
    col2 = fits.Column(name='RA_OBJ',format='D',unit='deg',array=np.concatenate((ra[idx],ra_full[idx_ni])))
    col3 = fits.Column(name='DEC_OBJ',format='D',unit='deg',array=np.concatenate((dec[idx],dec_full[idx_ni])))
    #col4 = fits.Column(name='z_tonry',format='E',array=z_tonry[idx])
    col5 = fits.Column(name='z_spec',format='E',unit='',array=np.concatenate((z_spec[idx],z_full[idx_ni])))
    col6 = fits.Column(name='FILLFLAG',format='I',unit='',array=np.concatenate((flag[idx],flag_full[idx_ni])))

    #new_cols = fits.ColDefs([col1,col2,col3,col4,col5,col6])
    new_cols = fits.ColDefs([col1,col2,col3,col5,col6])

    # make new table:
    new_hdu = fits.BinTableHDU.from_columns(new_cols)
    tab_hdr = new_hdu.header
    # add keyword
    add_keywords(tab_hdr,'InputCatFiller','Input Catalogue, filler','SAMI input catalogue of filler targets')
    # insert extra comment keywords:
    tab_hdr.insert('TFORM1',('TCOMM1','Object indentifier','Description for column 1'),after=True)
    tab_hdr.insert('TFORM2',('TCOMM2','RA of object','Description for column 2'),after=True)
    tab_hdr.insert('TFORM3',('TCOMM3','Dec. of object','Description for column 3'),after=True)
    tab_hdr.insert('TFORM4',('TCOMM4','Heliocentric redshift of object','Description for column 4'),after=True)
    tab_hdr.insert('TFORM5',('TCOMM5','Filler type flag','Description for column 5'),after=True)

    
    # write new file:
    new_hdu.writeto('InputCatFiller.fits',overwrite=True)
    
    

###############################################################
def cat_combine_offset_coords(catfile,offsetfile):
    """Take an original format SAMI catalogue file (particularly the GAMA
    format file), and merge in the offsets to define two lists of coordinates
    in the file.  These with be RA_obj and Dec_obj and RA_ifu and Dec_ifu.
    """

    # first copy the cat file to a new name:
    outfile = catfile.replace('.fits','_objifu_coords.fits')
    #shutil.copy(catfile,outfile)
    
    # read in the catalogue file.  We will assume it is in fits format:
    hdulist = fits.open(catfile)

    tab = hdulist[1].data
    tab_cols = tab.columns
    # get catid:
    catid = tab['CATID']
    ra_ifu = tab['RA']
    dec_ifu = tab['Dec']
    bad_class = tab['BAD_CLASS']

    # change the names of the RA/Dec cols:
    tab_cols[1].name = 'RA_IFU'
    tab_cols[2].name = 'DEC_IFU'
    
    # read the text file that contains offsets:
    off_data =ascii.read(offsetfile)

    catid_off = off_data['col1'].data
    flag_off  = off_data['col5'].data
    # the offset is in arcsec and is applied to the original coordinates by
    # adding the offset, without any cos(dec) term.  See line ~592 of
    # sami_sel_gama.f
    ra_off  = off_data['col6'].data
    dec_off  = off_data['col7'].data

    #check sizes match:
    ng_off = np.size(catid_off)    
    ng = np.size(catid)
    print('Number of rows in cat:',ng)
    print('Number of rows in offset file:',ng_off)

    ra_obj = np.copy(ra_ifu)
    dec_obj = np.copy(dec_ifu)

    print('Objects with BAD_CLASS=5:')
    nchange = 0
    for i in range(ng):
        if (bad_class[i] == 5):
            for j in range(ng_off):
                if (catid[i] == catid_off[j]):
                    print(catid[i],bad_class[i],catid_off[j],flag_off[j],ra_off[j],dec_off[j])
                    ra_obj[i] = ra_obj[i] - ra_off[j]/3600.0
                    dec_obj[i] = dec_obj[i] - dec_off[j]/3600.0
                    nchange = nchange + 1

            
    # put the new columns in the table:
    new_cols = fits.ColDefs([
        fits.Column(name='RA_OBJ',format='D',array=ra_obj,unit='deg'),
        fits.Column(name='DEC_OBJ',format='D',array=dec_obj,unit='deg'),
        ])

    # make new table:
    new_hdu = fits.BinTableHDU.from_columns(tab_cols[0] + new_cols + tab_cols[1:])

    # write new file:
    new_hdu.writeto(outfile,overwrite=True)

    print('Number of coordinates changed:',nchange)

############################################################
# Clean up input catalogue and remove extra stuff in the
# fITS file that is not needed
#
def clean_input_cat(infile,extname='InputCat_GAMA',dmuname='Input Catalogue, GAMA',description='SAMI input catalogue in GAMA regions'):

    outfile = infile.replace('.fits','_clean.fits')

    catname = infile.replace('.fits','')
    
    # open original file:
    hdulist = fits.open(infile)

    # define a new HDU list:
    new_hdulist = fits.HDUList()

    # append HDU from table to new HDU list:
    new_hdulist.append(hdulist[1])

    # close old file:
    #hdulist.close()
    
    # set up extra keywords:
    tab_hdr = new_hdulist[1].header
    tab_hdr['EXTNAME'] = extname
    tab_hdr['DMU_NAME'] = (dmuname,'DMU Name')
    
    # datetime object containing current date and time
    now = datetime.now()
    # dd-mm-YY H:M:S
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    print("date and time =", dt_string)	

    tab_hdr['DATE'] = (dt_string,'Date and time DMU was generated')
    tab_hdr['AUTHOR'] = ('Scott Croom','Catalogue author')
    tab_hdr['CONTACT'] = ('scott.croom@sydney.edu.au','Contact for author')
    tab_hdr['DESCRIPT'] = (description,'Short description')

    tab_hdr.set('TCOMM1','Object identifier',after='TFORM1')

    if ('DATE-HDU' in tab_hdr):
        del tab_hdr['DATE-HDU']
        
    if ('STILVERS' in tab_hdr):
        del tab_hdr['STILVERS']

    if ('STILCLAS' in tab_hdr):
        del tab_hdr['STILCLAS']
        
    if ('MU_NAME' in tab_hdr):
        del tab_hdr['MU_NAME']

    # now set to Nan all undefined values:
    tab_data = new_hdulist[1].data
    cols = tab_data.columns
    cnames = cols.names
    cformats = cols.formats

    print('Number of Nans flagged in each column:')
    ncol = np.size(cnames)
    for i in range(ncol):

        if (cformats[i] == 'E'):
            tab_data[cnames[i]][tab_data[cnames[i]]<-90.0] = np.nan

            n_nan = np.count_nonzero(np.isnan(tab_data[cnames[i]]))
            print(cnames[i],cformats[i],n_nan)
    
    
    
    new_hdulist.writeto(outfile,overwrite=True)

############################################################
# clean up short filler catalogue
#
def clean_filler_cat(infile):

    outfile = infile.replace('.fits','_clean.fits')

    # open original file:
    hdulist = fits.open(infile)

    # define a new HDU list:
    new_hdulist = fits.HDUList()

    # append HDU from table to new HDU list:
    new_hdulist.append(hdulist[1])

    # close old file:
    #hdulist.close()
    
    # set up extra keywords:
    tab_hdr = new_hdulist[1].header

    add_keywords(tab_hdr,'InputCat_filler','Input Catalogue, filler','SAMI input catalogue of filler targets')

    new_hdulist.writeto(outfile,overwrite=True)

    

#################################################################
# add regular keywords:
#
def add_keywords(hdr,extname,dmuname,desc):   

    hdr['EXTNAME'] = extname
    hdr['DMU_NAME'] = (dmuname,'DMU Name')
    
    # datetime object containing current date and time
    now = datetime.now()
    # dd-mm-YY H:M:S
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    print("date and time =", dt_string)	

    hdr['DATE'] = (dt_string,'Date and time DMU was generated')
    hdr['AUTHOR'] = ('Scott Croom','Catalogue author')
    hdr['CONTACT'] = ('scott.croom@sydney.edu.au','Contact for author')
    hdr['DESCRIPT'] = (desc,'Short description')

    
    return
    
############################################################
# Add in morphology info into main input catalogue
#
def cat_add_morphology(infile,morphfile):

    """Take an original format SAMI catalogue file (particularly the GAMA
    format file), and merge in the offsets to define two lists of coordinates
    in the file.  These with be RA_obj and Dec_obj and RA_ifu and Dec_ifu.
    """

    # first copy the cat file to a new name:
    outfile = catfile.replace('.fits','_objifu_coords.fits')
    #shutil.copy(catfile,outfile)
    
    # read in the catalogue file.  We will assume it is in fits format:
    #hdulist = fits.open(catfile)

    #tab = hdulist[1].data
    #tab_cols = tab.columns
    # get catid:
    #catid = tab['CATID']
    #ra_ifu = tab['RA']
    #dec_ifu = tab['Dec']
    #bad_class = tab['BAD_CLASS']

    # change the names of the RA/Dec cols:
    #tab_cols[1].name = 'RA_IFU'
    #tab_cols[2].name = 'DEC_IFU'
    
    # read the text file that contains offsets:
    #off_data =ascii.read(offsetfile)

    #catid_off = off_data['col1'].data
    #flag_off  = off_data['col5'].data
