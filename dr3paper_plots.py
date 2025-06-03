###############################################################
# Script to generate plots for SAMI DR3 paper

import shutil
import numpy as np
import pylab as py
import scipy as sp
import astropy.io.fits as fits
import astropy.io.ascii as ascii
from scipy import stats
import wedgez.wedge
from scipy import ndimage

import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
#                                                  mark_inset)
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
#from matplotlib._png import read_png

import matplotlib.ticker as ticker
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, AnchoredOffsetbox
import matplotlib.transforms as mtransforms
from matplotlib.artist import Artist  
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from matplotlib.transforms import Bbox

#from matplotlib import pyplot, transforms
#from matplotlib.lines import Line2D
#import matplotlib.ticker as ticker
#import matplotlib.patches as patches


from datetime import datetime

from astropy.io import ascii,fits
from astropy.table import Column
from astropy import table

from sklearn import linear_model, datasets

from sami_tools_smc.dr_tools.sami_fluxcal import sami_read_apspec
from sami_dr_smc.sami_utils import spectres

import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator,
                                                 DictFormatter)
import matplotlib.gridspec as gridspec

from PIL import Image

###############################################################

def setup_plots():

    # set up formating:
    py.rc('text', usetex=True)
    py.rcParams.update({'font.size': 14})
    #py.rcParams.update({'lines.linewidth': 1})
    py.rcParams.update({'figure.autolayout': True})
    # this to get sans-serif latex maths:
    py.rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmathfonts}',  # load up the sansmath so that math -> helvet
       #r'\sansmathfonts'               # <- tricky! -- gotta actually tell tex to use!
        ]  

    

###############################################################

def completeness_corner(clusters=False,primary=True):

    """plot a corner plot for GAMA region completeness
    In various parameters.  If clusters is True, plot for
    clusters, otherwise plot for GAMA.."""

    # set alpha:
    a1 = 1.0

    
    # read in CubeObs file:
    hdulist = fits.open('/Users/scroom/data/sami/dr3/cats/all_cats_dr3/CubeObs.fits')
    tab_obs = hdulist[1].data
    catid_obs = tab_obs['CATID']
    cubeidpub = tab_obs['CUBEIDPUB']
    ncube = np.size(catid_obs)

    if (not clusters):
        # read in GAMA catalogue file:
        hdulist = fits.open('/Users/scroom/data/sami/dr3/cats/all_cats_dr3/InputCatGAMADR3.fits')
        tab_gama = hdulist[1].data
        catid_gama = tab_gama['CATID']
        pri_gama = tab_gama['SURV_SAMI']
        badflag_gama = tab_gama['BAD_CLASS']
        ra_gama = tab_gama['RA_OBJ']
        mstar = tab_gama['Mstar']
        z = tab_gama['z_spec']
        re = tab_gama['r_e']
        ngama = np.size(catid_gama)
        obsflag_gama = np.zeros(ngama)
        dens = np.empty(ngama)
        dens.fill(np.nan)
        
        # read in densities:
        #hdulist = fits.open('/Users/scroom/data/sami/cats/sami_sel_updatedJuly2015_v0.9_matched_EnvironmentMeasuresGamav01.fits')
        hdulist = fits.open('/Users/scroom/data/sami/dr3/cats/all_cats_dr3/DensityCatDR3.fits')
        tab_dens = hdulist[1].data
        catid_dens = tab_dens['CATID']
        dens_dens = tab_dens['SurfaceDensity']
        ndens = np.size(dens_dens)


        # read the text file that contains offsets:
        #dens_data =ascii.read('/Users/scroom/data/sami/dr3/cats/env_cats/EnvironmentMeasuresGAMAv01_020316_n5.dat')
        #catid_dens = dens_data['CATAID']
        #dens_dens = dens_data['SurfaceDensity']
        #ndens = np.size(dens_dens)
        #print(dens_data['SurfaceDensity'])

        
        
    if (clusters):
        # read in GAMA catalogue file:
        hdulist = fits.open('/Users/scroom/data/sami/dr3/cats/all_cats_dr3/InputCatClustersDR3.fits')
        tab_gama = hdulist[1].data
        catid_gama = tab_gama['CATID']
        pri_gama = tab_gama['SURV_SAMI']
        badflag_gama = tab_gama['BAD_CLASS']
        ra_gama = tab_gama['RA_OBJ']
        mstar = tab_gama['Mstar']
        z = tab_gama['z_spec']
        re = tab_gama['r_e']
        ngama = np.size(catid_gama)
        obsflag_gama = np.zeros(ngama)
        dens = -99*np.ones(ngama)

        # read in densities:
        hdulist = fits.open('/Users/scroom/data/sami/dr3/cats/all_cats_dr3/DensityCatDR3.fits')
        tab_dens = hdulist[1].data
        catid_dens = tab_dens['CATID']
        dens_dens = tab_dens['SurfaceDensity']
        ndens = np.size(dens_dens)
        #hdulist = fits.open('/Users/scroom/data/sami/dr3/cats/env_cats/DensityCatDR3.fits')
        #tab_dens = hdulist[1].data
        #catid_dens = tab_dens['CATID']
        #dens_dens = tab_dens['density']
        #ndens = np.size(dens_dens)


    nfound = 0
    nfound_unique = 0

    for i in range(ngama):
        for j in range(ncube):
            catid = catid_obs[j]
            if (catid == catid_gama[i]):
                nfound = nfound + 1
                obsflag_gama[i] = 1

        if (obsflag_gama[i] == 1):
            nfound_unique = nfound_unique + 1
            
        for j in range(ndens):
            catid = catid_dens[j]
            if (catid == catid_gama[i]):
                dens[i] = np.log10(dens_dens[j])
                break
                
        if (obsflag_gama[i] == 1):
            nfound_unique = nfound_unique + 1

    print('Number of targets:',ngama)
    print('Number of input gals found in obs cat:',nfound)
    print('Number of unique input gals found in obs cat:',nfound_unique)

    if (primary):
        prim = np.where((pri_gama > 6) & ((badflag_gama == 0) | (badflag_gama == 5)))
        prim_gooddens = np.where((pri_gama > 6)  & (np.isfinite(dens)) & ((badflag_gama == 0) | (badflag_gama == 5) | (badflag_gama == 8)))
        prim_obs =  np.where((pri_gama > 6) & (obsflag_gama == 1) & ((badflag_gama == 0) | (badflag_gama == 5) | (badflag_gama == 8)))
        prim_notobs =  np.where((pri_gama > 6) & (obsflag_gama == 0) & ((badflag_gama == 0) | (badflag_gama == 5) | (badflag_gama == 8)))
        prim_obs_gooddens =  np.where((pri_gama > 6) & (obsflag_gama == 1) &  (np.isfinite(dens)) & ((badflag_gama == 0) | (badflag_gama == 5) | (badflag_gama == 8)))
        prim_notobs_gooddens =  np.where((pri_gama > 6) & (obsflag_gama == 0)  & (np.isfinite(dens)) & ((badflag_gama == 0) | (badflag_gama == 5) | (badflag_gama == 8)))
    else:
        prim = np.where((pri_gama < 7) & ((badflag_gama == 0) | (badflag_gama == 5)))
        prim_gooddens = np.where((pri_gama < 7) & (np.isfinite(dens)) & ((badflag_gama == 0) | (badflag_gama == 5) | (badflag_gama == 8)))
        prim_obs =  np.where((pri_gama < 7) & (obsflag_gama == 1) & ((badflag_gama == 0) | (badflag_gama == 5) | (badflag_gama == 8)))
        prim_notobs =  np.where((pri_gama < 7) & (obsflag_gama == 0) & ((badflag_gama == 0) | (badflag_gama == 5) | (badflag_gama == 8)))
        prim_obs_gooddens =  np.where((pri_gama < 7) & (obsflag_gama == 1) & (np.isfinite(dens)) & ((badflag_gama == 0) | (badflag_gama == 5) | (badflag_gama == 8)))
        prim_notobs_gooddens =  np.where((pri_gama < 7)  & (np.isfinite(dens)) & (obsflag_gama == 0) & ((badflag_gama == 0) | (badflag_gama == 5) | (badflag_gama == 8)))

    # do some analysis on the comleteness in clusters per cluster
    if (clusters):
        catid_str = np.empty(ngama,dtype='U16')
        clus_str = np.empty(ngama,dtype='U16')
        for i in range(ngama):
            catid_str[i] = str(catid_gama[i])
            clus_str[i] = catid_str[i][1:5]

        cluster_list = np.unique(clus_str)
        print(cluster_list)

        # for each cluster, count the number of objects:
        for clus in cluster_list:
            ic_prim = np.where((clus_str == clus) & (pri_gama > 6))
            ic_prim_good = np.where((clus_str == clus) & (pri_gama > 6) & ((badflag_gama == 0) | (badflag_gama == 5)  | (badflag_gama == 8)))
            ic_prim_good_obs = np.where((clus_str == clus) & (pri_gama > 6) & (obsflag_gama == 1) & ((badflag_gama == 0) | (badflag_gama == 5)  | (badflag_gama == 8)))
            ic_sec = np.where((clus_str == clus) & (pri_gama < 7))
            ic_sec_good = np.where((clus_str == clus) & (pri_gama < 7) & ((badflag_gama == 0) | (badflag_gama == 5)  | (badflag_gama == 8)))
            ic_sec_good_obs = np.where((clus_str == clus) & (pri_gama < 7) & (obsflag_gama == 1) & ((badflag_gama == 0) | (badflag_gama == 5)  | (badflag_gama == 8)))

            comp = np.size(clus_str[ic_prim_good_obs])/np.size(clus_str[ic_prim_good])
            print(clus,np.size(clus_str[ic_prim]),np.size(clus_str[ic_prim_good]),np.size(clus_str[ic_prim_good_obs]),comp)
            print(clus,np.size(clus_str[ic_sec]),np.size(clus_str[ic_sec_good]),np.size(clus_str[ic_sec_good_obs]),'sec')
            
        ic_prim = np.where((pri_gama > 6))
        ic_prim_good = np.where((pri_gama > 6) & ((badflag_gama == 0) | (badflag_gama == 5)  | (badflag_gama == 8)))
        ic_prim_good_obs = np.where((pri_gama > 6) & (obsflag_gama == 1) & ((badflag_gama == 0) | (badflag_gama == 5)  | (badflag_gama == 8)))
        ic_sec = np.where((pri_gama < 7))
        ic_sec_good = np.where((pri_gama < 7) & ((badflag_gama == 0) | (badflag_gama == 5)  | (badflag_gama == 8)))
        ic_sec_good_obs = np.where((pri_gama < 7) & (obsflag_gama == 1) & ((badflag_gama == 0) | (badflag_gama == 5)  | (badflag_gama == 8)))

        comp = np.size(clus_str[ic_prim_good_obs])/np.size(clus_str[ic_prim_good])
        print('All:',np.size(clus_str[ic_prim]),np.size(clus_str[ic_prim_good]),np.size(clus_str[ic_prim_good_obs]),comp)
        print('All:',np.size(clus_str[ic_sec]),np.size(clus_str[ic_sec_good]),np.size(clus_str[ic_sec_good_obs]),'sec')
        
        
    # set up PDF
    if (clusters):
        pdf = PdfPages('clusters_comp_corner.pdf')
    else:
        pdf = PdfPages('gama_comp_corner.pdf')

    # start plotting:
    setup_plots()
    py.rcParams.update({'font.size': 8})
    py.minorticks_on()
    
    fig, axs = py.subplots(4, 4,gridspec_kw={'hspace': 0, 'wspace': 0})
    #(ax1, ax2), (ax3, ax4) = axs
    #fig.suptitle('Sharing x per column, y per row')

    axs[0,1].axis('off')
    axs[0,2].axis('off')
    axs[0,3].axis('off')
    axs[1,2].axis('off')
    axs[1,3].axis('off')
    axs[2,3].axis('off')

    # define ranges for parameters:
    m_range = [7.5,12.0]
    d_range = [-2.0,3.0]
    z_range = [0.0,0.1]
    r_range = [0.0,20.0]

    # flag to rescale histogram:
    rescale = True
    pt = '.'
    ms = 0.5
    
    # plot mass vs other parameters:
    primcol = 'k'
    if (clusters):
        obscol = 'r'
    else:
        obscol = 'r'
        
    # sequence with redshift:
    axs[3,0].plot(z[prim_notobs],mstar[prim_notobs],pt,color=primcol,markersize=ms)
    axs[3,0].plot(z[prim_obs],mstar[prim_obs],pt,color=obscol,markersize=ms,alpha=a1)
    axs[3,0].set(xlim=z_range,ylim=m_range,xlabel='z',ylabel='log(M$^*$/M$_\odot$)')
    #axs[3,0].xaxis.set_major_locator(MultipleLocator(0.02))
    axs[3,0].xaxis.set_major_locator(ticker.FixedLocator([0.0,0.02,0.04,0.06,0.08]))
    axs[3,0].yaxis.set_major_locator(ticker.FixedLocator([8,9,10,11]))
    #axs[3,0].xaxis.set_major_locator(ticker.MaxNLocator(prune='upper'))
    #axs[3,0].yaxis.set_major_locator(ticker.MaxNLocator(prune='upper'))
    #py.setp(axs[3,0].get_xticklabels()[-1], visible=False)    
    #py.setp(axs[3,0].get_yticklabels()[-1], visible=False)
    
    axs[2,0].plot(z[prim_notobs],re[prim_notobs],pt,color=primcol,markersize=ms)
    axs[2,0].plot(z[prim_obs],re[prim_obs],pt,color=obscol,markersize=ms,alpha=a1)
    axs[2,0].set(xlim=z_range,ylim=r_range,ylabel='R$_e$ [arcsec]')
    py.setp(axs[2,0].get_yticklabels()[-1], visible=False)    
    
    axs[1,0].plot(z[prim_notobs],dens[prim_notobs],pt,color=primcol,markersize=ms)
    axs[1,0].plot(z[prim_obs],dens[prim_obs],pt,color=obscol,markersize=ms,alpha=a1)
    axs[1,0].set(xlim=z_range,ylim=d_range,ylabel='log($\Sigma_5$) [Mpc$^{-2}$]')
    axs[1,0].yaxis.set_major_locator(ticker.FixedLocator([-2.0,-1.0,0.0,1.0,2.0]))

    # set weights to get proper normalization:
    weights_prim = np.ones_like(z[prim])/float(len(z[prim]))
    weights_prim_obs = np.ones_like(z[prim_obs])/float(len(z[prim_obs]))

    #axs[0,0].hist(z[prim],bins=10,range=z_range,histtype='step',color=primcol,density=rescale)
    #axs[0,0].hist(z[prim_obs],bins=10,range=z_range,histtype='step',color=obscol,density=rescale)
    axs[0,0].hist(z[prim],bins=10,range=z_range,histtype='step',color=primcol,weights=weights_prim)
    axs[0,0].hist(z[prim_obs],bins=10,range=z_range,histtype='step',color=obscol,weights=weights_prim_obs)
    axs[0,0].set(xlim=z_range,ylabel='Fractional number')
    d,p = stats.ks_2samp(z[prim_obs],z[prim_notobs])
    print('Redshift K-S test (d,p):',d,p)
    print('median redshifts (all,obs):',np.nanmedian(z[prim]),np.nanmedian(z[prim_obs]))
    print('')
    
    # sequence with density:
    axs[3,1].plot(dens[prim_notobs],mstar[prim_notobs],pt,color=primcol,markersize=ms)
    axs[3,1].plot(dens[prim_obs],mstar[prim_obs],pt,color=obscol,markersize=ms,alpha=a1)
    axs[3,1].set(xlim=d_range,ylim=m_range,xlabel='log($\Sigma_5$) [Mpc$^{-2}$]')
    axs[3,1].xaxis.set_major_locator(ticker.FixedLocator([-2.0,-1.0,0.0,1.0,2.0]))
    axs[3,1].yaxis.set_major_locator(ticker.FixedLocator([8,9,10,11]))

    axs[2,1].plot(dens[prim_notobs],re[prim_notobs],pt,color=primcol,markersize=ms)
    axs[2,1].plot(dens[prim_obs],re[prim_obs],pt,color=obscol,markersize=ms,alpha=a1)
    axs[2,1].set(xlim=d_range,ylim=r_range)
    axs[2,1].xaxis.set_major_locator(ticker.FixedLocator([-2.0,-1.0,0.0,1.0,2.0]))

    # set weights to get proper normalization:
    weights_prim = np.ones_like(dens[prim_gooddens])/float(len(dens[prim_gooddens]))
    weights_prim_obs = np.ones_like(dens[prim_obs_gooddens])/float(len(dens[prim_obs_gooddens]))
    
    axs[1,1].hist(dens[prim_gooddens],bins=10,range=d_range,histtype='step',color=primcol,weights=weights_prim)
    axs[1,1].hist(dens[prim_obs_gooddens],bins=10,range=d_range,histtype='step',color=obscol,weights=weights_prim_obs)
    axs[1,1].set(xlim=d_range)

    d,p = stats.ks_2samp(dens[prim_notobs_gooddens],dens[prim_obs_gooddens])
    med_pri = np.nanmedian(dens[prim])
    med_obs = np.nanmedian(dens[prim_obs])
    med_notobs = np.nanmedian(dens[prim_notobs])
    med_pri_err = 1.2533*np.nanstd(dens[prim])/np.sqrt(np.size(dens[prim_gooddens]))
    med_obs_err = 1.2533*np.nanstd(dens[prim_obs])/np.sqrt(np.size(dens[prim_obs_gooddens]))
    med_notobs_err = 1.2533*np.nanstd(dens[prim_notobs])/np.sqrt(np.size(dens[prim_notobs_gooddens]))
    print('Density K-S test (obs vs notobs) d,p:',d,p)
    print('median density (all,obs,notobs):',med_pri,med_obs,med_notobs)
    print('median density errors (all,obs,notobs):',med_pri_err,med_obs_err,med_notobs_err)
    print('diff (all-obs):',med_pri-med_obs,'+-',np.sqrt(med_pri_err**2+med_obs_err**2))
    print('diff (obs-notobs):',med_obs-med_notobs,'+-',np.sqrt(med_obs_err**2+med_notobs_err**2))
    print('number with good densities (all,obs,notobs):',np.size(dens[prim_gooddens]),np.size(dens[prim_obs_gooddens]),np.size(dens[prim_notobs_gooddens]))
    print('')

    # sequence with re:
    axs[3,2].plot(re[prim_notobs],mstar[prim_notobs],pt,color=primcol,markersize=ms)
    axs[3,2].plot(re[prim_obs],mstar[prim_obs],pt,color=obscol,markersize=ms,alpha=a1)
    axs[3,2].set(xlim=r_range,ylim=m_range,xlabel='R$_e$ [arcsec]')
    axs[3,2].yaxis.set_major_locator(ticker.FixedLocator([8,9,10,11]))

    # set weights to get proper normalization:
    weights_prim = np.ones_like(re[prim])/float(len(re[prim]))
    weights_prim_obs = np.ones_like(re[prim_obs])/float(len(re[prim_obs]))
    
    axs[2,2].hist(re[prim],bins=10,range=r_range,histtype='step',color=primcol,weights=weights_prim)
    axs[2,2].hist(re[prim_obs],bins=10,range=r_range,histtype='step',color=obscol,weights=weights_prim_obs)
    axs[2,2].set(xlim=r_range)
    d,p = stats.ks_2samp(re[prim_obs],re[prim_notobs])
    print('Re K-S test d,p:',d,p)
    print('median Re (all,obs):',np.nanmedian(re[prim]),np.nanmedian(re[prim_obs]))
    print('\n')


    # sequence with mstar:

    # set weights to get proper normalization:
    weights_prim = np.ones_like(mstar[prim])/float(len(mstar[prim]))
    weights_prim_obs = np.ones_like(mstar[prim_obs])/float(len(mstar[prim_obs]))

    axs[3,3].hist(mstar[prim],bins=11,range=m_range,histtype='step',color=primcol,weights=weights_prim)
    axs[3,3].hist(mstar[prim_obs],bins=11,range=m_range,histtype='step',color=obscol,weights=weights_prim_obs)
    axs[3,3].set(xlim=m_range,xlabel='log(M$^*$/M$_\odot$)')
    d,p = stats.ks_2samp(mstar[prim_obs],mstar[prim_notobs])
    print('Mstar K-S test d,p:',d,p)
    med_pri = np.nanmedian(mstar[prim])
    med_obs = np.nanmedian(mstar[prim_obs])
    med_notobs = np.nanmedian(mstar[prim_notobs])
    med_pri_err = 1.2533*np.nanstd(mstar[prim])/np.sqrt(np.size(mstar[prim]))
    med_obs_err = 1.2533*np.nanstd(mstar[prim_obs])/np.sqrt(np.size(mstar[prim_obs]))
    med_notobs_err = 1.2533*np.nanstd(mstar[prim_notobs])/np.sqrt(np.size(mstar[prim_notobs]))
    print('median Mstar (all,obs):',med_pri,med_obs,med_notobs)
    print('median Mstar errors (all,obs):',med_pri_err,med_obs_err,med_notobs)
    print('diff (all-obs):',med_pri-med_obs,'+-',np.sqrt(med_pri_err**2+med_obs_err**2))
    print('diff (obs-notobs):',med_obs-med_notobs,'+-',np.sqrt(med_obs_err**2+med_notobs_err**2))
    print('\n')

    # make sure labels only on outer axes:
    for ax in axs.flat:
        ax.label_outer()
        ax.tick_params(which='both', direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        #ax.xaxis.set_minor_locator(MultipleLocator(5))
        #ax.yaxis.set_minor_locator(MultipleLocator(5))
        
    py.savefig(pdf, format='pdf')        
    pdf.close()

    fig3 = py.figure(3)
    ax3 = fig3.add_subplot(111)
    ax3.plot(mstar[prim],dens[prim],'.')

        
###############################################################

def fibre_profiles(col=1300,row=2210,ncol=50,gapnum=6):

    """Plot data frame, fibre profiles and scattered light fit"""

    # set up formating:
    py.rc('text', usetex=True)
    py.rc('font', family='sans-serif')
    py.rcParams.update({'font.size': 8})
    py.rcParams.update({'lines.linewidth': 1})

    # locations of gaps:
    gaploc=[63,126,189,252,315,378,441,504,567,630,693,756]

    # set up PDF 
    pdf = PdfPages('dr3_fibre_profiles.pdf')

    # Read in specific files.  We will use the flat from a specific
    # field on disk:
    path = '/Users/scroom/data/sami/test_data/scattered_light/vbright_gal/ccd_2/'
    frame = '21apr20008'
    imfile = path+frame+'im.fits'
    tlmfile = path+frame+'tlm.fits'
    mimfile =  path+frame+'_outdir/'+frame+'mim.fits'
    mslfile =  path+frame+'_outdir/'+frame+'msl.fits'
    resfile =  path+frame+'_outdir/'+frame+'res.fits'

    # read data:
    im = fits.getdata(imfile)
    tlm = fits.getdata(tlmfile)
    sig = fits.getdata(tlmfile,extname='SIGMAPRF')
    mim = fits.getdata(mimfile)
    msl = fits.getdata(mslfile)
    res = fits.getdata(resfile)

    ny,nx = np.shape(im)
    print(ny,nx)
    print(np.shape(tlm))
    
    # define column to cut at
    xax = np.linspace(1,ny,ny)
    xax = np.linspace(1,ny,ny)
    y1 = row-ncol
    y2 = row+ncol
    x1 = col-ncol
    x2 = col+ncol

    vmin = 0.0
    vmax = 50000.0

    # get tlm around gap:
    tlm1 = tlm[gaploc[gapnum]-1,:]
    tlm2 = tlm[gaploc[gapnum],:]
    sig1 = sig[gaploc[gapnum]-1,:]
    sig2 = sig[gaploc[gapnum],:]
    tlmax = np.linspace(1,nx,nx)
    
    # start plotting setup:
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(2,2,1)
    cax1 = ax1.imshow(im,origin='lower',interpolation='nearest',cmap='gray',vmin=vmin,vmax=vmax)
    pos1 = ax1.get_position() # get the original position 
    cbaxes = fig1.add_axes([pos1.x0+0.1,pos1.y0+0.03,0.01,pos1.height]) 
    cbar1 = py.colorbar(cax1, cax = cbaxes)
    #cbar1 = fig1.colorbar(cax1)
    cbar1.set_label('Counts', rotation=90)
    ax1.axvline(col,color='r',linestyle=':')
    p = py.Rectangle((x1, y1), ncol*2, ncol*2, fill=False,linestyle=':',color='r')
    #p.set_transform(ax.transAxes)
    #p.set_clip_on(False)
    ax1.add_patch(p)
    ax1.set(xlabel='Spectral pixels',ylabel='Spatial pixels')
    # For the minor ticks, use no labels; default NullFormatter.
    ax1.xaxis.set_major_locator(MultipleLocator(1000))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.text(0.04, 0.93,'a)',horizontalalignment='left',verticalalignment='center',transform = ax1.transAxes,bbox=dict(facecolor='white',alpha=0.6,edgecolor='none', pad=2.0))
    #t1.set_bbox(dict(facecolor='white', edgecolor='white'))
    print(ax1.get_position())
    pos1 = ax1.get_position() # get the original position 
    pos2 = [pos1.x0 - 0.05, pos1.y0 + 0.03,  pos1.width, pos1.height] 
    ax1.set_position(pos2) # set a new position
    print(ax1.get_position())
    #pos1 = cax1.get_position() # get the original position 
    #pos2 = [pos1.x0 - 0.05, pos1.y0 + 0.05,  pos1.width, pos1.height] 
    #cax1.set_position(pos2) # set a new position

    
    ax2 = fig1.add_subplot(2,2,2)
    cax2 = ax2.imshow(im,origin='lower',interpolation='nearest',cmap='gray',vmin=vmin,vmax=vmax)
    ax2.axvline(col,color='r',linestyle=':')
    ax2.plot(tlmax,tlm1+sig1*4,'--',color='c')
    ax2.plot(tlmax,tlm2-sig2*4,'--',color='c')
    ax2.set(xlim=[x1,x2],ylim=[y1,y2])
    ax2.set(xlabel='Spectral pixels',ylabel='Spatial pixels')
    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.text(0.04, 0.93,'b)',horizontalalignment='left',verticalalignment='center',transform = ax2.transAxes,bbox=dict(facecolor='white',alpha=0.6,edgecolor='none', pad=2.0))
    #ax2.text(0.03, 0.93,'b)',color='white',horizontalalignment='left',verticalalignment='center',transform = ax2.transAxes)
    print(ax2.get_position())
    pos1 = ax2.get_position() # get the original position 
    pos2 = [pos1.x0, pos1.y0 + 0.03,  pos1.width, pos1.height] 
    ax2.set_position(pos2) # set a new position

    #
    ax3 = fig1.add_subplot(2,2,3)
    ax3.plot(xax,im[:,col],'-',color='b')
    ax3.plot(xax,msl[:,col],'-',color='r')
    ax3.axvline(y1,color='r',linestyle=':')
    ax3.axvline(y2,color='r',linestyle=':')
    ax3.set(xlabel='Spatial pixels',ylabel='Counts',ylim=[vmin,vmax+5000],xlim=[0,ny])
    print(vmax)
    ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax3.text(0.03, 0.93,'c)',horizontalalignment='left',verticalalignment='center',transform = ax3.transAxes)
    print(ax3.get_position())
    
    ax4 = fig1.add_subplot(2,2,4)
    ax4.plot(xax[y1:y2],im[y1:y2,col],'-',color='b',label='Data')
    ax4.plot(xax[y1:y2],mim[y1:y2,col]+msl[y1:y2,col],'-',color='g',label='Fit')
    ax4.plot(xax[y1:y2],msl[y1:y2,col],'-',color='r',label='Background')
    ax4.axvline(tlm1[col]+sig1[col]*4,ymax=0.75,linestyle='--',color='c',label='Gap boundary')
    ax4.axvline(tlm2[col]-sig2[col]*4,ymax=0.75,linestyle='--',color='c')
    #ax2.plot(tlmax,tlm1+sig1*4,'--',color='g')
    #ax2.plot(tlmax,tlm2-sig2*4,'--',color='g')
    ax4.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax4.set(xlabel='Spatial pixels',ylim=[vmin,vmax+5000],xlim=[y1,y2])
    ax4.text(0.03, 0.93,'d)',horizontalalignment='left',verticalalignment='center',transform = ax4.transAxes)
    py.legend(prop={'size':7},ncol=2,loc='upper right')

    print(ax4.get_position())

    # put in connecting lines:
    #ax1.annotate('', xy=(col,0), xytext=(0,vmax+5000), xycoords=ax1.transData, 
    #     textcoords=ax3.transData,arrowprops=dict(color='red', linestyle=':',
    #    arrowstyle='-',alpha=0.5,clip_on=False))
    # from ac1 to ax3:
    ax3.annotate('', xy=(col,0), xytext=(0,vmax+5000), xycoords=ax1.transData, 
         textcoords=ax3.transData, 
         arrowprops=dict(color='red', linestyle=':',arrowstyle='-',alpha=0.5))
    ax3.annotate('', xy=(col,0), xytext=(ny,vmax+5000), xycoords=ax1.transData, 
         textcoords=ax3.transData, 
         arrowprops=dict(color='red', linestyle=':',arrowstyle='-',alpha=0.5))

    # from ax1 to ax2:
    ax2.annotate('', xy=(x2,y1), xytext=(x1,y1), xycoords=ax1.transData, 
         textcoords=ax2.transData, 
         arrowprops=dict(color='red', linestyle=':',arrowstyle='-',alpha=0.5))
    ax2.annotate('', xy=(x2,y2), xytext=(x1,y2), xycoords=ax1.transData, 
         textcoords=ax2.transData, 
         arrowprops=dict(color='red', linestyle=':',arrowstyle='-',alpha=0.5))
    
    # from ax2 to ax4:
    ax4.annotate('', xy=(col,y1), xytext=(y1,vmax+5000), xycoords=ax2.transData, 
         textcoords=ax4.transData, 
         arrowprops=dict(color='red', linestyle=':',arrowstyle='-',alpha=0.5))
    ax4.annotate('', xy=(col,y1), xytext=(y2,vmax+5000), xycoords=ax2.transData, 
         textcoords=ax4.transData, 
         arrowprops=dict(color='red', linestyle=':',arrowstyle='-',alpha=0.5))

    
    #py.tight_layout()
    
    py.savefig(pdf, format='pdf')        
    pdf.close()
            
##########################################################################
# plot example spectra for 5577 line fixes.
#
def sky5577_comp():

    """Plot comparison between old and new SAMI spectra and SDSS for scattered light
    around 5577 night sky line."""

    setup_plots()
    
    # set up main path:
    path = '/Users/scroom/data/sami/fluxcal/'
    
    # define files needed:
    samidr3_blue = path+'dr3_aperture_spec_v8/289198_blue_6_Y14SAR1_P006_12T046_2015_02_16-2015_02_25_apspec.fits'
    #samidr3_red = path+'dr3_aperture_spec_v8/289198_red_6_Y14SAR1_P006_12T046_2015_02_16-2015_02_25_apspec.fits'
    
    samidr2_blue = path+'dr2_aperture_spec_v2/289198_blue_6_Y14SAR1_P006_12T046_aperture_spec.fits'
    #samidr2_red = path+'dr2_aperture_spec_v2/289198_red_6_Y14SAR1_P006_12T046_aperture_spec.fits'

    sdss_spec = path+'sdss_spec2/289198_1_spec-517-52024-189.fits'

    # set up PDF 
    pdf = PdfPages('dr3_sky5577.pdf')

    # read files:
    apname='3_ARCSECOND'
    hdulist = fits.open(samidr3_blue)
    samidr3_flux_blue,samidr3_lam_blue = sami_read_apspec(hdulist,apname,doareacorr=False)
    # SDSS has flux units of 1E-17 erg/cm^2/s/Ang, so *10 for SAMI:
    samidr3_flux_blue = samidr3_flux_blue * 10.0
    hdulist.close()
    
    hdulist = fits.open(samidr2_blue)
    samidr2_flux_blue,samidr2_lam_blue = sami_read_apspec(hdulist,apname,doareacorr=False)
    samidr2_flux_blue = samidr2_flux_blue * 10.0
    hdulist.close()

    # read SDSS spec
    hdulist = fits.open(sdss_spec)
    sdss_spec_table = hdulist['COADD'].data
    sdss_flux = sdss_spec_table['flux']
    sdss_loglam = sdss_spec_table['loglam']
    sdss_lam = 10.0**sdss_loglam
    sdss_lam_air = sdss_lam/(1.0 +2.735182e-4 + 131.4182/sdss_lam**2 + 2.76249e8/sdss_lam**4)

    # SDSS spectra (as of DR6) are normalized to PSF mags
    # these have more flux in them than fibre mags, so there is a correction.
    # the difference is 0.35 mags (see http://classic.sdss.org/dr6/products/spectra/spectrophotometry.html)
    # this correction is only true in the average, as it also depends on seeing.
    psf2fib = 10.0**(-0.4*0.35)
    sdss_flux = sdss_flux * psf2fib
    sdss_flux_blue = spectres(samidr3_lam_blue,sdss_lam_air,sdss_flux)

    # start plotting:
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(sdss_lam_air,sdss_flux,'-',color='k',label='SDSS')
    ax1.plot(samidr2_lam_blue,samidr2_flux_blue,'-',color='c',label='SAMI DR2')
    ax1.plot(samidr3_lam_blue,samidr3_flux_blue,'-',color='b',label='SAMI DR3')    
    ax1.set(ylabel='Flux [1E-17 erg/cm$^2$/s/\AA]',xlabel='Wavelength [\AA]',xlim=[3700.0,5750.0],ylim=[0.0,30.0])
    ax1.legend(prop={'size':10},ncol=1,loc='upper right')


    ax2 = py.axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.1,0.65,0.5,0.3])
    ax2.set_axes_locator(ip)
    ax2.plot(sdss_lam_air,sdss_flux,'-',color='k')
    ax2.plot(samidr2_lam_blue,samidr2_flux_blue,'-',color='c')
    ax2.plot(samidr3_lam_blue,samidr3_flux_blue,'-',color='b')    
    ax2.set(xlim=[5450.0,5700.0],ylim=[12.0,22.0])

    
    py.savefig(pdf, format='pdf')        
    pdf.close()

##########################################################################
# plot distribution of ratios between SDSS and SAMI spectra
#
def sami_sdss_comp():

    """Plot comparisons between SDSS and SAMI spectra, particularly ratios
    of the spectra"""

    # configure formatting:
    setup_plots()

    # table file:
    infile = '/Users/scroom/data/sami/fluxcal/sami_sdss_comp_tab_final.fits'
    
    tab = table.Table.read(infile)
    tab.info()

    # set up PDF 
    pdf = PdfPages('dr3_sami_sdss_fluxratio.pdf')

    # set up plotting:
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(1,1,1)
    ax1.hist(tab['RatioMed'],bins=50,range=(0.0,2.0),histtype='step',color='k',density=True,label='SAMI DR3')
    ax1.hist(tab['RatioMed2'],bins=50,range=(0.0,2.0),histtype='step',color='k',linestyle=':',density=True,label='SAMI DR2')
    ax1.text(0.04, 0.93,'a)',horizontalalignment='left',verticalalignment='center',transform = ax1.transAxes)

    # get median values:
    minus = (100-68.27)/2.0
    plus = 100-(100-68.27)/2.0
    med1 = np.nanmedian(tab['RatioMed'])
    med2 = np.nanmedian(tab['RatioMed2'])
    p1_plus = np.nanpercentile(tab['RatioMed'],plus)
    p1_minus = np.nanpercentile(tab['RatioMed'],minus)
    p1 = (p1_plus-p1_minus)/2.0
    p2_plus = np.nanpercentile(tab['RatioMed2'],plus)
    p2_minus = np.nanpercentile(tab['RatioMed2'],minus)
    p2 = (p2_plus-p2_minus)/2.0
    ax1.axvline(med1,linestyle='-',color='k')
    ax1.axvline(med2,linestyle=':',color='k')
    ax1.set(xlabel='SAMI/SDSS flux',ylabel='Normalized count')
    ax1.legend(prop={'size':10},ncol=1,loc='upper right')
    
    print('new data median: ',med1,p1)
    print('old data median: ',med2,p2)
    
    py.savefig(pdf, format='pdf')        
    pdf.close()
    
    # set up PDF 
    pdf = PdfPages('dr3_sami_sdss_fluxratioblue.pdf')

    # set up plotting:
    fig2 = py.figure(2)
    ax2 = fig2.add_subplot(1,1,1)
    ax2.hist(tab['RatioMedBlue'],bins=50,range=(0.0,2.0),histtype='step',color='b',density=True,label='SAMI DR3')
    ax2.hist(tab['RatioMedBlue2'],bins=50,range=(0.0,2.0),histtype='step',color='b',linestyle=':',density=True,label='SAMI DR2')

    # get median values:
    minus = (100-68.27)/2.0
    plus = 100-(100-68.27)/2.0
    med1 = np.nanmedian(tab['RatioMedBlue'])
    med2 = np.nanmedian(tab['RatioMedBlue2'])
    p1_plus = np.nanpercentile(tab['RatioMedBlue'],plus)
    p1_minus = np.nanpercentile(tab['RatioMedBlue'],minus)
    p1 = (p1_plus-p1_minus)/2.0
    p2_plus = np.nanpercentile(tab['RatioMedBlue2'],plus)
    p2_minus = np.nanpercentile(tab['RatioMedBlue2'],minus)
    p2 = (p2_plus-p2_minus)/2.0
    ax2.axvline(med1,linestyle='-',color='k')
    ax2.axvline(med2,linestyle=':',color='k')
    ax2.set(xlabel='SAMI/SDSS flux (blue)',ylabel='Normalized count')
    ax2.legend(prop={'size':10},ncol=1,loc='upper right')
    
    print('new data median: ',med1,p1)
    print('old data median: ',med2,p2)
    
    py.savefig(pdf, format='pdf')        
    pdf.close()

    # set up PDF 
    pdf = PdfPages('dr3_sami_sdss_fluxratiored.pdf')

    # set up plotting:
    fig3 = py.figure(3)
    ax3 = fig3.add_subplot(1,1,1)
    ax3.hist(tab['RatioMedRed'],bins=50,range=(0.0,2.0),histtype='step',color='r',density=True,label='SAMI DR3')
    ax3.hist(tab['RatioMedRed2'],bins=50,range=(0.0,2.0),histtype='step',color='r',linestyle=':',density=True,label='SAMI DR2')

    # get median values:
    minus = (100-68.27)/2.0
    plus = 100-(100-68.27)/2.0
    med1 = np.nanmedian(tab['RatioMedRed'])
    med2 = np.nanmedian(tab['RatioMedRed2'])
    p1_plus = np.nanpercentile(tab['RatioMedRed'],plus)
    p1_minus = np.nanpercentile(tab['RatioMedRed'],minus)
    p1 = (p1_plus-p1_minus)/2.0
    p2_plus = np.nanpercentile(tab['RatioMedRed2'],plus)
    p2_minus = np.nanpercentile(tab['RatioMedRed2'],minus)
    p2 = (p2_plus-p2_minus)/2.0
    ax3.axvline(med1,linestyle='-',color='k')
    ax3.axvline(med2,linestyle=':',color='k')
    ax3.set(xlabel='SAMI/SDSS flux (red)',ylabel='Normalized count')
    ax3.legend(prop={'size':10},ncol=1,loc='upper right')
    
    print('new data median: ',med1,p1)
    print('old data median: ',med2,p2)
    
    py.savefig(pdf, format='pdf')        
    pdf.close()

    # set up PDF 
    pdf = PdfPages('dr3_sami_sdss_flux_seeing.pdf')

    # set up plotting:
    fig4 = py.figure(4)
    ax4 = fig4.add_subplot(1,1,1)
    ax4.plot(tab['SAMIPSFFWHM'],tab['RatioMed'],'.',color='b')

    ax4.set(xlabel='SAMI cube PSF FWHM [arcsec]',ylabel='SAMI/SDSS flux ratio',xlim=[1.0,4.0],ylim=[0.5,2.0])
    ax4.legend(prop={'size':10},ncol=1,loc='upper right')
    ax4.axhline(1.0,linestyle='-',color='k')
    ax4.text(0.04, 0.93,'b)',horizontalalignment='left',verticalalignment='center',transform = ax4.transAxes)
    
    med_seeing_sdss = np.nanmedian(tab['SDSSPSFFWHM'])
    med_seeing_sami = np.nanmedian(tab['SAMIPSFFWHM'])

    print('median SDSS seeing:',med_seeing_sdss)
    print('median SAMI seeing:',med_seeing_sami)

    good = np.where(np.isfinite(tab['RatioMed']))
    x = tab['SAMIPSFFWHM'][good]
    y = tab['RatioMed'][good]
    bin_med, bin_edges, binnumber = stats.binned_statistic(x,y,statistic='mean', bins=40,range=[0.0,4.0])
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_cent = bin_edges[1:] - bin_width/2

    ax4.plot(bin_cent,bin_med,'o',color='r')

    good = np.where((bin_cent > 1.3) & (bin_cent < 3.5))
    print(bin_med[good])
    print(bin_cent[good])
    z = np.polyfit(bin_cent[good],bin_med[good],1)
    p = np.poly1d(z)
    fit = p(bin_cent) 
    print(z)

    ratio_corrected = tab['RatioMed'] - p(tab['SAMIPSFFWHM'])
    ratio_corrected_2 = tab['RatioMed2'] - p(tab['SAMIPSFFWHM'])

    med,pwidth = med_percentile_err(ratio_corrected)
    med_2,pwidth_2 = med_percentile_err(ratio_corrected_2)
    print('median and width corrected ratio (DR3)',med,pwidth) 
    print('median and width corrected ratio (DR2)',med_2,pwidth_2) 
    
    
    ax4.plot(bin_cent,fit,'-',color='r')
    
    py.savefig(pdf, format='pdf')        
    pdf.close()
    
    # now start with colours:
    # set up PDF 
    pdf = PdfPages('dr3_sami_sdss_4000_5450.pdf')
    
    fig5 = py.figure(5)
    ax5 = fig5.add_subplot(1,1,1)
    
    ratio4000_5500 = tab['Ratio4000']/tab['Ratio5450']
    ratio4000_5500_2 = tab['Ratio40002']/tab['Ratio54502']
    ax5.hist(ratio4000_5500,bins=50,range=(0.0,2.0),histtype='step',color='b',density=True,label='SAMI DR3')
    ax5.hist(ratio4000_5500_2,bins=50,range=(0.0,2.0),histtype='step',color='b',linestyle=':',density=True,label='SAMI DR2')
    ax5.set(xlabel='R(4000\AA)/R(5450\AA)',ylabel='Normalized count',xlim=[0.2,1.8])
    ax5.legend(prop={'size':10},ncol=1,loc='upper right')
    ax5.text(0.04, 0.93,'a)',horizontalalignment='left',verticalalignment='center',transform = ax5.transAxes)

    med,pwidth = med_percentile_err(ratio4000_5500)
    med_2,pwidth_2 = med_percentile_err(ratio4000_5500_2)

    ax5.axvline(med,linestyle='-',color='b')
    ax5.axvline(med_2,linestyle=':',color='b')

    print('med,pwidth for 4000/5450 (DR3):',med,pwidth)
    print('med,pwidth for 4000/5450 (DR2):',med_2,pwidth_2)
    
    py.savefig(pdf, format='pdf')        
    pdf.close()

    # set up PDF 
    pdf = PdfPages('dr3_sami_sdss_5450_7000.pdf')
    
    fig6 = py.figure(6)
    ax6 = fig6.add_subplot(1,1,1)
    
    ratio = tab['Ratio5450']/tab['Ratio7000']
    ratio_2 = tab['Ratio54502']/tab['Ratio70002']
    ax6.hist(ratio,bins=50,range=(0.0,2.0),histtype='step',color='r',density=True,label='SAMI DR3')
    ax6.hist(ratio_2,bins=50,range=(0.0,2.0),histtype='step',color='r',linestyle=':',density=True,label='SAMI DR2')
    ax6.set(xlabel='R(5450\AA)/R(7000\AA)',ylabel='Normalized count',xlim=[0.2,1.8])
    ax6.legend(prop={'size':10},ncol=1,loc='upper right')
    ax6.text(0.04, 0.93,'b)',horizontalalignment='left',verticalalignment='center',transform = ax6.transAxes)

    med,pwidth = med_percentile_err(ratio)
    med_2,pwidth_2 = med_percentile_err(ratio_2)

    ax6.axvline(med,linestyle='-',color='r')
    ax6.axvline(med_2,linestyle=':',color='r')
    
    print('med,pwidth for 5450/7000 (DR3):',med,pwidth)
    print('med,pwidth for 5450/7000 (DR2):',med_2,pwidth_2)
    
    py.savefig(pdf, format='pdf')        
    pdf.close()

    # plot location of galaxies with offsets:
    vmin=0.6
    vmax=1.4
    # set up PDF 
    pdf = PdfPages('dr3_blueratio_radec.pdf')
    
    fig7 = py.figure(7)
    ax7_1 = fig7.add_subplot(3,1,1)
    cax7_1 = ax7_1.scatter(tab['RA'],tab['DEC'],c=ratio4000_5500,marker='o',vmin=vmin,vmax=vmax)
    cbar7_1 = fig7.colorbar(cax7_1,spacing='proportional', format='%5.1f')
    ax7_1.set(xlim=[129.0,141.0],ylim=[-1.0,3.0])

    ax7_2 = fig7.add_subplot(3,1,2)
    cax7_2 = ax7_2.scatter(tab['RA'],tab['DEC'],c=ratio4000_5500,marker='o',vmin=vmin,vmax=vmax)
    cbar7_2 = fig7.colorbar(cax7_2,spacing='proportional', format='%5.1f')
    ax7_2.set(xlim=[174.0,186.0],ylim=[-2.0,2.0])
        
    ax7_3 = fig7.add_subplot(3,1,3)
    cax7_3 = ax7_3.scatter(tab['RA'],tab['DEC'],c=ratio4000_5500,marker='o',vmin=vmin,vmax=vmax)
    cbar7_3 = fig7.colorbar(cax7_3,spacing='proportional', format='%5.1f')
    ax7_3.set(xlim=[211.5,223.5],ylim=[-2.0,2.0])

    
   # get unique SDSS plates:
    unique_pnames, uidx_list = np.unique(tab['SDSSPlateName'],return_index=True)
    print('unique SDSS plate names:')
    print(unique_pnames)

    nup = np.size(unique_pnames)
    pratio = np.zeros(nup)
    psig = np.zeros(nup)
    pratio_bad = np.zeros(nup)
    psig_bad = np.zeros(nup)
    pfld_bad = np.zeros(nup)
    pnum = np.zeros(nup)
    pfld = np.zeros(nup)

    # get the median 4000/5500 ratio :
    med,pwidth = med_percentile_err(ratio4000_5500)
    # set up array of flags:
    igood = np.zeros(np.size(tab['SDSSPlateName']),dtype=bool)
    # loop through unique plate names:
    
    fnp = 0
    fnp_bad = 0
    for uidx in uidx_list:
        fname = tab['SDSSPlateName'][uidx]
        idx = np.where(tab['SDSSPlateName']==fname)
        nn =  np.size(ratio4000_5500[idx])
        if (nn < 10):
            continue
        pratio[fnp] = np.nanmedian(ratio4000_5500[idx])
        psig[fnp] = np.nanstd(ratio4000_5500[idx])/np.sqrt(nn)
        pnum[fnp] = nn
        pfld[fnp] = fnp
        if (abs(pratio[fnp]-med) > psig[fnp]*3):
            igood[idx] = False
            pratio_bad[fnp_bad] = pratio[fnp]
            psig_bad[fnp_bad] = psig[fnp]
            pfld_bad[fnp_bad] = pfld[fnp]
            fnp_bad = fnp_bad + 1
        else:
            igood[idx] = True
            
        print(fname,pratio[fnp],psig[fnp],pnum[fnp])
        # plot a circle for the fields:
        circle1 = py.Circle((tab['RAPLATESDSS'][uidx],tab['DECPLATESDSS'][uidx]), 1.5, color='r',fill=False)
        circle2 = py.Circle((tab['RAPLATESDSS'][uidx],tab['DECPLATESDSS'][uidx]), 1.5, color='r',fill=False)
        circle3 = py.Circle((tab['RAPLATESDSS'][uidx],tab['DECPLATESDSS'][uidx]), 1.5, color='r',fill=False)
        ax7_1.add_artist(circle1)
        ax7_2.add_artist(circle2)
        ax7_3.add_artist(circle3)
        
        fnp = fnp+1
        

    py.savefig(pdf, format='pdf')        
    pdf.close()
    
    # get the median 4000/5500 ratio after removing bad fields:
    idx = np.where(igood)
    med_nobad,pwidth_nobad = med_percentile_err(ratio4000_5500[idx])

    print('median and percentile width:',med,pwidth)
    print('median and percentile width (no bad):',med_nobad,pwidth_nobad)
    
    fig8 = py.figure(8)
    ax8 = fig8.add_subplot(1,1,1)
    ax8.errorbar(pfld[0:fnp],pratio[0:fnp],yerr=psig[0:fnp],marker='o',linestyle='',color='b')
    ax8.errorbar(pfld_bad[0:fnp_bad],pratio_bad[0:fnp_bad],yerr=psig_bad[0:fnp_bad],marker='o',linestyle='',color='r')
    ax8.axhline(med,linestyle=':',color='b')
    ax8.axhline(1.0,linestyle=':',color='k')

    

##########################################################
# plot flux offset from SDSS imaging
#
def sdss_sami_flux_hist():

    # configure formatting:
    setup_plots()

    pdf = PdfPages('dr3_sdss_sami_flux_hist.pdf')

    dr3file = '/Users/scroom/data/sami/dr3/wcs_flux_tests/WCS_crval_crosscorr_dr3.fits'
    dr2file = '/Users/scroom/data/sami/dr3/wcs_flux_tests/WCS_crval_crosscorr_dr2.fits'
    
    tab_dr2 = table.Table.read(dr2file)
    tab_dr3 = table.Table.read(dr3file)
    tab_dr2.info()

    ratio_dr2 = tab_dr2['SAMI_flux_sum']/tab_dr2['SDSS_flux_sum']
    ratio_dr3 = tab_dr3['SAMI_flux_sum']/tab_dr3['SDSS_flux_sum']

    idx_dr2 = np.where(tab_dr2['SAMI_flux_sum'] > 100)
    idx_dr3 = np.where(tab_dr3['SAMI_flux_sum'] > 100)
    
    # start plotting:
    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(1,1,1)
    ax1.hist(ratio_dr2[idx_dr2], bins=100, range=(0.0,2.0),histtype='step',color='k',density=True,linestyle=':')
    ax1.hist(ratio_dr3[idx_dr3], bins=100, range=(0.2,2.0),histtype='step',color='k',density=True,linestyle='-')
    ax1.set(xlabel='SAMI/SDSS 8 arcsec diameter flux ratio',ylabel='Normalized count',xlim=[0.4,1.6])

    med_dr2 = np.nanmedian(ratio_dr2[idx_dr2])
    med_dr3 = np.nanmedian(ratio_dr3[idx_dr3])

    print('DR2 median:',med_dr2)
    print('DR3 median:',med_dr3)
    
    ax1.axvline(med_dr2,linestyle=':',color='k',label='DR2')
    ax1.axvline(med_dr3,linestyle='-',color='k',label='DR3')
    ax1.legend(loc='upper right')
    
    py.savefig(pdf, format='pdf')        
    pdf.close()
    
##########################################################
#
def med_percentile_err(dat):

    minus = (100-68.27)/2.0
    plus = 100-(100-68.27)/2.0
    med1 = np.nanmedian(dat)
    n = np.size(dat)
    p1_plus = np.nanpercentile(dat,plus)
    p1_minus = np.nanpercentile(dat,minus)
    p1 = (p1_plus-p1_minus)/2.0

    return med1,p1
        
############################################################
# plot angular distribution of observed sources for GAMA
#
def plot_ang_GAMA():

    # configure formatting:
    setup_plots()
    py.rcParams.update({'font.size': 8})


    # input cat:
    incatfile = '/Users/scroom/data/sami/dr3/cats/all_cats_dr3/InputCatGAMADR3.fits'
    incat = table.Table.read(incatfile)
    incat.info()

    # observed cat:
    obscatfile = '/Users/scroom/data/sami/dr3/cats/all_cats_dr3/CubeObs.fits'
    obscat = table.Table.read(obscatfile)
    obscat.info()
    obscat_unique = table.unique(obscat, keys='CATID')

    # read in GAMA redshift catalogue:
    gamazcatfile = '/Users/scroom/data/sami/gama/DistancesFramesv08.fits'
    gamazcat = table.Table.read(gamazcatfile)
    
    # join table:
    obsall = table.join(incat,obscat_unique,keys='CATID',join_type='left')
    obsall.info()

    #obsall.write('test.fits', overwrite=True)  

    # plot regions:
    pdf = PdfPages('dr3_GAMA_ang.pdf')

    fig1 = py.figure(1)
    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1])

    glowz = np.where(gamazcat['Z_HELIO']<0.115)
    ax1 = fig1.add_subplot(gs[0,0:1])
    prim_all = np.where((obsall['SURV_SAMI']==8) & ((obsall['BAD_CLASS']==0) | (obsall['BAD_CLASS']==5) | (obsall['BAD_CLASS']==8))) 
    sec_all = np.where((obsall['SURV_SAMI']<8) & ((obsall['BAD_CLASS']==0) | (obsall['BAD_CLASS']==5) | (obsall['BAD_CLASS']==8))) 
    prim_obs = np.where((obsall['SURV_SAMI']==8) & ((obsall['BAD_CLASS']==0) | (obsall['BAD_CLASS']==5) | (obsall['BAD_CLASS']==8)) & (obsall['CATSOURCE']==1))
    sec_obs = np.where((obsall['SURV_SAMI']<8) & ((obsall['BAD_CLASS']==0) | (obsall['BAD_CLASS']==5) | (obsall['BAD_CLASS']==8)) & (obsall['CATSOURCE']==1))
    ax1.plot(gamazcat['RA'][glowz],gamazcat['DEC'][glowz],',',color='k',zorder=1.0)
    ax1.plot(obsall['RA_OBJ'][prim_all],obsall['DEC_OBJ'][prim_all],'o',color='b',markersize=2)
    ax1.plot(obsall['RA_OBJ'][prim_obs],obsall['DEC_OBJ'][prim_obs],'o',color='r',markersize=2)
    ax1.set_aspect('equal')
    ax1.set(xlim=[129.0,141.0],ylim=[-1.0,3.0],xlabel='RA (deg)',ylabel='Dec (deg)')

    py.savefig(pdf, format='pdf')        
    pdf.close()

    
    ax2 = fig1.add_subplot(gs[0,2:3])
    ax2.plot(gamazcat['RA'][glowz],gamazcat['DEC'][glowz],',',color='k',zorder=1.0)
    ax2.plot(obsall['RA_OBJ'][prim_all],obsall['DEC_OBJ'][prim_all],'o',color='b',markersize=2)
    ax2.plot(obsall['RA_OBJ'][prim_obs],obsall['DEC_OBJ'][prim_obs],'o',color='r',markersize=2)
    ax2.set_aspect('equal')
    ax2.set(xlim=[174.0,186.0],ylim=[-2.0,2.0],xlabel='RA (deg)')
        
    ax3 = fig1.add_subplot(gs[1:,:])
    pos1 = ax3.get_position()
    print(pos1)
    pos2 = [pos1.x0, pos1.y0,  pos1.width, pos1.height]
    print(pos2)
    ax3.set_position(pos2) # set a new position
    ax3.plot(gamazcat['RA'][glowz],gamazcat['DEC'][glowz],',',color='k',zorder=1.0)
    ax3.plot(obsall['RA_OBJ'][prim_all],obsall['DEC_OBJ'][prim_all],'o',color='b',markersize=2)
    ax3.plot(obsall['RA_OBJ'][prim_obs],obsall['DEC_OBJ'][prim_obs],'o',color='r',markersize=2)
    ax3.set_aspect('equal')
    ax3.set(xlim=[211.5,223.5],ylim=[-2.0,2.0],xlabel='RA (deg)',ylabel='Dec (deg)')




    #####################################
    pdf = PdfPages('dr3_GAMA_wedge.pdf')

    fig2 = py.figure(2)

    ra_min = 129.0
    ra_max = 223.5
    
    ax22, aux_ax22 = setup_axes3(fig2, 111,0.0,0.1,ra_min,ra_max)

    
    aux_ax22.plot(gamazcat['RA'],gamazcat['Z_HELIO'],',',color='k',zorder=1.0)
    aux_ax22.plot(obsall['RA_OBJ'][prim_all],obsall['z_spec'][prim_all],'o',color='b',markersize=2,zorder=1.1)
    aux_ax22.plot(obsall['RA_OBJ'][prim_obs],obsall['z_spec'][prim_obs],'o',color='r',markersize=2,zorder=1.2)

    
    py.savefig(pdf, format='pdf')        
    pdf.close()

    ########################################
    
    pdf = PdfPages('dr3_GAMA_wedge3.pdf')
    fig3 = py.figure(3)


    n = 3
    ra_mins = [129.0,174.0,211.5]
    ra_maxs = [141.0,186.0,223.5]
    rec = [311,312,313]
    labels = ['GAMA 09h','GAMA 12h','GAMA 15h']
    for i in range(n):
        ra_min = ra_mins[i]
        ra_max = ra_maxs[i]
        rot_deg = -1.0*(ra_min+ra_max)/2.0
        ax31, aux_ax31 = setup_axes3(fig3, rec[i],0.0,0.115,ra_min,ra_max,rot_deg=rot_deg)
        print(i,ra_min,ra_max,rot_deg)

        ax31.text(0.08, 0.7, labels[i], horizontalalignment='center',verticalalignment='center', transform=ax31.transAxes)
        
        aux_ax31.plot(gamazcat['RA'],gamazcat['Z_HELIO'],',',color='0.5',zorder=1.0)
        aux_ax31.plot(obsall['RA_OBJ'][sec_all],obsall['z_spec'][sec_all],'o',color='cyan',markersize=2,zorder=1.1)
        aux_ax31.plot(obsall['RA_OBJ'][prim_all],obsall['z_spec'][prim_all],'o',color='b',markersize=2,zorder=1.2)
        aux_ax31.plot(obsall['RA_OBJ'][sec_obs],obsall['z_spec'][sec_obs],'o',color='magenta',markersize=2,zorder=1.3)
        aux_ax31.plot(obsall['RA_OBJ'][prim_obs],obsall['z_spec'][prim_obs],'o',color='r',markersize=2,zorder=1.4)


    # place custom legend:
    axf = fig3.add_subplot(1,1,1)
    axf.set_axis_off()
    legend_elements = [Line2D([0], [0], marker='o',color='r', linestyle='None',markersize=4,label='SAMI primary observed'),
                        Line2D([0], [0], marker='o',color='b', linestyle='None', markersize=4,label='SAMI primary unobserved'),
                        Line2D([0], [0], marker='o',color='magenta', linestyle='None',markersize=4,label='SAMI secondary observed'),
                        Line2D([0], [0], marker='o',color='cyan', linestyle='None', markersize=4,label='SAMI secondary unobserved'),
                        Line2D([0], [0], marker='o',color='0.5', linestyle='None', markersize=1,label='GAMA sample')
                        ]

    axf.legend(handles=legend_elements,loc=(0.0,0.59),labelspacing=0.4)


        
    py.savefig(pdf, format='pdf')        
    pdf.close()

####################################################
# plot distribution of points in mass/env
#
def mass_env():

    # configure formatting:
    setup_plots()
    # this is the figure is across 2 cols, so not scaled down:
    py.rcParams.update({'font.size': 8})


    # input cat:
    catfile = '/Users/scroom/data/sami/disk_fading/merged_kin_conc_sfr_v0.12.fits'
    cat = table.Table.read(catfile)
    cat.info()

    # plot mass/enve disk:
    pdf = PdfPages('dr3_mass_env.pdf')
    fig1 = py.figure(1)

    ax1 = fig1.add_subplot(1,1,1)
    logden = np.log10(cat['ENV_SURFDENS'])
    good = np.where(np.isfinite(cat['LAMBDAR_RE']) & np.isfinite(logden))
    #p1 = ax1.scatter(cat['LMSTAR'][good],logden[good],cmap='rainbow')
    p1 = ax1.scatter(cat['LMSTAR'][good],logden[good],c=cat['LAMBDAR_RE'][good],cmap='rainbow')
    ax1.set(xlabel='log(M$^*$/M$_\odot$)',ylabel='log($\Sigma_5$)')
    cbar1 = fig1.colorbar(p1, ax=ax1)
    cbar1.set_label('$\lambda_{R_e}$', rotation=90)

#################################################################
# plot various SAMI maps in the mass-env plane
def mass_env_maps(dogrid=True,maptype=0,fluxsnlim=4.0,logim=True,clean=False):
    
    # configure formatting:
    setup_plots()
    # this figure is across 2 cols, so not scaled down:
    py.rcParams.update({'font.size': 8})

    # read input catalogues:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/InputCatGAMADR3.fits'
    cat_gama = table.Table.read(catfile)
    
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/InputCatClustersDR3.fits'
    cat_clusters = table.Table.read(catfile)

    # merge inout cats together:
    cat_gamacluster = table.vstack([cat_gama,cat_clusters])

    # read in environments:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/DensityCatDR3.fits'
    cat_env = table.Table.read(catfile)
    
    cat_allenv = table.join(cat_gamacluster,cat_env, keys='CATID',join_type='left')

    # read in obs cat:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/CubeObs.fits'
    cat_obs = table.Table.read(catfile)
    
    # generate a mask for the ISBEST flag, to only select the best cube for
    # each object:
    mask = cat_obs['ISBEST'] == True

    # do a join with the masked CubeObs file:
    cat_allenvobs = table.join(cat_allenv,cat_obs[mask], keys='CATID',join_type='left')

    # make new table of only observed objects:

    mask = cat_allenvobs['ISBEST'] == True    
    cat_obsonly = cat_allenvobs[mask]
    cat_obsonly.info()
    
    # read in stellar kinematics cat:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/samiDR3Stelkin.fits'
    cat_kin = table.Table.read(catfile)
    
    # read in gas kinematic PA cat:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/samiDR3gaskinPA.fits'
    cat_gaskin = table.Table.read(catfile)

    # get rid of leading white spaces in cat_obsonly and cat_kin for CUBEIDPUB:
    print(len(cat_obsonly))
    for i in range(len(cat_obsonly)):
        cat_obsonly['CUBEIDPUB'][i] = cat_obsonly['CUBEIDPUB'][i].strip()

    for i in range(len(cat_kin)):
        cat_kin['CUBEIDPUB'][i] = cat_kin['CUBEIDPUB'][i].strip()
        
    for i in range(len(cat_gaskin)):
        cat_gaskin['CUBEIDPUB'][i] = cat_gaskin['CUBEIDPUB'][i].strip()
        
    # match kinematic data to the rest:
    cat_allenvobskin = table.join(cat_obsonly,cat_kin, keys='CUBEIDPUB',join_type='left')
    
    # match gas kinematic data to the rest:
    cat_allenvobskin_gas = table.join(cat_allenvobskin,cat_gaskin, keys='CUBEIDPUB',join_type='left')

    # get the morphology catalogue:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/VisualMorphologyDR3.fits'
    cat_morph = table.Table.read(catfile)
    cat_morph.info()

    
    # match morph:
    #cat_allenvobskin_gas.rename_column('CATID_1','CATID')
    cat_allenvobskin_gasmorph = table.join(cat_allenvobskin_gas,cat_morph, keys='CATID',join_type='left')
    
    
    cat_allenvobskin_gasmorph.info()

    # write a fitsd file of the full table as a test:
    cat_allenvobskin_gasmorph.write('test_table.fits', overwrite=True)  

    # give a shorter name to the cat:
    cat_all = cat_allenvobskin_gasmorph
    ncat = len(cat_all)
    
    # catalogue data is now all merged into a single table, cat_allenv.  
    ########################

    # set up the plotting:
    pdf = PdfPages('dr3_mass_env_maps.pdf')
    fig1 = py.figure(1,clear=True)
    ax1 = fig1.add_subplot(1,1,1)

    ax1.set_facecolor('0.85')

    #limits for axes:
    xminl = 7.8
    xmaxl = 11.7
    yminl = -1.6
    ymaxl = 2.8

    # plots of plot:
    ax1.set(xlim=[xminl,xmaxl],ylim=[yminl,ymaxl])

    # adjust the grid to be slightly smaller than the axes:
    xmin = xminl+0.05
    xmax = xmaxl-0.05
    ymin = yminl+0.05
    ymax = ymaxl-0.05

    # steps for grid to set location of maps:
    xstep = 0.045
    ystep = 0.07
    maxdiff=10
    #maxdiff=100
    doff = np.zeros(10000)

    # set up locations array:
    xsize = int((xmax-xmin)/xstep)
    ysize = int((ymax-ymin)/ystep)
    xvec = np.arange(xsize+1)*xstep + xmin
    yvec = np.arange(ysize+1)*ystep + ymin
    print('size of locations array:',xsize+1,' x ',ysize+1)
    locarr = np.zeros((xsize+1,ysize+1))

    # define some useful values and masks before the main plotting loop:
    logden = np.log10(cat_all['SurfaceDensity'])
    goodlam = np.where(np.isfinite(cat_all['LAMBDAR_RE']) & np.isfinite(logden))
    goodstelflag = np.zeros(ncat)
    goodstelflag = np.where((np.isfinite(cat_all['LAMBDAR_RE'])),1,0)
    #p1 = ax1.scatter(cat['LMSTAR'][good],logden[good],cmap='rainbow')
    #p1 = ax1.scatter(cat['LMSTAR'][good],logden[good],c=cat['LAMBDAR_RE'][good],cmap='rainbow')
    #ax1.set(xlabel='log(M$^*$/M$_\odot$)',ylabel='log($\Sigma_5$)')
    #cbar1 = fig1.colorbar(p1, ax=ax1)
    #cbar1.set_label('$\lambda_{R_e}$', rotation=90)

    # main plotting loop:
    noff = 0
    nskip = 0
    for i in range(ncat):
    #for i in range(200):

        # get the actual value:
        xval = cat_all['Mstar'][i]
        yval = logden[i]

        # skip if not a good value:
        if (not np.isfinite(yval)):
            continue
        
        # if we are going to move objects to a grid, do this here:
        if (dogrid):
            # get the index of the closest and define this as the best to
            # start with:
            xmatch = int((xval - xmin)/xstep)
            ymatch = int((yval - ymin)/ystep)
            xbest = xmatch
            ybest = ymatch
            # Check to see if the point is within the allowed range:
            if ((xmatch > xsize) | (ymatch > ysize) | (xmatch < 0) | (ymatch < 0)):
                continue
            # next check to see if that bin is taken already.  If it is
            # then look for the next closest that is not taken:
            if (locarr[xmatch,ymatch] == 0):
                xmatch2 = xmatch
                ymatch2 = ymatch
            else:
                mindiff = 1.0e10
                for ix in range(int(xsize)):
                    for iy in range(int(ysize)):
                        diff = np.sqrt((xmatch - ix)**2 + (ymatch-iy)**2)
             #           print ix,iy,diff
                        
                        if ((diff < mindiff) & (locarr[ix,iy] == 0)):
                            mindiff = diff
                            xbest = ix
                            ybest = iy
                
                if (mindiff < maxdiff):
                    xmatch2 = xbest
                    ymatch2 = ybest
                    doff[noff] = mindiff
                    noff = noff + 1
                else:
                    print('No location to plot, skipping')
                    nskip = nskip + 1
                    continue
                
                
            locarr[xmatch2,ymatch2]=1
            xval2 = xvec[xmatch2]
            yval2 = yvec[ymatch2]
            #print('plotting location for galaxy ',cat_all['CATID_1'][i],': ',xval2,yval2)

            #line = str(object_name) + ' ' + str(xval2) + ' ' + str(yval2) + '\n'
            #fpos.write(line)
            #print 'test:',xval,yval,xmatch,ymatch,xval2,yval2,xbest,ybest
            # define location:
            x1 = xval2
            y1 = yval2
            xy = [xval2,yval2]

        else:
            x1 = xval
            y1 = yval
            xy = [xval,yval]


        # if we have a good position, then we have got to here, and we can read in the
        # map we will plot:
        repflag = cat_all['CUBEIDPUB'][i].strip()[-1]

        # get the PA and make this fixed for all cases:
        if (cat_all['PA_STELKIN_ERR'][i] < 20.0):
            ang = cat_all['PA_STELKIN'][i] + 45
        elif (cat_all['PA_GASKIN_ERR'][i] < 20.0):
            ang = cat_all['PA_GASKIN'][i] + 45
        else:
            ang = cat_all['PA'][i] + 45
            
            
        # stellar kinematics:
        if (maptype == 0):
            # to stellar for early types:
            if ((cat_all['TYPE'][i] < 1.75) & (cat_all['TYPE'][i] > -1.0)):
                mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_stellar-velocity_default_two-moment_'+repflag+'.fits'
                # derive the velocity limit based on the TF relation, but only 0.7 of it, not the full value:
                vlim = 0.7*10.0**(0.31*cat_all['Mstar'][i]-0.93)
                zmin = -1.0*vlim
                zmax = vlim        
                # get the map data:
                imm = fits.getdata(mapfile)
                imm_er = fits.getdata(mapfile,ext=1)
                # get the medin velocity and change the scale to be around this:
                vmed = np.nanmedian(imm[imm_er<30])
                zmax=zmax+vmed
                zmin=zmin+vmed
                if (vmed > 50.0):
                    print('vmed:',vmed,mapfile)
                # clip the image at the min and max values:
                im_a = np.where((imm>zmax),zmax,imm)
                im_a = np.where((im_a<zmin),zmin,im_a)
                # remove pixels with large errors:
                im_a = np.where((imm_er>30.0),np.nan,im_a)
                # define colour map:
                cmap=py.cm.RdYlBu_r
            elif ((cat_all['TYPE'][i] > 1.75) & (cat_all['TYPE'][i] < 4.0)):
                mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_gas-velocity_default_1-comp_'+repflag+'.fits'
                # derive the velocity limit based on the TF relation:
                vlim = 0.7*10.0**(0.31*cat_all['Mstar'][i]-0.93)
                zmin = -1.0*vlim
                zmax = vlim        
                # get the map data:
                try:
                    imm_t = fits.getdata(mapfile)
                except:
                    print('FILE NOT FOUND: ',mapfile)
                    continue
                imm_er_t = fits.getdata(mapfile,ext=1)
                imm = imm_t[0,:,:]
                imm_er = imm_er_t[0,:,:]
                # get the medin velocity and change the scale to be around this:
                vmed = np.nanmedian(imm[imm_er<20])
                zmax=zmax+vmed
                zmin=zmin+vmed
                if (vmed > 50.0):
                    print('vmed:',vmed,mapfile)
                # clip the image at the min and max values:
                im_a = np.where((imm>zmax),zmax,imm)
                im_a = np.where((im_a<zmin),zmin,im_a)
                # remove pixels with large errors:
                im_a = np.where((imm_er>20.0),np.nan,im_a)

                # read in the Ha map, and use the S/N in Ha to aditionally clip the kinematics:
                mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_Halpha_default_1-comp_'+repflag+'.fits'
                # try reading the image, but if not found skip to the next:
                try:
                    imm_t = fits.getdata(mapfile)
                except:
                    print('FILE NOT FOUND: ',mapfile)
                    continue
                # Halpha maps are 2x50x50, so take the first element:
                imm = imm_t[0,:,:]
                imm_er_t = fits.getdata(mapfile,ext=1)
                imm_er = imm_er_t[0,:,:]
                hasn = imm/imm_er
                # set to Nan for low S/N:
                im_a = np.where((hasn > fluxsnlim),im_a,np.nan)
                
                # define colour map:
                cmap=py.cm.PiYG_r
            else:
                continue
                
        # gas kinematics:
        if (maptype == 2):
            mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_gas-velocity_default_1-comp_'+repflag+'.fits'
            # derive the velocity limit based on the TF relation:
            vlim = 10.0**(0.31*cat_all['Mstar'][i]-0.93)
            zmin = -1.0*vlim
            zmax = vlim        
            # get the map data:
            try:
                imm_t = fits.getdata(mapfile)
            except:
                print('FILE NOT FOUND: ',mapfile)
                continue
            imm_er_t = fits.getdata(mapfile,ext=1)
            imm = imm_t[0,:,:]
            imm_er = imm_er_t[0,:,:]
            
            # clip the image at the min and max values:
            im_a = np.where((imm>zmax),zmax,imm)
            im_a = np.where((im_a<zmin),zmin,im_a)
            # remove pixels with large errors:
            im_a = np.where((imm_er>10.0),np.nan,im_a)
            # define colour map:
            cmap=py.cm.PiYG_r
            
        # Halpha maps:
        elif (maptype == 1):
            mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_Halpha_default_1-comp_'+repflag+'.fits'
            # try reading the image, but if not found skip to the next:
            try:
                imm_t = fits.getdata(mapfile)
            except:
                print('FILE NOT FOUND: ',mapfile)
                continue
            # Halpha maps are 2x50x50, so take the first element:
            imm = imm_t[0,:,:]
            imm_er_t = fits.getdata(mapfile,ext=1)
            imm_er = imm_er_t[0,:,:]
            # set to Nan for low S/N:
            imm = np.where((imm/imm_er > fluxsnlim),imm,np.nan)
            if (logim):
                imm = np.log10(imm)
                zmax = np.nanmax(imm)
                zmin = zmax-2.0
                
            # clip the image at the min and max values:
            im_a = np.where((imm>zmax),zmax,imm)
            im_a = np.where((im_a<zmin),zmin,im_a)

            # define the angle to rotate the images
            #ang = cat_allenvobskin['PA'][i]+90
            
            # define colour map:
            cmap=py.cm.YlOrRd

        # [NII]/Halpha maps:
        elif (maptype == 3):
            mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_Halpha_default_1-comp_'+repflag+'.fits'
            # try reading the image, but if not found skip to the next:
            try:
                imm_t = fits.getdata(mapfile)
            except:
                print('FILE NOT FOUND: ',mapfile)
                continue
            # Halpha maps are 2x50x50, so take the first element:
            imm = imm_t[0,:,:]
            imm_er_t = fits.getdata(mapfile,ext=1)
            imm_er = imm_er_t[0,:,:]
            # get [NII] map:
            mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_NII6583_default_1-comp_'+repflag+'.fits'
            # try reading the image, but if not found skip to the next:
            try:
                imm2 = fits.getdata(mapfile)
            except:
                print('FILE NOT FOUND: ',mapfile)
                continue
            # Halpha maps are 2x50x50, so take the first element:
            imm2_er = fits.getdata(mapfile,ext=1)
             # define the ratio:
            ratio = np.where(((abs(imm/imm_er) > fluxsnlim) & (abs(imm2/imm2_er) > fluxsnlim)),imm2/imm,np.NaN) 
            if (logim):
                imm = np.log10(ratio)
                zmax = 0.4
                zmin = -1.2
                # if making the plot with the clean flag on, which is used for a smaller region,
                # reduce the range:
                if (clean):
                    zmin=-0.85
                
            # clip the image at the min and max values:
            im_a = np.where((imm>zmax),zmax,imm)
            im_a = np.where((im_a<zmin),zmin,im_a)

            # define the angle to rotate the images
            #ang = cat_allenvobskin['PA'][i]+90
            
            # define colour map:
            cmap=py.cm.rainbow
            
        # call function to read map and return an artist object:
        zoom=0.15
        
        art = map_to_artist(ax1,im_a,zoom,xy,zmin,zmax,cmap=cmap,ang=ang)
        ax1.add_artist(art)

        # if not a clean version, add in the zooms:

        if (not clean):
            # add label:
            #ax1.text(xy[0],xy[1],str(cat_all['CATID_1'][i]), fontsize=1)
            # add a zoom in for selected galaxies:
            #if (cat_all['CATID_1'][i] == 9016800074):
            #zoomgal = [9016800074,9008500205,388424,9388000001]
            zoomgal = [9016800074,100192,388424,9388000001]
            #zoomgal = [9016800074,9008500205,388424,84677]
            #zoomgal = [8485,8570,23593,24433]
            zoomgalx = [8.05,8.40,8.75,9.10]
            zoomcol =['cyan','k','magenta','red']
            nzoomgal = np.size(zoomgal)
            for iz in range(nzoomgal):
                if (cat_all['CATID_1'][i] == zoomgal[iz]):
                    print(cat_all['CATID_1'][i],cat_all['RA_OBJ'][i],cat_all['DEC_OBJ'][i])
                    zoomz = 0.8
                    xyz = [zoomgalx[iz],2.4]
                    art = map_to_artist(ax1,im_a,zoomz,xyz,zmin,zmax,cmap=cmap,ang=ang)
                    ax1.add_artist(art)
                    ax1.scatter(xyz[0],xyz[1], s=1000,linewidths=1.0,edgecolor=zoomcol[iz], facecolor="none",zorder=0)
                    ax1.scatter(xy[0],xy[1], s=40, linewidths=1.5,edgecolor=zoomcol[iz], facecolor="none",zorder=0)
                    ax1.text(xyz[0],xyz[1]+0.25,str(cat_all['CATID_1'][i]),horizontalalignment='center',fontsize=6)
            
        
    
    print('Number of galaxies skipped for plotting:',nskip)
    print('Number of galaxies offset for plotting:',noff)

    # look at the distribution of offsets
    print(doff)



    if (not clean):
        # add SAMI logo to the plot:
        #arr_lena = read_png("/Users/scroom/data/sami/lzifu/SAMI-Survey-Logo-small.png")
        arr_lena = py.imread("/Users/scroom/data/sami/lzifu/SAMI-Survey-Logo-small.png")

        #ax1.imshow(arr_lena,aspect='auto', extent=(8.0,8.5,2.5,2.8), zorder=-1)
        # upper left:
        imagebox = OffsetImage(arr_lena,zoom=0.2)
        ab = AnnotationBbox(imagebox,(8.15,1.9),
                        box_alignment=(0.5, 0.5),
                        frameon=False,
                        boxcoords="data",
                        pad=0.3)
        #ab = AnchoredOffsetbox(2,child=imagebox,frameon=False)
        ax1.add_artist(ab)

        ax1.set(xlabel='log(stellar mass/M$_\odot$)',ylabel='log(local density, $\\Sigma_5$/Mpc$^{-2}$)')

        # add colourbar axis, depending on plot:
        if (maptype == 0):
            cax = fig1.add_axes([0.92, 0.15, 0.01, 0.2])
            cb = mpl.colorbar.ColorbarBase(cax, orientation='vertical',
                                    cmap=py.cm.PiYG_r,norm=mpl.colors.Normalize(-1,1),)
                                    #ticks=['Vmin','Vmax'])
            cb.set_ticks([])
            cax.set_title('$v_{gas}$',fontsize=6)
        
            cax = fig1.add_axes([0.955, 0.15, 0.01, 0.2])
            cb = mpl.colorbar.ColorbarBase(cax, orientation='vertical',
                                    cmap=py.cm.RdYlBu_r,norm=mpl.colors.Normalize(-1,1),)
                                    #ticks=['Vmin','Vmax'])
            cb.set_ticks([])
            cax.set_title('$v_{stel}$',fontsize=6)

        if (maptype == 1):
            cax = fig1.add_axes([0.92, 0.15, 0.01, 0.2])
            cb = mpl.colorbar.ColorbarBase(cax, orientation='vertical',
                                    cmap=py.cm.YlOrRd,norm=mpl.colors.Normalize(0,-2),
                                    ticks=[0,-0.5,-1.0,-1.5,-2.0])
            cax.tick_params(labelsize=6)
            cax.set_title('log(H$\\alpha$)',fontsize=6)
        
        if (maptype == 3):
            cax = fig1.add_axes([0.92, 0.15, 0.01, 0.2])
            cb = mpl.colorbar.ColorbarBase(cax, orientation='vertical',
                                    cmap=py.cm.rainbow,norm=mpl.colors.Normalize(-1.2,0.4),
                                    ticks=[-1.2,-0.8,-0.4,0.0,0.4])
            cax.tick_params(labelsize=6)
            cax.set_title('log([NII]/H$\\alpha$)',fontsize=6)


        
    # note dpi=1000 to make sure final image has good resolution.

    # Message for Lexi about format for School pf physics picturesL
    # Current print requirements are for 100cm wide prints.
    # This quality should be at least 11000 by 11000px across by 300dpi.
    # Note - Sometime the printer can rescan and improve quality of pieces at least 4000px across - so ok if you can only get around that resolution.

    #
            
    # define a bounding box to get just some of the image:
    #bbox = Bbox([[xminl,yminl], [xmaxl, ymaxl]])
    # 1 galaxy is about ~0.1 in the BBox units
    if (clean):

        # first get the default size:
        def_size = fig1.get_size_inches()
        print('default fig size in inches:',def_size)
        # 1m is 39.4 inches

        # these values work for pdf at 1000 dpi:
        bsize = 1.97 # this range gives 31x31 galaxies, so each galaxy is
                     # about 0.0635 in these units.
        bxmin = 3.39
        bymin = 2.03
        # move lower by 10 galaxies in y (env):
        bymin = bymin -0.635
        bbox = Bbox([[bxmin,bymin], [bxmin+bsize, bymin+bsize]])
        print('Bounding box:',bbox)
        py.savefig(pdf, format='pdf',dpi=1000,bbox_inches=bbox)        
        pdf.close()

        # need to adjust bbox values for high res png (not obvious why this is needed
        # it should not matter).
        #note at 5800 dpi, 1 pixel is 0.000172 inches
        # seem to need to set dpi before changed bbox...
        # this does not work - breaks the file generation:
        fig1.set_dpi(5800)
        #bymin = bymin
        bxmin = bxmin + 460*0.000172
        bymin=bymin + 600*0.000172
        bbox = Bbox([[bxmin,bymin], [bxmin+bsize, bymin+bsize]])

        py.savefig('dr3_mass_env_maps.png', format='png',dpi=5800,bbox_inches=bbox)        
    else:
        py.savefig(pdf, format='pdf',dpi=1000)        
        pdf.close()

    fig1.clear()
        
    fig2 = py.figure(2)
    ax2 = fig2.add_subplot(1,1,1)
    ax2.hist(doff[0:noff],bins=20,range=[0,20.0],histtype='step')
    fig2.clear()

    
#################################################################
# plot various SAMI maps stepping through each galaxy
def mass_env_maps_individual(maptype=0,fluxsnlim=3.0,logim=True):

    "plot maps for individual galaxies"
    
    # configure formatting:
    setup_plots()
    # this figure is across 2 cols, so not scaled down:
    py.rcParams.update({'font.size': 8})

    # read input catalogues:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/InputCatGAMADR3.fits'
    cat_gama = table.Table.read(catfile)
    
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/InputCatClustersDR3.fits'
    cat_clusters = table.Table.read(catfile)

    # merge inout cats together:
    cat_gamacluster = table.vstack([cat_gama,cat_clusters])

    # read in environments:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/DensityCatDR3.fits'
    cat_env = table.Table.read(catfile)
    
    cat_allenv = table.join(cat_gamacluster,cat_env, keys='CATID',join_type='left')

    # read in obs cat:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/CubeObs.fits'
    cat_obs = table.Table.read(catfile)
    
    # generate a mask for the ISBEST flag, to only select the best cube for
    # each object:
    mask = cat_obs['ISBEST'] == True

    # do a join with the masked CubeObs file:
    cat_allenvobs = table.join(cat_allenv,cat_obs[mask], keys='CATID',join_type='left')

    # make new table of only observed objects:

    mask = cat_allenvobs['ISBEST'] == True    
    cat_obsonly = cat_allenvobs[mask]
    cat_obsonly.info()
    
    # read in stellar kinematics cat:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/samiDR3Stelkin.fits'
    cat_kin = table.Table.read(catfile)
    
    # read in gas kinematic PA cat:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/samiDR3gaskinPA.fits'
    cat_gaskin = table.Table.read(catfile)

    # get rid of leading white spaces in cat_obsonly and cat_kin for CUBEIDPUB:
    print(len(cat_obsonly))
    for i in range(len(cat_obsonly)):
        cat_obsonly['CUBEIDPUB'][i] = cat_obsonly['CUBEIDPUB'][i].strip()

    for i in range(len(cat_kin)):
        cat_kin['CUBEIDPUB'][i] = cat_kin['CUBEIDPUB'][i].strip()
        
    for i in range(len(cat_gaskin)):
        cat_gaskin['CUBEIDPUB'][i] = cat_gaskin['CUBEIDPUB'][i].strip()
        
    # match kinematic data to the rest:
    cat_allenvobskin = table.join(cat_obsonly,cat_kin, keys='CUBEIDPUB',join_type='left')
    
    # match gas kinematic data to the rest:
    cat_allenvobskin_gas = table.join(cat_allenvobskin,cat_gaskin, keys='CUBEIDPUB',join_type='left')

    # get the morphology catalogue:
    catfile = '/Users/scroom/data/sami/dr3/cats/all_cats_DR3/VisualMorphologyDR3.fits'
    cat_morph = table.Table.read(catfile)
    cat_morph.info()

    
    # match morph:
    #cat_allenvobskin_gas.rename_column('CATID_1','CATID')
    cat_allenvobskin_gasmorph = table.join(cat_allenvobskin_gas,cat_morph, keys='CATID',join_type='left')
    
    
    cat_allenvobskin_gasmorph.info()

    # write a fitsd file of the full table as a test:
    cat_allenvobskin_gasmorph.write('test_table.fits', overwrite=True)  

    # give a shorter name to the cat:
    cat_all = cat_allenvobskin_gasmorph
    ncat = len(cat_all)
    
    # catalogue data is now all merged into a single table, cat_allenv.  
    ########################

    fig1 = py.figure(1)
    ax1 = fig1.add_subplot(2,2,1)
    ax2 = fig1.add_subplot(2,2,2)
    ax3 = fig1.add_subplot(2,2,3)
    ax4 = fig1.add_subplot(2,2,4)

    ax1.set_facecolor('0.8')

    # define some useful values and masks before the main plotting loop:
    logden = np.log10(cat_all['SurfaceDensity'])
    goodlam = np.where(np.isfinite(cat_all['LAMBDAR_RE']) & np.isfinite(logden))
    goodstelflag = np.zeros(ncat)
    goodstelflag = np.where((np.isfinite(cat_all['LAMBDAR_RE'])),1,0)
    #p1 = ax1.scatter(cat['LMSTAR'][good],logden[good],cmap='rainbow')
    #p1 = ax1.scatter(cat['LMSTAR'][good],logden[good],c=cat['LAMBDAR_RE'][good],cmap='rainbow')
    #ax1.set(xlabel='log(M$^*$/M$_\odot$)',ylabel='log($\Sigma_5$)')
    #cbar1 = fig1.colorbar(p1, ax=ax1)
    #cbar1.set_label('$\lambda_{R_e}$', rotation=90)

    # main plotting loop:
    noff = 0
    nskip = 0
    for i in range(ncat):
    #for i in range(100):

        # get the actual value:
        xval = cat_all['Mstar'][i]
        yval = logden[i]

        print(i,cat_all['CATID_1'][i])
        # check for particular regions of parameter space:
        #if ((xval < 9.0) or (xval > 9.5) or (yval > 0)):
        #    continue
        #if ((xval < 11.0) or (xval > 12.0) or (yval < 1.5) or (cat_all['CATID_1'][i] < 10000000)):
        #    continue
        if ((xval < 11.0) or (xval > 12.0) or (yval < 0.0) or (yval > 0.5)  or (cat_all['CATID_1'][i] != 41059)):
            continue
        #if ((xval < 11.5)):
        #    continue

        #if (cat_all['CATID_1'][i] != 567676):
        #    continue
        
        # skip if not a good value:
        if (not np.isfinite(yval)):
            continue
        
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        # if we have a good position, then we have got to here, and we can read in the
        # map we will plot:
        repflag = cat_all['CUBEIDPUB'][i].strip()[-1]

        # get the PA and make this fixed for all cases:
        if (cat_all['PA_STELKIN_ERR'][i] < 20.0):
            ang = cat_all['PA_STELKIN'][i] + 45
        elif (cat_all['PA_GASKIN_ERR'][i] < 20.0):
            ang = cat_all['PA_GASKIN'][i] + 45
        else:
            ang = cat_all['PA'][i] + 45
                        
        # stellar kinematics:
        # to stellar for early types:
        mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_stellar-velocity_default_two-moment_'+repflag+'.fits'
        # derive the velocity limit based on the TF relation:
        vlim = 10.0**(0.31*cat_all['Mstar'][i]-0.93)
        zmin = -1.0*vlim
        zmax = vlim        
        # get the map data:
        imm = fits.getdata(mapfile)
        imm_er = fits.getdata(mapfile,ext=1)
        # get the medin velocity and change the scale to be around this:
        vmed = np.nanmedian(imm[imm_er<30])
        zmax=zmax+vmed
        zmin=zmin+vmed
        print(vmed,zmin,zmax)
        if (vmed > 50.0):
            print('vmed:',vmed,mapfile)
        # clip the image at the min and max values:
        im_a = np.where((imm>zmax),zmax,imm)
        im_a = np.where((im_a<zmin),zmin,im_a)
        # remove pixels with large errors:
        # note vel error of 30 seems okay..
        im_a = np.where((imm_er>30.0),np.nan,im_a)
        # define colour map:
        cmap=py.cm.RdYlBu_r

        p1 = ax1.imshow(im_a,origin='lower',interpolation='nearest',cmap=cmap,vmin=zmin,vmax=zmax)
        cbar1 = fig1.colorbar(p1, ax=ax1)
        
        # Halpha maps:
        mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_Halpha_default_1-comp_'+repflag+'.fits'
        # try reading the image, but if not found skip to the next:
        try:
            imm_t = fits.getdata(mapfile)
        except:
            print('FILE NOT FOUND: ',mapfile)
            continue
        # Halpha maps are 2x50x50, so take the first element:
        imm = imm_t[0,:,:]
        imm_er_t = fits.getdata(mapfile,ext=1)
        imm_er = imm_er_t[0,:,:]
        # set to Nan for low S/N:
        hasn = np.copy(imm/imm_er)
        imm = np.where((imm/imm_er > fluxsnlim),imm,np.nan)
        if (logim):
            imm = np.log10(imm)
            zmax = np.nanmax(imm)
            zmin = zmax-3.0
                
        # clip the image at the min and max values:
        im_a = np.where((imm>zmax),zmax,imm)
        im_a = np.where((im_a<zmin),zmin,im_a)

        # define the angle to rotate the images
        #ang = cat_allenvobskin['PA'][i]+90
        
        # define colour map:
        cmap=py.cm.YlOrRd

        ax3.imshow(im_a,origin='lower',interpolation='nearest',cmap=cmap)
        
        # gas kinematics:
        mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_gas-velocity_default_1-comp_'+repflag+'.fits'
        # derive the velocity limit based on the TF relation:
        vlim = 10.0**(0.31*cat_all['Mstar'][i]-0.93)
        zmin = -1.0*vlim
        zmax = vlim        
        # get the map data:
        try:
            imm_t = fits.getdata(mapfile)
        except:
            print('FILE NOT FOUND: ',mapfile)
            continue
        imm_er_t = fits.getdata(mapfile,ext=1)
        imm = imm_t[0,:,:]
        imm_er = imm_er_t[0,:,:]
        # get the medin velocity and change the scale to be around this:
        vmed = np.nanmedian(imm[imm_er<10])
        zmax=zmax+vmed
        zmin=zmin+vmed
        # clip the image at the min and max values:
        im_a = np.where((imm>zmax),zmax,imm)
        im_a = np.where((im_a<zmin),zmin,im_a)
        # remove pixels with large errors:
        im_a = np.where((imm_er>20.0),np.nan,im_a)
        # remove pixels with low S/N Ha:
        im_a = np.where((hasn > fluxsnlim),im_a,np.nan)
        # define colour map:
        cmap=py.cm.PiYG_r

        ax2.imshow(im_a,origin='lower',interpolation='nearest',cmap=cmap)

        
        # [NII]/Halpha maps:
        mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_Halpha_default_1-comp_'+repflag+'.fits'
        # try reading the image, but if not found skip to the next:
        try:
            imm_t = fits.getdata(mapfile)
        except:
            print('FILE NOT FOUND: ',mapfile)
            continue
        # Halpha maps are 2x50x50, so take the first element:
        imm = imm_t[0,:,:]
        imm_er_t = fits.getdata(mapfile,ext=1)
        imm_er = imm_er_t[0,:,:]
        # get [NII] map:
        mapfile = '/Users/scroom/data/sami/dr3/maps/'+str(cat_all['CATID_1'][i])+'_NII6583_default_1-comp_'+repflag+'.fits'
        # try reading the image, but if not found skip to the next:
        try:
            imm2 = fits.getdata(mapfile)
        except:
            print('FILE NOT FOUND: ',mapfile)
            continue
        # Halpha maps are 2x50x50, so take the first element:
        imm2_er = fits.getdata(mapfile,ext=1)
        # define the ratio:
        ratio = np.where(((abs(imm/imm_er) > fluxsnlim) & (abs(imm2/imm2_er) > fluxsnlim)),imm2/imm,np.NaN) 
        if (logim):
            imm = np.log10(ratio)
            zmax = 0.4
            zmin = -1.2
                
        # clip the image at the min and max values:
        im_a = np.where((imm>zmax),zmax,imm)
        im_a = np.where((im_a<zmin),zmin,im_a)

        # define the angle to rotate the images
        #ang = cat_allenvobskin['PA'][i]+90
            
        # define colour map:
        cmap=py.cm.rainbow
            
        # call function to read map and return an artist object:
        zoom=0.15

        ax4.imshow(im_a,origin='lower',interpolation='nearest',cmap=cmap)

        ax1.set(title=str(cat_all['CATID_1'][i]))
        ax2.set(title=str(xval)+' '+str(yval))

        py.draw()
        # pause for input if plotting all the spectra:
        print(i,cat_all['CATID_1'][i],ncat)
        yn = input('Continue? (y/n):')

        


    
####################################################
# convert map to artist object:
def map_to_artist(ax,im_a,zoom,xy,zmin,zmax,cmap=py.cm.RdYlBu,ang=0.0):

    # rotate the image of needed:
    if (ang != 0):
        im_b = ndimage.rotate(im_a, ang, reshape=False,order=0,cval=np.nan)
    else:
        im_b = im_a

    # convert to rgb image...
    # first scale between 0 and 1:
    im_c = (im_b - zmin) /(zmax-zmin)

    # then get the image with the correct colourmap:
    im_d = Image.fromarray(np.uint8(cmap(im_c)*255))
    
    # finally fix up the NaN values, so that the alpha (last slice) is zero. Note
    # the arder of indexing is different between array and image!!!!
    for i in range(50):
        for j in range(50):
            if (np.isnan(im_b[j,i])):
                im_d.putpixel((i,j),(0,0,0,0))
                
    # generate the image from the array:
    im = OffsetImage(im_d, zoom=zoom,origin='lower',interpolation='nearest',cmap=cmap)
    
    # place the image in an 
    ab = AnnotationBbox(im, xy,
#                        xybox=(-50., 50.),
                        #xycoords='data',
                        box_alignment=(0.5, 0.5),
                        #boxcoords = xy,
                        frameon=False,
                        boxcoords="data",
                        pad=0.3)
    
    return ab

####################
# a test plot:

def test_plot():
    delta = 0.25
  
    x = y = np.arange(-3.0, 3.0, delta)  
    X, Y = np.meshgrid(x, y)  
  
    Z1 = np.exp(-X**2 - Y**2)  
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)  
    Z = (Z1 - Z2)  
    
    transform = mtransforms.Affine2D().rotate_deg(30).scale(0.1,0.1)  
    fig, ax = py.subplots()  
        
    im = ax.imshow(Z, interpolation ='none',  
               origin ='lower',  
               clip_on = True)  

    print('im:',im)
    trans_data = transform + ax.transData  
    Artist.set_transform(im, trans_data)

    im2 = OffsetImage(Z,origin='lower',interpolation='nearest')
    print('im2:',im2)
    im2.set_transform(transform)
    print(im2.get_transform())
    ab = AnnotationBbox(im2, (10,10),
                        box_alignment=(0.5, 0.5),
                        frameon=True,
                        #boxcoords="data",
                        boxcoords = transform,
                        pad=0.3)

    print('ab:',ab)
    print(ab.get_transform())
    ab.set_transform(transform)
    print(ab.get_transform())
    ax.add_artist(ab)
    
    x1, x2, y1, y2 = im.get_extent()  
    ax.plot([x1, x2, x2, x1, x1],   
        [y1, y1, y2, y2, y1],  
        "ro-",  
        transform = trans_data)  
    
    #ax.set_xlim(-5, 5)  
    #ax.set_ylim(-4, 4)  
  
    py.title("Test", fontweight="bold") 
  
    py.show()

####################################################
# plot distribution of points in mass/env
#
def mass_env_wedge():

    # configure formatting:
    setup_plots()
    # this is the figure is across 2 cols, so not scaled down:
    py.rcParams.update({'font.size': 8})

    # input cat:
    catfile = '/Users/scroom/data/sami/disk_fading/merged_kin_conc_sfr_v0.12.fits'
    cat = table.Table.read(catfile)
    cat.info()

    # read in GAMA redshift catalogue:
    gamazcatfile = '/Users/scroom/data/sami/gama/DistancesFramesv08.fits'
    gamazcat = table.Table.read(gamazcatfile)
    
    # plot mass/enve disk:
    pdf = PdfPages('dr3_mass_env_wedge.pdf')
    fig1 = py.figure(1)

    # define good lambda_re
    good = np.where(np.isfinite(cat['LAMBDAR_RE']))

    n = 3
    ra_mins = [129.0,174.0,211.5]
    ra_maxs = [141.0,186.0,223.5]
    rec = [311,312,313]
    for i in range(n):
        ra_min = ra_mins[i]
        ra_max = ra_maxs[i]
        rot_deg = -1.0*(ra_min+ra_max)/2.0
        ax31, aux_ax31 = setup_axes3(fig1, rec[i],0.0,0.06,ra_min,ra_max,rot_deg=rot_deg)
        print(i,ra_min,ra_max,rot_deg)
    
        aux_ax31.plot(gamazcat['RA'],gamazcat['Z_HELIO'],',',color='k',zorder=1.0)
        p1 = aux_ax31.scatter(cat['RA_1'][good],cat['Z_SPEC'][good],c=cat['LAMBDAR_RE'][good],cmap='rainbow_r',s=4,zorder=1.1)
        #ux_ax31.plot(obsall['RA_1'][good],obsall['z_spec'][sec_all],'o',color='c',markersize=2,zorder=1.1)
        #aux_ax31.plot(obsall['RA_OBJ'][prim_all],obsall['z_spec'][prim_all],'o',color='b',markersize=2,zorder=1.2)
        #aux_ax31.plot(obsall['RA_OBJ'][sec_obs],obsall['z_spec'][sec_obs],'o',color='m',markersize=2,zorder=1.3)
        #aux_ax31.plot(obsall['RA_OBJ'][prim_obs],obsall['z_spec'][prim_obs],'o',color='r',markersize=2,zorder=1.4)

        # plot colourbar
        if (i == 0):      
            cbar1 = fig1.colorbar(p1, ax=aux_ax31,location='bottom',anchor=(0.0,1.0),fraction=0.1,pad=0.3)
            cbar1.set_label('$\lambda_{R_e}$')

    
    py.savefig(pdf, format='pdf')        
    pdf.close()


####################################################
#

def env_clusters():

    # configure formatting:
    setup_plots()
    # this is the figure is across 2 cols, so not scaled down:
    py.rcParams.update({'font.size': 8})

    # input cat:
    catfile = '/Users/scroom/data/sami/disk_fading/merged_kin_conc_sfr_v0.12.fits'
    cat = table.Table.read(catfile)
    cat.info()

    # get input cat:
    incatfile = '/Users/scroom/data/sami/dr3/cats/InputCat_Clusters/InputCat_Clusters.fits'
    incat = table.Table.read(incatfile)
    incat.info()
    
    # define good lambda_re
    good = np.where(np.isfinite(cat['LAMBDAR_RE']))
    
    # plot mass/enve disk:
    pdf = PdfPages('dr3_env_clusters.pdf')
    fig1 = py.figure(1,figsize=(4, 8))

    n = 8
    rac = [6.381, 10.460, 14.067, 18.740, 329.389, 336.977, 355.398, 356.895]
    decc = [-33.047, -9.303, -1.255, 0.431, -7.794, -30.575, -29.236, -28.125]
    name = ['EDCC0442','Abell0085','Abell0119','Abell0168','Abell2399','Abell3880','APMCC0917','Abell4038']
    size = [0.5,0.75,0.75,0.5,1.0,1.0,1.0,1.0]

    ax=[]
    for i in range(n):

        x1 = -1.0 * size[i]
        x2 = +1.0 * size[i]
        y1 = -1.0 * size[i]
        y2 = +1.0 * size[i]
        ax.append(fig1.add_subplot(4,2,i+1))
        delra = (cat['RA_1']-rac[i])*np.cos((cat['DEC_1']*np.pi/180))
        #delra_in = (incat['RA_OBJ']-rac[i])*np.cos((incat['DEC_OBJ']*np.pi/180))
        #ax[i].scatter(delra_in,incat['DEC_OBJ']-decc[i],c='k',s=1,zorder=1.1)
        ax[i].scatter(delra[good],cat['DEC_1'][good]-decc[i],c=cat['LAMBDAR_RE'][good],cmap='rainbow_r',s=4,zorder=1.2)
        #ax[i].scatter((cat['RA_1'][good]-rac[i])*np.cos(cat['DEC_1'][good]),cat['DEC_1'][good]-decc[i],c=cat['LAMBDAR_RE'][good],cmap='rainbow_r',s=4,zorder=1.1)
        ax[i].set(xlim=(x1,x2),ylim=(y1,y2),aspect='equal',title=name[i])

        #ax.append(fig1.add_subplot(4,4,8+i+1))
        #ax[i+].scatter(delra,cat['Z_SPEC'][good]-decc[i],c=cat['LAMBDAR_RE'][good],cmap='rainbow_r',s=4,zorder=1.1)
        #ax[i].scatter((cat['RA_1'][good]-rac[i])*np.cos(cat['DEC_1'][good]),cat['DEC_1'][good]-decc[i],c=cat['LAMBDAR_RE'][good],cmap='rainbow_r',s=4,zorder=1.1)
        #ax[i+8].set(xlim=(x1,x2),title=name[i])


        
    #ax[n-1].text(-1.4, -0.3,'dRA (deg)',horizontalalignment='center',verticalalignment='center',transform = ax[n-1].transAxes)
    #ax[n-1].text(-3.6, 1.1,'dDec (deg)',horizontalalignment='center',verticalalignment='center',transform = ax[n-1].transAxes,rotation=90)

    py.tight_layout()
    
#        text(0.5, 0.5, 'matplotlib', horizontalalignment='center',
#...      verticalalignment='center', transform=ax.transAxes)

        



    py.savefig(pdf, format='pdf')        
    pdf.close()
    


    
 
####################################################
def setup_axes3(fig, rect,min_rad,max_rad,ra_min,ra_max,rot_deg=270.0,ang_label='RA (deg)',rad_label='Redshift, z'):
    """
    Sometimes, things like axis_direction need to be adjusted.
    """

    # rotate a bit for better orientation
    tr_rotate = Affine2D().translate(rot_deg, 0)

    # scale degree to radians
    tr_scale = Affine2D().scale(np.pi/180., 1.)

    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    grid_locator1 = angle_helper.LocatorD(3)
    #tick_formatter1 = angle_helper.FormatterHMS()
    tick_formatter1 = None

    grid_locator2 = MaxNLocator(5)

    ra0 = ra_min
    ra1 = ra_max
    cz0 = min_rad
    cz1 = max_rad
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(ra0, ra1, cz0, cz1),
        grid_locator1=grid_locator1,
        grid_locator2=grid_locator2,
        tick_formatter1=tick_formatter1,
        tick_formatter2=None)

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    # adjust axis
    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["right"].set_axis_direction("top")

    ax1.axis["bottom"].set_visible(False)
    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")

    ax1.axis["left"].label.set_text(rad_label)
    ax1.axis["top"].label.set_text(ang_label)

    # create a parasite axes whose transData in RA, cz
    aux_ax = ax1.get_aux_axes(tr)

    aux_ax.patch = ax1.patch  # for aux_ax to have a clip path as in ax
    ax1.patch.zorder = 0.9  # but this has a side effect that the patch is
    # drawn twice, and possibly over some other
    # artists. So, we decrease the zorder a bit to
    # prevent this.

    # hide the first tick.  This does not work, not sure why....
    #xticks = aux_ax.xaxis.get_major_ticks()
    #xticks[0].set_visible(False)
    #yticks = aux_ax.yaxis.get_major_ticks()
    #yticks[0].set_visible(False)

    return ax1, aux_ax

#################################################
