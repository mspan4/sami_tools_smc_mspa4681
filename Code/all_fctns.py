import numpy as np
import astropy
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.table import join
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.coordinates import SkyCoord
from scipy.optimize import linear_sum_assignment


SAMI_Target_catalogues = ("InputCatGAMADR3.fits", "InputCatClustersDR3.fits", "InputCatFiller.fits")
SAMI_regions = {0: "GAMA", 1: "Clusters", 2: "Filler"}




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#--------------------------------------------------------- BPT FUNCTIONS --------------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# plot Kauffmann BPT line:
def k03_bpt_formula(xbpt):
    ybpt = 0.61/(xbpt-0.05) + 1.3 
    ybpt[xbpt>0.05] = -np.inf

    return ybpt

############################################################################
# plot Kewley 2001 BPT line:
def k01_bpt_formula(xbpt, metal='N II'):
    match metal:
        case 'N II':
            ybpt = 0.61/(xbpt-0.47) + 1.19
            ybpt[xbpt>0.47] = -np.inf

        case 'S II':
            ybpt = 0.72/(xbpt-0.31) + 1.30
            ybpt[xbpt>0.31] = -np.inf


        case 'O I':
            ybpt = 0.73/(xbpt+0.59) + 1.33
            ybpt[xbpt>-0.59] = -np.inf

        
        case _:
            print("Not a valid metal type")
            raise(TypeError)

    return ybpt

#getting AGN/LINER seperation line
def ka03_ke06_Seyfert_LINER_formula(xbpt, metal):
    match metal:
        case 'N II':
            start_coords = [-0.45, -0.5]
            gradient = np.tan(65 *np.pi/180)

            ybpt = gradient*(xbpt-start_coords[0]) + start_coords[1]
        
        case 'S II':
            ybpt = 1.89*xbpt + 0.76

        case 'O I':
            ybpt = 1.18*xbpt + 1.30
        
        case _:
            print("Not a valid metal type")
            raise(TypeError)
    return ybpt

k06_AGN_formula = lambda xbpt: -1.701*xbpt - 2.163
k06_Seyfert_LINER_formula = lambda xbpt: 1*xbpt +0.7

def plot_BPT_lines(ax, metal, have_legend = True, plot_xlims = {'N II': [-2, 1], 'S II': [-2, 0.5], 'O I': [-2.5, 0.5]}, plot_ylims = [-2.5,1.5], AGN_cutoffs= True):
        # x limits to stop log from changing sign and div by 0
    kewley_xlims = {'N II': 0.47, 'S II': 0.31, 'O I': -0.59} 
    kauffman_xlims = [-1.28, 0.05]
    Seyfert_LINER_line_startpoints = {'N II': -0.45, 'S II': -0.3, 'O I': -1.13}

    k01_metal_Ha_AGN_bounds = {'N II': 0.05, 'S II': -0.1, 'O I': -1.0}
    k01_OIII_Hb_AGN_bound = 1.0 #shouldn't vary with metal type
    k03_k01_intercept = -1.28 # intercept point of Kauffman 2003 and Kewley 2001 lines

        
        # kewley AGN/HII or composite line
    xbpt = np.linspace(plot_xlims[metal][0], kewley_xlims[metal], 100)
    ax.plot(xbpt,k01_bpt_formula(xbpt, metal),color='k',linestyle='-', label='(Kewley, 2001)')

    # Seyfert and LINER divider line
    paper = '(Kewley, 2006)'
    if metal == 'N II':
        paper = '(Kauffman, 2003)'

    xbpt= np.linspace(Seyfert_LINER_line_startpoints[metal], 4, 100)
    ax.plot(xbpt,ka03_ke06_Seyfert_LINER_formula(xbpt, metal),color='k',linestyle='--', label=paper)

    if metal == 'N II':
        xbpt = np.linspace(kauffman_xlims[0], kauffman_xlims[1], 100)
        ax.plot(xbpt,k03_bpt_formula(xbpt),color='k',linestyle=':', label='(Kauffman, 2003)')


        fontsize = 20
        ax.text(-0.5, 1.2, 'Seyferts', size = fontsize)
        ax.text(0.35, -1.5, 'LINERs', size=fontsize)
        ax.text(-0.1, -2.2, 'Comp', size=fontsize)
        ax.text(-1.2, -0.7, 'H II', size=fontsize)

    if AGN_cutoffs: # shade AGN cutoff regions
        # AGN cutoff regions
        ax.axhline(y=k01_OIII_Hb_AGN_bound, color='grey', linestyle='--', alpha=0.5)
        ax.axvline(x=k01_metal_Ha_AGN_bounds[metal], color='grey', linestyle='--', alpha=0.5)


        ax.fill_betweenx([k01_OIII_Hb_AGN_bound, plot_ylims[1]], plot_xlims[metal][0], plot_xlims[metal][1], color='grey', alpha=0.2, label='AGN Cutoff Region')
        ax.fill_betweenx([plot_ylims[0], k01_OIII_Hb_AGN_bound], k01_metal_Ha_AGN_bounds[metal], plot_xlims[metal][1], color='grey', alpha=0.2)


    ax.set(xlim=plot_xlims[metal], ylim=plot_ylims,xlabel=f'$log( [{metal}]/[H \\alpha] )$',ylabel='$Log( [OIII]/[H \\beta] )$')
    ax.grid()

    if have_legend:
        ax.legend()

    return

def plot_Oxygen_diagnostic_lines(ax, plot_xlims=[-2.5, 0.5], plot_ylims=[-2.5,1.5], have_legend=True, AGN_cutoffs=True):
    xbpt = np.linspace(plot_xlims[0], plot_xlims[1], 100)
    k06_OI_Ha_AGN_bound = -0.7 

    # AGN line
    ax.plot(xbpt, k06_AGN_formula(xbpt), color='k',linestyle='-', label='AGN Dividing Line')

    # Seyfert-LINER line
    xbpt=np.linspace(-1.06, plot_xlims[1], 100)
    ax.plot(xbpt, k06_Seyfert_LINER_formula(xbpt), color='k',linestyle='--', label='Seyfert-LINER Line')


    # adding region names
    fontsize = 20
    plt.text(-1, 1, 'Seyferts', size = fontsize)
    plt.text(-0.2, -1.5, 'LINERs', size=fontsize)
    plt.text(-2.4, -2.2, 'Star-forming and Composite', size=fontsize)

    ax.set(xlim=plot_xlims, ylim=plot_ylims,xlabel=f'$log( [O I]/[H \\alpha] )$',ylabel='$Log( [O III]/[O II] )$')
    ax.grid()


        # AGN cutoff line
    if AGN_cutoffs: # shade AGN cutoff regions
        # AGN cutoff regions
        ax.axvline(x=k06_OI_Ha_AGN_bound, color='grey', linestyle='--', alpha=0.5) 
        ax.fill_betweenx(plot_ylims, k06_OI_Ha_AGN_bound, plot_xlims[1], color='grey', alpha=0.2, label='AGN Cutoff Region')


    if have_legend:
        ax.legend()
    return

def get_BPT_AGN_classification(BPT_x, BPT_x_error, BPT_y, BPT_y_error, metal = 'N II', plot_type = 'BPT', SN_lim = 5):
    """
    Take flux emission line values and errors to determine if galaxy is an AGN based on Kauffman 2003 and Kewley 2001 \\
    NOTE - does not do S/N filtering, must be done seperately \\
    
    returns: array of galaxy_type, mask of is_AGN \\
        galaxy_types: \\
            -1 - S/N above threshold or no spectra for both axes \\
            0 - Star-forming \\
			1 - located on Star-forming / Composite region boundary \\
			2 - Composite \\
			3 - located on Composite / AGN region boundary \\
            4 - Located Across Star-forming / Composite / AGN regions \\
            5 - Located on Star-forming / AGN region boundary before intercept point in NII BPT\\
			6 - LINER \\
			7 - Seyfert \\
			8 - located on LINER / Seyfert boundary \\
            9 - AGN based only on O III / H\\beta \\
            10 - AGN based only on {metal} / H\\alpha \\
    """    

    # Use x values to set approximate cutoff points on x/y axis where AGN is 'guaranteed'
    k01_metal_Ha_AGN_bounds = {'N II': 0.05, 'S II': -0.1, 'O I': -1.0}
    k01_OIII_Hb_AGN_bound = 1.0 #shouldn't vary with metal type
    k03_k01_intercept = -1.28 # intercept point of Kauffman 2003 and Kewley 2001 lines

    k06_OI_Ha_AGN_bound = -0.7 

    # initialise array (set to -10 as this is not a valid galaxy type)
    galaxy_types = np.full(BPT_x.shape, -10, dtype=int)

    logBPT_y_max = np.log10(BPT_y + BPT_y_error)
    logBPT_y_min = np.log10(BPT_y - BPT_y_error)

    logBPT_x_max = np.log10(BPT_x + BPT_x_error)
    logBPT_x_min = np.log10(BPT_x - BPT_x_error)


    # check if S/N is above threshold for both axes (or NANs for both axes)
    is_low_SN_or_NaN = ( ( ( BPT_y / BPT_y_error <= SN_lim) |  np.isnan(BPT_y/BPT_y_error, dtype=bool) ) & ( ( BPT_x/BPT_x_error <= SN_lim) | np.isnan(BPT_x/BPT_x_error, dtype=bool)) )

    # mask to ensure both axis have valid values
    SN_mask = ( BPT_y / BPT_y_error> SN_lim) & ( BPT_x / BPT_x_error > SN_lim) & ~np.isnan(BPT_y/BPT_y_error, dtype=bool) & ~np.isnan(BPT_x/BPT_x_error, dtype=bool)


    # first check for cases in which only one axis has valid values
    is_only_BPT_x = (BPT_x / BPT_x_error > SN_lim) &( ( BPT_y / BPT_y_error <= SN_lim) | (np.isnan(BPT_y/BPT_y_error, dtype=bool) ))
    is_only_BPT_y = (BPT_y / BPT_y_error > SN_lim) &( ( BPT_x / BPT_x_error <= SN_lim) | (np.isnan(BPT_x/BPT_x_error, dtype=bool) ))
    

    if plot_type == 'BPT':
        # check if above k01 bounds set above
        is_above_k01_x_bound = (logBPT_x_min >= k01_metal_Ha_AGN_bounds[metal])
        is_above_k01_y_bound = (logBPT_y_min >= k01_OIII_Hb_AGN_bound)

        is_AGN_only_BPT_x = is_only_BPT_x & is_above_k01_x_bound
        is_AGN_only_BPT_y = is_only_BPT_y & is_above_k01_y_bound

    
        # get all relevant masks
        k01_logBPT_x_min = k01_bpt_formula(logBPT_x_min, metal)
        k01_logBPT_x_max = k01_bpt_formula(logBPT_x_max, metal)

        is_above_k01_line = (logBPT_y_min >= k01_logBPT_x_min) & (logBPT_y_min >= k01_logBPT_x_max)
        is_across_k01_line =( (logBPT_y_min < k01_logBPT_x_min) | (logBPT_y_min < k01_logBPT_x_max) ) & ( (logBPT_y_max >= k01_logBPT_x_max) | (logBPT_y_max >= k01_logBPT_x_min) )
        is_below_k01_line = (logBPT_y_max < k01_logBPT_x_min) & (logBPT_y_max < k01_logBPT_x_max)

        if metal == 'N II':
            k03_logBPT_x_min = k03_bpt_formula(logBPT_x_min)
            k03_logBPT_x_max = k03_bpt_formula(logBPT_x_max)

            is_above_k03_line = (logBPT_y_min >= k03_logBPT_x_min) & (logBPT_y_min >= k03_logBPT_x_max)
            is_across_k03_line = ( (logBPT_y_min < k03_logBPT_x_min) | (logBPT_y_min < k03_logBPT_x_max) ) & ( (logBPT_y_max >= k03_logBPT_x_max) | (logBPT_y_max >= k03_logBPT_x_min) )
            is_below_k03_line = (logBPT_y_max < k03_logBPT_x_min) & (logBPT_y_max < k03_logBPT_x_max)

            # Star-forming (below k03 and valid SN)
            is_star_forming =               is_below_k03_line    & is_below_k01_line  & SN_mask
            is_star_forming_or_composite =  is_across_k03_line   & is_below_k01_line  & SN_mask
            is_composite =                  is_above_k03_line    & is_below_k01_line  & SN_mask
            is_composite_or_AGN =           is_above_k03_line    & is_across_k01_line & SN_mask
            is_star_forming_or_AGN = is_across_k01_line & (logBPT_x_max < k03_k01_intercept) & SN_mask # across k01 line in region before intercept point
            is_star_forming_or_composite_or_AGN = is_across_k01_line & is_across_k03_line & (logBPT_x_max > k03_k01_intercept) & SN_mask # across both lines in region before intercept point


        elif metal in ('S II', 'O I'):
            is_star_forming =               np.full(len(BPT_x), False, dtype=bool) # no only star-forming region in these plots
            is_star_forming_or_composite =  is_below_k01_line  & SN_mask
            is_composite =                  np.full(len(BPT_x), False, dtype=bool) # no only composite region in these plots
            is_composite_or_AGN =           np.full(len(BPT_x), False, dtype=bool) # no only composite/AGN region in these plots
            is_star_forming_or_AGN =        np.full(len(BPT_x), False, dtype=bool) # no only star-forming/AGN region in these plots
            is_star_forming_or_composite_or_AGN = is_across_k01_line & SN_mask

        else:
            print("Not a valid BPT metal type")
            raise(TypeError)
        
    
        Seyfert_LINER_x_min = ka03_ke06_Seyfert_LINER_formula(logBPT_x_min, metal)
        Seyfert_LINER_x_max = ka03_ke06_Seyfert_LINER_formula(logBPT_x_max, metal)

        is_above_Seyfert_LINER_line = (logBPT_y_min >= Seyfert_LINER_x_min) & (logBPT_y_min >= Seyfert_LINER_x_max)
        is_across_Seyfert_LINER_line =( (logBPT_y_min < Seyfert_LINER_x_min) | (logBPT_y_min < Seyfert_LINER_x_max) ) & ( (logBPT_y_max >= Seyfert_LINER_x_max) | (logBPT_y_max >= Seyfert_LINER_x_min) )
        is_below_Seyfert_LINER_line = (logBPT_y_max < Seyfert_LINER_x_min) & (logBPT_y_max < Seyfert_LINER_x_max)

        is_Seyfert =        is_above_Seyfert_LINER_line     & is_above_k01_line & SN_mask
        is_LINER  =         is_below_Seyfert_LINER_line     & is_above_k01_line & SN_mask
        is_boundary_AGN =   is_across_Seyfert_LINER_line    & is_above_k01_line & SN_mask

            
    elif plot_type == 'O_diagram':
        is_above_k06_xbound = (logBPT_x_min >= k06_OI_Ha_AGN_bound)
        is_AGN_only_BPT_x = is_only_BPT_x & is_above_k06_xbound
        is_AGN_only_BPT_y = np.full(len(BPT_x), False, dtype=bool) # no distinct cutoof on y axes for O-diagram

        k06_AGN_x_min = k06_AGN_formula(logBPT_x_min)
        k06_AGN_x_max = k06_AGN_formula(logBPT_x_max)

        is_above_k06_line = (logBPT_y_min >= k06_AGN_x_min) & (logBPT_y_min >= k06_AGN_x_max)
        is_across_k06_line =( (logBPT_y_min < k06_AGN_x_min) | (logBPT_y_min < k06_AGN_x_max) ) & ( (logBPT_y_max >= k06_AGN_x_max) | (logBPT_y_max >= k06_AGN_x_min) )
        is_below_k06_line = (logBPT_y_max < k06_AGN_x_min) & (logBPT_y_max < k06_AGN_x_max)

        Seyfert_LINER_x_min = k06_Seyfert_LINER_formula(logBPT_x_min)
        Seyfert_LINER_x_max = k06_Seyfert_LINER_formula(logBPT_x_max)

        is_above_Seyfert_LINER_line = (logBPT_y_min >= Seyfert_LINER_x_min) & (logBPT_y_min >= Seyfert_LINER_x_max)
        is_across_Seyfert_LINER_line =( (logBPT_y_min < Seyfert_LINER_x_min) | (logBPT_y_min < Seyfert_LINER_x_max) ) & ( (logBPT_y_max >= Seyfert_LINER_x_max) | (logBPT_y_max >= Seyfert_LINER_x_min) )
        is_below_Seyfert_LINER_line = (logBPT_y_max < Seyfert_LINER_x_min) & (logBPT_y_max < Seyfert_LINER_x_max)


        is_star_forming =               np.full(len(BPT_x), False, dtype=bool) # no only star-forming region in O-diagram
        is_star_forming_or_composite =  is_below_k06_line  & SN_mask
        is_composite =                  np.full(len(BPT_x), False, dtype=bool) # no only composite region in O-diagram
        is_composite_or_AGN =           np.full(len(BPT_x), False, dtype=bool) # no only composite/AGN region in O-diagram
        is_star_forming_or_AGN =        np.full(len(BPT_x), False, dtype=bool) # no only star-forming/AGN region in O-diagram
        is_star_forming_or_composite_or_AGN = is_across_k06_line & SN_mask

        is_Seyfert =        is_above_Seyfert_LINER_line     & is_above_k06_line & SN_mask
        is_LINER  =         is_below_Seyfert_LINER_line     & is_above_k06_line & SN_mask
        is_boundary_AGN =   is_across_Seyfert_LINER_line    & is_above_k06_line & SN_mask


    else:
        print("Not a valid plot type")
        raise(TypeError)

    # inconclusive are all that cannot be classified on the diagram (even using one axis)
    galaxy_types[is_low_SN_or_NaN | (is_only_BPT_x & ~is_AGN_only_BPT_x)| (is_only_BPT_y & ~is_AGN_only_BPT_y)] = -1
    
    #could be classified
    galaxy_types[is_star_forming] = 0
    galaxy_types[is_star_forming_or_composite] = 1
    galaxy_types[is_composite] = 2
    galaxy_types[is_composite_or_AGN] = 3
    galaxy_types[is_star_forming_or_composite_or_AGN] = 4
    galaxy_types[is_star_forming_or_AGN] = 5

    # all AGN columns
    galaxy_types[is_LINER] =6
    galaxy_types[is_Seyfert] = 7
    galaxy_types[is_boundary_AGN] = 8
    galaxy_types[is_AGN_only_BPT_y] = 9 
    galaxy_types[is_AGN_only_BPT_x] = 10 

    is_AGN = is_LINER | is_Seyfert | is_boundary_AGN | is_AGN_only_BPT_x | is_AGN_only_BPT_y

    return galaxy_types, is_AGN


def get_flux_and_error_1_4_ARCSEC(SAMI_spectra_table_hdu, metal):
    """
    Returns the flux column and error table for the metal specified from the given SAMI_spectra_table \\
    valid metals: 'N II', 'S II', 'O I', 'O II', 'O III', 'H Alpha', 'H Beta'
    """
    SAMI_metal_columns = {'N II': ('NII6583_1_4_ARCSECOND',), 
                            'S II': ('SII6716_1_4_ARCSECOND', 'SII6731_1_4_ARCSECOND'),
                            'O I': ('OI6300_1_4_ARCSECOND',),
                            'O II': ('OII3726_1_4_ARCSECOND', 'OII3729_1_4_ARCSECOND'),
                            'O III': ('OIII5007_1_4_ARCSECOND',),
                            'H Alpha': ('HALPHA_1_4_ARCSECOND',),
                            'H Beta': ('HBETA_1_4_ARCSECOND',)}

    metal_flux = np.zeros(len(SAMI_spectra_table_hdu))
    metal_error = np.zeros(len(SAMI_spectra_table_hdu))

    for metal_colname in SAMI_metal_columns[metal]:
        metal_flux += SAMI_spectra_table_hdu[metal_colname]
        metal_error += SAMI_spectra_table_hdu[f'{metal_colname}_ERR']

    return metal_flux, metal_error


def plot_full_BPT_diagrams(relevant_SAMI_spectra_table, SN_lim=5, bpt_metals=('N II', 'S II', 'O I'), 
                           fig1 = None, fig_height=7, fig_width=21,
                            plot_xlims=None,
                            print_SN_counts=True,
                            all_galaxy_types_array=None, galaxy_type_labels=None, all_same_label=False):
    """
    Plots the full BPT diagrams for the given SAMI spectra table. \\
    
    Parameters: \\
    - relevant_SAMI_spectra_table: The relevant SAMI spectra table. \\
    - SN_lim: The signal-to-noise ratio limit for plotting (and classifying). \\
    - bpt_metals: The list of metals to use for the BPT diagrams. \\
    - fig1: The figure object to plot on. If None, a new figure will be created. \\
    - fig_height: Height of the figure. \\
    - fig_width: Width of the figure. \\
    - plot_xlims: Dictionary of x limits for each metal. \\
    - print_SN_counts: If True, will print the number of galaxies with S/N above the limit for each metal. \\
    - all_galaxy_types_array: The already classified point labels to use for the plot, their value should correspond to the index in the galaxy_type_labels list. If None, will classify based on each metals BPT. If 'N II', will classify based on N II BPT. If provided, will use these labels for the points.
    - galaxy_type_labels: The labels for the galaxy types. If None, will use the possible_galaxies values as labels (unless all_galaxy_types_array is None or 'N II'). \\
    - all_same_label: If True, all points will be labelled the same.
    """

    # figure setup / options    
    if fig1 is None:
        fig1, axs = plt.subplots(1, len(bpt_metals))

        fig1.set_figheight(fig_height)
        fig1.set_figwidth(fig_width)


    # BPT initialisation and limits
    if plot_xlims is None:
        plot_xlims = {'N II': [-2, 1], 'S II': [-2, 0.5], 'O I': [-2.5, 0.5]}
    

    # labels and possible galaxy types
    if all_same_label: # if all points are to be labelled the same, set all_galaxy_types_array to list of zeros and possible_galaxies to just 0
        all_galaxy_types_array = np.zeros(len(relevant_SAMI_spectra_table), dtype=int)
        possible_galaxies = (0, )

    if all_galaxy_types_array in (None, 'N II'): # if no galaxy types provided, classify based on NII BPT classification
        galaxy_type_labels = ('HII', 'HII/Comp', 'Comp', 'AGN/Comp', 'HII/Comp/AGN', 'HII/AGN', 'LINER', 'Seyfert', 'Boundary AGN', 'AGN OIII', 'AGN metal', 'Inconclusive')
        possible_galaxies = (0,1,2,3,4,5,6,7,8,9,10, -1)

        label_type = all_galaxy_types_array # need to store as will reuse all_galaxy_types_array for each metal BPT classification
        all_galaxy_types_array = np.zeros(len(relevant_SAMI_spectra_table), dtype=int) # reset to zeros


    else: # need to get new possible_galaxies values and corresponding galaxy_AGN_labels
        possible_galaxies = np.unique(all_galaxy_types_array)

        if galaxy_type_labels is None: # i.e. if no labels provided, use the possible_galaxies values as labels
            galaxy_type_labels = possible_galaxies
        else:
            if len(galaxy_type_labels) <= np.max(possible_galaxies):
                raise ValueError("galaxy_type_labels does not have enough labels for all possible galaxy types.")



    # construct BPT for 1.4 sec aperture removing anything with S/N ratio greater than limit
    OIII_flux, OIII_error = get_flux_and_error_1_4_ARCSEC(relevant_SAMI_spectra_table, 'O III')
    HBeta_flux, HBeta_error = get_flux_and_error_1_4_ARCSEC(relevant_SAMI_spectra_table, 'H Beta')

    HAlpha_flux, HAlpha_error = get_flux_and_error_1_4_ARCSEC(relevant_SAMI_spectra_table, 'H Alpha')
    if print_SN_counts:
        print(f"Number of galaxies with S/N > {SN_lim}:")
        print(f"{'O III:':10} {sum((OIII_flux/OIII_error > SN_lim))}/{len(OIII_flux)}")
        print(f"{'H beta:':10} {sum((HBeta_flux/HBeta_error > SN_lim))}/{len(HBeta_flux)}")
        print(f"{'H alpha:':10} {sum((HAlpha_flux/HAlpha_error > SN_lim))}/{len(HAlpha_flux)}\n")

    #testing
    for i, metal in enumerate(bpt_metals):
        #for i, metal in enumerate(bpt_metals):
        metal_flux, metal_error = get_flux_and_error_1_4_ARCSEC(relevant_SAMI_spectra_table, metal)

        SN_mask = (HBeta_flux/HBeta_error > SN_lim) & (OIII_flux/OIII_error > SN_lim) & (HAlpha_flux/HAlpha_error > SN_lim) & (metal_flux/metal_error > SN_lim)

        if print_SN_counts:
            print(f"{metal:10} {sum((metal_flux/metal_error > SN_lim))}/{len(metal_flux)}")
            print(f"{f'{metal} Total':10} {sum(SN_mask)}/{len(SN_mask)}\n")

        BPT_y = OIII_flux/HBeta_flux
        BPT_x = metal_flux/HAlpha_flux

        BPT_y_error = np.abs(BPT_y * np.sqrt( (OIII_error/OIII_flux)**2+(HBeta_error/HBeta_flux)**2))
        BPT_x_error = np.abs(BPT_x * np.sqrt( (metal_error/metal_flux)**2+(HAlpha_error/HAlpha_flux)**2))

        if i ==0 and label_type == 'N II':
            all_galaxy_types_array, _ = get_BPT_AGN_classification(BPT_x, BPT_x_error, BPT_y, BPT_y_error, metal=metal)
       
        elif label_type is None: # if no galaxy types provided, classify based on NII BPT classification
            all_galaxy_types_array, _ = get_BPT_AGN_classification(BPT_x, BPT_x_error, BPT_y, BPT_y_error, metal=metal)
        
        
        ax = axs.flatten()[i]


        for possible_type in possible_galaxies:
            
            galaxy_type_mask = all_galaxy_types_array == possible_type
        
            if type(galaxy_type_mask) == bool:
                galaxy_type_mask = (galaxy_type_mask,)

            if len(galaxy_type_mask)==0:
                continue
            
            ax.plot(np.log10(BPT_x[galaxy_type_mask]), np.log10(BPT_y[galaxy_type_mask]),'.', markersize=3, label=galaxy_type_labels[possible_type]) 
        plot_BPT_lines(ax, metal, have_legend=True)

    return fig1, axs, relevant_SAMI_spectra_table, all_galaxy_types_array




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#------------------------------------------------- CROSSMATCHING FUNCTIONS ------------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def angular_dist(r1, d1, r2, d2):    
    a = np.sin(np.abs(d1 - d2)/2)**2
    b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2
    
    d_rad = 2*np.arcsin((a+b)**(1/2))
    d_deg = np.degrees(d_rad)
    return d_deg

def get_SAMI_region(catalogues_filepath, CATIDs):
    """
    Function to get the SAMI region for a given set of CATIDs.\\
    0 - GAMA\\
    1 - Clusters\\
    2 - Filler\\
    """
    SAMI_Target_catalogues = ("InputCatGAMADR3.fits", "InputCatClustersDR3.fits", "InputCatFiller.fits")

    region_array = np.zeros(len(CATIDs))

    for i, SAMI_Target_catalogue in enumerate(SAMI_Target_catalogues):
        with fits.open(catalogues_filepath + SAMI_Target_catalogue) as hdul:
            Region_CATIDs = Table(hdul[1].data)['CATID']

        in_region = np.isin(CATIDs, Region_CATIDs)

        region_array[in_region] = i
    print(region_array)
    return region_array


def get_RA_Dec_cutout_fits_table(table_hdu: Table, bounds: tuple, bound_type = 'rect', col_names = ('RA', 'Dec')):
    
    if bound_type in ('rect', 'GAMA'):
        # Get rid of all not within RA range
        RA_array = np.array(table_hdu[col_names[0]])
        sort_ind_RA_removed = (RA_array > bounds[0][0]) & (RA_array < bounds[1][0] )
        table_hdu_RA_removed = table_hdu[sort_ind_RA_removed]


        # Get rid of all values not within Dec range
        Dec_array = np.array(table_hdu_RA_removed[col_names[1]])
        sort_ind_both_removed = (Dec_array > bounds[0][1]) & (Dec_array < bounds[1][1] )
        table_hdu_both_removed = table_hdu_RA_removed[sort_ind_both_removed]
    
    elif bound_type in ('circ', 'Cluster'): # need to convert degrees to radians for np.sin and np.cos
        RA_array = np.array(table_hdu[col_names[0]]) * np.pi/180
        Dec_array = np.array(table_hdu[col_names[1]]) * np.pi/180

        r_bound = bounds[1]
        circle_centre_RA = bounds[0][0] * np.pi/180
        circle_centre_Dec = bounds[0][1] * np.pi/180

        mask_within_radius = angular_dist(RA_array, Dec_array, circle_centre_RA, circle_centre_Dec) < r_bound
        table_hdu_both_removed = table_hdu[mask_within_radius]

    else:
        raise TypeError(f"Invalid bound type: {bound_type}. Valid types are \'rect\' and \'circ\'")

    return table_hdu_both_removed

def get_fits_table_SAMI_target_regions_cutout(catalogues_filepath, fits_filename, col_names, save_file = True):

    GAMA_region_bounds = [((129,-2), (141, 3)), ((174, -3), (186, 2)), ((211.5, -2), (223.5, 3))] # bounds of the GAMA target regions in the SAMI data including filler which are rectangular in shape (bottom left to top right coords) all in degrees

    Cluster_region_bounds = [((355.397880, -29.236351), 2), ((18.815777, 0.213486), 1.5), ((18.739974, 0.430807), 1), 
                            ((356.937810, -28.140661), 3), ((6.380680, -33.046570), 1.5), ((336.977050, -30.575371), 2), 
                            ((329.372605, -7.795692), 2), ((14.067150, -1.255370), 2), ((10.460211, -9.303184), 3)] # bounds of the Cluster target regions (including filler targets) circular in shape  ((x_centre,y_centre), r_approx) all in degrees


    with fits.open(catalogues_filepath+fits_filename) as hdul:
        cutout_tables = []

        table_hdu = Table(hdul[1].data)


        bound_type = 'rect'

        for bounds in GAMA_region_bounds:
            table_hdu_both_cutout = get_RA_Dec_cutout_fits_table(table_hdu, bounds, bound_type=bound_type, col_names=col_names)
            cutout_tables.append(table_hdu_both_cutout)


        bound_type = 'circ'

        for bounds in Cluster_region_bounds:
            table_hdu_both_cutout = get_RA_Dec_cutout_fits_table(table_hdu, bounds, bound_type=bound_type, col_names=col_names)
            cutout_tables.append(table_hdu_both_cutout)
  

    table_hdu_allcutouts = astropy.table.vstack(cutout_tables) # stack tables together
    unique_table_hdu_allcutouts = astropy.table.unique(table_hdu_allcutouts) # remove duplicates created from vstacking

    if save_file:
        unique_table_hdu_allcutouts.write(catalogues_filepath+"SAMI_target_region_cutout_"+fits_filename)
        savepath = str(catalogues_filepath+"SAMI_target_regions_cutout_"+fits_filename)
        print(f"Cutout saved to {savepath}")

    return unique_table_hdu_allcutouts


def get_all_SAMI_targets_catalogue(catalogue_filepath, save_file = False):
    SAMI_Target_catalogues = ("InputCatClustersDR3.fits", "InputCatFiller.fits", "InputCatGAMADR3.fits")
    SAMI_regions = ("Clusters", "Filler", "GAMA")

    SAMI_tables = []
    for k, SAMI_Target_catalogue in enumerate(SAMI_Target_catalogues):
        with fits.open(catalogue_filepath + SAMI_Target_catalogue) as SAMI_Target_hdul:
            SAMI_Target_table_hdu = Table(SAMI_Target_hdul[1].data)

        SAMI_Target_table_hdu.add_column(astropy.table.Column([SAMI_regions[k]]*len(SAMI_Target_table_hdu), name='SAMI_region'))

        SAMI_tables.append(SAMI_Target_table_hdu)

    all_SAMI_Targets_table = astropy.table.vstack(SAMI_tables)


    if save_file:
        all_SAMI_Targets_table_filename = f"ALL_SAMI_TARGETS.fits"

        all_SAMI_Targets_fpath = catalogue_filepath+all_SAMI_Targets_table_filename
        all_SAMI_Targets_table.write(all_SAMI_Targets_fpath)
        print(f"Combine SAMI Targets Table Saved to: {all_SAMI_Targets_fpath}")

    return all_SAMI_Targets_table


def get_crossmatched_fits_table(catalogue_filepath, 
                                SAMI_target_table, 
                                survey_cutout_table_hdu, 
                                sep_arcsec = 20, 
                                crossmatching_colnames = ('RA', 'Dec'), 
                                save_file = False, 
                                cat_names = ('All_SAMI', 'RACS-mid1_sources'),
                                only_closest = False,
                                SAMI_cols_to_keep = ('CATID', 'RA_OBJ', 'DEC_OBJ'),
                                survey_cols_to_keep = 'ALL'):
    """
    Returns and Saves a crossmatched catalogue of a SAMI_Target_catalogue and the desired crossmatching_catalogue when separation of objects is less than sep_arcsec.
    """

    max_separation = sep_arcsec*u.arcsec #max distance to be considered a match in arcsec

    SAMI_RA_array = np.array(SAMI_target_table['RA_OBJ'])
    SAMI_Dec_array = np.array(SAMI_target_table['DEC_OBJ'])

    SAMI_coords = SkyCoord(SAMI_RA_array, SAMI_Dec_array, unit='deg')
    
    survey_cutout_RA_array = np.array(survey_cutout_table_hdu[crossmatching_colnames[0]])
    survey_cutout_Dec_array = np.array(survey_cutout_table_hdu[crossmatching_colnames[1]])

    survey_cutout_coords = SkyCoord(survey_cutout_RA_array, survey_cutout_Dec_array, unit='deg')


    # initialise reduced version of SAMI default is keeping only CATID, RA_OBJ and DEC_OBJ columns
    reduced_SAMI_target_table = SAMI_target_table.copy()    
    if not SAMI_cols_to_keep == 'ALL':
        reduced_SAMI_target_table.keep_columns(SAMI_cols_to_keep)

    # now initialise a reduced version of the matching catalogue default is to keep all columns
    reduced_survey_cutout_table_hdu = survey_cutout_table_hdu.copy()
    if not survey_cols_to_keep == 'ALL':
        reduced_survey_cutout_table_hdu.keep_columns(survey_cols_to_keep)


    # now, do crossmatching with Skycoord
    matched_indices, separation, _ = SAMI_coords.match_to_catalog_sky(survey_cutout_coords)
    
    
    survey_cutout_matched_table = reduced_survey_cutout_table_hdu[matched_indices]

    crossmatched_table = astropy.table.hstack([reduced_SAMI_target_table, survey_cutout_matched_table, Table([separation.arcsec], names=['sep_arcsec'])])
    reduced_crossmatched_table = crossmatched_table[crossmatched_table['sep_arcsec']*u.arcsec<=max_separation]

    # ensure only closest match is kept in table
    if only_closest:
        sorted_reduced_crossmatched_table = reduced_crossmatched_table[reduced_crossmatched_table.argsort('sep_arcsec')]
        sep_sorted_survey_RA = sorted_reduced_crossmatched_table[crossmatching_colnames[0]]
        unique_sep_sorted_survey_RA, unique_index = np.unique(sep_sorted_survey_RA, return_index = True)

        reduced_crossmatched_table= sorted_reduced_crossmatched_table[unique_index]



    if save_file:
        crossmatched_filename = f"{cat_names[0]}_matched_{cat_names[1]}.fits"

        crossmatched_filepath = catalogue_filepath+"Crossmatched\\"+crossmatched_filename
        reduced_crossmatched_table.write(crossmatched_filepath)
        print(f"Crossmatched Table Saved to: {crossmatched_filepath}")

    return reduced_crossmatched_table

def get_all_SAMI_crossmatched_fits_table(catalogue_filepath, 
                                crossmatching_catalogue, 
                                sep_arcsec = 20, 
                                crossmatching_colnames = ('RA', 'Dec'), 
                                separation_column = True, 
                                save_file = False, 
                                only_closest = True,
                                SAMI_cols_to_keep = ('CATID', 'RA_OBJ', 'DEC_OBJ'),
                                survey_cols_to_keep = 'ALL'):
    
    SAMI_Target_catalogues = ("InputCatClustersDR3.fits", "InputCatFiller.fits", "InputCatGAMADR3.fits")
    SAMI_regions = ("Clusters", "Filler", "GAMA")


    # test if the combined SAMI target catalogue exists
    try: 
        with fits.open(catalogue_filepath + "ALL_SAMI_TARGETS.fits") as all_SAMI_targets_hdul:
            None
    except FileNotFoundError: # if not, produce one
        print("All SAMI Targets Catalogue does not already exist.")
        get_all_SAMI_targets_catalogue(catalogue_filepath)


    # test if a cutout of the survey exists
    try:
        with fits.open(catalogue_filepath + "SAMI_target_region_cutout_" + crossmatching_catalogue) as survey_cutout_hdul:
            None
    except FileNotFoundError: # if not, produce one
        print("SAMI Target Region Cutout does not already exist.")
        get_fits_table_SAMI_target_regions_cutout(catalogue_filepath, crossmatching_catalogue, col_names=crossmatching_colnames)

    with fits.open(catalogue_filepath + "ALL_SAMI_TARGETS.fits") as all_SAMI_targets_hdul, fits.open(catalogue_filepath + "SAMI_target_region_cutout_" + crossmatching_catalogue) as survey_cutout_hdul:
        all_SAMI_target_table = Table(all_SAMI_targets_hdul[1].data)
        survey_cutout_table_hdu = Table(survey_cutout_hdul[1].data)

    all_SAMI_crossmatched_table = get_crossmatched_fits_table(catalogue_filepath, 
                                                        all_SAMI_target_table, 
                                                        survey_cutout_table_hdu, 
                                                        save_file=False, 
                                                        sep_arcsec=sep_arcsec,
                                                        only_closest = only_closest,
                                                        crossmatching_colnames=crossmatching_colnames, 
                                                        SAMI_cols_to_keep = SAMI_cols_to_keep,
                                                        survey_cols_to_keep = survey_cols_to_keep)

    if save_file:
        crossmatched_filename = f"all_SAMI_target_matched_{crossmatching_catalogue}"

        crossmatched_filepath = catalogue_filepath+"Crossmatched\\"+crossmatched_filename
        all_SAMI_crossmatched_table.write(crossmatched_filepath)
        print(f"Crossmatched Table Saved to: {crossmatched_filepath}")


    return all_SAMI_crossmatched_table




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#------------------------------------------------------- SFR functions ----------------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def SFR_1_4GHz_ho03(reduced_SAMI_AGN_summary_table, CATID_redshifts, survey = 'RACS'):
    """
    SFR estimated from 1.4GHz radio luminosity, from (Hopkins et al. 2003)\\
    input: reduced_SAMI_AGN_summary_table, relevant luminosity distances\\
    returns: SFR in $M _\\odot yr^-1$
    """
    if survey == 'RACS':
        radio_fluxes_Jy = reduced_SAMI_AGN_summary_table['RACS_TOTALFLUX']       *u.mJy   # in Jy
        radio_errors_Jy = reduced_SAMI_AGN_summary_table['RACS_TOTALFLUX_ERR']   *u.mJy   # in Jy

    elif survey == 'FIRST':
        radio_fluxes_Jy = reduced_SAMI_AGN_summary_table['NVSS_TOTALFLUX']       *u.mJy   # in Jy
        radio_errors_Jy = np.zeros(len(reduced_SAMI_AGN_summary_table))  *u.mJy   # in Jy

    else:
        raise TypeError(f"Invalid radio survey {survey}, accepted surveys are RACS or FIRST")
    

    # convert Jy to W /m^2 /Hz
    radio_fluxes = radio_fluxes_Jy.to(u.W / u.m**2 /u.Hz) # in W/m^2 /Hz
    radio_errors = radio_errors_Jy.to(u.W / u.m**2/u.Hz) 

    CATID_luminosity_distances = get_luminosity_distance(CATID_redshifts) # in m

    # Convert to Luminosity, Sectiion 3.2 of Pracy et al. 2016, in W 
    alpha_spectral_index = -0.7
    L_radio = 4 * np.pi * (CATID_luminosity_distances**2) * 1 / ( (1+CATID_redshifts)** (1+alpha_spectral_index) ) * radio_fluxes  # in W
    L_radio_error = radio_errors/radio_fluxes * L_radio # in W

    SN = L_radio / L_radio_error # S/N ratio

    Lc = 6.4e21 *u.W/u.Hz   #Lc from Hopkins et al. 2003, in W/Hz and the RACS-mid bandwidth in Hz

    f_prefactor=np.ones(len(L_radio))

    if len(f_prefactor[L_radio<=Lc]) >0:
        f_prefactor[L_radio<=Lc] = 1/(0.1 + 0.9 * (L_radio[L_radio<=Lc]/Lc)**0.3)

    SFR_array =f_prefactor* L_radio/(1.81e21 * u.W/u.Hz)

    # convert to approriate IMF (this was all done in the Chabrier IMF but SAMI SFRs are in Salpeter IMF)
    SFR_array = SFR_array / 0.62 # convert to Chabrier IMF from Salpeter IMF (Madusha sent this through)
    return SFR_array, SN


def SFR_Halpha_ho03(reduced_SAMI_spectra_table_hdu, CATID_redshifts):
    """
    SFR estimated from $H\\alpha$ luminosity, from (Hopkins et al. 2003)\\
    input: L_Halpha in W \\
    returns: SFR in $M _\\odot yr^-1$
    """
    #get the Halpha fluxes and errors for each CATID: *2 to account for RE_MGE is only half of galaxy light
    HAlpha_flux_ergs = reduced_SAMI_spectra_table_hdu['HALPHA_RE_MGE']  *2        *1e-16 * u.erg / u.cm**2 / u.s # in 1e-16 erg/cm^2/s
    HAlpha_error_ergs =  reduced_SAMI_spectra_table_hdu['HALPHA_RE_MGE_ERR'] *2   *1e-16 * u.erg / u.cm**2 / u.s # in 1e-16 erg/cm^2/s

    # convert erg/s/cm^2 to W /m^2 
    HAlpha_flux = HAlpha_flux_ergs.to(u.W / u.m**2) # in W/m^2
    HAlpha_error = HAlpha_error_ergs.to(u.W / u.m**2) # in W/m^2

    CATID_luminosity_distances = get_luminosity_distance(CATID_redshifts) #then get the luminosity distance for each CATID

    L_Halpha = HAlpha_flux          * 4 * np.pi * (CATID_luminosity_distances**2) # in W
    L_Halpha_error = HAlpha_error   * 4 * np.pi * (CATID_luminosity_distances**2) # in W


    Halpha_SN = L_Halpha_error/L_Halpha

    # relationship from Hopkins et al. 2003, in W
    SFR_HAlpha = L_Halpha/ (1.27e34 * u.W)

    SFR_HAlpha = SFR_HAlpha / 0.62 # convert to Chabrier IMF from Salpeter IMF (Madusha sent this through)
    return SFR_HAlpha, Halpha_SN


def get_luminosity_distance(redshifts, desired_units=u.m):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    return cosmo.luminosity_distance(redshifts * cu.redshift).to(u.m)


def get_z_best(catalogues_filepath, CATIDs):
    """
    Returns the best available redshift for given CATIDs from the Input catalogues.
    Uses z_tonry For the GAMA catalogue, and z_spec for Clusters and FIller.
    """
    SAMI_Target_catalogues = ("InputCatGAMADR3.fits", "InputCatClustersDR3.fits", "InputCatFiller.fits")

    z_spec_array = np.full(len(CATIDs), np.nan)  # Initialize with NaN

    for SAMI_Target_catalogue in SAMI_Target_catalogues:
        with fits.open(catalogues_filepath + SAMI_Target_catalogue) as hdul:
            catalogue_table = Table(hdul[1].data)
            catalogue_CATIDs = catalogue_table['CATID']
            
            # Use z_tonry for GAMA catalogue, otherwise use z_spec
            if SAMI_Target_catalogue == "InputCatGAMADR3.fits":
                catalogue_z_spec = catalogue_table['z_tonry']
            else:
                catalogue_z_spec = catalogue_table['z_spec']

        # Find indices of matching CATIDs
        matching_indices = np.isin(CATIDs, catalogue_CATIDs)

        # Update z_spec_array for matching CATIDs
        z_spec_array[matching_indices] = catalogue_z_spec[np.isin(catalogue_CATIDs, CATIDs[matching_indices])]

    return z_spec_array



def SFR_comparison_plot(ax, catalogue_filepath, SAMI_AGN_summary_table, SAMI_SFR_table_hdu, relevant_CATIDs, SAMI_SFR_table_type = 'SFR', survey_type='Radio',label='',marker = '.', SN_lim=5):
    
    SFR_survey, SFR_SAMI = get_SFRs(catalogue_filepath, SAMI_AGN_summary_table, SAMI_SFR_table_hdu, relevant_CATIDs, SAMI_SFR_table_type = SAMI_SFR_table_type, survey_type=survey_type,label=label,marker = marker, SN_lim=SN_lim)
      

    #SN_mask = (Halpha_SN<SN_lim) & (radio_SN<SN_lim)

    ax.plot(np.log10(SFR_survey), np.log10(SFR_SAMI), marker,label=label)

    return SFR_survey, SFR_SAMI


def get_SFRs(catalogue_filepath, SAMI_AGN_summary_table, SAMI_SFR_table_hdu, relevant_CATIDs, SAMI_SFR_table_type = 'SFR', survey_type='Radio',label='',marker = '.', SN_lim=5):
    """
    
    returns: SFR_survey, SFR_SAMI for each CATID
    """
    
    CATID_mask = np.isin(SAMI_AGN_summary_table['CATID'], relevant_CATIDs)
    reduced_SAMI_AGN_summary_table = SAMI_AGN_summary_table[CATID_mask]

    # turn all the filled values into nans
    reduced_SAMI_AGN_summary_table = reduced_SAMI_AGN_summary_table.filled(np.nan)


    #get the redshift for each CATID:
    CATID_redshifts = get_z_best(catalogue_filepath, reduced_SAMI_AGN_summary_table['CATID'])     #first get the redshift for each CATID

    # add temporary CATID_redshifts column to the reduced summary table
    reduced_SAMI_AGN_summary_table.add_column(astropy.table.Column(CATID_redshifts, name='CATID_redshifts'))


    if SAMI_SFR_table_type == 'SFR':
        reduced_SAMI_AGN_summary_table.rename_column('SFR_SAMI', 'SFR_SAMI_temporary')
        reduced_SAMI_AGN_summary_table['SFR_SAMI_temporary'] = 10**reduced_SAMI_AGN_summary_table['SFR_SAMI_temporary']

    elif SAMI_SFR_table_type in ('Halpha', 'SFR_spectra'):
        SFR_SAMI = np.full(len(reduced_SAMI_AGN_summary_table), np.nan)
        
        # need to reduce the spectra table to only those that are in the reduced summary table (the is_best cubes)
        # first need to strip the spaces from the CUBEIDPUB column in the spectra table
        SAMI_spectra_CUBEIDPUBS = [cubeid.strip() for cubeid in SAMI_SFR_table_hdu['CUBEIDPUB']]
        reduced_SAMI_SFR_table_hdu = SAMI_SFR_table_hdu[np.isin(SAMI_spectra_CUBEIDPUBS, reduced_SAMI_AGN_summary_table['CUBEIDPUB'])]
        reduced_SAMI_SFR_table_hdu = reduced_SAMI_SFR_table_hdu.filled(np.nan)

        SAMI_spectra_mask = np.isin(reduced_SAMI_AGN_summary_table['CATID'], reduced_SAMI_SFR_table_hdu['CATID'])

        if SAMI_SFR_table_type == 'SFR_spectra':
            SFR_SAMI[SAMI_spectra_mask] = reduced_SAMI_SFR_table_hdu['SFR_RE_MGE']*2

        else:
            SFR_Halpha, Halpha_SN = SFR_Halpha_ho03(reduced_SAMI_SFR_table_hdu, reduced_SAMI_AGN_summary_table['CATID_redshifts'])
            SFR_SAMI[SAMI_spectra_mask] = SFR_Halpha

        reduced_SAMI_AGN_summary_table.add_column(astropy.table.Column(SFR_SAMI, name='SFR_SAMI_temporary'))
    

    if survey_type =='Radio':        
        SFR_survey, radio_SN = SFR_1_4GHz_ho03(reduced_SAMI_AGN_summary_table ,reduced_SAMI_AGN_summary_table['CATID_redshifts'])  

    if survey_type == 'LARGESS':
        SFR_survey, LARGESS_SN = SFR_1_4GHz_ho03(reduced_SAMI_AGN_summary_table ,reduced_SAMI_AGN_summary_table['CATID_redshifts'], survey='FIRST')  

    reduced_SAMI_AGN_summary_table.add_column(astropy.table.Column(SFR_survey, name='SFR_survey'))

    reduced_SAMI_AGN_summary_table = reduced_SAMI_AGN_summary_table.filled(np.nan)



    return reduced_SAMI_AGN_summary_table['SFR_survey'], reduced_SAMI_AGN_summary_table['SFR_SAMI_temporary']


def get_SAMI_SFRs(Summary_table, SAMI_SFR_table_hdu):
    """
    Returns the SFRs for the given CATIDs from the SAMI SFR table.
    """
    
    # reduce summary table to only the CATID column
    CATID_table = Summary_table.copy()
    CATID_table.keep_columns('CATID')

    # rename the catid column to CATID
    SAMI_SFR_table_hdu.rename_column('catid', 'CATID')

    # only keep relevant columns in the SAMI SFR table
    SAMI_SFR_table = SAMI_SFR_table_hdu.copy()
    SAMI_SFR_table.keep_columns(['CATID', 'SFR_best', 'SFR_best_flag'])


    # join in order to ensure matching of CATIDs order
    CATID_SFR_table = join(CATID_table, SAMI_SFR_table, 'CATID', join_type='left')
    
    return CATID_SFR_table


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#------------------------------------------------- Galaxy Matching functions ----------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def minimising_fctn(parameter_table1, parameter_table2, mass_colname='M_STAR', SFR_colname='SFR_SAMI', match_single_to_multiple=False, log_scale =False):
    """
    Function to minimise the difference between two parameter tables.
    """
   
    m1 = parameter_table1[mass_colname]
    m2 = parameter_table2[mass_colname]

    SFR1 = parameter_table1[SFR_colname]
    SFR2 = parameter_table2[SFR_colname]

    if log_scale:
        m1 = 10**m1
        m2 = 10**m2
        SFR1 = 10**SFR1
        SFR2 = 10**SFR2


    if match_single_to_multiple: # If matching a single galaxy to multiple, need to 'extend the first table so it has the same number of rows as the second table.
        m1 = np.repeat(m1, len(parameter_table2))
        SFR1 = np.repeat(parameter_table1[SFR_colname], len(parameter_table2))


    m_sqrdiff_normalised = ( (m1-m2 ))**2
    SFR_sqrdiff_normalised = ( (SFR1-SFR2) )**2

    weighting_factors = (1, 1)
    
    diff = np.sqrt(weighting_factors[0] * m_sqrdiff_normalised + weighting_factors[1] * SFR_sqrdiff_normalised)
    return diff


def get_closest_galaxy_match(summary_table, input_CATID, possible_CATIDs, mass_colname='M_STAR', SFR_colname='SFR_SAMI', log_scale=False):
    """
    Function to find the closest galaxy match in the summary table based on mass and SFR.
    """

    input_galaxy = summary_table[summary_table['CATID'] == input_CATID]
    
    if len(input_galaxy) == 0:
        print(f"Input CATID {input_CATID} not found in summary table.")
        return None
    
    reduced_summary_table = summary_table[np.isin(summary_table['CATID'], possible_CATIDs)]

    diff_array = minimising_fctn(input_galaxy, reduced_summary_table, mass_colname, SFR_colname, match_single_to_multiple=True, log_scale=log_scale)

    if np.all(np.isnan(diff_array)):
        print("All differences are NaN. No valid matches found.")
        return None

    min_diff = np.nanmin(diff_array)
    closest_match_index = np.nanargmin(diff_array)
    closest_match_CATID = reduced_summary_table['CATID'][closest_match_index]

    return closest_match_CATID, min_diff


def get_multiple_galaxy_matches(summary_table, input_CATIDs, possible_CATIDs, mass_colname='M_STAR', SFR_colname='SFR_SAMI', log_scale=False):
    """
    Matches each galaxy in set A to one in set B, minimizing a cost function based on mass and SFR.
    """



    N = len(input_CATIDs)
    M = len(possible_CATIDs)

    cost_matrix = np.zeros((N, M))

    # Default cost function: Euclidean distance in log space
    for i in range(N):
        cost_matrix[i, :] = minimising_fctn(summary_table[summary_table['CATID'] == input_CATIDs[i]],
                                             summary_table[np.isin(summary_table['CATID'], possible_CATIDs)],
                                             mass_colname=mass_colname, SFR_colname=SFR_colname, match_single_to_multiple=True, log_scale=log_scale)


    # Solve assignment problem (Hungarian algorithm)
    indices_A, indices_B = linear_sum_assignment(cost_matrix)

    matched_galaxies_table =  Table(names=('Input_CATID', 'Matched_CATID', 'min_diff'), dtype=(int, int, float))

    for i, j in zip(indices_A, indices_B):
        matched_galaxies_table.add_row((input_CATIDs[i], possible_CATIDs[j], cost_matrix[i, j]))

    return matched_galaxies_table


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#-------------------------------------------------- Summary Table function ------------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



