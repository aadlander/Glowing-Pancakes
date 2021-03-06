from __future__ import print_function, division
from astropy.visualization import simple_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes
import pandas as pd
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from glowing_waffles.differential_photometry import catalog_search, calculate_transform_coefficients

# This function really should be somewhere else eventually.
def scale_and_downsample(data, downsample=4,
                         min_percent=20,
                         max_percent=99.5):

    norm = simple_norm(data,
                              min_percent=min_percent,
                              max_percent=max_percent)
    scaled_data = norm(data)
    if downsample > 1:
        scaled_data = block_reduce(scaled_data,
                                   block_size=(downsample, downsample))
    return scaled_data

def source_ra(source_number):                #Returns the the column heading for a stars RA in the measurements file
    col_name = 'RA_C'+str(source_number)
    return col_name

def source_error(source_number):             #Returns the column heading for a star's source-sky error in the measurements file
    col_name = 'Source_Error_C'+str(source_number)
    return col_name

def source_column(source_number):            #REturns the column heading for a star's source-sky in the measurements file
    col_name = 'Source-Sky_C'+str(source_number)
    return col_name

def uniformize_source_names(aij_tbl):        #Uniformizes the source names in the columns of the measurements file
    import re                                #and creates a list of source numbers starting at 1

    data_col = re.compile(r'Source-Sky_[TC](\d+)')
    sources = []
    for c in aij_tbl.colnames:
        match = re.search(data_col, c)
        if match:
            sources.append(len(sources) + 1)
            source_number =  match.groups()[0]
            try:
                aij_tbl.rename_column(c, 'Source-Sky_C' + source_number)
            except KeyError:
                # Column already exists s there was no need to change the name
                pass
            try:
                aij_tbl.rename_column('Source_Error_T' + source_number,
                                      'Source_Error_C' + source_number)
            except KeyError:
                pass
            try:
                aij_tbl.rename_column('Peak_T' + source_number,
                                      'Peak_C' + source_number)
            except KeyError:
                pass
            try:
                aij_tbl.rename_column('RA_T' + source_number,
                                      'RA_C' + source_number)
            except KeyError:
                pass
            try:
                aij_tbl.rename_column('DEC_T' + source_number,
                                      'DEC_C' + source_number)
            except KeyError:
                pass
            try:
                aij_tbl.rename_column('X(FITS)_T' + source_number,
                                      'X(FITS)_C' + source_number)
            except KeyError:
                pass
            try:
                aij_tbl.rename_column('Y(FITS)_T' + source_number,
                                      'Y(FITS)_C' + source_number)
            except KeyError:
                pass

    return sources


#Define the function to be used to plot the differential magnitudes
def plot_magnitudes(mags=None, errors=None, times=None, source=None, night=None, ref_mag=0, color=None):
    #calcualte the mean of the magnitudes passed
    mean = np.nanmean(mags)
    #calcualte the standard deviation of the magntiudes passed
    std = np.nanstd(mags)
    #plot the magnitudes vs time
    plt.errorbar(times, mags, yerr=errors, fmt='o',
                 label='{}, stdev: {:5.3f}\nnight: {}'.format(source, std, night))
    #change the xlims of the plot to reflect the times
    plt.xlim(times.min(), times.max())
    #Plots a line correspinding to the mean
    #plt.plot(plt.xlim(), [mean, mean], 'k--', )
    plt.axvline(mean, color='gray', linewidth=2)
    #plots a line corresponding to the upper limit of the mean
    #plt.plot(plt.xlim(), [mean + std, mean + std], 'k:')
    plt.axvline(mean + std, color='gray', linewidth=2)
    #plots a line corresponding to the lower limit of the mean
    #plt.plot(plt.xlim(), [mean - std, mean - std], 'k:')
    plt.axvline(mean - std, color='gray', linewidth=2)
    #Following Line was commented out:
    plt.plot(pd.rolling_mean(times, 20, center=True),
             pd.rolling_mean(mags, 20, center=True),
             color='gray', linewidth=3)
    # Make sure plot range is at least 0.1 mag...
    min_range = 0.1
    #find the ylim of the plot
    ylim = plt.ylim()
    #check if the difference in the limits is less then the min range
    if ylim[1] - ylim[0] < min_range:
        #if less then the mid range then change the y limits to be min_range different
        plt.ylim(mean - min_range/2, mean + min_range/2)

    #find the new ylim of the plot
    ylim = plt.ylim()
    # Reverse vertical axis so brighter is higher
    plt.ylim((ylim[1], ylim[0]))

    size = 1000./(mean - ref_mag + 0.1)**2
    plt.scatter([0.8*(plt.xlim()[1]-plt.xlim()[0]) + plt.xlim()[0]],
                [0.8*(plt.ylim()[1] - plt.ylim()[0]) + plt.ylim()[0]],
                c='red', marker='o', s=size)

    plt.title(color)
    plt.legend()
    #send back the mean and the standard deviation of the plot
    return mean, std


def find_apass_stars(image):
    #use the catalog_search function to find the apass stars in the frame of the image read above
    apass, apass_x, apass_y = catalog_search(image.wcs, image.shape, 'II/336/apass9', 'RAJ2000', 'DEJ2000', 1, False)

    #Creates a boolean array of the apass stars that have well defined magnitudes and color
    apass_bright = (apass['e_r_mag'] < 0.05) & (apass['e_B-V'] < 0.1) #& (apass['u_e_r_mag'] == 0)

    #create new lists of apass stars and x y pixel coordinates using boolean array
    apass_in_bright, in_apass_x, in_apass_y = apass[apass_bright], apass_x[apass_bright], apass_y[apass_bright]

    return apass, apass_x, apass_y, apass_in_bright, in_apass_x, in_apass_y

def find_known_variables(image):
    #Get any known variable stars from a new catalog search of VSX
    vsx, vsx_x, vsx_y = catalog_search(image.wcs, image.shape, 'B/vsx/vsx', 'RAJ2000', 'DEJ2000')
    vsx_names = vsx['Name']      #Get the names of the variables
    return vsx, vsx_x, vsx_y, vsx_names

def find_stars_from_catalog(image, catatalog):
    cat, x, y = catalog_search(image.wcs, image.shap, catalog, 'RAJ2000', 'DEJ2000')
    return 0

def plot_apass_variables(image, disp, vsx_x, vsx_y, vsx_names, apass, in_apass_x, in_apass_y, apass_x, apass_y):
    plt.figure(figsize=(12, 7))
    plt.imshow(disp, cmap='gray', origin='lower')
    plt.scatter(vsx_x, vsx_y, c='none', s=100, edgecolor='cyan')
    plt.title('Blue: VSX, Yellow: APASS', fontsize=20)

    for x, y, m in zip(vsx_x, vsx_y, vsx_names):
        plt.text(x, y, str(m), fontsize=18, color='cyan')

    plt.scatter(in_apass_x, in_apass_y, c='none', s=50, edgecolor='yellow', alpha=0.5, marker='o')

    apass_in = apass['recno']
    for x, y, c in zip(apass_x, apass_y, apass_in):
        plt.text(x, y, str(c), fontsize=12, color='yellow')

    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])

#Get the RA and Dec of a selected list
def get_RA_Dec(name):
    #Get the RA of the well defined apass stars (in degrees)
    ra = name['RAJ2000']
    #Get the DEC of the well defined apass stars (in degrees)
    dec = name['DEJ2000']
    return ra, dec

def get_color(name):
    #Get the color of all of the apass stars
    color = name['B-V']
    #get the color error of all of the apass stars
    color_error = name['e_B-V']
    return color, color_error

def color_corrections(aij_stars, aij_mags, apass_index, apass_color, apass_R_mags, good_match,
                      order=1):
    # Changing from huber to astropy fit
    # Create empty list for the corrections (slope and intercept)
    corrections = []
    # Create empy list for the error in the corrections
    all_aij_mags = np.zeros_like(aij_mags)
    for idx, star in enumerate(aij_stars):
        all_aij_mags[idx, :] = star.magnitude
    all_aij_mags = np.ma.masked_invalid(all_aij_mags)
    # loop over all images
    for idx in range(aij_mags.shape[1]):
        these_mags = all_aij_mags[:, idx]
        BminusV = apass_color[apass_index][good_match]
        r = these_mags[good_match]
        r = np.ma.masked_invalid(r)
        R = apass_R_mags[apass_index][good_match]
        R = np.ma.masked_invalid(R)
        mask = r.mask | R.mask
        r.mask = mask
        R.mask = mask
        BminusV = BminusV.compress(~mask)
        R = R.compressed()
        r = r.compressed()
        # Astropy Fit

        filtered_data, or_fitted_model = calculate_transform_coefficients(r, R, BminusV,
                                                                          order=order)
        if order == 1:
            fit_list = [or_fitted_model.c1.value, or_fitted_model.c0.value]
        elif order == 2:
            fit_list = [or_fitted_model.c2.value, or_fitted_model.c1.value, or_fitted_model.c0.value]

        corrections.append(fit_list)

        plt.ylim(or_fitted_model.c0.value - 0.5,
                 or_fitted_model.c0.value + 0.5)
        plt.title(idx)
        plt.plot(BminusV, R - r, 'o')
        plt.plot(BminusV, filtered_data, '+')
        plt.plot(BminusV, or_fitted_model(BminusV), 'gx', label="model fitted w/ filtered data")
        plt.show()
        print(or_fitted_model.c1.value)

    return corrections, BminusV, R - r

def mag_error(aij_raw, gain, read_noise, sources):
    #create a magnitude error list
    mag_err = []
    for source in sources:
        #use source_error function to get error in source-sky
        err = aij_raw[source_error(source)]
        #calculate the signal to noise ratio
        snr = gain * aij_raw[source_column(source)] / err
        #calculate magnitude error
        mag_e = 1/snr
        #add magnitude error to list mag_err
        mag_err.append(mag_e)
    return mag_err


def corrected_curveses(aij_mags, aij_stars, all_apass_color, all_apass_color_error,
                       apass_index_for_color, BminusV, corrections,
                       include_color_term=True):
    corrected_curves = np.zeros_like(aij_mags)
    corrections_good = np.array(corrections)

    all_aij_mags = np.zeros_like(aij_mags)
    for idx, star in enumerate(aij_stars):
        all_aij_mags[idx, :] = star.magnitude
    all_aij_mags = np.ma.masked_invalid(all_aij_mags)
    # loop over all of the stars
    for obj in range(len(aij_stars)):
        # get the apass color for the star
        BminusV = np.array(all_apass_color[apass_index_for_color[obj]])
        color_term = 0
        if include_color_term:
            n_terms = len(corrections_good[0, :]) - 1
            for idx in range(n_terms):
                power = n_terms - idx
                color_term += corrections_good[:, idx] * BminusV**power

        old_mag = all_aij_mags[obj, :]
        new_mag = old_mag + color_term + corrections_good[:, -1]
        corrected_curves[obj] = np.array(new_mag)
    return corrected_curves
