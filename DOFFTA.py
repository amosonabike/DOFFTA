import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.stats import cauchy
from scipy import signal
from scipy.special import legendre

### MATPLOTLIB PLOTTING PARAMETERS

figure_width = 3 * 160 / 25.4 #conversion to mm is 25.4
figure_height = 3 * 100 / 25.4 #conversion to mm is 25.4
figure_size = (figure_width, figure_height)

resolution = 300 #dpi
tick_size = 18
fontlabel_size = 18


params = {
    'lines.markersize' : 2,
    'axes.labelsize': fontlabel_size,
    'legend.fontsize': fontlabel_size,
    'xtick.labelsize': tick_size,
    'ytick.labelsize': tick_size,
    'figure.figsize': figure_size,
    'xtick.direction':     'in',     # direction: {in, out, inout}
    'ytick.direction':     'in',     # direction: {in, out, inout}
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.pad':  8,
    'ytick.major.pad':  8,
    'font.family' : 'serif,',
    'ytick.labelsize' : fontlabel_size,
    'xtick.labelsize' : fontlabel_size,
    'axes.linewidth' : 1.2
}
plt.rcParams.update(params)


#reference properties of pure water
mu_water = 1.0016e-3 #Pa.s dynamic viscosity water
rho_water = 998.23 #kg / m^3 density
sigma_water = 72.8e-3#72.8e-3 #N / m surface tnesion

### DATA IMPORT, REVIEW & CLEAN

def import_data(filepath):
    '''imports data from instrument, drops nan rows and adds column names.
    Returns dataframe of experimental data'''
    #column names of the data files
    column_names = ['time_s', 'x1', 'x2', 'x3', 'aspect_ratio', 'diameter_um', 'x4', 'centre_offset_x', 'centre_offset_y']

    #read data
    df =  pd.read_csv(filepath,
                      sep = '\t',
                      header=None,
                      names=column_names)

    df.dropna(inplace = True)
    
    return df

def review_data(df):
    '''given a df in the with columns:
        time_s	x1	x2	x3	aspect_ratio	diameter_um	x4	centre_offset_x	centre_offset_y
    this will plot  diameter, aspect ratio and 2 centre offsets for this data over time for user to review'''
    
    fig = plt.figure(figsize = (19,17))
    gs = fig.add_gridspec(2, 2,hspace=0.2, wspace=0.1)

    ((ax1, ax2), (ax3, ax4)) = gs.subplots(sharex='col')

    ax1.plot(df.time_s/1e-6,df.diameter_um, '#785EF0')
    ax2.plot(df.time_s/1e-6,df.aspect_ratio, '#DC267F')
    ax3.plot(df.time_s/1e-6,df.centre_offset_x, '#FE6100')
    ax4.plot(df.time_s/1e-6,df.centre_offset_y, '#FFB000')

    ax1.set_title( 'Diameter / µm')
    ax2.set_title( 'Aspect Ratio / µm')
    ax3.set_title( 'Centre offsett x / pixels')
    ax4.set_title( 'Centre offsett y / pixels')

    ax3.set(xlabel = 'Time / µs')
    ax4.set(xlabel = 'Time / µs')

    plt.show()
    
    return


def remove_unecessary_data(df, aspect_ratio_limit = 5, end_time_limit = 1):
    '''Takes experimental data df and removes data outside of aspect ratio and final time limit'''
    
    df = df.loc[df.aspect_ratio < aspect_ratio_limit]
    
    df.reset_index(inplace = True, drop = True)

    df = df.loc[df.time_s < end_time_limit]
    
    return df


### ROLLING AVERAGE FUNCTION

def rolling_average(data, window):
    '''Rolling average function that uses padding to avoid clipping of data.
    Returns series of rolling average'''
    if window % 2 == 0:
        window += 1
    if window < 4:
        window = 5

    edge_width = int((window - 1) / 2)
        
    s = pd.DataFrame(np.pad(data, edge_width, mode = 'edge'))
    s = s.rolling(window, center = True).mean()
    s = s.iloc[edge_width : - edge_width]
    s.reset_index(drop = True, inplace = True)
    return s.iloc[:,0].values


def data_cleaning(df):
    '''Dummy function returning same df as received, in place for any future developments of cleaning.
    It is essential that any cleaning method maintains spacing of data for FFT. Otherwise new FFT method will be required:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html'''
    
    return df

def review_clean_data(df, df_clean):
    '''Takes experimental data and cleaned data and plots both for comparison'''
    #review selected data
    plt.scatter(df.time_s/1e-6, df.diameter_um, s = 15, c = 'k')
    plt.scatter(df_clean.time_s/1e-6, df_clean.diameter_um, c = 'r', s = 3)

    rolling_window = 5
    #rolling average data
    plt.plot(rolling_average(df_clean.time_s,rolling_window)/1e-6, rolling_average(df_clean.diameter_um, rolling_window), c= 'k')

    plt.ylabel('Diameter$_{horizontal}$ / µm')
    plt.xlabel('Time / µs')
    plt.show()

    return

### DIAMETER CALIBRATION

def get_calibrated_diameter(df, equilibrium_data_points = 100):
    '''Takes experimental data and calculates diamter correction based uppon:
    Blur_width = (width_observed/AR) - width_observed
    Assumes that the equillibrium_data_points are at final equillibrium shape/size'''
    
    width_final = df.diameter_um.tail(equilibrium_data_points).mean() * 1e-6
    AspectRatio_final = df.aspect_ratio.tail(equilibrium_data_points).mean()

    width_correction = (width_final/AspectRatio_final - width_final)
    width_calibrated = width_final + width_correction
    
    A_0_calculated = (df.diameter_um.values[0]/1e6 - width_final) / 2 
    
    '''print('Measured final diameter / µm = ' + str(round(1e6* width_final,3)) + '\nCalculated diameter error / µm = ' + str(round(1e6*width_correction,3)) + '\nCalculated final diameter / µm = ' + str(round(1e6*width_calibrated,3)))'''

    
    return width_final, AspectRatio_final, width_correction, width_calibrated, A_0_calculated

def get_calibrated_data(df, diameter_correction):
    '''Takes experimetnal data and blur corrction and adds this to all the diameter values
    Returns calibrated data df'''
    
    df_calibrated = df.copy()
    df_calibrated.diameter_um = df_calibrated.diameter_um + diameter_correction*1e6
    
    return df_calibrated

### PEAK LOCATION

def find_peaks(xdata, ydata, window_points):
    '''Takes arrays of x and y data and a window size. Finds peaks within rolling window and
    returns x and y data of peaks and troughs'''
    peaks = []
    for i in range(n, len(ydata) - n):
        if ydata[i-n:i+n].max() == ydata[i]:
            peaks.append(i)
            
    troughs = []
    for i in range(n, len(ydata) - n):
        if ydata[i-n:i+n].min() == ydata[i]:
            troughs.append(i)
            
    
    return xdata[peaks], ydata[peaks], xdata[troughs], ydata[troughs]

### FFT

def get_fft(x_data, y_data, lower_frequency_cutoff = 7000):
    '''Takes x and y arrays and performs fft.
    Returns FFT in df of freq and amplitude without any values of freq lower than specified cut off'''
    
    yf = np.abs(fft(y_data))

    N = len(y_data)
    duration = x_data.max() - x_data.min()
    sample_rate = N / duration
    
    xf = fftfreq(N, 1/sample_rate)
    
    df = pd.DataFrame({'frequency' : xf, 'amplitude' : yf})
    
    df = df.loc[(df.frequency > lower_frequency_cutoff) & (df.frequency < 100000)  ]
    return df

### LORENTZIAN FITTING

def lorentzian_fit_func(x_data ,scale,offset, width, vert_offset):
    '''Describes Lorentzian function with a vertical offset'''
    dist = cauchy(offset, width)
    return scale * dist.pdf(x_data) + vert_offset

def get_lorentzian_fit(frequency_data, frequency_amplitude):
    '''Takes FFT data and 
    returns fitted function, as a data set and as a list of parameters'with errors.
    If fit failes, returns no data and params of zeros'''
    
    fit_freq_range = np.linspace(frequency_data.min(),frequency_data.max(),1000)
    
    try:
        lorentzian_parameters, lpcov = curve_fit(lorentzian_fit_func,
                                 frequency_data,
                                 frequency_amplitude,
                                 p0 = [10,40e3,10e3,0],
                                bounds = ([1,0,1, -50],
                                          [1e6,100e3,100e3, 50]))

        fitted_lorentzian = lorentzian_fit_func(fit_freq_range, lorentzian_parameters[0], lorentzian_parameters[1], lorentzian_parameters[2], lorentzian_parameters[3])
        
        lorentzian_parameters_errors = np.sqrt(np.diag(lpcov))
        
        return fit_freq_range, fitted_lorentzian, lorentzian_parameters, lorentzian_parameters_errors
    
    except:
        print ('failed to fit! review data!')
        return frequency_data, frequency_amplitude, np.array([0,0,0,0]), np.array([0,0,0,0])
    
### CALCULATE SURFACE TENSION

def calc_suraface_tension(R_0,density, frequency, n):
    '''Calculates surface tension, given equilibrium raidus, droplet density, peak frequency and mode order'''
    return ((R_0**3)*(density)*(frequency**2)) / (n*(n-1)*(n+2))

def calc_suraface_tension_with_correction(R_0, density, viscosity, frequency, n):
    '''
    Calculates surface tension, given equilibrium raidus, droplet density, peak frequency and mode order.
        Includes 'viscosity correction' of w_obs^2 = w_n^2 - B_n^2
    '''
    return ((density * R_0 **3)/(n * (n-1) * (n+1)) * (frequency + (((n-1)**2 * (2*n + 1)**2 * viscosity**2)/(density**2 * R_0**6))))

### FIND X OF PEAK Y IN DATASET

def get_peak_frequency(x,y):
    '''Takes x and y arrays and returns x value of peak y'''
    return x[np.argmax(y)]

### IDENTIFY AND REMOVE HIGH ORDER MODES

def remove_high_order_data(df, rolling_average_window):
    '''Takes experimental data df, finds equilibrium value and std dev.
    Returns data where rolling average is within limit'''
    
    offset_limit = 3.5
    
    C_offset_x_std = np.std(rolling_average(df.centre_offset_x, rolling_average_window)[-250:])
    C_offset_x_mean = np.mean(rolling_average(df.centre_offset_x, rolling_average_window)[-250:])
    
    offset_data = rolling_average(df.centre_offset_x,rolling_average_window)[(rolling_average(df.centre_offset_x,rolling_average_window) > C_offset_x_mean+offset_limit*C_offset_x_std) | (rolling_average(df.centre_offset_x,rolling_average_window) < C_offset_x_mean-offset_limit*C_offset_x_std)]
    time_data = rolling_average(df.time_s,rolling_average_window)[(rolling_average(df.centre_offset_x,rolling_average_window) > C_offset_x_mean+offset_limit*C_offset_x_std) | (rolling_average(df.centre_offset_x,rolling_average_window) < C_offset_x_mean-offset_limit*C_offset_x_std)]
    
    return time_data, offset_data, C_offset_x_mean, C_offset_x_std

### GETTING TRIMMED DATA

def review_centre_offset(df, average_window = 11, save_figures = 'n', save_location = 'NaN'):
    '''Takes Experimental data df.
    Returns only second order data in trimmed df after reviewing and seeking user confirmation.
    Saves figures by default.'''
    centre_offset_equilib = remove_high_order_data(df, average_window)[2]
    centre_offset_std = remove_high_order_data(df, average_window)[3]
    
    centre_offset_figure, ax = plt.subplots()

    #raw data
    ax.scatter(df.time_s/1e-6, df.centre_offset_x, s = 1, c = 'purple', label = 'x offset' )
    ax.scatter(df.time_s/1e-6, df.centre_offset_y, s = 1, c = 'orange', label = 'y offset' )
    
    #values outside threshold
    ax.scatter(remove_high_order_data(df,average_window)[0]/1e-6,
                remove_high_order_data(df,average_window)[1],
                s = 50, marker = 's', c = 'violet')

    #smoothed data
    ax.plot(rolling_average(df.time_s/1e-6, average_window),
            rolling_average(df.centre_offset_x,average_window),
            lw = 3, c = 'purple', label = 'rolling average x' )
    ax.plot(rolling_average(df.time_s/1e-6, average_window),
            rolling_average(df.centre_offset_y, average_window),
            lw = 1, c = 'orange', label = 'rolling average y' )
    
    ax.axhline(centre_offset_equilib, c = 'darkviolet')
    ax.axhspan(centre_offset_equilib-centre_offset_std, centre_offset_equilib+centre_offset_std, color = 'darkviolet', alpha = 0.2)
    ax.axhspan(centre_offset_equilib-2*centre_offset_std, centre_offset_equilib+2*centre_offset_std, color = 'darkviolet', alpha = 0.2)
    ax.axhspan(centre_offset_equilib-3*centre_offset_std, centre_offset_equilib+3*centre_offset_std, color = 'darkviolet', alpha = 0.2)
    ax.axhspan(centre_offset_equilib-3.5*centre_offset_std, centre_offset_equilib+3.5*centre_offset_std, color = 'darkviolet', alpha = 0.2)


    plt.ylabel('Centre Offset')
    plt.xlabel('Time / µs')
    plt.ylim(-3,3)

    ax.legend()
    plt.show()
    
    #calculate end of 3rd+ order modes
    calculated_high_order_cut_off = remove_high_order_data(df,average_window)[0].max()
    print('Calculated cut off (µs): ' + str(round(calculated_high_order_cut_off * 1e6,2)))
    print('Is this cut off sensible? \n\nIf yes, press enter. \n\nIf no, enter a suggested value below (in µs):')

    suggested_cut_off = input()
    
    if suggested_cut_off  == '':
        high_order_cut_off = calculated_high_order_cut_off
    else:
        high_order_cut_off = float(suggested_cut_off)/1e6
        
    df_trimmed  = df.loc[df.time_s > high_order_cut_off].copy()
    
    fig_c_offset_cutoff, ax = plt.subplots()

    #data
    ax.scatter(df.time_s/1e-6, df.centre_offset_x, s = 1, c = 'purple', label = 'x offset' )

    #smoothed data
    ax.plot(rolling_average(df.time_s/1e-6, average_window), rolling_average(df.centre_offset_x,average_window), lw = 3, c = 'purple', label = 'Rolling average' )

    #threshold
    ax.axhline(centre_offset_equilib, c = 'darkviolet')
    ax.axhspan(centre_offset_equilib-centre_offset_std, centre_offset_equilib+centre_offset_std, color = 'darkviolet', alpha = 0.2)
    ax.axhspan(centre_offset_equilib-2*centre_offset_std, centre_offset_equilib+2*centre_offset_std, color = 'darkviolet', alpha = 0.2)
    ax.axhspan(centre_offset_equilib-3*centre_offset_std, centre_offset_equilib+3*centre_offset_std, color = 'darkviolet', alpha = 0.2)
    ax.axhspan(centre_offset_equilib-3.5*centre_offset_std, centre_offset_equilib+3.5*centre_offset_std, color = 'darkviolet', alpha = 0.2)


    #values outside threshold
    ax.scatter(remove_high_order_data(df,average_window)[0]/1e-6,
                remove_high_order_data(df,average_window)[1],
                s = 50, marker = 's', c = 'violet')

    #display cut off
    ax.axvline(high_order_cut_off/1e-6, linestyle = '--', color = 'k', label = 'Cut off time')
    ax.axvspan(0,high_order_cut_off/1e-6,color = 'k', alpha = 0.1, label = 'Data to be removed')


    # view raw y data
    ax.scatter(df.time_s/1e-6, df.centre_offset_y, s = 1, c = 'orange', label = 'y offset' )
    # view smoothed y data
    ax.plot(rolling_average(df.time_s/1e-6, average_window),
            rolling_average(df.centre_offset_y, average_window),
            lw = 1, c = 'orange', label = 'rolling average y' )

    ax.set_ylabel('Centre Offset')
    ax.set_xlabel('Time / µs')
    ax.set_ylim(-3,3)

    ax.legend()
    
    if save_figures == 'y':
        plt.savefig(save_location + '/Centre Offset Figure.jpg', dpi = 300)
    
    plt.show()


    ################################


    fig_diameter_cutoff, ax = plt.subplots()

    # diameter data
    ax.plot(rolling_average(df.time_s/1e-6,average_window),
            rolling_average(df.diameter_um/1e6,average_window),
            ':', c = 'k')

    ax.plot(rolling_average(df_trimmed.time_s/1e-6,average_window),
            rolling_average(df_trimmed.diameter_um/1e6,average_window),
            linewidth = 3, c = 'k')



    #display cut off
    ax.axvline(high_order_cut_off/1e-6, linestyle = '--', color = 'k', label = 'Cut off time')
    ax.axvspan(0,high_order_cut_off/1e-6,color = 'k', alpha = 0.1, label = 'Data to be removed')


    ax.set_ylabel('Diameter / µm')
    ax.set_xlabel('Time / µs')

    ax.legend()
    
    if save_figures == 'y':
        plt.savefig(save_location + '/Diameter Figure.jpg', dpi = 300)
    
    plt.show()


    ############################


    fig_AR_cutoff, ax = plt.subplots()

    # AR data
    ax.plot(rolling_average(df.time_s/1e-6,average_window),
            rolling_average(df.aspect_ratio/1e6,average_window),
            ':', c = 'mediumblue')

    ax.plot(rolling_average(df_trimmed.time_s/1e-6,average_window),
            rolling_average(df_trimmed.aspect_ratio/1e6,average_window),
            linewidth = 3, c = 'mediumblue')



    #display cut off
    ax.axvline(high_order_cut_off/1e-6, linestyle = '--', color = 'k', label = 'Cut off time')
    ax.axvspan(0,high_order_cut_off/1e-6,color = 'k', alpha = 0.1, label = 'Data to be removed')


    ax.set_ylabel('Aspect Ratio')
    ax.set_xlabel('Time / µs')

    ax.legend()

    
    if save_figures == 'y':
        plt.savefig(save_location + '/Aspect Ratio Figure.jpg', dpi = 300)

    plt.show()

    return df_trimmed



### RUNNING FULL SURFACE TENSION CALCULATION


def full_surface_tension_calculation(time_data, oscillation_data, equilibrium_radius, mode_order = 2, density = rho_water, low_freq_cut_off = 0, high_freq_cut_off = 100e3, save_figures = 'n', save_location = 'NaN'):
    '''Takes time and oscillation data, equilibrium radius and
    Returns Surface tension + error + figure of fit + data.
    Takes optional mode order argument, density parameter and frquency limits over which to fit in case of poor data quality.'''
    fft = get_fft(time_data, oscillation_data)
    
    fft = fft[(fft.frequency > low_freq_cut_off) & (fft.frequency < high_freq_cut_off)]
    
    lorentzian_fit = get_lorentzian_fit(fft.frequency, fft.amplitude)
    
    figure, ax = plt.subplots()
    #fit
    ax.plot(lorentzian_fit[0], lorentzian_fit[1], c = 'r', linewidth = 5, label = 'Lorentzian Fit', zorder = 0)

    # experimental data
    ax.scatter(fft.frequency, fft.amplitude, s = 50, c = 'k', marker = 's', label = 'Experimental FFT')

    #peak
    ax.axvline(lorentzian_fit[2][1], linestyle = '--', c = 'r', label = 'Peak Frequency')
    ax.axvspan(lorentzian_fit[2][1] - lorentzian_fit[3][1], lorentzian_fit[2][1] + lorentzian_fit[3][1], color = 'r', alpha = 0.2)
    
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency / Hz')
    
    ax.legend()
    
    
    peak_frequency = 2 * np.pi * lorentzian_fit[2][1]
    peak_frequency_err = 2 * np.pi * lorentzian_fit[3][1]
    
    surface_tension = calc_suraface_tension(equilibrium_radius,density, peak_frequency, mode_order)
    surface_tension_err = 2 * peak_frequency_err * calc_suraface_tension(equilibrium_radius,density,np.sqrt(peak_frequency), mode_order)    
    
    
    if save_figures == 'y':
        plt.savefig(save_location + '/FFT Fit Figure.jpg', dpi = 300)
    
    plt.show()
    
    return surface_tension, surface_tension_err, lorentzian_fit, fft

def full_surface_tension_calculation_with_correction(time_data, oscillation_data, equilibrium_radius, mode_order = 2, density = rho_water, viscosity = mu_water, low_freq_cut_off = 0, high_freq_cut_off = 100e3, save_figures = 'n', save_location = 'NaN'):
    '''Takes time and oscillation data, equilibrium radius and
    Returns Surface tension + error + figure of fit + data.
    Takes optional mode order argument, density parameter and frquency limits over which to fit in case of poor data quality.'''
    fft = get_fft(time_data, oscillation_data)
    
    fft = fft[(fft.frequency > low_freq_cut_off) & (fft.frequency < high_freq_cut_off)]
    
    lorentzian_fit = get_lorentzian_fit(fft.frequency, fft.amplitude)
    
    figure, ax = plt.subplots()
    #fit
    ax.plot(lorentzian_fit[0], lorentzian_fit[1], c = 'r', linewidth = 5, label = 'Lorentzian Fit', zorder = 0)

    # experimental data
    ax.scatter(fft.frequency, fft.amplitude, s = 50, c = 'k', marker = 's', label = 'Experimental FFT')

    #peak
    ax.axvline(lorentzian_fit[2][1], linestyle = '--', c = 'r', label = 'Peak Frequency')
    ax.axvspan(lorentzian_fit[2][1] - lorentzian_fit[3][1], lorentzian_fit[2][1] + lorentzian_fit[3][1], color = 'r', alpha = 0.2)
    
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency / Hz')
    
    ax.legend()
    
    
    peak_frequency = 2 * np.pi * lorentzian_fit[2][1]
    peak_frequency_err = 2 * np.pi * lorentzian_fit[3][1]
    
    surface_tension = calc_suraface_tension_with_correction(equilibrium_radius,density, viscosity, peak_frequency, mode_order)
    surface_tension_err = 2 * peak_frequency_err * calc_suraface_tension_with_correction(equilibrium_radius,density, viscosity, np.sqrt(peak_frequency), mode_order)    
    
    
    if save_figures == 'y':
        plt.savefig(save_location + '/FFT Fit Figure.jpg', dpi = 300)
    
    plt.show()
    
    return surface_tension, surface_tension_err, lorentzian_fit, fft

def get_water_true_diameter(sigma_observed, r_observed, sigma_true = sigma_water):
    '''Takes observed sigma and r for water droplet and calculates what the true r should be for a droplet with that frequency. This is used for instrument calibration'''
    return np.cbrt((sigma_true/sigma_observed) * r_observed ** 3)

def save_experiment_parameters(save_location, df, df_trimmed):
    
    '''
    Surface tension ± Error
    Initial data point, high order cut off, final data point
    Lorentzian parameters
    Lorentzian parameters errors
    Observed final diameter, blur correction, calculated final diameter
    If sample is water, this is what true diameter should be.
    '''
    
    #Saving experimental data
    df.to_csv(save_location + '/Calibrated Data.txt', sep = '\t')
    df_trimmed.to_csv(save_location + '/Trimmed Calibrated Data.txt', sep = '\t')
    
    d_final, AR_final, d_correction, d_calibrated, A_0_estimate = get_calibrated_diameter(df)
    
    sig, sig_err, lor_fit, exp_fft = full_surface_tension_calculation(df_trimmed.time_s, df_trimmed.aspect_ratio.values, d_calibrated/2)
    
    #saving lorentzian data
    lorentzian_data = np.array([lor_fit[0], lor_fit[1]])
    df_lorentzian = pd.DataFrame(lorentzian_data.T, columns = ['Frequency', 'Amplituded'])
    df_lorentzian.to_csv(save_location + '/Lorentzian Data.txt', sep = '\t')
    #saving fft data
    exp_fft.to_csv(save_location + '/FFT Data.txt', sep = '\t')
    
    #saving experimental parameters file
    lines = ['Surface Tension / N/m ±\tSurface Tension Error / N/m:',
             str(sig)+'\t'+str(sig_err),
             '\nExperiment Start / s \tExperiment Cut Off / s \tExperiment End / s',
              str(df.time_s.values[0]) + '\t' + str(df_trimmed.time_s.values[0]) + '\t' + str(df.time_s.values[-1]),
             '\nLorentzian Parameters\nScale \tHorizontal Offset / Hz \tWidth \t Vertical Offset',
            str(lor_fit[2][0])+'\t'+str(lor_fit[2][1])+'\t'+str(lor_fit[2][2])+'\t'+str(lor_fit[2][3]),
             'Lorentzian Parameters Uncertainties\nScale \tHorizontal Offset / Hz \tWidth \t Vertical Offset',
            str(lor_fit[3][0])+'\t'+str(lor_fit[3][1])+'\t'+str(lor_fit[3][2])+'\t'+str(lor_fit[3][3]),
             '\nIf sample was water the expected diameter of a droplet with this oscillation frequency / m is:',
             str(2*get_water_true_diameter(sig, d_calibrated/2)),
            '\nObserved final diameter / m \tCalculated Blur Correction /m \t Calculated Diameter / m',
            str(d_final)+'\t'+str(d_correction)+'\t'+str(d_calibrated)]
    
    
    with open(save_location + '/Experiment Parameters.txt', 'w') as f:
        for line in lines:
            f.writelines(line)
            f.writelines('\n')
    
    return 