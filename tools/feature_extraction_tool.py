"""
Feature Extraction refers to the process of transforming raw data into numerical features
that can be processed while preserving the information in the original data set. 
"""
import os, pickle
# Imports
import numpy as np
import math
import mne
import matplotlib.pyplot as plt

def calc_Mean(segment:np):
    """
    Calculate the mean of each channel of the segment.

    Args:
    segment (numpy array): Segment of shape (n_samples, n_channels).

    Returns:
    numpy array: Mean of each channel in the segment, shape: (n_channels,).
    """
    return np.mean(segment, axis=0)

def calc_Median(segment:np):
    """
    Calculate the median of each channel in an EEG segment.

    Args:
    segment (numpy array): Segment of shape (n_samples,n_channels).

    Returns:
    numpy array: Median of each channel in the segment, shape: (n_channels,).
    """
    return np.median(segment, axis=0)

def calc_RMS(segment:np):
    for channel in range(segment.shape[1]):
        print(math.sqrt(np.mean(np.power(segment[:,channel],2))))




def calc_pk2pk(segment:np):
    """
    Calculate the Peak-to-peak value of each channel in the segment.
    
    Args:
    segment (numpy array): Segment of shape (n_samples,n_channels).

    Returns:
    numpy array: Peak-to-peak value of each channel in the segment, shape: (n_channels,).
    """
    return np.max(segment, axis=0) - np.min(segment, axis=0)

def calc_STD(segment:np):
    """
    Calculate the Standard Deviation of each channel in the segment.

    Args:
    segment (numpy array): Segment of shape (n_samples, n_channels).

    Returns:
    numpy array: Standard Deviation of each channel in the segment, shape: (n_channels,).
    """
    return np.std(segment, axis=0)

def calc_Variance(segment:np):
    """
    Calculate the Variance of each channel in the segment.

    Args:
    segment (numpy array): Segment of shape (n_samples, n_channels).

    Returns:
    numpy array: Variance of each channel in the segment, shape: (n_channels,).
    """
    return np.var(segment, axis=0)

def calc_Skewness(segment:np):
    """
    Calculate the Skewness of each channel in the segment.

    Args:
    segment (numpy array): Segment of shape (n_samples, n_channels).

    Returns:
    numpy array: Skewness of each channel in the segment, shape: (n_channels,).
    """
    return np.apply_along_axis(lambda x: np.nan_to_num(np.mean((x - np.mean(x))**3) / (np.mean((x - np.mean(x))**2)**(3/2))), 0, segment)
    
def calc_Kurtosis(segment:np):
    """
    Calculate the Kurtosis of each channel in the segment.

    Args:
    segment (numpy array): Segment of shape (n_samples, n_channels).

    Returns:
    numpy array: Kurtosis of each channel in the segment, shape: (n_channels,).
    """
    return np.apply_along_axis(lambda x: np.nan_to_num(np.mean((x - np.mean(x))**4) / (np.mean((x - np.mean(x))**2)**2)), 0, segment)

def calc_Energy(segment:np):
    """
    Calculate Energy of each channel in the segment.
    
    Args:
        segment (numpy array): Segment of shape (n_samples, n_channels).
        
    Returns:
        energy (list of floats): List of energy values for each channel.
    """
    return np.apply_along_axis(lambda x: np.sum(np.square(x)), 0, segment)

def hjorth_params(segment:np):
    """
    Compute Hjorth parameters for an EEG segment.
    
    Parameters:
    -----------
    segment : numpy array, shape (n_samples, n_channels)
        EEG segment to compute Hjorth parameters for.
        
    Returns:
    --------
    activity : numpy array, shape (n_channels,)
        Hjorth activity parameter for each channel.
    mobility : numpy array, shape (n_channels,)
        Hjorth mobility parameter for each channel.
    complexity : numpy array, shape (n_channels,)
        Hjorth complexity parameter for each channel.
    """
    # Compute first derivative
    diff1 = np.diff(segment, axis=0)
    diff1 = np.vstack([diff1[0], diff1])

    # Compute second derivative
    diff2 = np.diff(segment, n=2, axis=0)
    diff2 = np.vstack([diff2[0], diff2[0], diff2])

    # Compute Hjorth activity, mobility, and complexity
    activity = np.var(segment, axis=0)
    mobility = np.sqrt(np.var(diff1, axis=0) / activity)
    complexity = np.sqrt(np.var(diff2, axis=0) / np.var(diff1, axis=0) * mobility)
    
    return activity, mobility, complexity









def extract_bands(segment, sfreq):
    """
    Extracts frequency bands from a given EEG segment.

    Parameters:
    -----------
    segment (numpy array): EEG segment of shape (n_samples, n_channels).
    sfreq (float): Sampling frequency of the EEG segment.

    Returns:
    --------
    bands (numpy array): Array of shape (n_samples, n_channels) containing
                        the amplitude of the following frequency bands for each
                        channel in the input segment: delta (0.5-4 Hz), theta (4-8 Hz),
                        alpha (8-13 Hz), and beta (13-30 Hz).
    """
    
    # Define the frequency bands of interest
    bands = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "alpha": [8, 13],
        "beta": [13, 30]
    }
    filter_length = 'auto' # Automatic filter length
    fir_design = 'firwin' # Use finite impulse response (FIR) filter design
    fir_window = 'hamming' # Use Hann window

    # Create an empty array to hold the band-limited signals
    band_signals = np.zeros((segment.shape[0], segment.shape[1] * len(bands)))
    
    # Iterate over each channel
    for i in range(segment.shape[1]):
        # Get the current channel's signal
        signal = segment[:, i]
        
        # Apply a bandpass filter to the signal for each frequency band
        bandpass_signals = []
        for band in bands.values():
            bandpass_signal = mne.filter.filter_data(signal, sfreq = sfreq,
                                            l_freq = l_freq, h_freq = h_freq,
                                            filter_length = filter_length, fir_design = fir_design,
                                            fir_window = fir_window)
            bandpass_signals.append(bandpass_signal)
            
        # Concatenate the bandpass signals into a single array for this channel
        channel_signals = np.concatenate(bandpass_signals)
        
        # Add the channel signals to the overall array
        band_signals[:, i * len(bands): (i + 1) * len(bands)] = channel_signals.reshape(-1, len(bands))
    
    return band_signals



###############################################################
# Set the file path
file_path = "trial_001_08.12.19_18.02.37.pkl"
fragments, fragments_id = [], []

try:
    with open(file_path,'rb') as f:
        fragment_dict = pickle.load(f)
        for fragment in fragment_dict["df_lsit"]:
            fragments.append(fragment)
        for label in fragment_dict["df_labels"]:
            fragments_id.append(label)
except ValueError:
    print("Oops! There was a problem with file: ")

print("Total number of Fragmnets: " + str(len(fragments)))




nnn = 9
print(fragments_id[nnn])

print(calc_pk2pk(fragments[nnn].to_numpy()))
