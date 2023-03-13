import os, pickle, mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler
from tools.segmetation_tool import signal_Segmentation



"""---------------------------------------------------------SETUP------------------------------------------------------------"""

FRAGMENTS_FOLDER_NAME = 'XAXA'                              # Set the name of the folder in which the fragments of the signals will be stored.
FILE_TYPE = "EEG"                                           # Set the file type (Can be "EEG" or "GYROSCOPE").
EEG_CHANNELS = ['EEG.F7', 'EEG.F8', 'EEG.TP9', 'EEG.TP10']  # Set the EEG channels u want to use
FS = 128                                                    # Set Sample Rate (Hz) || If APPLY_RESAMPLING = True then FS is the new FS

APPLY_RESAMPLING = False
INITIAL_FS = FS                                             # If APPLY_RESAMPLING = True: Set Initail Sample Rate (Hz) || If APPLY_RESAMPLING = False: INITIAL_FS = FS

APPLY_FILTERING = True
NOTCH_FILTER = 50                                           # Set Notch Filter (Hz)
LOWER_FREQUENCY_BOUND = 0.5                                 # Set Bandpass Filter Low Bound (Hz)
UPPER_FREQUENCY_BOUND = 8                                   # Set Bandpass Filter High Bound (Hz)

APPLY_SEGMENTATION = True
WINDOW_LENGHT = 4                                           # Set Segmentation Window Lenght (sec)

APPLY_NORMALIZATION = False
REMOVE_DC_OFFSET = True

"""--------------------------------------------------------------------------------------------------------------------------"""

def preprocessing_flag():
    """
    Description
    ----------
    Asks the user if they want to edit the file (Yes/No).
    
    """
    while True:
        processing_flag = input('Would you like to preprocess this file?(Yes/No): ')
        yeah = ["Yes","y","Y","YES","yes"]
        nope = ["no","NO", "n","N","No"]
        if (processing_flag in yeah): return True
        elif (processing_flag in nope): return False
        else: print("Please type Yes/No")

def load_file():
    # User enters the file path
    file_path = input('Enter a file path: ')
    filename = os.path.splitext(os.path.basename(file_path))[0]
    try:
        if(FILE_TYPE == "EEG"):
            dataframe = pd.read_csv(file_path, header = 1, usecols = EEG_CHANNELS) 
            dataframe = dataframe.loc[:, EEG_CHANNELS]
        elif(FILE_TYPE == "GYROSCOPE"):
            dataframe = pd.read_csv(file_path,header = 1, usecols = ['X','Y','Z']) 
            dataframe = dataframe.loc[:, ['X','Y','Z']]
        return dataframe, filename
    except ValueError:
        print("Oops! There was a problem with this file ")
        exit()

"""--------------------------------------------------------------------------------------------------------------------------"""

def plot_EEG_MNE(data_EEG:pd, fs=FS, annotations = []):
    plt.style.use('default')
    mne.set_log_level('ERROR')  # To avoid flooding the cell outputs with messages
    if(annotations):
        annots = mne.Annotations(annotations[0], annotations[1], annotations[2])

    signals = data_EEG.transpose().to_numpy()

    ch_names = [item.split('.')[1] for item in EEG_CHANNELS]
    ch_types = ['eeg' for chanel in ch_names]

    montage = mne.channels.make_standard_montage("standard_1020") # We can use mne.channels.get_builtin_montages() to see all different types of montages in mne.

    # Create Info
    info = mne.create_info(ch_names= ch_names, ch_types= ch_types, sfreq= fs)
    Eeg_signals = mne.io.RawArray(signals, info)
    Eeg_signals.set_montage(montage = montage)
    if(annotations):
        Eeg_signals.set_annotations(annots, emit_warning=False)

    scalled_Eeg_signals = Eeg_signals.copy().apply_function(lambda x: x * 1e-6)
    # Ploting
    scalled_Eeg_signals.plot(scalings=220e-6,duration=200) 
    print("MNE figure displayed...")
    plt.show()

def calc_psd(df,fs=FS,get=False):
    """
    Description
    ----------
    Calculate & Plot the PSD for each EEG signal.
    
    Parameters
    ----------
    df (pandas.DataFrame): A dataframe containing EEG signalsn.
    get (boolean) : Set True if you want to return PSD and Freq.
    
    Returns
    ----------
    If get = True : returns PSD and Freq.
    """
    nperseg = 4*fs # Length of each segment for Welch's method
    window = 'hann' # type of window for Welch's method

    # Set the style to use a gray background and gridlines
    plt.style.use('ggplot')

    # Loop through each column of the dataframe and calculate PSD using Welch's method
    for col in df.columns:
        f, Pxx = signal.welch(df[col], fs=fs, nperseg=nperseg, window=window)
        print(f.shape,Pxx.shape)
        plt.semilogy(f, Pxx, label = col)  # Use semilog to plot the PSD in dB.
        
    # Set plot labels and title
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('PSD of EEG Signals')
    
    # Add legend, gridlines, and display the plot
    plt.legend()
    plt.grid(True, alpha=0.8)
    plt.show()
    if(get): return f, Pxx

"""--------------------------------------------------------------------------------------------------------------------------"""

def resample_signals(df, old_fs, new_fs):
    """
    Resample each EEG signal in a pandas dataframe from an old sampling frequency
    to a new sampling frequency using scipy's resample function.
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing columns of EEG signals
    old_fs (int): The original sampling frequency of the signals
    new_fs (int): The desired sampling frequency of the signals after resampling
    
    Returns:
    pandas.DataFrame: A new dataframe with the resampled signals
    """
    # Calculate the resampling factor
    resample_factor = new_fs / old_fs
    # Calculate the new length of each signal after resampling
    new_length = int(len(df) * resample_factor)
    # Resample each signal using scipy's resample function
    resampled_signals = []
    for signal in df.values.T:
        resampled_signal = resample(signal, new_length)
        resampled_signals.append(resampled_signal)
    # Convert the resampled signals back to a pandas dataframe
    resampled_df = pd.DataFrame(np.array(resampled_signals).T, columns=df.columns)
    
    return resampled_df

"""--------------------------------------------------------------------------------------------------------------------------"""

def notch_filter(signal):
    filtered_signal = mne.filter.notch_filter(signal, Fs = FS,
                                            freqs=NOTCH_FILTER,
                                            filter_length = 'auto', fir_design = 'firwin')
    return filtered_signal

def bandpass_filter(signal):
    # Define filter parameters
    filter_length = 'auto' # Automatic filter length
    fir_design = 'firwin' # Use finite impulse response (FIR) filter design
    fir_window = 'hamming' # Use Hann window
    filtered_signal = mne.filter.filter_data(signal, sfreq = FS,
                                            l_freq = LOWER_FREQUENCY_BOUND, h_freq = UPPER_FREQUENCY_BOUND,
                                            filter_length = filter_length, fir_design = fir_design,
                                            fir_window = fir_window)
    return filtered_signal

def denoising(dataframe:pd):
    signals = dataframe.transpose().values
    for channel_num in range(signals.shape[0]):
        signals[channel_num] = bandpass_filter(notch_filter(signals[channel_num]))

    # Creating a new dataframe with filtered EEG Signals
    return pd.DataFrame(data = signals.transpose(),  columns = EEG_CHANNELS)

"""--------------------------------------------------------------------------------------------------------------------------"""

def normalization(signals_df:pd,feature_range = (-1, 1)):
    scaler = MinMaxScaler(feature_range)
    norm_signals = pd.DataFrame(scaler.fit_transform(signals_df))
    return norm_signals

"""--------------------------------------------------------------------------------------------------------------------------"""

def remove_DC_offset(fragment_dataset:pd):
    """
    Description
    ----------
    Remove the DC offset from each EEG signal fragmnet (using SciPy's signal.detrend function).
    
    Parameters
    ----------
    fragment_dataset (pandas.DataFrame): A dataframe containing a window of EEG signals.
    
    Returns
    -------
    pandas.DataFrame: A new dataframe with the DC offset removed from each signal.
    """

    detrended = signal.detrend(fragment_dataset, axis=0)
    detrended_df = pd.DataFrame(detrended, columns=fragment_dataset.columns)
    return detrended_df

"""--------------------------------------------------------------------------------------------------------------------------"""

def main():
    print("Preprocesing Script Started")
    
    # Load the channels you choose from Raw File
    dataframe, filename = load_file()
    
    # Plot the Raw channels and the PSD
    plot_EEG_MNE(dataframe,fs=INITIAL_FS)
    calc_psd(dataframe,fs=INITIAL_FS)

    # Clear terminal
    os.system('cls' if os.name=='nt' else 'clear')
    
    # Start Preprocessing
    if(preprocessing_flag()):

        # 1.Resampling and replot
        if(APPLY_RESAMPLING): 
            dataframe = resample_signals(dataframe, 128, 4)
            print("Resampling applied")
            plot_EEG_MNE(dataframe)
            calc_psd(dataframe)

        # 2. Filtering (bandpass) / Denoisng and plot filtered signals 
        filtered_signals_df = denoising(dataframe)
        print("Filtering applied")
        calc_psd(dataframe)

        # 3. Signal Segmantation
        fragments_dataset = signal_Segmentation(filtered_signals_df, Fs=FS,window_lenght=WINDOW_LENGHT)
        if(fragments_dataset.empty): 
            print("No Signal Fragments - Exit.")
            exit()
        else: print("Signal Segmentation applied")

        signals_df, info_df = fragments_dataset.drop(['Sample_Counter','Label'],axis=1), fragments_dataset.drop(EEG_CHANNELS,axis=1)

        # 3. Normalization
        if(APPLY_NORMALIZATION): 
            signals_df = normalization(signals_df)
            print("Normalization Applied")
            plot_EEG_MNE(signals_df)
            calc_psd(signals_df)

        # Merge the 2 dataframes (signals & infos) horizontally
        merged_df = pd.merge(signals_df, info_df, left_index=True, right_index=True)

        # 4. Split the dataframe into windows
        window_size = FS * WINDOW_LENGHT
        
        fragment_dfs, labels = [], []
        for i in range(0, len(merged_df), window_size):
            window_label = merged_df['Label'].iloc[i]
            window_df = merged_df.iloc[i:i+window_size].drop(['Sample_Counter','Label'],axis=1).reset_index(drop=True)
            
            # 5. Remove DC Offset
            if(REMOVE_DC_OFFSET):
                window_df = remove_DC_offset(window_df)


            fragment_dfs.append(window_df)
            labels.append(window_label)
        plt.show()

        # Save Fragments
        data = {"df_lsit" : fragment_dfs , "df_labels" : labels}
        if not os.path.exists(FRAGMENTS_FOLDER_NAME):
            os.makedirs(FRAGMENTS_FOLDER_NAME)
            print(f"Folder '{FRAGMENTS_FOLDER_NAME}' created.")
        file_path = FRAGMENTS_FOLDER_NAME + "//" + filename + ".pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data,f)

    else: exit()

if __name__ == "__main__":
    main()