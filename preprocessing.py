import os
import pandas as pd
import mne
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tools.segmetation_tool import signal_Segmentation
from tools.figures_tool import MNE_show, test_figure

""""--------------------------------------------------------------------"""

def preprocessing_flag():
    while True:
        processing_flag = input('Would you like to preprocess (Denoising-Segmentation-Normalization) this file?(Yes/No)')
        yeah = ["Yes","y","Y","YES","yes"]
        nope = ["no","NO", "n","N","No"]
        if (processing_flag in yeah): return True
        elif (processing_flag in nope): return False
        else: print("Please type Yes/No")

def load_file():
    # User enters the file type
    while True:
        file_type = input('Enter a file type (EEG or GS (Gyroscope): ')
        accepted_file_types_eeg = ["EEG","eeg"]
        accepted_file_types_gyroscope = ["GS","gs","GYROSCOPE","gyroscope","Gyroscope"]
        if (file_type in accepted_file_types_eeg): 
            file_type = "EEG"
            break
        elif (file_type in accepted_file_types_gyroscope): 
            file_type = "GYROSCOPE"
            break
        else: print("Wrong file type")
    # User enters the file path
    file_path = input('Enter a file path: ')
    filename = os.path.splitext(os.path.basename(file_path))[0]
    try:
        if(file_type == "EEG"):
            dataframe = pd.read_csv(file_path,header = 1,usecols = ['EEG.F7','EEG.F8','EEG.TP9','EEG.TP10']) 
            dataframe = dataframe.loc[:, ['EEG.F7','EEG.F8','EEG.TP9','EEG.TP10']]
        elif(file_type == "GYROSCOPE"):
            dataframe = pd.read_csv(file_path,header = 1,usecols = ['X','Y','Z']) 
            dataframe = dataframe.loc[:,['X','Y','Z']]
    except ValueError:
        print("Oops! There was a problem with this file ")
    return dataframe, file_type, filename

""""--------------------------------------------------------------------"""

def bandpass_filter(signal):
    filtered_signal = mne.filter.filter_data(signal,sfreq=128, l_freq=0.5, h_freq=12.5,fir_design='firwin')
    return filtered_signal

def denoising(dataframe:pd):
    signals = dataframe.transpose().values
    for channel_num in range(signals.shape[0]):
        signals[channel_num] = bandpass_filter(signals[channel_num])

    # Creating a new dataframe with filtered EEG Signals
    column_values = ['EEG.F7','EEG.F8','EEG.TP9','EEG.TP10']
    return pd.DataFrame(data = signals.transpose(),  columns = column_values)

""""--------------------------------------------------------------------"""

def normalization(fragments_dataset:pd,feature_range = (-1, 1)):
    signals, info = fragments_dataset.drop(['Sample_Counter','Label'],axis=1), fragments_dataset.drop(['EEG.F7','EEG.F8','EEG.TP9','EEG.TP10'],axis=1)
    scaler = MinMaxScaler(feature_range)
    norm_signals = pd.DataFrame(scaler.fit_transform(signals))
    # Merge the 2 dataframes (signals & infos) horizontally
    merged_df = pd.merge(norm_signals, info, left_index=True, right_index=True)
    return merged_df

""""--------------------------------------------------------------------"""

def remove_DC_offset(fragment_dataset:pd):
    return (fragment_dataset - fragment_dataset.mean()) # Mean Filter

""""--------------------------------------------------------------------"""

def main():
    print("Preprocesing Script Started")
    # Load and plot Raw File
    dataframe, file_type, filename = load_file()
    MNE_show(dataframe,show_psd = True, show_sensors = True)
    os.system('cls' if os.name=='nt' else 'clear')
    # Start Preprocessing
    if(preprocessing_flag()):

        # 1. Filtering(bandpass) / Denoisng and plot filtered signals 
        filtered_signals_df = denoising(dataframe)
        MNE_show(filtered_signals_df,show_psd = True)

        # 2. Segmantation
        fragments = signal_Segmentation(filtered_signals_df)

        # 3. Normalization
        if(not(fragments.empty)): ppp = normalization(fragments)
        else: 
            print("No Signal Fragments - Exit.")
            exit()

        # 4. Split the dataframe into windows
        fs = 128
        window_lenght = 4
        window_size = fs * window_lenght
        
        # Plot all the fragments as 1 signal
        test_figure(ppp.drop(ppp.columns[-2:], axis=1),window_size)

        fragment_dfs, labels = [], []
        for i in range(0, len(ppp), window_size):
            window_label = ppp['Label'].iloc[i]
            window_df = ppp.iloc[i:i+window_size].drop(['Sample_Counter','Label'],axis=1).reset_index(drop=True)
            # 5. Remove DC Offset
            window_df = remove_DC_offset(window_df)

            print(window_label), test_figure(window_df,window_size)
            fragment_dfs.append(window_df)
            labels.append(window_label)
        plt.show()

        # Save Fragments
        data = {"df_lsit" : fragment_dfs , "df_labels" : labels}
        folder_name = 'Labeled_Fragments_Data'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")
        file_path = folder_name + "//" + filename + ".pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data,f)

    else: exit()

if __name__ == "__main__":
    main()