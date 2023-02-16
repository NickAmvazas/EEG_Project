import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def MNE_show(EEG_dataset:pd, annotations = [], channels_only = True, show_psd = False, show_sensors = False, muse_mode = "EEG", sampling_freq = 128):
    
    mne.set_log_level('ERROR')  # To avoid flooding the cell outputs with messages
    if(annotations):
        #onset = [0, 45, 55, 60, 65, 70, 75, 80 , 85 , 90]
        #duration = [45, 10, 5, 5, 5, 5, 5, 5, 5, 5]
        #description = ['Base Line','Frontal Open','F to R','R to F','F to L','L to F','F to Up','Up to F','F to Down','Down to F']
        annots = mne.Annotations(annotations[0], annotations[1], annotations[2])

    if(channels_only):
        signals = EEG_dataset.transpose().to_numpy()
    else:
        signals = EEG_dataset
        
    # Assigning the channel type when initializing the Info object
    if (muse_mode == "EEG"):
        ch_names = ['F7','F8','TP9','TP10']
        ch_types = ['eeg', 'eeg', 'eeg', 'eeg']

    montage = mne.channels.make_standard_montage("standard_1020") # We can use mne.channels.get_builtin_montages() to see all different types of montages in mne.

    # Create Info
    info = mne.create_info(ch_names= ch_names, ch_types= ch_types, sfreq= sampling_freq)
    Eeg_Raw = mne.io.RawArray(signals, info)
    Eeg_Raw.set_montage(montage = montage)
    if(annotations):
        Eeg_Raw.set_annotations(annots, emit_warning=False)

    #show sensors positions
    if(show_sensors):
        Eeg_Raw.plot_sensors(show_names=True, kind='topomap')
    
    scalled_Eeg_Raw = Eeg_Raw.copy().apply_function(lambda x: x * 1e-6)

    #Plots
    scalled_Eeg_Raw.plot(scalings=250e-6,duration=140)#title = files_info.iloc[file_id][1])#,group_by='position')
    if(show_psd):
        scalled_Eeg_Raw.plot_psd()
    print("MNE figure displayed...")
    plt.show()
    

def test_figure(EEG_dataset,window_size = 0):

    signal_names = ["F7","F8","TP9","TP10"]
    # Create a figure and axes
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    #fig.subplots_adjust(left=0.25), fig.subplots_adjust(right=0.975)
    fig.subplots_adjust(top=0.93), fig.subplots_adjust(bottom=0.09)
    fig.text(0.5, 0.04, 'Samples', ha='center')
    fig.text(0.05, 0.5, 'Amplitube (Voltage [$\mu V$])', va='center', rotation='vertical')
    fig.suptitle('EEG Signals Segmentation/Labeling', fontsize=16)
    
    # Plot signals
    ax1.plot(EEG_dataset[0],label=signal_names[0],color="blue")
    ax2.plot(EEG_dataset[1],label=signal_names[1],color="darkcyan")
    ax3.plot(EEG_dataset[2],label=signal_names[2],color="red")
    ax4.plot(EEG_dataset[3],label=signal_names[3],color="orange")
    ax1.legend(loc='upper left'),ax2.legend(loc='upper left'),ax3.legend(loc='upper left'),ax4.legend(loc='upper left')
    # Set Grids
    ax1.grid(), ax2.grid(), ax3.grid(), ax4.grid()
    
    if (not(window_size == 0)):
        for i in range(0, len(EEG_dataset), window_size):
            ax1.axvline(x=i),ax2.axvline(x=i),ax3.axvline(x=i),ax4.axvline(x=i)
            ax1.axvline(x=i+window_size),ax2.axvline(x=i+window_size),ax3.axvline(x=i+window_size),ax4.axvline(x=i+window_size)

    plt.show()

