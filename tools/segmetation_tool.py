from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tkinter import Tk, Text, TOP, BOTH, INSERT

""""##############################################################################################################################################################################"""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def show_description_window(event):
    my_text = ("<Right Click> : Create fragment window" +
                "\n<Backspace> : Delete last window" +
                "\n<Enter> or <Esc> : Submit windows & Close\n\n")
    root = Tk()
    root.geometry("370x200")
    root.title("Description")
    text = Text(root, background="lightyellow")
    text.tag_configure("black", foreground="black")
    text.tag_configure("darkgrey", foreground="darkgrey")
    text.tag_configure("red", foreground="red")
    text.tag_configure("green", foreground="green")
    text.tag_configure("blue", foreground="blue")
    text.tag_configure("orange", foreground="orange")
    text.insert(INSERT, my_text,"black")
    text.insert(INSERT, "• <0> : Ιnactivity (default)\n", "darkgrey")
    text.insert(INSERT, "• <1> : Left Movement\n", "red")
    text.insert(INSERT, "• <2> : Right Movement\n", "green")
    text.insert(INSERT, "• <3> : Up Movement\n", "blue")
    text.insert(INSERT, "• <4> : Down Movement\n", "orange")
    text.config(state="disabled")
    text.pack(side=TOP, fill=BOTH, expand=True)
    root.mainloop()


def on_right_click(event,fig,ax_list,signals,movements_list,points,windows,label_id,sample_rate,window_lenght) -> None:
    """
    Function to handle right click events.    
    """
    # Set label ID
    if label_id[0] == 0 : color = "darkgrey"
    elif label_id[0] == 1 : color = "red"
    elif label_id[0] == 2 : color = "green"
    elif label_id[0] == 3 : color = "blue"
    elif label_id[0] == 4 : color = "orange"
    # Check if the event was a right click
    if event.button == 3:
        # Get the x-axis position of the right click
        x = int(event.xdata)
        # Create window borders
        span_start = x - (window_lenght/2)*sample_rate
        span_end = x + (window_lenght/2)*sample_rate
        # Check if window borders are correct
        if(span_start>=0 and span_end<=len(signals[0])):
            # Add movement
            movements_list[0].append(label_id[0]), movements_list[1].append(x)
            # Add a colored dot to each signal plot
            for i in range(len(ax_list)):
                points[i].append(ax_list[i].scatter(x, signals[i][x], c=color, zorder=2))
            # Add a vertical span to each signal plot
            for i in range(len(ax_list)):
                windows[i].append(ax_list[i].axvspan(span_start, span_end, color=color, alpha=0.4))
            # Print message to console
            print("Signal Fragment Added")
            # Update plots
            fig.canvas.draw()
        else: print(f"{bcolors.WARNING} WARNING ---> The window you define is out of the signal borders. Please re-define your window.{bcolors.ENDC}")


def on_key_press(event,fig,movements_list,points,windows,label_id) -> None:
    """
    Connect the key press event to the function.
    Keyboard buttons description:
        - Backspace: Erase last segmentation
        - ESC or Enter: Close plot / Submit your signal windows
        - Numbers 0-4: Set label    
    """
    if event.key == 'backspace':
        if len(points[0])>0:
            for i in range(len(points)):
                point = points[i].pop()
                point.remove()
                window = windows[i].pop()
                window.remove()
            movements_list[0].pop(), movements_list[1].pop()
            print("Last Signal Fragment Erased")
        else:
            print("Empty - No fragment to delete")
        fig.canvas.draw()
    elif event.key == 'escape' or event.key == 'enter':
        plt.close()
    elif event.key == '0':
        label_id.pop()
        label_id.append(0)
    elif event.key == '1':
        label_id.pop()
        label_id.append(1)
    elif event.key == '2':
        label_id.pop()
        label_id.append(2)
    elif event.key == '3':
        label_id.pop()
        label_id.append(3)
    elif event.key == '4':
        label_id.pop()
        label_id.append(4)
         

def create_figure(signals, muse_type, Fs, window_lenght):
    if(muse_type == "EEG" and len(signals) == 4):
        signal_names = ["F7","F8","TP9","TP10"]
        # Create a figure and axes
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        #fig.subplots_adjust(left=0.25), fig.subplots_adjust(right=0.975)
        fig.subplots_adjust(top=0.93), fig.subplots_adjust(bottom=0.09)
        fig.text(0.5, 0.04, 'Samples', ha='center')
        fig.text(0.05, 0.5, 'Amplitube (Voltage [$\mu V$])', va='center', rotation='vertical')
        fig.suptitle('EEG Signals Segmentation/Labeling', fontsize=16)
    
        # Plot signals
        ax1.plot(signals[0],label=signal_names[0],color="blue")
        ax2.plot(signals[1],label=signal_names[1],color="darkcyan")
        ax3.plot(signals[2],label=signal_names[2],color="red")
        ax4.plot(signals[3],label=signal_names[3],color="orange")
        ax1.legend(loc='upper left'),ax2.legend(loc='upper left'),ax3.legend(loc='upper left'),ax4.legend(loc='upper left')
        # Set Grids
        ax1.grid(), ax2.grid(), ax3.grid(), ax4.grid()
        axs = [ax1,ax2,ax3,ax4]

        # Keep track of the points and windows for 4 signals
        # Init points and windows lists
        ax1_points, ax2_points, ax3_points, ax4_points = [], [], [], []
        ax1_windows, ax2_windows, ax3_windows, ax4_windows = [], [], [], []
        points = [ax1_points, ax2_points, ax3_points, ax4_points]
        windows = [ax1_windows, ax2_windows, ax3_windows, ax4_windows]
    elif( (muse_type == "Gyroscope" or muse_type == "gyroscope") and len(signals) == 3):
        signal_names = ["X","Y","Z"]
        # Create a figure and axes
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.subplots_adjust(top=0.93), fig.subplots_adjust(bottom=0.09)
        fig.text(0.5, 0.04, 'Samples', ha='center')
        fig.text(0.05, 0.5, 'Amplitube ', va='center', rotation='vertical')
        fig.suptitle('Gyroscope Signals Segmentation/Labeling', fontsize=16)
        # Plot signals
        ax1.plot(signals[0],label=signal_names[0],color="red")
        ax2.plot(signals[1],label=signal_names[1],color="green")
        ax3.plot(signals[2],label=signal_names[2],color="blue")
        ax1.legend(loc='upper left'),ax2.legend(loc='upper left'),ax3.legend(loc='upper left')
        # Set Grids
        ax1.grid(), ax2.grid(), ax3.grid()
        axs = [ax1,ax2,ax3]
        ax1_points, ax2_points, ax3_points = [], [], []
        ax1_windows, ax2_windows, ax3_windows = [], [], []
        points = [ax1_points, ax2_points, ax3_points]
        windows = [ax1_windows, ax2_windows, ax3_windows]
    else:
        print("ERROR")
        return [-1],[-1]
    
    # Init movements list
    movements_list = [[],[]]
    # Init label ID
    label_id = [0]

    # Create Description Button
    plt.subplots_adjust(bottom=0.2)
    callback = show_description_window
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
    button = Button(ax_button, 'Description')
    button.on_clicked(callback)

    # Connect the keyboard events to the function
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event,fig,movements_list,points,windows,label_id))
    # Connect the right click event to the function
    fig.canvas.mpl_connect("button_press_event", lambda event: on_right_click(event,fig,axs,signals,movements_list,points,windows,label_id, Fs, window_lenght))

    # Maximize fig window
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # Show the plot
    plt.show()
    
    # Return movements list
    return movements_list[0], movements_list[1] 


def create_dataframe(dataframe,muse_mode, movement_marker_list, movement_label_list, fs, window_lenght):
    if(muse_mode=="EEG"): columns = ['Sample_Counter','EEG.F7','EEG.F8','EEG.TP9','EEG.TP10','Label']
    elif(muse_mode=="Gyroscope"): columns = ['Sample_Counter','X','Y','Z','Label']
    df = pd.DataFrame(columns = columns)
    row = 0
    for i in range(len(movement_marker_list)):
        left_window_border = movement_marker_list[i] - ((fs * (window_lenght/2)) - 1)
        signal_window = [left_window_border]
        for j in range((fs * window_lenght) - 1):
            left_window_border = left_window_border + 1
            signal_window.append(left_window_border)
        data_window = dataframe.take(signal_window)

        for sample in range(len(data_window)): # len(data_window) = 512 samples
            if(muse_mode=="EEG"):
                info = [sample, data_window.iloc[sample][0], data_window.iloc[sample][1],data_window.iloc[sample][2],data_window.iloc[sample][3], movement_label_list[i]]
            elif(muse_mode=="Gyroscope"):
                 info = [sample, data_window.iloc[sample][0], data_window.iloc[sample][1],data_window.iloc[sample][2], movement_label_list[i]]
            df.loc[row] = info
            row += 1
    return df

""""##############################################################################################################################################################################"""

def signal_Segmentation(signal_df:pd, Fs=128, window_lenght = 4):

    """
    Description
    ----------
    A tool that splits the given signal dataframe into labeled fragments based on markers you set.

    Parameters
    ----------
    signal_df : pandas.DataFrame
        The signal dataframe must containing exactly 
            4 columns: EEG signals or 
            3 columns: Gyroscope signals
    *Fs : int
        Sample rate (128 Hz by default).
    *window_lenght : int
        The length of each window in seconds (4 sec by default)

    Returns
    -------
    pandas.DataFrame
        A dataFrame which contains each window/fragment and its label.
        
    Notes
    -----
    - Depending on the dataframe structure that is imported the mode can be: EEG (4 signals) or Gyroscope(3 signals).

    """
    signals = signal_df.transpose().values
    if (len(signals)==4): muse_mode = "EEG"   
    elif (len(signals)==3) : muse_mode = "Gyroscope"
    else:
        empty_df = pd.DataFrame() 
        print("WARNING ---> The dataframe structure being imported is incorrect.\n" +
                "The dataframe must have:\n" +
                "Exactly 4 columns/signals (F7,F8,TP9,TP10) in EEG mode or\n Exactly 3 columns/signals (X,Y,Z) in Gyroscope mode.")
        return empty_df
    print("Segmentation tool Mode: " + f"{bcolors.OKGREEN}"  + muse_mode + f"{bcolors.ENDC}")

    labels_list, window_markers_list = create_figure(signals, muse_mode, Fs, window_lenght)
    if (labels_list and window_markers_list):
        # Combine the two lists into a list of tuples
        zipped_lists = list(zip(labels_list, window_markers_list))
        # Sort the list of tuples based on the first element of each tuple
        zipped_lists.sort(key=lambda x: x[0])
        # Unpack the sorted list of tuples back into two separate lists
        labels_list, window_markers_list = zip(*zipped_lists)

    return create_dataframe(signal_df,muse_mode, window_markers_list, labels_list,Fs,window_lenght)
