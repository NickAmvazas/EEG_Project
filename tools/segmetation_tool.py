import matplotlib.pyplot as plt
from matplotlib.patches import BoxStyle
#import matplotlib.widgets as widgets
#from matplotlib.widgets import TextBox

def on_right_click(event,fig,ax_list,signals,x_lst,points,windows,set_label_id) -> None:
    """
    Function to handle right click events.    
    """
    sample_rate = 256
    
    if set_label_id[0] == 0 : color = "black"
    elif set_label_id[0] == 1 : color = "red"
    elif set_label_id[0] == 2 : color = "green"
    elif set_label_id[0] == 3 : color = "blue"
    # Check if the event was a right click
    if event.button == 3:
        # Get the x-axis position of the right click
        x = int(event.xdata)
        # Add a x
        x_lst.append(x)
        # Add a colored dot to the plot
        for i in range(len(ax_list)):
            points[i].append(ax_list[i].scatter(x, signals[i][x], c=color, zorder=2))

        # Add a vertical span to the plot
        span_start = max(x - 2*sample_rate, 0)
        span_end = min(x + 2*sample_rate, len(signals[0]))
        for i in range(len(ax_list)):
            windows[i].append(ax_list[i].axvspan(span_start, span_end, color='yellow', alpha=0.4))

        # Update the plot
        fig.canvas.draw()
        
def on_key_press(event,fig,x_lst,points,windows,set_label_id):
    """
    Connect the key press event to the function.    
    """
    if event.key == 'backspace':
        if len(points[0])>0:
            for i in range(len(points)):
                point = points[i].pop()
                point.remove()
                window = windows[i].pop()
                window.remove()
            x_lst.pop()
        else:
            print("Empty")
        fig.canvas.draw()
    elif event.key == 'escape' or event.key == 'enter':
        plt.close()
    elif event.key == '0':
        set_label_id.pop()
        set_label_id.append(0)
    elif event.key == '1':
        set_label_id.pop()
        set_label_id.append(1)
    elif event.key == '2':
        set_label_id.pop()
        set_label_id.append(2)
    elif event.key == '3':
        set_label_id.pop()
        set_label_id.append(3)
        
        
def signal_segmentation(signals,signal_names = ["EEG.F7","EEG.F8","TP9","TP10"]):

    # Create a figure and two axes
    fig, (ax1, ax2, ax3, ax4 ) = plt.subplots(4, 1)
    fig.subplots_adjust(left=0.25), fig.subplots_adjust(right=0.975)
    fig.subplots_adjust(top=0.95), fig.subplots_adjust(bottom=0.09)

 
    # add text description in a box
    title = "Muse 2 WM"
    description = ("Description: " +
                "\n<Right Click> : Create fragment window" +
                "\n<Backspace> : Delete last window"+
                "\n<Enter> or <Esc> : Submit windows & Close"+
                "\n<0> : " +
                "\n<1> : " +
                "\n<2> : " +
                "\n<3> : ") 
    plt.figtext(0.07, 0.65, title, bbox=dict(facecolor='green', alpha=0.5, boxstyle=BoxStyle("round", pad=0.5)), fontsize=17,va='center', ha='left')
    plt.figtext(0.02, 0.5, description, bbox=dict(facecolor='lightblue', alpha=0.5, boxstyle=BoxStyle("round", pad=0.5)), fontsize=10,va='center', ha='left')
    
    # Plot signals
    ax1.plot(signals[0],label=signal_names[0])
    ax1.legend()
    ax2.plot(signals[1])
    ax3.plot(signals[2])
    ax4.plot(signals[3])
    #Set Grids
    ax1.grid(), ax2.grid(), ax3.grid(), ax4.grid()

    # Keep track of the points and windows for 4 signals
    ax1_points, ax2_points, ax3_points, ax4_points = [], [], [], []
    ax1_windows, ax2_windows, ax3_windows, ax4_windows = [], [], [], []
    points = [ax1_points, ax2_points, ax3_points, ax4_points]
    windows = [ax1_windows, ax2_windows, ax3_windows, ax4_windows]
    x_lst = []
    set_label_id = [0]
     
    # Connect the keyboard events to the function
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event,fig,x_lst,points,windows,set_label_id))
    # Connect the right click event to the function
    fig.canvas.mpl_connect("button_press_event", lambda event: on_right_click(event,fig,[ax1,ax2,ax3,ax4],signals,x_lst,points,windows,set_label_id ))

    # Show the plot
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    return x_lst


def main():
    print("Hello World!")
        # Create a sample signal
    samples = 128
    signals = []
    signal1 = [i for i in range(samples*100)]
    signals.append(signal1)
    signal2 = [i**2 for i in range(samples*100)]
    signals.append(signal2)
    signals.append(signal1)
    signals.append(signal1)
    
      # Create a sample signal
    print(signal_segmentation(signals))

if __name__ == "__main__":
    main()