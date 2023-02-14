import pandas as pd
from tools.segmetation_tool import signal_Segmentation
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    dataframe = pd.read_csv("testfile.csv",
                header = 1,usecols = ['EEG.F7','EEG.F8','EEG.TP9','EEG.TP10']) 
    dataframe = dataframe.loc[:, ['EEG.F7','EEG.F8','EEG.TP9','EEG.TP10']]
except ValueError:
    print("Oops! There was a problem with file: ")
kapa = signal_Segmentation(dataframe).drop(['Sample_Counter','Label'],axis=1)
print(kapa)
plt.plot(kapa)
plt.show()

scaler = MinMaxScaler(feature_range=(-100, 100))

plt.plot(scaler.fit_transform(kapa))
plt.show()