from functools import total_ordering
from operator import sub
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf

from data_management import clean_chest_data

input_window_length = 60 # taking the fist minute 
output_window_lengt = 1 # predictin the next second
batch_size = 32

windows_list = []

for data in clean_chest_data:

    features = data[0]
    target = data[1]
    subject = data[2]

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123, shuffle=False)

    train_window = TimeseriesGenerator(x_train, y_train, length=input_window_length, sampling_rate=output_window_lengt, batch_size=batch_size)
    test_window = TimeseriesGenerator(x_test, y_test, length=input_window_length, sampling_rate=output_window_lengt, batch_size=batch_size)

    windows_list.append((train_window, test_window, subject))

if __name__ == "__main__":
    print(windows_list[0][2])
    print(windows_list[0][1][0])




