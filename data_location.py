import pandas as pd
import numpy as np
df = pd.read_csv('iphone_14_r.csv')
file = open("twt.txt", "r")
df_4 = file.read()
data = pd.read_csv('ankwe1.csv')
data_portland = pd.read_csv('temperature.csv')


data['tempc'] = data.apply(lambda x: (x['temp']-32)*(5/9), axis=1)
data.index = pd.to_datetime(data['datetime'], format='%Y.%m.%d')
data_core = data[['tempc']].copy()


data_portland['temp'] = data_portland.apply(lambda x: (x['Portland']-273.15), axis=1)
data_portland.index = pd.to_datetime(data_portland['datetime'], format='%Y.%m.%d %H:%M:%S')
data_core2 = data_portland[['temp']].copy()
data_core2 = data_core2[1:].copy()


x_in = np.array([34, 37, 35, 33, 34, 37, 35, 33, 34, 34, 30, 27, 26, 24, 27, 29, 31, 32,
                 30, 25, 29, 34, 35, 35, 27, 27, 23, 19, 19, 20])
x_in = x_in.reshape((1, 30, 1))

expect = [24, 28, 31, 30, 27, 28, 29, 28, 26, 24, 27, 25, 22, 22, 22, 22, 21, 24, 24, 24, 22, 19, 18, 18,
          19, 18, 21, 22, 22, 21]

x_in2 = np.array([17, 16, 16, 16, 16, 17, 17, 16, 16, 16, 17, 17, 18, 19, 19, 20, 20, 21, 20, 19, 19, 18, 17, 16])
x_in3 = x_in2.reshape((1, 24, 1))
