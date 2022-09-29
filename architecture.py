import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import *
from data_location import df

sentiment_factor = df.Analysis.factorize()
twt = df.Tweet.values
tokenizer = Tokenizer(5000)
tokenizer.fit_on_texts(twt)
enco_d = tokenizer.texts_to_sequences(twt)
pad_se = pad_sequences(enco_d, maxlen=200)
voc_s = len(tokenizer.word_index) + 1


class sentiment:
    def __init__(self, embed_v_l, metrics, epoch, batch_sizer, predict_data):
        self.embed_v_l = embed_v_l
        self.metrics = metrics
        self.epoch = epoch
        self.batch_sizer = batch_sizer
        self.predict_data = predict_data

    def model_sentiment(self):
        self.model = Sequential()
        self.model.add(Embedding(voc_s, self.embed_v_l, input_length=200))
        self.model.add(LSTM(50))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[self.metrics])
        print(self.model.summary())

    def fitter(self):
        history = self.model.fit(pad_se, sentiment_factor[0], validation_split=0.2, epochs=self.epoch, batch_size=self.batch_sizer)

        return history

    def predict_sentiment(self):
        result = []
        df_list = self.predict_data.split("\n")
        for i in range(len(df_list)):

            tok = tokenizer.texts_to_sequences([df_list[i]])
            tok = pad_sequences(tok, maxlen=200)
            pred = int(self.model.predict(tok).round().item())
            result.append(sentiment_factor[1][pred])

        neg = (result.count('Negative'))
        percentagen = round((neg / len(result)*100))
        pos = (result.count('Positive'))
        percentagep = round((pos / len(result)*100))
        print('%', percentagen, 'Negative')
        print('%', percentagep, 'Positive')
        print(result)

    def predict_sentence(self, text):
        tok = tokenizer.texts_to_sequences([text])
        tok = pad_sequences(tok, maxlen=200)
        pred = int(self.model.predict(tok).round().item())
        print((sentiment_factor[1][pred]))


class weather:
    def __init__(self, data, prediction_data, epoch, batch_size, steps, date_scala, expectetion):
        self.data = data
        self.prediction_data = prediction_data
        self.epoch = epoch
        self.batch_size = batch_size
        self.steps = steps
        self.date_scala = date_scala
        self.expectetion = expectetion

    def tensorize(self):
        data_core_np = self.data.iloc[:, 0].to_numpy()
        X = []
        y = []
        for i in range(len(data_core_np) - self.steps):
            inpt = [[x] for x in data_core_np[i:i+self.steps]]
            X.append(inpt)
            outpt = data_core_np[i+self.steps]
            y.append(outpt)
        part_1 = int(0.75 * len(self.data))
        part_2 = int(0.95*len(self.data))
        self.X_train, self.y_train = X[:part_2], y[:part_2]
        self.X_val, self.y_val = X[part_1:part_2], y[part_1:part_2]
        self.X_test, self.y_test = X[part_2:], y[part_2:]

    def model_weather(self):
        self.model1 = Sequential()
        self.model1.add(tf.keras.layers.InputLayer((self.steps, 1)))
        self.model1.add(tf.keras.layers.LSTM(128))
        self.model1.add(tf.keras.layers.Dense(64, 'relu'))
        self.model1.add(tf.keras.layers.Dense(1, 'linear'))
        self.model1.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                            metrics=[tf.keras.metrics.RootMeanSquaredError()])
        self.model1.summary()

    def fitter1(self):
        history1 = self.model1.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                                   epochs=self.epoch, batch_size=self.batch_size)
        return history1

    def predicter(self):
        import datetime
        if self.date_scala == 'day':
            n = self.steps
            prediction_scenerio = []
            for i in range(n):
                prep1 = self.model1.predict(self.prediction_data, verbose=0)
                self.prediction_data = np.delete(self.prediction_data, 0)
                prediction_scenerio = np.append(prediction_scenerio, prep1)
                self.prediction_data = np.append(self.prediction_data, [prep1])
                self.prediction_data = self.prediction_data.reshape((1, len(self.prediction_data), 1))

            print(prediction_scenerio)
            date_start = datetime.datetime(2022, 9, 24)
            end = datetime.datetime(2022, 10, 24)
            step = datetime.timedelta(days=1)
            date_list = []

            while date_start < end:
                date_list.append(date_start.strftime('%Y-%m-%d'))
                date_start += step

            x_past = self.data['tempc'].copy()
            x_past = x_past[:5312].copy()
            x_past = pd.DataFrame(x_past)
            prediction_scenerio = pd.DataFrame(prediction_scenerio)
            self.expectetion = pd.DataFrame(self.expectetion)
            self.expectetion.columns = ['expectetion']
            prediction_scenerio.columns = ['prediction']
            x_past.columns = ['past']

            prediction_scenerio.index = pd.to_datetime(date_list, format='%Y.%m.%d')
            self.expectetion.index = pd.to_datetime(date_list, format='%Y.%m.%d')

            bx = self.expectetion.plot()
            ax = prediction_scenerio.plot()
            x_past[5250:].plot(ax=ax)
            x_past[5250:].plot(ax=bx)
            plt.show()
        elif self.date_scala == 'hour':
            day_n = 60
            n = day_n*24
            prediction_scenerio = []
            for i in range(n):
                prep1 = self.model1.predict(self.prediction_data, verbose=0)
                self.prediction_data = np.delete(self.prediction_data, 0)
                prediction_scenerio = np.append(prediction_scenerio, prep1)
                self.prediction_data = np.append(self.prediction_data, [prep1])
                self.prediction_data = self.prediction_data.reshape((1, self.steps, 1))
            daily_pred = []
            predector = prediction_scenerio.tolist()
            for i in range(day_n):
                day = predector[0:24]
                del predector[:24]
                total = max(day)
                daily_pred = np.append(daily_pred, total)
            print(daily_pred)
            plt.plot(daily_pred)
            plt.show()

        else:
            print('wrong date scala')


def plotter(history, metric, val_metric):
        plt.plot(history.history['loss'], 'g', label='Training loss')
        plt.plot(history.history['val_loss'], 'b', label='validation loss')
        plt.plot(history.history[val_metric], 'r', label=val_metric)
        plt.plot(history.history[metric], 'y', label=metric)
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Scala')
        plt.legend()
        plt.show()
