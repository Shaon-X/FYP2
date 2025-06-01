# # resource: https://colab.research.google.com/drive/1HxPsJvEAH8L7XTmLnfdJ3UQx7j0o1yX5?usp=sharing#scrollTo=vdaqGHG4YZkN
#
# # import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # import tensorflow as tf
# # import keras
# from keras.models import Sequential
# from keras.layers import *
# from keras.callbacks import ModelCheckpoint
# from keras.losses import MeanSquaredError
# from keras.metrics import RootMeanSquaredError
# from keras.optimizers import Adam
# import time
# import math
# from keras.models import load_model
# import random
# import winsound
#
# def idk(x):
#     x.append(4)
#     return x
#
# def plot_graph(x, num, show):
#     tt = [0]*len(x)
#     for y in range(len(x)):
#         tt[y] = x[y][num]
#     plt.plot(tt)
#     if show:
#         plt.show()
#
#
# def plot_output(actual, prediction, show, title):
#
#     dat = [[0] * len(actual), [0] * len(actual)]
#     dat2 = [[0] * len(prediction), [0] * len(prediction)]
#     for y in range(len(actual)):
#         for z in range(len(actual[0])):
#             dat[z][y] = actual[y][z]
#     for y in range(len(prediction)):
#         for z in range(len(prediction[0])):
#             dat2[z][y] = prediction[y][z]
#
#     figure, axis = plt.subplots(2, 1)
#
#     axis[0].plot(dat[0])
#     axis[0].plot(dat2[0])
#
#     axis[1].plot(dat[1])
#     axis[1].plot(dat2[1])
#
#     figure.suptitle(title)
#
#     if show:
#         plt.show()
#
#
# def print_nice(x):
#     for y in x:
#         print(y)
#
#
# def shuffle(x, y):
#     data = []
#     for num in range(len(x)):
#         data.append([x[num], y[num]])
#     random.shuffle(data)
#     for num in range(len(x)):
#         x.pop()
#         y.pop()
#     for num in range(len(data)):
#         x.append(data[num][0])
#         y.append(data[num][1])
#
#
# def split_input(x, y, window, prediction):
#     empty = []
#     for z in range(len(x) - window - prediction + 1):
#         y.append(empty[:])
#         for a in range(window):
#             y[z].append(x[z+a][:])
#
# def reference_pos(x,o):
#     buffer = []
#     buffero = []
#     for y in range(len(x)):
#         buffer.append([])
#         for z in range(len(x[0])):
#             buffer[y].append(x[y][z][:])
#     for y in range(len(o)):
#         buffero.append(o[y][:])
#     for y in range(len(buffer)):
#         for z in range(1, len(buffer[y])):
#             buffer[y][z][0] = buffer[y][z][0] - buffer[y][0][0]
#             buffer[y][z][1] = buffer[y][z][1] - buffer[y][0][1]
#         buffero[y][0] = buffero[y][0] - buffer[y][0][0]
#         buffero[y][1] = buffero[y][1] - buffer[y][0][1]
#         buffer[y].pop(0)
#     for y in range(len(x)):
#         x.pop()
#     for y in range(len(o)):
#         o.pop()
#     for y in range(len(buffer)):
#         x.append([])
#         for z in range(len(buffer[0])):
#             x[y].append(buffer[y][z][:])
#     for y in range(len(buffero)):
#         o.append(buffero[y][:])
#
#
# def split_output(x, y, window, prediction):
#     for a in range(prediction + window - 1, len(x)):
#         y.append(x[a])
#
#
# def split_training(x_ori, y_ori, x_train, y_train, x_val, y_val, x_test, y_test, num1, num2):
#     for a in range(num1):
#         x_train.append(x_ori[a])
#         y_train.append(y_ori[a])
#     for a in range(num2 - num1):
#         x_val.append(x_ori[num1 + a])
#         y_val.append(y_ori[num1 + a])
#     for a in range(len(x_ori) - num2):
#         x_test.append(x_ori[num2 + a])
#         y_test.append(y_ori[num2 + a])
#
#
# def get_true_pos(x, y, split):
#     for z in range(split, len(x)):
#         y.append(x[z][0][:])
#
#
# def get_param(x, y, param):
#     dat = []
#     x_no = 3
#     y_no = 1
#
#
#     for z in range(x_no + y_no):
#         dat.append([])
#     for b in range(len(x)):
#         for a in range(len(x[0])):
#             for z in range(x_no):
#                 dat[z].append(x[b][a][z])
#     for b in range(len(y)):
#         for z in range(y_no):
#             dat[x_no + z].append(y[b][z])
#
#     for a in range(len(dat)):
#         param[a * 2] = np.mean(dat[a])
#         param[a * 2 + 1] = np.std(dat[a])
#
#
# def preprocess_input(x, param):
#
#     for a in range(len(x)):
#         for b in range(len(x[0])):
#             x[a][b][0] = (x[a][b][0] - param[0]) / param[1]
#             x[a][b][1] = (x[a][b][1] - param[2]) / param[3]
#             x[a][b][2] = (x[a][b][2] - param[4]) / param[5]
#
# def preprocess_input_live(x, param):
#
#     x[len(x) - 1][0] = (x[len(x) - 1][0] - param[0]) / param[1]
#     x[len(x) - 1][1] = (x[len(x) - 1][1] - param[2]) / param[3]
#     # x[len(x) - 1][2] = (x[len(x) - 1][2] - param[4]) / param[5]
#
#
# def preprocess_output(x, param):
#
#     for a in range(len(x)):
#         x[a][0] = (x[a][0] - param[6]) / param[7]
#
#
# def postprocess_output(x, param):
#     for y in range(len(x)):
#         # x[y][0] = (x[y][0] * param[len(param) - 3]) + param[len(param) - 4]
#         # x[y][1] = (x[y][1] * param[len(param) - 1]) + param[len(param) - 2]
#         x[y][0] = x[y][0] * param[7] + param[6]
#
# def postprocess_input(x, param):
#
#     for a in range(len(x)):
#         for b in range(len(x[0])):
#             x[a][b][0] = (x[a][b][0] * param[1]) + param[0]
#             x[a][b][1] = (x[a][b][1] * param[3]) + param[2]
#             x[a][b][2] = (x[a][b][2] * param[5]) + param[4]
#
#
# def train(x_train, y_train, x_val, y_val, window, epoch, name, drop):
#     model = Sequential()
#     model.add(InputLayer((window, len(x_train[0][0]))))
#     # model.add(LSTM(32))
#
#     # model.add(LSTM(32, return_sequences=True))
#     # model.add(LSTM(16))
#
#     # model.add(Conv1D(32, kernel_size=2, activation='relu'))
#     # model.add(Flatten())
#
#     # model.add((LSTM(256, return_sequences=True)))
#     # model.add(Dropout(drop))
#     # model.add((LSTM(128, return_sequences=True)))
#     # model.add(Dropout(drop))
#
#     # model.add(GRU(512, return_sequences=True))
#     # model.add(Dropout(drop))
#     # model.add(GRU(64, return_sequences=True))
#     # model.add(Dropout(drop))
#     # model.add(GRU(64, return_sequences=True))
#     # model.add(Dropout(drop))
#     # model.add(GRU(64, return_sequences=True))
#     # model.add(Dropout(drop))
#     model.add(GRU(32, return_sequences=True))
#     # model.add(Dropout(drop))
#     model.add(GRU(16, return_sequences=True))
#     # model.add(Dropout(drop))
#     model.add(GRU(8))
#     # model.add(Dropout(drop))
#
#
#     # model.add(Dense(8, 'linear'))
#     model.add(Dense(len(y_train[0]), 'linear'))
#
#     cp = ModelCheckpoint(name, save_best_only=True)
#     model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
#     history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epoch, callbacks=[cp])
#     winsound.Beep(500, 1500)
#     print(history.history.keys())
#
#     figure, axis = plt.subplots(2, 1)
#
#     axis[0].plot(history.history['root_mean_squared_error'])
#     axis[0].plot(history.history['val_root_mean_squared_error'])
#     plt.title('model root_mean_squared_error')
#     plt.ylabel('error')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#
#     axis[1].plot(history.history['loss'])
#     axis[1].plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
#
#
# def use_model(x, y, model):
#     print('Starting Prediction...')
#     predictions = model.predict(x)
#     # print(predictions)
#     for a in range(len(predictions)):
#         y.append(predictions[a])
#
#
# def predict(s, model, duration, param):
#     if time.time() - s.ait > duration:
#         s.prediction = model.predict([s.ai])
#         postprocess_output(s.prediction, param)
#         # print(round(s.prediction[0][0]))
#         s.ait = time.time()
#
#
# def design_filter(filter_cof, order, cutoff):
#     for x in range(round(order/2)):
#         filter_cof.append(math.sin((cutoff * math.pi) * (x - (order / 2))) * (0.54 - 0.46*math.cos(2*math.pi*x/order)) / (math.pi * (x - (order / 2))))
#     filter_cof.append(cutoff)
#     for x in range(round(order/2)):
#         filter_cof.append(filter_cof[round(order/2) - 1 - x])
#
#
# def apply_filter(x, filter_cof, order, output):
#     data = []
#     summ = 0
#     for a in range(len(x)):
#         if len(data) > order:
#             data.pop(0)
#         data.append(x[a])
#         for b in range(len(data)):
#             summ = summ + data[len(data) - 1 - b] * filter_cof[b]
#         output.append(summ)
#         summ = 0
#
#
# def input_split_xy(ori, x, y):
#     for z in range(len(ori)):
#         x.append([])
#         y.append([])
#         for a in range(len(ori[0])):
#             x[z].append([ori[z][a][0], ori[z][a][2], ori[z][a][4]])
#             y[z].append([ori[z][a][1], ori[z][a][3], ori[z][a][5]])
#
# def output_split_xy(ori, x, y):
#     for z in range(len(ori)):
#         x.append([ori[z][0]])
#         y.append([ori[z][1]])
#
# def boxify(input):
#     #start end gap must be ngam ngam
#     start = [-50, -10]
#     end = [40, 50]
#     gap = [30, 30]
#
#     y = []
#     for x in range(len(input)):
#         y.append([input[x][0], input[x][1]])
#
#     for x in range(len(y)):
#         num = 1
#         for z in range(start[0] + gap[0], end[0], gap[0]):
#             if y[x][0] < z:
#                 y[x][0] = num
#                 break
#             num = num + 1
#         if num == (end[0] - start[0]) / gap[0]:
#             y[x][0] = num
#
#         num = 1
#         for z in range(start[1] + gap[1], end[1], gap[1]):
#             if y[x][1] < z:
#                 y[x][1] = num
#                 break
#             num = num + 1
#         if num == (end[1] - start[1]) / gap[1]:
#             y[x][1] = num
#
#
#     for x in range(len(input)):
#         input.pop()
#
#     for x in range(len(y)):
#         input.append([y[x][0], y[x][1]])
#
#
# def unboxify(x):
#     start = [-50, -10]
#
#     data = []
#     for y in range(len(x)):
#         data.append([round(x[y][0]), round(x[y][1])])
#
#     for y in range(len(x)):
#         data[y][0] = start[0] + 5 + (data[y][0] - 1) * 10
#         data[y][1] = start[1] + 5 + (data[y][1] - 1) * 10
#
#     for y in range(len(x)):
#         x.pop()
#
#     for y in range(len(data)):
#         x.append([data[y][0], data[y][1]])
#
# # df = pd.read_csv('C:\\Users\\Asus\\.keras\\datasets\\jena_climate_2009_2016_extracted\\jena_climate_2009_2016.csv')
# # df = df[5::6]
# # df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
# # temp = df['T (degC)']
# #
# # temp_df = pd.DataFrame({'Temperature': temp})
# # temp_df['Seconds'] = temp_df.index.map(pd.Timestamp.timestamp)
# #
# # day = 24 * 60 * 60
# # year = (365.2425) * day
# #
# # temp_df['Day sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / day))
# # temp_df['Day cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / day))
# # temp_df['Year sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / year))
# # temp_df['Year cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / year))
# # temp_df = temp_df.drop('Seconds', axis=1)
# #
# # p_temp_df = pd.concat([df['p (mbar)'], temp_df], axis=1)
# #
# #
# # def df_to_X_y3(df, window_size=7):
# #     df_as_np = df.to_numpy()
# #     X = []
# #     y = []
# #     for i in range(len(df_as_np)-window_size):
# #       row = [r for r in df_as_np[i:i+window_size]]
# #       X.append(row)
# #       label = [df_as_np[i+window_size][0], df_as_np[i+window_size][1]]
# #       y.append(label)
# #     return np.array(X), np.array(y)
# #
# #
# # X3, y3 = df_to_X_y3(p_temp_df)
# # X3_train, y3_train = X3[:60000], y3[:60000]
# # X3_val, y3_val = X3[60000:65000], y3[60000:65000]
# # X3_test, y3_test = X3[65000:], y3[65000:]
# #
# # p_training_mean3 = np.mean(X3_train[:, :, 0])
# # p_training_std3 = np.std(X3_train[:, :, 0])
# #
# # temp_training_mean3 = np.mean(X3_train[:, :, 1])
# # temp_training_std3 = np.std(X3_train[:, :, 1])
# #
# #
# # def preprocess3(X):
# #     X[:, :, 0] = (X[:, :, 0] - p_training_mean3) / p_training_std3
# #     X[:, :, 1] = (X[:, :, 1] - temp_training_mean3) / temp_training_std3
# #     return X
# #
# #
# # def preprocess_output(y):
# #     y[:, 0] = (y[:, 0] - p_training_mean3) / p_training_std3
# #     y[:, 1] = (y[:, 1] - temp_training_mean3) / temp_training_std3
# #     return y
# #
# #
# # preprocess3(X3_train)
# # preprocess3(X3_val)
# # preprocess3(X3_test)
# #
# # preprocess_output(y3_train)
# # preprocess_output(y3_val)
# # preprocess_output(y3_test)
# #
# # # model5 = Sequential()
# # # model5.add(InputLayer((7, 6)))
# # # model5.add(LSTM(64))
# # # # model6.add(LSTM(32, return_sequences=True))
# # # # model6.add(LSTM(16))
# # # model5.add(Dense(8, 'relu'))
# # # model5.add(Dense(2, 'linear'))
# # #
# # # cp5 = ModelCheckpoint('model5/', save_best_only=True)
# # # model5.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# # # model5.fit(X3_train, y3_train, validation_data=(X3_val, y3_val), epochs=10, callbacks=[cp5])
# # model5 = load_model('model5/')
# #
# #
# # def postprocess_temp(arr):
# #     arr = (arr*temp_training_std3) + temp_training_mean3
# #     return arr
# #
# #
# # def postprocess_p(arr):
# #     arr = (arr*p_training_std3) + p_training_mean3
# #     return arr
# #
# #
# # def get_predictions_postprocessed(model, X, y):
# #     predictions = model.predict(X)
# #     p_preds, temp_preds = postprocess_p(predictions[:, 0]), postprocess_temp(predictions[:, 1])
# #     p_actuals, temp_actuals = postprocess_p(y[:, 0]), postprocess_temp(y[:, 1])
# #     df = pd.DataFrame(data={'Temperature Predictions': temp_preds,
# #                             'Temperature Actuals':temp_actuals,
# #                             'Pressure Predictions': p_preds,
# #                             'Pressure Actuals': p_actuals
# #                             })
# #     return df
# #
# #
# # post_processed_df = get_predictions_postprocessed(model5, X3_test, y3_test)
# #
# # start, end = 0, 100
# # # plt.plot(post_processed_df['Temperature Predictions'][start:end])
# # # plt.plot(post_processed_df['Temperature Actuals'][start:end])
# #
# # plt.plot(post_processed_df['Pressure Predictions'][start:end])
# # plt.plot(post_processed_df['Pressure Actuals'][start:end])
# #
# # plt.show()

# resource: https://colab.research.google.com/drive/1HxPsJvEAH8L7XTmLnfdJ3UQx7j0o1yX5?usp=sharing#scrollTo=vdaqGHG4YZkN

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
import time
import math
from keras.models import load_model
import random
import winsound

def idk(x):
    x.append(4)
    return x

def plot_graph(x, num, show):
    tt = [0]*len(x)
    for y in range(len(x)):
        tt[y] = x[y][num]
    plt.plot(tt)
    if show:
        plt.show()


def plot_output(actual, prediction, show, title):

    dat = [[0] * len(actual), [0] * len(actual)]
    dat2 = [[0] * len(prediction), [0] * len(prediction)]
    for y in range(len(actual)):
        for z in range(len(actual[0])):
            dat[z][y] = actual[y][z]
    for y in range(len(prediction)):
        for z in range(len(prediction[0])):
            dat2[z][y] = prediction[y][z]

    figure, axis = plt.subplots(2, 1)

    axis[0].plot(dat[0])
    axis[0].plot(dat2[0])

    axis[1].plot(dat[1])
    axis[1].plot(dat2[1])

    figure.suptitle(title)

    if show:
        plt.show()


def print_nice(x):
    for y in x:
        print(y)


def shuffle(x, y):
    data = []
    for num in range(len(x)):
        data.append([x[num], y[num]])
    random.shuffle(data)
    for num in range(len(x)):
        x.pop()
        y.pop()
    for num in range(len(data)):
        x.append(data[num][0])
        y.append(data[num][1])


def split_input(x, y, window, prediction):
    empty = []
    for z in range(len(x) - window - prediction + 1):
        y.append(empty[:])
        for a in range(window):
            y[z].append(x[z+a][:])

def reference_pos(x,o):
    buffer = []
    buffero = []
    for y in range(len(x)):
        buffer.append([])
        for z in range(len(x[0])):
            buffer[y].append(x[y][z][:])
    for y in range(len(o)):
        buffero.append(o[y][:])
    for y in range(len(buffer)):
        for z in range(1, len(buffer[y])):
            buffer[y][z][0] = buffer[y][z][0] - buffer[y][0][0]
            buffer[y][z][1] = buffer[y][z][1] - buffer[y][0][1]
        buffero[y][0] = buffero[y][0] - buffer[y][0][0]
        buffero[y][1] = buffero[y][1] - buffer[y][0][1]
        buffer[y].pop(0)
    for y in range(len(x)):
        x.pop()
    for y in range(len(o)):
        o.pop()
    for y in range(len(buffer)):
        x.append([])
        for z in range(len(buffer[0])):
            x[y].append(buffer[y][z][:])
    for y in range(len(buffero)):
        o.append(buffero[y][:])


def split_output(x, y, window, prediction):
    for a in range(prediction + window - 1, len(x)):
        y.append(x[a])


def split_training(x_ori, y_ori, x_train, y_train, x_val, y_val, num):
    border = round(num * len(x_ori))
    for a in range(border):
        x_train.append(x_ori[a])
        y_train.append(y_ori[a])
    for a in range(len(x_ori) - border):
        x_val.append(x_ori[border + a])
        y_val.append(y_ori[border + a])


def get_true_pos(x, y, split):
    for z in range(split, len(x)):
        y.append(x[z][0][:])


def get_param(x, y, param):
    dat = []
    x_no = 2
    y_no = 2


    for z in range(x_no + y_no):
        dat.append([])
    for b in range(len(x)):
        for a in range(len(x[0])):
            for z in range(x_no):
                dat[z].append(x[b][a][z])
    for b in range(len(y)):
        for z in range(y_no):
            dat[x_no + z].append(y[b][z])

    for a in range(len(dat)):
        param[a * 2] = np.mean(dat[a])
        param[a * 2 + 1] = np.std(dat[a])


def preprocess_input(x, param):

    for a in range(len(x)):
        for b in range(len(x[0])):
            x[a][b][0] = (x[a][b][0] - param[0]) / param[1]
            x[a][b][1] = (x[a][b][1] - param[2]) / param[3]

def preprocess_input_live(x, param):

    for y in range(2):
        x[y][0] = (x[y][0] - param[0]) / param[1]
        x[y][1] = (x[y][1] - param[2]) / param[3]
    # x[len(x) - 1][2] = (x[len(x) - 1][2] - param[4]) / param[5]


def preprocess_output(x, param):

    for a in range(len(x)):
        x[a][0] = (x[a][0] - param[4]) / param[5]
        x[a][1] = (x[a][1] - param[6]) / param[7]


def postprocess_output(x, param):
    for y in range(len(x)):
        # x[y][0] = (x[y][0] * param[len(param) - 3]) + param[len(param) - 4]
        # x[y][1] = (x[y][1] * param[len(param) - 1]) + param[len(param) - 2]
        x[y][0] = x[y][0] * param[5] + param[4]
        x[y][1] = x[y][1] * param[7] + param[6]

def postprocess_input(x, param):

    for a in range(len(x)):
        for b in range(len(x[0])):
            x[a][b][0] = (x[a][b][0] * param[1]) + param[0]
            x[a][b][1] = (x[a][b][1] * param[3]) + param[2]
            x[a][b][2] = (x[a][b][2] * param[5]) + param[4]


def train(x_train, y_train, x_val, y_val, window, epoch, name, drop):
    model = Sequential()
    model.add(InputLayer((len(x_train[0]), len(x_train[0][0]))))
    # model.add(LSTM(32))

    # model.add(LSTM(32, return_sequences=True))
    # model.add(LSTM(16))

    # model.add(Conv1D(32, kernel_size=2, activation='relu'))
    # model.add(Flatten())

    # model.add((LSTM(256, return_sequences=True)))
    # model.add(Dropout(drop))
    # model.add((LSTM(128, return_sequences=True)))
    # model.add(Dropout(drop))

    # model.add(GRU(512, return_sequences=True))
    # model.add(Dropout(drop))
    # model.add(GRU(256, return_sequences=True))
    # model.add(Dropout(drop))
    model.add(GRU(128, return_sequences=True))
    # model.add(Dropout(drop))
    model.add(GRU(64, return_sequences=True))
    # model.add(Dropout(drop))
    model.add(GRU(32, return_sequences=True))
    # model.add(Dropout(drop))
    model.add(GRU(16, return_sequences=True))
    # model.add(Dropout(drop))
    model.add(GRU(8))
    # model.add(Dropout(drop))



    # model.add(Dense(8, 'linear'))
    model.add(Dense(len(y_train[0]), 'linear'))

    cp = ModelCheckpoint(name, save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epoch, callbacks=[cp])
    winsound.Beep(500, 1500)
    print(history.history.keys())

    figure, axis = plt.subplots(2, 1)

    axis[0].plot(history.history['root_mean_squared_error'])
    axis[0].plot(history.history['val_root_mean_squared_error'])
    plt.title('model root_mean_squared_error')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    axis[1].plot(history.history['loss'])
    axis[1].plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def use_model(x, y, model):
    print('Starting Prediction...')
    predictions = model.predict(x)
    # print(predictions)
    for a in range(len(predictions)):
        y.append(predictions[a])


def predict(s, model, duration, param):
    if time.time() - s.ait >= duration:
        s.ait = time.time()
        aibuf = []
        if len(s.ai) >= 2:
            for x in range(len(s.ai) - 2, -1, -1):
                if s.ai[len(s.ai) - 1][2] - s.ai[x][2] >= 0.01 * s. width * s.window:
                    aibuf.append([s.ai[x][0], s.ai[x][1]])
                    aibuf.append([s.ai[len(s.ai) - 1][0], s.ai[len(s.ai) - 1][1]])
                    preprocess_input_live(aibuf, param)
                    buffer = model.predict([aibuf])
                    postprocess_output(buffer, param)
                    prev_pred = s.prediction[0][0]
                    if 0 < round(buffer[0][0]) < 4 and 0 < round(buffer[0][1]) < 3:
                        s.prediction[0][0] = 3 + round(buffer[0][0])
                        if round(buffer[0][1]) == 2:
                            s.prediction[0][0] = s.prediction[0][0] - 3
                    else:
                        s.prediction[0][0] = 0

                    if prev_pred != s.prediction[0][0]:
                        s.prediction[0][1] = s.ai[len(s.ai) - 1][2] + s.delay

                    for y in range(x+1):
                        s.ai.pop(0)
                    break

def design_filter(filter_cof, order, cutoff):
    for x in range(round(order/2)):
        filter_cof.append(math.sin((cutoff * math.pi) * (x - (order / 2))) * (0.54 - 0.46*math.cos(2*math.pi*x/order)) / (math.pi * (x - (order / 2))))
    filter_cof.append(cutoff)
    for x in range(round(order/2)):
        filter_cof.append(filter_cof[round(order/2) - 1 - x])


def apply_filter(x, filter_cof, order, output):
    data = []
    summ = 0
    for a in range(len(x)):
        if len(data) > order:
            data.pop(0)
        data.append(x[a])
        for b in range(len(data)):
            summ = summ + data[len(data) - 1 - b] * filter_cof[b]
        output.append(summ)
        summ = 0


def input_split_xy(ori, x, y):
    for z in range(len(ori)):
        x.append([])
        y.append([])
        for a in range(len(ori[0])):
            x[z].append([ori[z][a][0], ori[z][a][2], ori[z][a][4]])
            y[z].append([ori[z][a][1], ori[z][a][3], ori[z][a][5]])

def output_split_xy(ori, x, y):
    for z in range(len(ori)):
        x.append([ori[z][0]])
        y.append([ori[z][1]])

def boxify(input):
    #start end gap must be ngam ngam
    start = [-50, -10]
    end = [40, 50]
    gap = [30, 30]

    y = []
    for x in range(len(input)):
        y.append([input[x][0], input[x][1]])

    for x in range(len(y)):
        num = 1
        for z in range(start[0] + gap[0], end[0], gap[0]):
            if y[x][0] < z:
                y[x][0] = num
                break
            num = num + 1
        if num == (end[0] - start[0]) / gap[0]:
            y[x][0] = num

        num = 1
        for z in range(start[1] + gap[1], end[1], gap[1]):
            if y[x][1] < z:
                y[x][1] = num
                break
            num = num + 1
        if num == (end[1] - start[1]) / gap[1]:
            y[x][1] = num


    for x in range(len(input)):
        input.pop()

    for x in range(len(y)):
        input.append([y[x][0], y[x][1]])


def unboxify(x):
    start = [-50, -10]

    data = []
    for y in range(len(x)):
        data.append([round(x[y][0]), round(x[y][1])])

    for y in range(len(x)):
        data[y][0] = start[0] + 5 + (data[y][0] - 1) * 10
        data[y][1] = start[1] + 5 + (data[y][1] - 1) * 10

    for y in range(len(x)):
        x.pop()

    for y in range(len(data)):
        x.append([data[y][0], data[y][1]])

# df = pd.read_csv('C:\\Users\\Asus\\.keras\\datasets\\jena_climate_2009_2016_extracted\\jena_climate_2009_2016.csv')
# df = df[5::6]
# df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
# temp = df['T (degC)']
#
# temp_df = pd.DataFrame({'Temperature': temp})
# temp_df['Seconds'] = temp_df.index.map(pd.Timestamp.timestamp)
#
# day = 24 * 60 * 60
# year = (365.2425) * day
#
# temp_df['Day sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / day))
# temp_df['Day cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / day))
# temp_df['Year sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / year))
# temp_df['Year cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / year))
# temp_df = temp_df.drop('Seconds', axis=1)
#
# p_temp_df = pd.concat([df['p (mbar)'], temp_df], axis=1)
#
#
# def df_to_X_y3(df, window_size=7):
#     df_as_np = df.to_numpy()
#     X = []
#     y = []
#     for i in range(len(df_as_np)-window_size):
#       row = [r for r in df_as_np[i:i+window_size]]
#       X.append(row)
#       label = [df_as_np[i+window_size][0], df_as_np[i+window_size][1]]
#       y.append(label)
#     return np.array(X), np.array(y)
#
#
# X3, y3 = df_to_X_y3(p_temp_df)
# X3_train, y3_train = X3[:60000], y3[:60000]
# X3_val, y3_val = X3[60000:65000], y3[60000:65000]
# X3_test, y3_test = X3[65000:], y3[65000:]
#
# p_training_mean3 = np.mean(X3_train[:, :, 0])
# p_training_std3 = np.std(X3_train[:, :, 0])
#
# temp_training_mean3 = np.mean(X3_train[:, :, 1])
# temp_training_std3 = np.std(X3_train[:, :, 1])
#
#
# def preprocess3(X):
#     X[:, :, 0] = (X[:, :, 0] - p_training_mean3) / p_training_std3
#     X[:, :, 1] = (X[:, :, 1] - temp_training_mean3) / temp_training_std3
#     return X
#
#
# def preprocess_output(y):
#     y[:, 0] = (y[:, 0] - p_training_mean3) / p_training_std3
#     y[:, 1] = (y[:, 1] - temp_training_mean3) / temp_training_std3
#     return y
#
#
# preprocess3(X3_train)
# preprocess3(X3_val)
# preprocess3(X3_test)
#
# preprocess_output(y3_train)
# preprocess_output(y3_val)
# preprocess_output(y3_test)
#
# # model5 = Sequential()
# # model5.add(InputLayer((7, 6)))
# # model5.add(LSTM(64))
# # # model6.add(LSTM(32, return_sequences=True))
# # # model6.add(LSTM(16))
# # model5.add(Dense(8, 'relu'))
# # model5.add(Dense(2, 'linear'))
# #
# # cp5 = ModelCheckpoint('model5/', save_best_only=True)
# # model5.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# # model5.fit(X3_train, y3_train, validation_data=(X3_val, y3_val), epochs=10, callbacks=[cp5])
# model5 = load_model('model5/')
#
#
# def postprocess_temp(arr):
#     arr = (arr*temp_training_std3) + temp_training_mean3
#     return arr
#
#
# def postprocess_p(arr):
#     arr = (arr*p_training_std3) + p_training_mean3
#     return arr
#
#
# def get_predictions_postprocessed(model, X, y):
#     predictions = model.predict(X)
#     p_preds, temp_preds = postprocess_p(predictions[:, 0]), postprocess_temp(predictions[:, 1])
#     p_actuals, temp_actuals = postprocess_p(y[:, 0]), postprocess_temp(y[:, 1])
#     df = pd.DataFrame(data={'Temperature Predictions': temp_preds,
#                             'Temperature Actuals':temp_actuals,
#                             'Pressure Predictions': p_preds,
#                             'Pressure Actuals': p_actuals
#                             })
#     return df
#
#
# post_processed_df = get_predictions_postprocessed(model5, X3_test, y3_test)
#
# start, end = 0, 100
# # plt.plot(post_processed_df['Temperature Predictions'][start:end])
# # plt.plot(post_processed_df['Temperature Actuals'][start:end])
#
# plt.plot(post_processed_df['Pressure Predictions'][start:end])
# plt.plot(post_processed_df['Pressure Actuals'][start:end])
#
# plt.show()
