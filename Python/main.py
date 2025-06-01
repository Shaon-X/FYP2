# import time
#
# # import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#
# from interface import *
# from prediction import *
# # import multiprocessing
#
# if __name__ == '__main__':
#
#     # MODE = "Get_Training_Data"
#     MODE = "Train_Data"
#     # MODE = "Run_Model"
#     # MODE =  "Plot_Model"
#     # MODE = "Normal"
#     # MODE = "test"
#
#     window = 8
#     prediction = 5
#     width = 10
#     robot_vel = 50 #cm/s
#     delay = 0.5
#
#     take_val = 10
#     take_val_acc = 10
#     split = [3600, 4800]
#     epoch = 400
#     params_to_process = 4
#     # try one layer only but wider
#     output_file = 'output - input 2 - data4 - box 15 - window 0.8 - GRU 64, 64, 64 ~ 8'
#
#     if MODE == "Get_Training_Data":
#
#         s = SerialClass("COM3", 10, width, window)
#         calibrate(s, 5)
#         collect_data(s, 'raw_data_4_box', 600)
#         beep()
#
#     elif MODE == "Train_Data":
#         # 59966 / 10 samples
#         #
#
#         # filt = []
#         # smoothx = []
#         # smoothy = []
#         # order = 30
#         # design_filter(filt, order, 0.1)
#         output_param = [0] * (params_to_process*2)
#         box_length = [26, 29]
#         LSTM_data = []
#         LSTM_output = []
#         LSTM_data_train = []
#         LSTM_data_val = []
#         LSTM_data_test = []
#         LSTM_output_train = []
#         LSTM_output_val = []
#         LSTM_output_test = []
#         raw_data = []
#         raw_output = []
#         read_csv(raw_data, 'raw_data_4_box')
#         # filter for only 2 input 2 output
#         # xdat = [item[0] for item in raw_data]
#         # ydat = [item[1] for item in raw_data]
#         # apply_filter(xdat, filt, order, smoothx)
#         # apply_filter(ydat, filt, order, smoothy)
#         # raw_data = []
#         # for a in range(len(smoothx)):
#         #     raw_data.append([smoothx[a], smoothy[a]])
#         # for a in raw_data:
#         #     raw_output.append(a[:])
#
#         # for num in range(len(raw_data)):
#         #     while len(raw_data[num]) > 2:
#         #         raw_data[num].pop(2)
#
#         test_data = []
#         for x in raw_data:
#             test_data.append(x[:])
#         i = 0
#         new_l = int(len(test_data) / width)
#         while len(test_data) > new_l + width:
#             for x in range(width - 1):
#                 test_data.pop(i + 1)
#             i = i + 1
#         while len(test_data) > new_l:
#             test_data.pop(new_l)
#         raw_data = []
#         for x in test_data:
#             raw_data.append(x)
#
#         for x in range(len(raw_data)):
#             if 25 <= raw_data[x][1] < 40:
#                 if -20 <= raw_data[x][0] < -5:
#                     raw_data[x][2] = 4
#                 elif -5 <= raw_data[x][0] < 10:
#                     raw_data[x][2] = 5
#                 elif 10 <= raw_data[x][0] < 25:
#                     raw_data[x][2] = 6
#             elif 40 <= raw_data[x][1] < 55:
#                 if -20 <= raw_data[x][0] < -5:
#                     raw_data[x][2] = 1
#                 elif -5 <= raw_data[x][0] < 10:
#                     raw_data[x][2] = 2
#                 elif 10 <= raw_data[x][0] < 25:
#                     raw_data[x][2] = 3
#
#         for x in range(len(raw_data)):
#             raw_output.append([raw_data[x][2]])
#
#         # x = -40 to 40, y = -10 to 50
#         # 8 x 6 box
#
#         # boxify(raw_output)
#
#         # for x in range(take_val):
#         #     raw_data[x].append(0)
#         #     raw_data[x].append(0)
#         # for x in range(take_val, len(raw_data)):
#         #     raw_data[x].append((raw_data[x][0] - raw_data[x - take_val][0])/(0.01*width*take_val))
#         #     raw_data[x].append((raw_data[x][1] - raw_data[x - take_val][1]) / (0.01 * width * take_val))
#         #
#         # for x in range(take_val_acc):
#         #     raw_data[x].append(0)
#         #     raw_data[x].append(0)
#         # for x in range(take_val_acc, len(raw_data)):
#         #     raw_data[x].append((raw_data[x][2] - raw_data[x - take_val_acc][2]) / (0.01 * width * take_val_acc))
#         #     raw_data[x].append((raw_data[x][3] - raw_data[x - take_val_acc][3]) / (0.01 * width * take_val_acc))
#
#         split_input(raw_data, LSTM_data, window, prediction)
#         split_output(raw_output, LSTM_output, window, prediction)
#         # reference_pos(LSTM_data, LSTM_output)
#         split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_val, LSTM_output_val,
#                        LSTM_data_test, LSTM_output_test, split[0], split[1])
#         get_param(LSTM_data_train, LSTM_output_train, output_param)
#         preprocess_input(LSTM_data_train, output_param)
#         preprocess_input(LSTM_data_val, output_param)
#         preprocess_input(LSTM_data_test, output_param)
#         preprocess_output(LSTM_output_train, output_param)
#         preprocess_output(LSTM_output_val, output_param)
#         preprocess_output(LSTM_output_test, output_param)
#
#         shuffle(LSTM_data_train, LSTM_output_train)
#
#         for x in range(len(LSTM_data_train)):
#             for a in range(len(LSTM_data_train[0])):
#                 LSTM_data_train[x][a].pop(2)
#
#         for x in range(len(LSTM_data_val)):
#             for a in range(len(LSTM_data_val[0])):
#                 LSTM_data_val[x][a].pop(2)
#
#         for x in range(len(LSTM_data_test)):
#             for a in range(len(LSTM_data_test[0])):
#                 LSTM_data_test[x][a].pop(2)
#
#         tt = time.time()
#         train(LSTM_data_train, LSTM_output_train, LSTM_data_val, LSTM_output_val, window, epoch, 'model/', 0.2)
#         print(time.time() - tt)
#         beep()
#
#     elif MODE == "Run_Model":
#         # 59966 / 10 samples
#         #
#
#         # filt = []
#         # smoothx = []
#         # smoothy = []
#         # order = 30
#         # design_filter(filt, order, 0.1)
#         output_param = [0] * (params_to_process * 2)
#         box_length = [26, 29]
#         LSTM_data = []
#         LSTM_output = []
#         LSTM_data_train = []
#         LSTM_data_val = []
#         LSTM_data_test = []
#         LSTM_output_train = []
#         LSTM_output_val = []
#         LSTM_output_test = []
#         raw_data = []
#         raw_output = []
#         read_csv(raw_data, 'raw_data_4_box')
#         # filter for only 2 input 2 output
#         # xdat = [item[0] for item in raw_data]
#         # ydat = [item[1] for item in raw_data]
#         # apply_filter(xdat, filt, order, smoothx)
#         # apply_filter(ydat, filt, order, smoothy)
#         # raw_data = []
#         # for a in range(len(smoothx)):
#         #     raw_data.append([smoothx[a], smoothy[a]])
#         # for a in raw_data:
#         #     raw_output.append(a[:])
#
#         # for num in range(len(raw_data)):
#         #     while len(raw_data[num]) > 2:
#         #         raw_data[num].pop(2)
#
#         test_data = []
#         for x in raw_data:
#             test_data.append(x[:])
#         i = 0
#         new_l = int(len(test_data) / width)
#         while len(test_data) > new_l + width:
#             for x in range(width - 1):
#                 test_data.pop(i + 1)
#             i = i + 1
#         while len(test_data) > new_l:
#             test_data.pop(new_l)
#         raw_data = []
#         for x in test_data:
#             raw_data.append(x)
#
#         for x in range(len(raw_data)):
#             if 25 <= raw_data[x][1] < 40:
#                 if -20 <= raw_data[x][0] < -5:
#                     raw_data[x][2] = 4
#                 elif -5 <= raw_data[x][0] < 10:
#                     raw_data[x][2] = 5
#                 elif 10 <= raw_data[x][0] < 25:
#                     raw_data[x][2] = 6
#             elif 40 <= raw_data[x][1] < 55:
#                 if -20 <= raw_data[x][0] < -5:
#                     raw_data[x][2] = 1
#                 elif -5 <= raw_data[x][0] < 10:
#                     raw_data[x][2] = 2
#                 elif 10 <= raw_data[x][0] < 25:
#                     raw_data[x][2] = 3
#
#         for x in range(len(raw_data)):
#             raw_output.append([raw_data[x][2]])
#
#         # x = -40 to 40, y = -10 to 50
#         # 8 x 6 box
#
#         # boxify(raw_output)
#
#         # for x in range(take_val):
#         #     raw_data[x].append(0)
#         #     raw_data[x].append(0)
#         # for x in range(take_val, len(raw_data)):
#         #     raw_data[x].append((raw_data[x][0] - raw_data[x - take_val][0])/(0.01*width*take_val))
#         #     raw_data[x].append((raw_data[x][1] - raw_data[x - take_val][1]) / (0.01 * width * take_val))
#         #
#         # for x in range(take_val_acc):
#         #     raw_data[x].append(0)
#         #     raw_data[x].append(0)
#         # for x in range(take_val_acc, len(raw_data)):
#         #     raw_data[x].append((raw_data[x][2] - raw_data[x - take_val_acc][2]) / (0.01 * width * take_val_acc))
#         #     raw_data[x].append((raw_data[x][3] - raw_data[x - take_val_acc][3]) / (0.01 * width * take_val_acc))
#
#         split_input(raw_data, LSTM_data, window, prediction)
#         split_output(raw_output, LSTM_output, window, prediction)
#         # reference_pos(LSTM_data, LSTM_output)
#         split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_val, LSTM_output_val,
#                        LSTM_data_test, LSTM_output_test, split[0], split[1])
#         get_param(LSTM_data_train, LSTM_output_train, output_param)
#         preprocess_input(LSTM_data_train, output_param)
#         preprocess_input(LSTM_data_val, output_param)
#         preprocess_input(LSTM_data_test, output_param)
#         preprocess_output(LSTM_output_train, output_param)
#         preprocess_output(LSTM_output_val, output_param)
#         preprocess_output(LSTM_output_test, output_param)
#         shuffle(LSTM_data_train, LSTM_output_train)
#
#         for x in range(len(LSTM_data_train)):
#             for a in range(len(LSTM_data_train[0])):
#                 LSTM_data_train[x][a].pop(2)
#
#         for x in range(len(LSTM_data_val)):
#             for a in range(len(LSTM_data_val[0])):
#                 LSTM_data_val[x][a].pop(2)
#
#         for x in range(len(LSTM_data_test)):
#             for a in range(len(LSTM_data_test[0])):
#                 LSTM_data_test[x][a].pop(2)
#
#         model_output = []
#         print('Starting...')
#         tt = time.time()
#         model = load_model('model/')
#         use_model(LSTM_data_test, model_output, model)
#         print(time.time() - tt)
#         write_csv(model_output, output_file)
#         beep()
#
#     elif MODE == "Plot_Model":
#         # 59966 / 10 samples
#         #
#
#         # filt = []
#         # smoothx = []
#         # smoothy = []
#         # order = 30
#         # design_filter(filt, order, 0.1)
#         output_param = [0] * (params_to_process * 2)
#         box_length = [26, 29]
#         LSTM_data = []
#         LSTM_output = []
#         LSTM_data_train = []
#         LSTM_data_val = []
#         LSTM_data_test = []
#         LSTM_output_train = []
#         LSTM_output_val = []
#         LSTM_output_test = []
#         raw_data = []
#         raw_output = []
#         read_csv(raw_data, 'raw_data_4_box')
#         # filter for only 2 input 2 output
#         # xdat = [item[0] for item in raw_data]
#         # ydat = [item[1] for item in raw_data]
#         # apply_filter(xdat, filt, order, smoothx)
#         # apply_filter(ydat, filt, order, smoothy)
#         # raw_data = []
#         # for a in range(len(smoothx)):
#         #     raw_data.append([smoothx[a], smoothy[a]])
#         # for a in raw_data:
#         #     raw_output.append(a[:])
#
#         # for num in range(len(raw_data)):
#         #     while len(raw_data[num]) > 2:
#         #         raw_data[num].pop(2)
#
#         test_data = []
#         for x in raw_data:
#             test_data.append(x[:])
#         i = 0
#         new_l = int(len(test_data) / width)
#         while len(test_data) > new_l + width:
#             for x in range(width - 1):
#                 test_data.pop(i + 1)
#             i = i + 1
#         while len(test_data) > new_l:
#             test_data.pop(new_l)
#         raw_data = []
#         for x in test_data:
#             raw_data.append(x)
#
#         for x in range(len(raw_data)):
#             if 25 <= raw_data[x][1] < 40:
#                 if -20 <= raw_data[x][0] < -5:
#                     raw_data[x][2] = 4
#                 elif -5 <= raw_data[x][0] < 10:
#                     raw_data[x][2] = 5
#                 elif 10 <= raw_data[x][0] < 25:
#                     raw_data[x][2] = 6
#             elif 40 <= raw_data[x][1] < 55:
#                 if -20 <= raw_data[x][0] < -5:
#                     raw_data[x][2] = 1
#                 elif -5 <= raw_data[x][0] < 10:
#                     raw_data[x][2] = 2
#                 elif 10 <= raw_data[x][0] < 25:
#                     raw_data[x][2] = 3
#
#         for x in range(len(raw_data)):
#             raw_output.append([raw_data[x][2]])
#
#         # x = -40 to 40, y = -10 to 50
#         # 8 x 6 box
#
#         # boxify(raw_output)
#
#         # for x in range(take_val):
#         #     raw_data[x].append(0)
#         #     raw_data[x].append(0)
#         # for x in range(take_val, len(raw_data)):
#         #     raw_data[x].append((raw_data[x][0] - raw_data[x - take_val][0])/(0.01*width*take_val))
#         #     raw_data[x].append((raw_data[x][1] - raw_data[x - take_val][1]) / (0.01 * width * take_val))
#         #
#         # for x in range(take_val_acc):
#         #     raw_data[x].append(0)
#         #     raw_data[x].append(0)
#         # for x in range(take_val_acc, len(raw_data)):
#         #     raw_data[x].append((raw_data[x][2] - raw_data[x - take_val_acc][2]) / (0.01 * width * take_val_acc))
#         #     raw_data[x].append((raw_data[x][3] - raw_data[x - take_val_acc][3]) / (0.01 * width * take_val_acc))
#
#         split_input(raw_data, LSTM_data, window, prediction)
#         split_output(raw_output, LSTM_output, window, prediction)
#         # reference_pos(LSTM_data, LSTM_output)
#         split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_val, LSTM_output_val,
#                        LSTM_data_test, LSTM_output_test, split[0], split[1])
#         get_param(LSTM_data_train, LSTM_output_train, output_param)
#
#         # read_csv(model_output, 'output - complicated model')
#         # read_csv(model_output, 'output - simple model')
#         # read_csv(model_output, 'output - complicated model - reference')
#
#         # for x in range(len(LSTM_output_test)):
#         #     LSTM_output_test[x][0] = LSTM_output_test[x][0] + true_pos[x][0]
#         #     LSTM_output_test[x][1] = LSTM_output_test[x][1] + true_pos[x][1]
#
#         # write_csv(LSTM_output_test, 'plot_data_4')
#         # # write_csv(true_pos, 'pos_data_3')
#         # buffer = [output_param, [0, 0, 0]]
#         # write_csv(buffer, 'param_data_4')
#         # print('Done')
#         # stop()
#
#         model_output = []
#         buffer_param = []
#         output_param = []
#         LSTM_output_test = []
#         true_pos = []
#         read_csv(model_output,
#                  'output - input 2 - data4 - box 15 - window 0.8 - GRU 32 ~ 8')
#         read_csv(LSTM_output_test, 'plot_data_4')
#         # read_csv(true_pos, 'pos_data_3')
#         read_csv(buffer_param, 'param_data_4')
#         for x in range(len(buffer_param[0])):
#             output_param.append(buffer_param[0][x])
#         postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#
#
#         a = []
#         for x in range(len(LSTM_output_test)):
#             a.append(LSTM_output_test[x][0])
#         plt.plot(a)
#
#         a = []
#         for x in range(len(model_output)):
#             a.append(round(model_output[x][0]))
#         plt.plot(a)
#
#         num1 = 0
#         num2 = 0
#         for x in range(len(model_output)):
#             if LSTM_output_test[x][0] == round(model_output[x][0]):
#                 num1 = num1 + 1
#             else:
#                 num2 = num2 + 1
#         print(num1 / (num1 + num2))
#
#
#         # plot_output(LSTM_output_test, model_output, 0, 'GRU 16 8 data3 box')
#
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - pos reference - velocity 0.05 - acc 0.05 - window 0.75 - 64')
#         # read_csv(LSTM_output_test, 'plot_data')
#         # read_csv(true_pos, 'pos_data')
#         # read_csv(buffer_param, 'param_data')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, '64')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - pos reference - velocity 0.05 - acc 0.05 - window 0.75 - 64 32')
#         # read_csv(LSTM_output_test, 'plot_data')
#         # read_csv(true_pos, 'pos_data')
#         # read_csv(buffer_param, 'param_data')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, '64 32')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - pos reference - velocity 0.05 - acc 0.05 - window 0.75 - 64 32 16')
#         # read_csv(LSTM_output_test, 'plot_data')
#         # read_csv(true_pos, 'pos_data')
#         # read_csv(buffer_param, 'param_data')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, '64 32 16')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - pos reference - velocity 0.05 - acc 0.05 - window 0.75 - 64 32 16 8')
#         # read_csv(LSTM_output_test, 'plot_data')
#         # read_csv(true_pos, 'pos_data')
#         # read_csv(buffer_param, 'param_data')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, '64 32 16 8')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - data2 - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - 64 32 16 8')
#         # read_csv(LSTM_output_test, 'plot_data_2')
#         # read_csv(true_pos, 'pos_data_2')
#         # read_csv(buffer_param, 'param_data_2')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, '64 32 16 8 data2')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - 64 32 16 8')
#         # read_csv(LSTM_output_test, 'plot_data_2')
#         # read_csv(true_pos, 'pos_data_2')
#         # read_csv(buffer_param, 'param_data_2')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, '64 32 16 8 data2 split')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 64 32 16 8')
#         # read_csv(LSTM_output_test, 'plot_data_2')
#         # read_csv(true_pos, 'pos_data_2')
#         # read_csv(buffer_param, 'param_data_2')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, 'GRU 64 32 16 8 data2 split')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 32 16 8')
#         # read_csv(LSTM_output_test, 'plot_data_2')
#         # read_csv(true_pos, 'pos_data_2')
#         # read_csv(buffer_param, 'param_data_2')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, 'GRU 32 16 8 data2 split')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 256 64 16 8')
#         # read_csv(LSTM_output_test, 'plot_data_2')
#         # read_csv(true_pos, 'pos_data_2')
#         # read_csv(buffer_param, 'param_data_2')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, 'GRU 256 64 16 8 data2 split')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 512 256 128 64 32 16 8')
#         # read_csv(LSTM_output_test, 'plot_data_2')
#         # read_csv(true_pos, 'pos_data_2')
#         # read_csv(buffer_param, 'param_data_2')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, 'GRU 512 256 128 64 32 16 8 data2 split')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 8')
#         # read_csv(LSTM_output_test, 'plot_data_2')
#         # read_csv(true_pos, 'pos_data_2')
#         # read_csv(buffer_param, 'param_data_2')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, 'GRU 8 data2 split')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - data3 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 64 32 16 8')
#         # read_csv(LSTM_output_test, 'plot_data_3')
#         # read_csv(true_pos, 'pos_data_3')
#         # read_csv(buffer_param, 'param_data_3')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, 'GRU 64 32 16 8 data3 split')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - data3 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 256 128 64 32 16 8')
#         # read_csv(LSTM_output_test, 'plot_data_3')
#         # read_csv(true_pos, 'pos_data_3')
#         # read_csv(buffer_param, 'param_data_3')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # plot_output(LSTM_output_test, model_output, 0, 'GRU 256 128 64 32 16 8 data3 split')
#         #
#         # model_output = []
#         # buffer_param = []
#         # output_param = []
#         # LSTM_output_test = []
#         # true_pos = []
#         # read_csv(model_output,
#         #          'output - data3 - box 30 - window 0.5 - GRU 16 8')
#         # read_csv(LSTM_output_test, 'plot_data_3')
#         # # read_csv(true_pos, 'pos_data_3')
#         # read_csv(buffer_param, 'param_data_3_box')
#         # for x in range(len(buffer_param[0])):
#         #     output_param.append(buffer_param[0][x])
#         # postprocess_output(model_output, output_param)
#         # # for x in range(len(model_output)):
#         # #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
#         # #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
#         # for x in range(len(model_output)):
#         #     model_output[x][0] = -20 + (round(model_output[x][0]) - 1) * 30
#         # plot_output(LSTM_output_test, model_output, 0, 'GRU 16 8 data3 box')
#
#
#         plt.show()
#
#         # filter for only 2 input 2 output
#         # xdat = [item[0] for item in model_output]
#         # ydat = [item[1] for item in model_output]
#         # apply_filter(xdat, filt, order, smoothx)
#         # apply_filter(ydat, filt, order, smoothy)
#         # model_output = []
#         # for a in range(len(smoothx)):
#         #     model_output.append([smoothx[a], smoothy[a]])
#
#
#
#     elif MODE == "Normal":
#
#         s = SerialClass("COM3", 10, width, window, robot_vel, delay)
#         calibrate(s, 5)
#         model = load_model('32 ~ 8 model/')
#         buffer_param = []
#         output_param = []
#         read_csv(buffer_param, 'param_data_4')
#         for x in range(len(buffer_param[0])):
#             output_param.append(buffer_param[0][x])
#
#         # visibility_robot(s, 0)
#         # visibility_predictor(s, 0)
#         # visibility_arm_delay(s, 0)
#         while True:
#             #add actual robot arm, 4dof
#             #checkbox: visibility, on/off prediction
#             #text input: delay amount
#             #radio buttons: camera pos
#             update_serial(s, output_param)
#             # a = time.time()
#             predict(s, model, 0.1, output_param)
#
#             # if time.time() - a > 0.05:
#             #     print(time.time() - a)
#
#     else:
#         pass


import time

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import silence_tensorflow.auto
from interface import *
from prediction import *
# import multiprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sn


if __name__ == '__main__':

    # MODE = "Get_Training_Data"
    # MODE = "Train_Data"
    # MODE = "Run_Model"
    # MODE =  "Plot_Model"
    MODE = "Normal"
    # MODE = "test"

    window = 3
    prediction = 8
    width = 10
    robot_vel = 50 #cm/s
    delay = 0.3

    empty_window = 1
    idle_rate = 0.3 #how much percentage of idle data
    idle_bias = 1 #how much should (prediction + window) multiply with
    take_val = 10
    take_val_acc = 10
    split = 0.8
    epoch = 200
    params_to_process = 4

    file_name = 'Data 5 - 300ms - empty window 3 - GRU 128~8'

    if MODE == "Get_Training_Data":

        s = SerialClass("COM3", 10, width, window, robot_vel, delay)
        visibility_arm_delay(s, 0)
        s.label.visible = True
        calibrate(s, 5)
        collect_data(s, 'raw_data_5_box', 600)
        beep()

    elif MODE == "Train_Data":
        # 59966 / 10 samples
        #

        # filt = []
        # smoothx = []
        # smoothy = []
        # order = 30
        # design_filter(filt, order, 0.1)
        output_param = [0] * (params_to_process*2)
        box_length = [26, 29]
        LSTM_data = []
        LSTM_output = []
        LSTM_data_train = []
        LSTM_data_test = []
        LSTM_output_train = []
        LSTM_output_test = []
        raw_data = []
        raw_output = []
        read_csv(raw_data, 'raw_data_5_box')
        # filter for only 2 input 2 output
        # xdat = [item[0] for item in raw_data]
        # ydat = [item[1] for item in raw_data]
        # apply_filter(xdat, filt, order, smoothx)
        # apply_filter(ydat, filt, order, smoothy)
        # raw_data = []
        # for a in range(len(smoothx)):
        #     raw_data.append([smoothx[a], smoothy[a]])
        # for a in raw_data:
        #     raw_output.append(a[:])

        # for num in range(len(raw_data)):
        #     while len(raw_data[num]) > 2:
        #         raw_data[num].pop(2)


        test_data = []
        for x in raw_data:
            test_data.append(x[:])
        i = 0
        new_l = int(len(test_data) / width)
        while len(test_data) > new_l + width:
            for x in range(width - 1):
                test_data.pop(i + 1)
            i = i + 1
        while len(test_data) > new_l:
            test_data.pop(new_l)
        raw_data = []
        for x in test_data:
            raw_data.append(x)

        for x in range(len(raw_data)):
            raw_data[x].pop(2)

        for x in range(len(raw_data)):
            if 25 <= raw_data[x][1] < 40:
                if -20 <= raw_data[x][0] < -5:
                    raw_output.append([1, 1])
                elif -5 <= raw_data[x][0] < 10:
                    raw_output.append([2, 1])
                elif 10 <= raw_data[x][0] < 25:
                    raw_output.append([3, 1])
                else:
                    raw_output.append([0, 0])
            elif 40 <= raw_data[x][1] < 55:
                if -20 <= raw_data[x][0] < -5:
                    raw_output.append([1, 2])
                elif -5 <= raw_data[x][0] < 10:
                    raw_output.append([2, 2])
                elif 10 <= raw_data[x][0] < 25:
                    raw_output.append([3, 2])
                else:
                    raw_output.append([0, 0])
            else:
                raw_output.append([0, 0])


        # x = -40 to 40, y = -10 to 50
        # 8 x 6 box

        # boxify(raw_output)

        # for x in range(take_val):
        #     raw_data[x].append(0)
        #     raw_data[x].append(0)
        # for x in range(take_val, len(raw_data)):
        #     raw_data[x].append((raw_data[x][0] - raw_data[x - take_val][0])/(0.01*width*take_val))
        #     raw_data[x].append((raw_data[x][1] - raw_data[x - take_val][1]) / (0.01 * width * take_val))
        #
        # for x in range(take_val_acc):
        #     raw_data[x].append(0)
        #     raw_data[x].append(0)
        # for x in range(take_val_acc, len(raw_data)):
        #     raw_data[x].append((raw_data[x][2] - raw_data[x - take_val_acc][2]) / (0.01 * width * take_val_acc))
        #     raw_data[x].append((raw_data[x][3] - raw_data[x - take_val_acc][3]) / (0.01 * width * take_val_acc))

        split_input(raw_data, LSTM_data, window, prediction)
        split_output(raw_output, LSTM_output, window, prediction)

        if empty_window:
            for x in range(len(LSTM_data)):
                while len(LSTM_data[x]) > 2:
                    LSTM_data[x].pop(1)

        # x = 0
        # while x < len(LSTM_output):
        #
        # stop()
        # reference_pos(LSTM_data, LSTM_output)
        split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_test, LSTM_output_test,
                       split)

        x_prev = 0
        y_prev = 0
        include = 0
        output_idle = []
        output_changing = []
        input_idle = []
        input_changing = []
        for x in range(len(LSTM_output_train)):
            if x_prev != LSTM_output_train[x][0] or y_prev != LSTM_output_train[x][1]:
                input_changing.append(LSTM_data_train[x])
                output_changing.append(LSTM_output_train[x])
                include = round((prediction + window) * idle_bias)
            elif include > 0:
                input_changing.append(LSTM_data_train[x])
                output_changing.append(LSTM_output_train[x])
                include = include - 1
            else:
                input_idle.append(LSTM_data_train[x])
                output_idle.append(LSTM_output_train[x])
            x_prev = LSTM_output_train[x][0]
            y_prev = LSTM_output_train[x][1]

        num = [0, 0, 0, 0, 0, 0]
        balanced_data = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]
        for x in range(len(output_changing)):
            if output_changing[x][0] == 1 and output_changing[x][1] == 1:
                num[0] = num[0] + 1
                balanced_data[0][0].append(input_changing[x])
                balanced_data[0][1].append(output_changing[x])
            elif output_changing[x][0] == 2 and output_changing[x][1] == 1:
                num[1] = num[1] + 1
                balanced_data[1][0].append(input_changing[x])
                balanced_data[1][1].append(output_changing[x])
            elif output_changing[x][0] == 3 and output_changing[x][1] == 1:
                num[2] = num[2] + 1
                balanced_data[2][0].append(input_changing[x])
                balanced_data[2][1].append(output_changing[x])
            elif output_changing[x][0] == 1 and output_changing[x][1] == 2:
                num[3] = num[3] + 1
                balanced_data[3][0].append(input_changing[x])
                balanced_data[3][1].append(output_changing[x])
            elif output_changing[x][0] == 2 and output_changing[x][1] == 2:
                num[4] = num[4] + 1
                balanced_data[4][0].append(input_changing[x])
                balanced_data[4][1].append(output_changing[x])
            elif output_changing[x][0] == 3 and output_changing[x][1] == 2:
                num[5] = num[5] + 1
                balanced_data[5][0].append(input_changing[x])
                balanced_data[5][1].append(output_changing[x])
            else:
                print('Error: Output out of range')
                stop()

        for x in range(len(num)):
            if num[x] != min(num):
                while len(balanced_data[x][0]) > min(num):
                    random_delete = random.randint(0, len(balanced_data[x][0])-1)
                    balanced_data[x][0].pop(random_delete)
                    balanced_data[x][1].pop(random_delete)

        LSTM_data_train = []
        LSTM_output_train = []
        for x in range(len(balanced_data)):
            for y in range(len(balanced_data[0][0])):
                LSTM_data_train.append(balanced_data[x][0][y])
                LSTM_output_train.append(balanced_data[x][1][y])

        num = [0, 0, 0, 0, 0, 0]
        balanced_data = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]
        for x in range(len(output_idle)):
            if output_idle[x][0] == 1 and output_idle[x][1] == 1:
                num[0] = num[0] + 1
                balanced_data[0][0].append(input_idle[x])
                balanced_data[0][1].append(output_idle[x])
            elif output_idle[x][0] == 2 and output_idle[x][1] == 1:
                num[1] = num[1] + 1
                balanced_data[1][0].append(input_idle[x])
                balanced_data[1][1].append(output_idle[x])
            elif output_idle[x][0] == 3 and output_idle[x][1] == 1:
                num[2] = num[2] + 1
                balanced_data[2][0].append(input_idle[x])
                balanced_data[2][1].append(output_idle[x])
            elif output_idle[x][0] == 1 and output_idle[x][1] == 2:
                num[3] = num[3] + 1
                balanced_data[3][0].append(input_idle[x])
                balanced_data[3][1].append(output_idle[x])
            elif output_idle[x][0] == 2 and output_idle[x][1] == 2:
                num[4] = num[4] + 1
                balanced_data[4][0].append(input_idle[x])
                balanced_data[4][1].append(output_idle[x])
            elif output_idle[x][0] == 3 and output_idle[x][1] == 2:
                num[5] = num[5] + 1
                balanced_data[5][0].append(input_idle[x])
                balanced_data[5][1].append(output_idle[x])
            else:
                print('Error: Output out of range')
                stop()

        if idle_rate < 1:
            total = round(len(LSTM_data_train) / (1 - idle_rate))
            y = 0
            while len(LSTM_data_train) < total:
                random_insert = random.randint(0, len(balanced_data[y][0])-1)
                LSTM_data_train.append(balanced_data[y][0][random_insert])
                LSTM_output_train.append(balanced_data[y][1][random_insert])
                balanced_data[y][0].pop(random_insert)
                balanced_data[y][1].pop(random_insert)
                y = y + 1
                if y == 6:
                    y = 0
        else:
            for x in range(len(balanced_data)):
                for y in range(len(balanced_data[x][0])):
                    LSTM_data_train.append(balanced_data[x][0][y])
                    LSTM_output_train.append(balanced_data[x][1][y])

        get_param(LSTM_data_train, LSTM_output_train, output_param)
        buffer = [output_param, [0, 0, 0]]
        write_csv(buffer, 'Param - ' + file_name)

        preprocess_input(LSTM_data_train, output_param)
        preprocess_output(LSTM_output_train, output_param)
        preprocess_input(LSTM_data_test, output_param)
        preprocess_output(LSTM_output_test, output_param)



        shuffle(LSTM_data_train, LSTM_output_train)

        tt = time.time()
        train(LSTM_data_train, LSTM_output_train, LSTM_data_test, LSTM_output_test, window, epoch, 'Model - ' + file_name, 0.25)
        print(time.time() - tt)
        beep()

    elif MODE == "Run_Model":
        # 59966 / 10 samples
        #

        # filt = []
        # smoothx = []
        # smoothy = []
        # order = 30
        # design_filter(filt, order, 0.1)
        output_param = [0] * (params_to_process * 2)
        box_length = [26, 29]
        LSTM_data = []
        LSTM_output = []
        LSTM_data_train = []
        LSTM_data_test = []
        LSTM_output_train = []
        LSTM_output_test = []
        raw_data = []
        raw_output = []
        read_csv(raw_data, 'raw_data_5_box')
        # filter for only 2 input 2 output
        # xdat = [item[0] for item in raw_data]
        # ydat = [item[1] for item in raw_data]
        # apply_filter(xdat, filt, order, smoothx)
        # apply_filter(ydat, filt, order, smoothy)
        # raw_data = []
        # for a in range(len(smoothx)):
        #     raw_data.append([smoothx[a], smoothy[a]])
        # for a in raw_data:
        #     raw_output.append(a[:])

        # for num in range(len(raw_data)):
        #     while len(raw_data[num]) > 2:
        #         raw_data[num].pop(2)

        test_data = []
        for x in raw_data:
            test_data.append(x[:])
        i = 0
        new_l = int(len(test_data) / width)
        while len(test_data) > new_l + width:
            for x in range(width - 1):
                test_data.pop(i + 1)
            i = i + 1
        while len(test_data) > new_l:
            test_data.pop(new_l)
        raw_data = []
        for x in test_data:
            raw_data.append(x)

        for x in range(len(raw_data)):
            raw_data[x].pop(2)

        for x in range(len(raw_data)):
            if 25 <= raw_data[x][1] < 40:
                if -20 <= raw_data[x][0] < -5:
                    raw_output.append([1, 1])
                elif -5 <= raw_data[x][0] < 10:
                    raw_output.append([2, 1])
                elif 10 <= raw_data[x][0] < 25:
                    raw_output.append([3, 1])
                else:
                    raw_output.append([0, 0])
            elif 40 <= raw_data[x][1] < 55:
                if -20 <= raw_data[x][0] < -5:
                    raw_output.append([1, 2])
                elif -5 <= raw_data[x][0] < 10:
                    raw_output.append([2, 2])
                elif 10 <= raw_data[x][0] < 25:
                    raw_output.append([3, 2])
                else:
                    raw_output.append([0, 0])
            else:
                raw_output.append([0, 0])

        # x = -40 to 40, y = -10 to 50
        # 8 x 6 box

        # boxify(raw_output)

        # for x in range(take_val):
        #     raw_data[x].append(0)
        #     raw_data[x].append(0)
        # for x in range(take_val, len(raw_data)):
        #     raw_data[x].append((raw_data[x][0] - raw_data[x - take_val][0])/(0.01*width*take_val))
        #     raw_data[x].append((raw_data[x][1] - raw_data[x - take_val][1]) / (0.01 * width * take_val))
        #
        # for x in range(take_val_acc):
        #     raw_data[x].append(0)
        #     raw_data[x].append(0)
        # for x in range(take_val_acc, len(raw_data)):
        #     raw_data[x].append((raw_data[x][2] - raw_data[x - take_val_acc][2]) / (0.01 * width * take_val_acc))
        #     raw_data[x].append((raw_data[x][3] - raw_data[x - take_val_acc][3]) / (0.01 * width * take_val_acc))

        split_input(raw_data, LSTM_data, window, prediction)
        split_output(raw_output, LSTM_output, window, prediction)

        if empty_window:
            for x in range(len(LSTM_data)):
                while len(LSTM_data[x]) > 2:
                    LSTM_data[x].pop(1)

        # x = 0
        # while x < len(LSTM_output):
        #
        # stop()
        # reference_pos(LSTM_data, LSTM_output)
        split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_test, LSTM_output_test,
                       split)

        output_param = []
        buffer_param = []
        read_csv(buffer_param, 'Param - ' + file_name)
        for x in range(len(buffer_param[0])):
            output_param.append(buffer_param[0][x])

        preprocess_input(LSTM_data_test, output_param)

        model_output = []
        print('Starting...')
        tt = time.time()
        model = load_model('Model - ' + file_name)
        use_model(LSTM_data_test, model_output, model)
        print(time.time() - tt)
        postprocess_output(model_output, output_param)
        write_csv(LSTM_output_test, 'True_output - ' + file_name)
        write_csv(model_output, 'Predicted - ' + file_name)
        beep()

    elif MODE == "Plot_Model":

        model_output = []
        LSTM_output_test = []
        read_csv(model_output, 'Predicted - ' + file_name)
        read_csv(LSTM_output_test, 'True_output - ' + file_name)

        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]

        figure, axis = plt.subplots(2, 1)
        a = []
        b = []
        for x in range(len(LSTM_output_test)):
            a.append(LSTM_output_test[x][0])
            b.append(round(model_output[x][0]))

        axis[0].plot(a)
        axis[0].plot(b)
        plt.title(file_name + ' - x')
        plt.ylabel('Distance (cm)')
        plt.xlabel('Data Sample')
        plt.legend(['Actual', 'Predicted'], loc='upper right')

        a = []
        b = []
        for x in range(len(LSTM_output_test)):
            b.append(round(model_output[x][1]))
            a.append(LSTM_output_test[x][1])

        axis[1].plot(a)
        axis[1].plot(b)
        plt.title(file_name + ' - y')
        plt.ylabel('Distance (cm)')
        plt.xlabel('Data Sample')
        plt.legend(['Actual', 'Predicted'], loc='upper right')
        plt.show()

        num1 = 0
        num2 = 0
        for x in range(len(model_output)):
            if LSTM_output_test[x][0] == round(model_output[x][0]):
                num1 = num1 + 1
            else:
                num2 = num2 + 1
        print(num1 / (num1 + num2))

        num1 = 0
        num2 = 0
        for x in range(len(model_output)):
            if LSTM_output_test[x][1] == round(model_output[x][1]):
                num1 = num1 + 1
            else:
                num2 = num2 + 1
        print(num1 / (num1 + num2))


        test_data= []
        predicted_data = []

        for x in range(len(model_output)):
            if 0 < LSTM_output_test[x][0]:
                test_data.append(int(LSTM_output_test[x][0]))
                if LSTM_output_test[x][1] == 1:
                    test_data[len(test_data)-1] = test_data[len(test_data)-1] + 3
            else:
                test_data.append(0)

            if 0 < round(model_output[x][0]) < 4 and 0 < round(model_output[x][1]) < 3:
                predicted_data.append(round(model_output[x][0]))
                if round(model_output[x][1]) == 1:
                    predicted_data[len(predicted_data) - 1] = predicted_data[len(predicted_data) - 1] + 3
            else:
                predicted_data.append(0)

        cm = confusion_matrix(test_data,predicted_data)
        accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        acc = accuracy_score(test_data, predicted_data)

        # test_data = []
        # predicted_data = []
        predicted_data_roc = []
        #
        for x in range(len(model_output)):
        #     test_data.append([])
        #     if 0 < LSTM_output_test[x][0]:
        #         if LSTM_output_test[x][1] == 1:
        #             for y in range(3):
        #                 test_data[len(test_data) - 1].append(0)
        #         for y in range(int(LSTM_output_test[x][0]) - 1):
        #             test_data[len(test_data) - 1].append(0)
        #         test_data[len(test_data) - 1].append(1)
        #         while len(test_data[len(test_data) - 1]) < 7:
        #             test_data[len(test_data) - 1].append(0)
        #     else:
        #         for y in range(7):
        #             test_data[len(test_data) - 1].append(0)
        #


            predicted_data_roc.append([])
            if 0 < round(model_output[x][0]) < 4 and 0 < round(model_output[x][1]) < 3:
                if round(model_output[x][1]) == 1:
                    for y in range(3):
                        predicted_data_roc[len(predicted_data_roc) - 1].append(0)
                for y in range(round(model_output[x][0]) - 1):
                    predicted_data_roc[len(predicted_data_roc) - 1].append(0)
                predicted_data_roc[len(predicted_data_roc) - 1].append(1)
                while len(predicted_data_roc[len(predicted_data_roc) - 1]) < 6:
                    predicted_data_roc[len(predicted_data_roc) - 1].append(0)
            else:
                for y in range(6):
                    predicted_data_roc[len(predicted_data_roc) - 1].append(1/6)

        roc = roc_auc_score(test_data, predicted_data_roc, multi_class='ovo')

        precision = precision_score(test_data, predicted_data, average='macro')
        recall = recall_score(test_data, predicted_data, average='macro')
        f1 = f1_score(test_data, predicted_data, average='macro')

        print('Acc:', accuracy)
        print('Lib Acc:', acc)
        print('roc:', roc)
        print('precision:', precision)
        print('recall:', recall)
        print('f1:', f1)

        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        # plt.title("Number of Trees: " + str(y))
        plt.show()


        # plot_output(LSTM_output_test, model_output, 0, 'GRU 16 8 data3 box')

        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - pos reference - velocity 0.05 - acc 0.05 - window 0.75 - 64')
        # read_csv(LSTM_output_test, 'plot_data')
        # read_csv(true_pos, 'pos_data')
        # read_csv(buffer_param, 'param_data')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, '64')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - pos reference - velocity 0.05 - acc 0.05 - window 0.75 - 64 32')
        # read_csv(LSTM_output_test, 'plot_data')
        # read_csv(true_pos, 'pos_data')
        # read_csv(buffer_param, 'param_data')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, '64 32')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - pos reference - velocity 0.05 - acc 0.05 - window 0.75 - 64 32 16')
        # read_csv(LSTM_output_test, 'plot_data')
        # read_csv(true_pos, 'pos_data')
        # read_csv(buffer_param, 'param_data')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, '64 32 16')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - pos reference - velocity 0.05 - acc 0.05 - window 0.75 - 64 32 16 8')
        # read_csv(LSTM_output_test, 'plot_data')
        # read_csv(true_pos, 'pos_data')
        # read_csv(buffer_param, 'param_data')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, '64 32 16 8')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - data2 - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - 64 32 16 8')
        # read_csv(LSTM_output_test, 'plot_data_2')
        # read_csv(true_pos, 'pos_data_2')
        # read_csv(buffer_param, 'param_data_2')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, '64 32 16 8 data2')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - 64 32 16 8')
        # read_csv(LSTM_output_test, 'plot_data_2')
        # read_csv(true_pos, 'pos_data_2')
        # read_csv(buffer_param, 'param_data_2')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, '64 32 16 8 data2 split')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 64 32 16 8')
        # read_csv(LSTM_output_test, 'plot_data_2')
        # read_csv(true_pos, 'pos_data_2')
        # read_csv(buffer_param, 'param_data_2')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, 'GRU 64 32 16 8 data2 split')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 32 16 8')
        # read_csv(LSTM_output_test, 'plot_data_2')
        # read_csv(true_pos, 'pos_data_2')
        # read_csv(buffer_param, 'param_data_2')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, 'GRU 32 16 8 data2 split')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 256 64 16 8')
        # read_csv(LSTM_output_test, 'plot_data_2')
        # read_csv(true_pos, 'pos_data_2')
        # read_csv(buffer_param, 'param_data_2')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, 'GRU 256 64 16 8 data2 split')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 512 256 128 64 32 16 8')
        # read_csv(LSTM_output_test, 'plot_data_2')
        # read_csv(true_pos, 'pos_data_2')
        # read_csv(buffer_param, 'param_data_2')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, 'GRU 512 256 128 64 32 16 8 data2 split')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - data2 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 8')
        # read_csv(LSTM_output_test, 'plot_data_2')
        # read_csv(true_pos, 'pos_data_2')
        # read_csv(buffer_param, 'param_data_2')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, 'GRU 8 data2 split')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - data3 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 64 32 16 8')
        # read_csv(LSTM_output_test, 'plot_data_3')
        # read_csv(true_pos, 'pos_data_3')
        # read_csv(buffer_param, 'param_data_3')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, 'GRU 64 32 16 8 data3 split')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - data3 - split - pos reference - velocity 0.25 - acc 0.35 - window 0.75 - GRU 256 128 64 32 16 8')
        # read_csv(LSTM_output_test, 'plot_data_3')
        # read_csv(true_pos, 'pos_data_3')
        # read_csv(buffer_param, 'param_data_3')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # plot_output(LSTM_output_test, model_output, 0, 'GRU 256 128 64 32 16 8 data3 split')
        #
        # model_output = []
        # buffer_param = []
        # output_param = []
        # LSTM_output_test = []
        # true_pos = []
        # read_csv(model_output,
        #          'output - data3 - box 30 - window 0.5 - GRU 16 8')
        # read_csv(LSTM_output_test, 'plot_data_3')
        # # read_csv(true_pos, 'pos_data_3')
        # read_csv(buffer_param, 'param_data_3_box')
        # for x in range(len(buffer_param[0])):
        #     output_param.append(buffer_param[0][x])
        # postprocess_output(model_output, output_param)
        # # for x in range(len(model_output)):
        # #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        # #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        # for x in range(len(model_output)):
        #     model_output[x][0] = -20 + (round(model_output[x][0]) - 1) * 30
        # plot_output(LSTM_output_test, model_output, 0, 'GRU 16 8 data3 box')

        # filter for only 2 input 2 output
        # xdat = [item[0] for item in model_output]
        # ydat = [item[1] for item in model_output]
        # apply_filter(xdat, filt, order, smoothx)
        # apply_filter(ydat, filt, order, smoothy)
        # model_output = []
        # for a in range(len(smoothx)):
        #     model_output.append([smoothx[a], smoothy[a]])



    elif MODE == "Normal":

        tf.keras.utils.disable_interactive_logging()

        s = SerialClass("COM3", 10, width, window, robot_vel, delay)
        calibrate(s, 5)
        model = load_model('Model - ' + file_name)
        buffer_param = []
        output_param = []
        read_csv(buffer_param, 'Param - ' + file_name)
        for x in range(len(buffer_param[0])):
            output_param.append(buffer_param[0][x])

        # visibility_robot(s, 0)
        # visibility_predictor(s, 0)
        # visibility_arm_delay(s, 0)
        ttt = 0
        while True:
            #add actual robot arm, 4dof
            #checkbox: visibility, on/off prediction
            #text input: delay amount
            #radio buttons: camera pos
            update_serial(s, output_param)
            # a = time.time()
            predict(s, model, 0.1, output_param)

            # if time.time() - a > 0.05:
            #     print(time.time() - a)

    else:
        pass
