import time

from interface import *
from prediction import *
import multiprocessing

if __name__ == '__main__':

    # MODE = "Get_Training_Data"
    # MODE = "Train_Data"
    # MODE = "Run_Model"
    MODE =  "Plot_Model"
    # MODE = "Normal"
    # MODE = "test"

    window = 8
    prediction = 5
    width = 10
    take_val = 10
    take_val_acc = 10
    split = [3600, 4800]
    epoch = 400
    params_to_process = 4
    # try one layer only but wider
    output_file = 'output - input 2 - data4 - box 15 - window 0.8 - GRU 32 ~ 8'

    if MODE == "Get_Training_Data":

        s = SerialClass("COM3", 10, width, window)
        calibrate(s, 5)
        collect_data(s, 'raw_data_4_box', 600)
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
        LSTM_data_val = []
        LSTM_data_test = []
        LSTM_output_train = []
        LSTM_output_val = []
        LSTM_output_test = []
        raw_data = []
        raw_output = []
        read_csv(raw_data, 'raw_data_4_box')
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
            if 25 <= raw_data[x][1] < 40:
                if -20 <= raw_data[x][0] < -5:
                    raw_data[x][2] = 4
                elif -5 <= raw_data[x][0] < 10:
                    raw_data[x][2] = 5
                elif 10 <= raw_data[x][0] < 25:
                    raw_data[x][2] = 6
            elif 40 <= raw_data[x][1] < 55:
                if -20 <= raw_data[x][0] < -5:
                    raw_data[x][2] = 1
                elif -5 <= raw_data[x][0] < 10:
                    raw_data[x][2] = 2
                elif 10 <= raw_data[x][0] < 25:
                    raw_data[x][2] = 3

        for x in range(len(raw_data)):
            raw_output.append([raw_data[x][2]])

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
        # reference_pos(LSTM_data, LSTM_output)
        split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_val, LSTM_output_val,
                       LSTM_data_test, LSTM_output_test, split[0], split[1])
        get_param(LSTM_data_train, LSTM_output_train, output_param)
        preprocess_input(LSTM_data_train, output_param)
        preprocess_input(LSTM_data_val, output_param)
        preprocess_input(LSTM_data_test, output_param)
        preprocess_output(LSTM_output_train, output_param)
        preprocess_output(LSTM_output_val, output_param)
        preprocess_output(LSTM_output_test, output_param)

        shuffle(LSTM_data_train, LSTM_output_train)

        for x in range(len(LSTM_data_train)):
            for a in range(len(LSTM_data_train[0])):
                LSTM_data_train[x][a].pop(2)

        for x in range(len(LSTM_data_val)):
            for a in range(len(LSTM_data_val[0])):
                LSTM_data_val[x][a].pop(2)

        for x in range(len(LSTM_data_test)):
            for a in range(len(LSTM_data_test[0])):
                LSTM_data_test[x][a].pop(2)

        tt = time.time()
        train(LSTM_data_train, LSTM_output_train, LSTM_data_val, LSTM_output_val, window, epoch, 'model/', 0.2)
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
        LSTM_data_val = []
        LSTM_data_test = []
        LSTM_output_train = []
        LSTM_output_val = []
        LSTM_output_test = []
        raw_data = []
        raw_output = []
        read_csv(raw_data, 'raw_data_4_box')
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
            if 25 <= raw_data[x][1] < 40:
                if -20 <= raw_data[x][0] < -5:
                    raw_data[x][2] = 4
                elif -5 <= raw_data[x][0] < 10:
                    raw_data[x][2] = 5
                elif 10 <= raw_data[x][0] < 25:
                    raw_data[x][2] = 6
            elif 40 <= raw_data[x][1] < 55:
                if -20 <= raw_data[x][0] < -5:
                    raw_data[x][2] = 1
                elif -5 <= raw_data[x][0] < 10:
                    raw_data[x][2] = 2
                elif 10 <= raw_data[x][0] < 25:
                    raw_data[x][2] = 3

        for x in range(len(raw_data)):
            raw_output.append([raw_data[x][2]])

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
        # reference_pos(LSTM_data, LSTM_output)
        split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_val, LSTM_output_val,
                       LSTM_data_test, LSTM_output_test, split[0], split[1])
        get_param(LSTM_data_train, LSTM_output_train, output_param)
        preprocess_input(LSTM_data_train, output_param)
        preprocess_input(LSTM_data_val, output_param)
        preprocess_input(LSTM_data_test, output_param)
        preprocess_output(LSTM_output_train, output_param)
        preprocess_output(LSTM_output_val, output_param)
        preprocess_output(LSTM_output_test, output_param)
        shuffle(LSTM_data_train, LSTM_output_train)

        for x in range(len(LSTM_data_train)):
            for a in range(len(LSTM_data_train[0])):
                LSTM_data_train[x][a].pop(2)

        for x in range(len(LSTM_data_val)):
            for a in range(len(LSTM_data_val[0])):
                LSTM_data_val[x][a].pop(2)

        for x in range(len(LSTM_data_test)):
            for a in range(len(LSTM_data_test[0])):
                LSTM_data_test[x][a].pop(2)

        model_output = []
        print('Starting...')
        tt = time.time()
        model = load_model('model/')
        use_model(LSTM_data_test, model_output, model)
        print(time.time() - tt)
        write_csv(model_output, output_file)
        beep()

    elif MODE == "Plot_Model":
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
        LSTM_data_val = []
        LSTM_data_test = []
        LSTM_output_train = []
        LSTM_output_val = []
        LSTM_output_test = []
        raw_data = []
        raw_output = []
        read_csv(raw_data, 'raw_data_4_box')
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
            if 25 <= raw_data[x][1] < 40:
                if -20 <= raw_data[x][0] < -5:
                    raw_data[x][2] = 4
                elif -5 <= raw_data[x][0] < 10:
                    raw_data[x][2] = 5
                elif 10 <= raw_data[x][0] < 25:
                    raw_data[x][2] = 6
            elif 40 <= raw_data[x][1] < 55:
                if -20 <= raw_data[x][0] < -5:
                    raw_data[x][2] = 1
                elif -5 <= raw_data[x][0] < 10:
                    raw_data[x][2] = 2
                elif 10 <= raw_data[x][0] < 25:
                    raw_data[x][2] = 3

        for x in range(len(raw_data)):
            raw_output.append([raw_data[x][2]])

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
        # reference_pos(LSTM_data, LSTM_output)
        split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_val, LSTM_output_val,
                       LSTM_data_test, LSTM_output_test, split[0], split[1])
        get_param(LSTM_data_train, LSTM_output_train, output_param)

        # read_csv(model_output, 'output - complicated model')
        # read_csv(model_output, 'output - simple model')
        # read_csv(model_output, 'output - complicated model - reference')

        # for x in range(len(LSTM_output_test)):
        #     LSTM_output_test[x][0] = LSTM_output_test[x][0] + true_pos[x][0]
        #     LSTM_output_test[x][1] = LSTM_output_test[x][1] + true_pos[x][1]

        # write_csv(LSTM_output_test, 'plot_data_4')
        # # write_csv(true_pos, 'pos_data_3')
        # buffer = [output_param, [0, 0, 0]]
        # write_csv(buffer, 'param_data_4')
        # print('Done')
        # stop()

        model_output = []
        buffer_param = []
        output_param = []
        LSTM_output_test = []
        true_pos = []
        read_csv(model_output,
                 'output - input 2 - data4 - box 15 - window 0.8 - GRU 128')
        read_csv(LSTM_output_test, 'plot_data_4')
        # read_csv(true_pos, 'pos_data_3')
        read_csv(buffer_param, 'param_data_4')
        for x in range(len(buffer_param[0])):
            output_param.append(buffer_param[0][x])
        postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]

        figure, axis = plt.subplots(2, 1)

        a = []
        for x in range(len(LSTM_output_test)):
            a.append(LSTM_output_test[x][0])
        axis[0].plot(a)

        a = []
        for x in range(len(model_output)):
            a.append(round(model_output[x][0]))
        axis[0].plot(a)

        num1 = 0
        num2 = 0
        for x in range(len(model_output)):
            if LSTM_output_test[x][0] == round(model_output[x][0]):
                num1 = num1 + 1
            else:
                num2 = num2 + 1
        print(num1 / (num1 + num2))

        model_output = []
        buffer_param = []
        output_param = []
        LSTM_output_test = []
        true_pos = []
        read_csv(model_output,
                 'output - input 2 - data4 - box 15 - window 0.8 - GRU 32 ~ 8')
        read_csv(LSTM_output_test, 'plot_data_4')
        # read_csv(true_pos, 'pos_data_3')
        read_csv(buffer_param, 'param_data_4')
        for x in range(len(buffer_param[0])):
            output_param.append(buffer_param[0][x])
        postprocess_output(model_output, output_param)
        # for x in range(len(model_output)):
        #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
        #     model_output[x][1] = model_output[x][1] + true_pos[x][1]
        a = []
        for x in range(len(LSTM_output_test)):
            a.append(LSTM_output_test[x][0])
        axis[1].plot(a)

        a = []
        for x in range(len(model_output)):
            a.append(round(model_output[x][0]))
        axis[1].plot(a)

        num1 = 0
        num2 = 0
        for x in range(len(model_output)):
            if LSTM_output_test[x][0] == round(model_output[x][0]):
                num1 = num1 + 1
            else:
                num2 = num2 + 1
        print(num1 / (num1 + num2))

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


        plt.show()

        # filter for only 2 input 2 output
        # xdat = [item[0] for item in model_output]
        # ydat = [item[1] for item in model_output]
        # apply_filter(xdat, filt, order, smoothx)
        # apply_filter(ydat, filt, order, smoothy)
        # model_output = []
        # for a in range(len(smoothx)):
        #     model_output.append([smoothx[a], smoothy[a]])



    elif MODE == "Normal":

        s = SerialClass("COM3", 10, width, window)
        calibrate(s, 5)
        model = load_model('model/')
        buffer_param = []
        output_param = []
        read_csv(buffer_param, 'param_data_4')
        for x in range(len(buffer_param[0])):
            output_param.append(buffer_param[0][x])
        while True:
            update_serial(s, output_param)
            predict(s, model, 2, output_param)
    #         about 0.07s

    else:
        pass
