
import time

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import silence_tensorflow.auto
from interface import *
from prediction import *
from run_from_data import *
# import multiprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sn


if __name__ == '__main__':

    # MODE = "Get_Training_Data"
    # MODE = "Train_Data"
    # MODE = "Run_Model"
    # MODE =  "Plot_Model"
    MODE = "Normal"
    # MODE = "Get_Robot_effectiveness"
    # MODE = 'Plot_effectiveness'
    # MODE = "Plot_All"
    # MODE
    # MODE = "test"

    prediction = 5
    hidden_layer = 5

    max_hidden_layer = 4
    for x in range(hidden_layer):
        max_hidden_layer = max_hidden_layer * 2
    max_hidden_layer = int(max_hidden_layer)
    window = 3
    width = 10
    prediction = int(prediction * 10 / width)
    window = int(window * 10 / width)
    robot_vel = 50 #cm/s

    empty_window = 1
    idle_rate = 0.3 #how much percentage of idle data
    idle_bias = 1 #how much should (prediction + window) multiply with
    take_val = 10
    take_val_acc = 10
    split = 0.8
    epoch = 200
    params_to_process = 4

    file_name = 'Data 5 - ' + str(prediction) + '00ms - empty window 3 - LSTM ' + str(max_hidden_layer) + '~8'
    delay = prediction / 10

    if MODE == "Get_Training_Data":

        s = SerialClass("COM3", 10, width, window, robot_vel, delay)
        visibility_arm_delay(s, 0)
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
                if len(balanced_data[y][0]) > 0:
                    random_insert = random.randint(0, len(balanced_data[y][0])-1)
                    LSTM_data_train.append(balanced_data[y][0][random_insert])
                    LSTM_output_train.append(balanced_data[y][1][random_insert])
                    balanced_data[y][0].pop(random_insert)
                    balanced_data[y][1].pop(random_insert)
                else:
                    a = 0
                    for x in range(len(balanced_data)):
                        if len(balanced_data[x][0]) <= 0:
                            a = a + 1
                    if a == len(balanced_data):
                        break
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
        train(LSTM_data_train, LSTM_output_train, LSTM_data_test, LSTM_output_test, epoch, 'Model - ' + file_name, 0, max_hidden_layer)
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
        print((time.time() - tt) / len(model_output))
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
        print('X-Accuracy' + str(num1 / (num1 + num2)))

        num1 = 0
        num2 = 0
        for x in range(len(model_output)):
            if LSTM_output_test[x][1] == round(model_output[x][1]):
                num1 = num1 + 1
            else:
                num2 = num2 + 1
        print('Y-Accuracy' + str(num1 / (num1 + num2)))


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
        total = 0
        for y in range(len(cm)):
            for x in range(len(cm[0])):
                total = total + cm[y][x]
        accuracy = 0
        for x in range(len(cm)):
            accuracy = accuracy + cm[x][x] / total
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


    elif MODE == "Normal":

        tf.keras.utils.disable_interactive_logging()
        delay = 0.3
        s = SerialClass("COM3", 10, width, window, robot_vel, delay)
        calibrate(s, 5)
        s.model = load_model('Model - Data 5 - 300ms - empty window 3 - LSTM 128~8')
        buffer_param = []
        s.output_param = []
        read_csv(buffer_param, 'Param - ' + file_name)
        for x in range(len(buffer_param[0])):
            s.output_param.append(buffer_param[0][x])

        # visibility_robot(s, 0)
        # visibility_robot_now(s, 0)
        # visibility_robot_delay(s, 0)
        # visibility_arm(s, 0)
        # visibility_arm_delay(s, 0)
        # visibility_indicator(s, 0)
        update_text(s)
        while True:
            #text input: delay amount
            #radio buttons: camera pos
            update_serial(s, s.output_param)
            keyboard_input(s)
            # a = time.time()
            predict(s, s.model, 0.1, s.output_param)

            # if time.time() - a > 0.05:
            #     print(time.time() - a)

    elif MODE == "Get_Robot_effectiveness":

        tf.keras.utils.disable_interactive_logging()

        s = SerialClass_rfd(10, width, window, robot_vel, delay, 'raw_data_5_box', split)
        model = load_model('Model - ' + file_name)
        buffer_param = []
        output_param = []
        read_csv(buffer_param, 'Param - ' + file_name)
        for x in range(len(buffer_param[0])):
            output_param.append(buffer_param[0][x])

        # visibility_robot(s, 0)
        # visibility_robot_now(s, 0)
        # visibility_robot_delay(s, 0)
        visibility_arm(s, 0)
        visibility_arm_delay(s, 0)
        # visibility_indicator(s, 0)

        total_len = len(s.raw_data)


        s.data_time = time.time()
        total_time = time.time()
        while True:
            # text input: delay amount
            # radio buttons: camera pos
            if len(s.raw_data) == 0:
                break
            update_serial_rfd(s, total_len)
            # a = time.time()
            predict(s, model, 0.1, output_param)

        total_time = time.time() - total_time

        total_len = len(s.rmse)
        num1 = 0
        num2 = 0
        num3 = 0
        num4 = 0
        for x in range(total_len):
            num1 = num1 + s.rmse[x][0]
            num2 = num2 + s.rmse[x][1]
            num3 = num3 + s.rmse[x][2]
            num4 = num4 + s.rmse[x][3]

        num1 = sqrt(num1 / total_len)
        num2 = sqrt(num2 / total_len)
        num3 = num3 / total_len
        num4 = num4 / total_len

        print("Total Time: " + str(total_time))
        print("RMSE with AI: " + str(num1))
        print("RMSE without AI: " + str(num2))
        print("Accuracy with AI: " + str(num3))
        print("Accuracy without AI: " + str(num4))

        stop()

    elif MODE == 'Plot_effectiveness':
        delay = [300, 500, 800, 1000, 1300, 1500, 1800, 2000, 2300, 2500, 2800, 3000, 4000, 6000, 8000, 10000]
        #GRU
        # rmse_without = [5.8197, 7.6195, 10.2113, 11.7967, 13.9221, 15.1773, 16.8583, 17.8085, 18.9430, 19.5205, 20.1291, 20.4309]
        # rmse_with = [4.0151, 4.5023, 5.8711, 7.3361, 8.6536, 9.8271, 11.1396, 12.1478, 13.3084, 13.8009, 14.0105, 13.9864]
        # acc_without = [0.7789, 0.6876, 0.5635, 0.4880, 0.3718, 0.2936, 0.1809, 0.1112, 0.0571, 0.0556, 0.0696, 0.0742]
        # acc_with = [0.8555, 0.8245, 0.7407, 0.6484, 0.5724, 0.4940, 0.4086, 0.3343, 0.2704, 0.2513, 0.2309, 0.2299]

        #LSTM
        rmse_with =[ 4.088339128364286, 4.4433493799865476, 5.949118891447315, 6.920611719569019, 8.575532077312332, 9.88205902213602, 12.078351409577133, 13.947773056264456, 15.100003091587649, 14.822069704053538, 14.805925089602583, 14.833600646796098, 15.042914423607161, 15.643639673856159, 15.927289405259637, 16.548714341921745]
        rmse_without = [ 5.933043706468459, 7.863522472293858, 10.514424751216625, 12.133060343166676, 14.462058154606082, 15.779694763025443, 17.59445584156312, 18.502086256073568, 19.75120511098072, 20.38330058630951, 21.011329847025127, 21.184660544670326, 20.784915917720554, 18.129599425670193, 19.408845792876456, 20.41830348747224]
        acc_with = [ 0.8708860759493671, 0.8489451476793249, 0.745531914893617, 0.6893617021276596, 0.5867346938775511, 0.5232067510548524, 0.3534923339011925, 0.2776371308016878, 0.21030405405405406, 0.21869639794168097, 0.21875, 0.21331058020477817, 0.1974025974025974, 0.20087336244541484, 0.22415291051259775, 0.18041237113402062]
        acc_without = [ 0.770464135021097, 0.6751054852320675, 0.5438297872340425, 0.4757446808510638, 0.35799319727891155, 0.280168776371308, 0.17291311754684838, 0.1021097046413502, 0.05320945945945946, 0.058319039451114926, 0.08023648648648649, 0.08703071672354949, 0.1264069264069264, 0.18427947598253275, 0.12510860121633363, 0.11855670103092783]

        # 1749427410.885838 - 1749407301.0685797

        figure, axis = plt.subplots(2, 1)

        axis[0].plot(delay, rmse_without, label="Without Prediction")
        axis[0].plot(delay, rmse_with, label="With Prediction")
        axis[0].set_title('Root Mean Squared Error')
        axis[0].set_ylabel('Magnitude (cm)')
        axis[0].set_xlabel('Delay Induced (ms)')

        axis[1].plot(delay, acc_without, label="Without Prediction")
        axis[1].plot(delay, acc_with, label="With Prediction")
        axis[1].set_title('Accuracy')
        axis[1].set_ylabel('Magnitude')
        axis[1].set_xlabel('Delay Induced (ms)')

        axis[0].legend(loc='best')
        axis[1].legend(loc='best')

        figure.tight_layout()

        ori_freq = []
        for x in range(len(delay)):
            ori_freq.append(0.6204)
        #GRU
        # ai_freq = [0.6928, 0.7346, 0.9524, 1.2364, 1.5887, 1.4716, 1.6736, 1.6653, 1.4070, 1.4154, 0.2096, 0.2096]
        #LSTM
        ai_freq = [ 0.7095158597662922, 0.7262103505843225, 1.1027568922305997, 1.2364243943191573, 1.638795986622108, 1.5635451505017053, 1.3807531380753428, 1.4058577405858037, 0.21775544388610174, 0.8542713567839375, 0.20955574182733047, 0.20955574182733047, 0.20990764063812362, 0.235888795282229, 0.18596787827557443, 0.1950805767599701]

        auc = [0.9644653843157595, 0.9640812206264648, 0.9258228873816275, 0.8806900756779292, 0.8203670822212351,
               0.7926100134173321, 0.6368710850607932, 0.584288064113863, 0.4972159090909091, 0.5000833836149101,
               0.5138541666666666, 0.5135606060606062, 0.5036079545454545, 0.49738136077758727, 0.5089251000571756,
               0.48966838193253287]
        all_data = []
        for x in range(len(rmse_without)):
            all_data.append(rmse_with[x])
            all_data.append(rmse_without[x])
        min_num = min(all_data)
        factor = 1 / (max(all_data) - min_num)
        for x in range(len(rmse_without)):
            rmse_without[x] = (rmse_without[x] - min_num) * factor
            rmse_with[x] = (rmse_with[x] - min_num) * factor

        all_data = []
        for x in range(len(acc_without)):
            all_data.append(acc_without[x])
            all_data.append(acc_with[x])
        min_num = min(all_data)
        factor = 1 / (max(all_data) - min_num)
        for x in range(len(acc_without)):
            acc_without[x] = (acc_without[x] - min_num) * factor
            acc_with[x] = (acc_with[x] - min_num) * factor

        figure, axis = plt.subplots(2, 1)

        axis[0].plot(delay, ori_freq, label="Without Prediction")
        axis[0].plot(delay, ai_freq, label="With Prediction")
        axis[0].set_title('Frequency of Outputs')
        axis[0].set_ylabel('Frequency (Hz)')
        axis[0].set_xlabel('Delay Induced (ms)')

        axis[1].plot(delay, rmse_without, label="RMSE without prediction", color = 'b', linestyle = '-')
        axis[1].plot(delay, rmse_with, label="RMSE with prediction", color = 'r', linestyle = '-')
        axis[1].plot(delay, acc_without, label="Accuracy without prediction", color = 'b', linestyle = '--')
        axis[1].plot(delay, acc_with, label="Accuracy with prediction", color = 'r', linestyle = '--')
        axis[1].set_title('Normalized RMSE and Accuracy')
        axis[1].set_ylabel('Magnitude')
        axis[1].set_xlabel('Delay Induced (ms)')

        axis[0].legend(loc='best')
        axis[1].legend(loc='best')

        figure.tight_layout()

        figure, axis = plt.subplots(1, 1)
        axis.plot(delay, auc)
        axis.set_title('AUC-ROC vs Delay Induced')
        axis.set_xlabel('Delay Induced (ms)')
        axis.set_ylabel('AUC-ROC')


        plt.show()

    elif MODE == 'Plot_All':

        base1 = 'Data 5 - '
        base2 = '00ms - empty window 3 - LSTM 128~8'

        num3 = [0, 0]
        for y in [3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30, 40, 60, 80, 100]:
            model_output = []
            LSTM_output_test = []
            read_csv(model_output, 'Predicted - ' + base1 + str(y) + base2)
            read_csv(LSTM_output_test, 'True_output - ' + base1 + str(y) + base2)
            time_stamp = []
            num = 0
            for z in range(len(LSTM_output_test)):
                time_stamp.append(num)
                num = num + 0.1

            # for x in range(len(model_output)):
            #     model_output[x][0] = model_output[x][0] + true_pos[x][0]
            #     model_output[x][1] = model_output[x][1] + true_pos[x][1]

            figure, axis = plt.subplots(2, 1)
            a = []
            b = []
            for x in range(len(LSTM_output_test)):
                a.append(LSTM_output_test[x][0])
                b.append(round(model_output[x][0]))

            axis[0].plot(time_stamp, a, label='Actual')
            axis[0].plot(time_stamp, b, label='Predicted')
            axis[0].set_title('X-axis - ' + str(y) + '00ms Delay')
            axis[0].set_ylabel('Coordinate (cm)')
            axis[0].set_xlabel('Time (s)')
            axis[0].legend(loc='best')

            a = []
            b = []
            for x in range(len(LSTM_output_test)):
                b.append(round(model_output[x][1]))
                a.append(LSTM_output_test[x][1])

            axis[1].plot(time_stamp, a, label='Actual')
            axis[1].plot(time_stamp, b, label='Predicted')
            axis[1].set_title('Y-axis - ' + str(y) + '00ms Delay')
            axis[1].set_ylabel('Coordinate (cm)')
            axis[1].set_xlabel('Time (s)')
            axis[1].legend(loc='upper right')

            figure.tight_layout()

            for x in range(len(model_output)):
                model_output[x][0] = round(model_output[x][0])
                model_output[x][1] = round(model_output[x][1])

            prev_actual = [LSTM_output_test[0][0], LSTM_output_test[0][1]]
            prev_predict = [model_output[0][0], model_output[0][1]]
            num1 = 0
            num2 = 0

            for x in range(len(LSTM_output_test)):
                if prev_actual[0] != LSTM_output_test[x][0] or prev_actual[1] != LSTM_output_test[x][1]:
                    num1 = num1 + 1
                if prev_predict[0] != model_output[x][0] or prev_predict[1] != model_output[x][1]:
                    num2 = num2 + 1
                prev_actual = [LSTM_output_test[x][0], LSTM_output_test[x][1]]
                prev_predict = [model_output[x][0], model_output[x][1]]
            print(str(y) + '00ms')
            print("Ori Freq: " + str(num1 / time_stamp[len(time_stamp) - 1]))
            print("AI Freq: " + str(num2 / time_stamp[len(time_stamp) - 1]) + '\n')
            num3[0] = num3[0] + num1 / time_stamp[len(time_stamp) - 1]
            num3[1] = num3[1] + 1

        print('Average Ori Freq: ' + str(num3[0] / num3[1]))
        plt.show()



    else:
        pass


#
# import time
#
# # import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# import silence_tensorflow.auto
# from interface import *
# from prediction import *
# from run_from_data import *
# # import multiprocessing
# from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
# import seaborn as sn
#
#
# if __name__ == '__main__':
#     print(time.time())
#     # MODE = "Get_Training_Data"
#     # MODE = "Train_Data"
#     # MODE = "Run_Model"
#     MODE =  "Plot_Model"
#     # MODE = "Normal"
#     # MODE = "Get_Robot_effectiveness"
#     # MODE = 'Plot_effectiveness'
#     # MODE = "Plot_All"
#     # MODE = "test"
#     the_data = []
#     tf.keras.utils.disable_interactive_logging()
#
#
#     epochhhh = 200
#     splittt = 0.8
#
#
#
#
#     for layer_loop in [2, 3, 4, 5, 6, 7, 8]:
#         for mode_loop in ['Train_Data', 'Run_Model', 'Plot_Model']:
#             # for prediction_loop in [3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]:
#
#             MODE = mode_loop
#             prediction = 5
#             hidden_layer = layer_loop
#             max_hidden_layer = 4
#             for x in range(hidden_layer):
#                 max_hidden_layer = max_hidden_layer * 2
#             max_hidden_layer = int(max_hidden_layer)
#             window = 3
#             width = 10
#             robot_vel = 50 #cm/s
#
#             empty_window = 1
#             idle_rate = 0.3 #how much percentage of idle data
#             idle_bias = 1 #how much should (prediction + window) multiply with
#             take_val = 10
#             take_val_acc = 10
#             split = 0.8
#             epoch = epochhhh
#             params_to_process = 4
#
#             file_name = 'Data 5 - ' + str(prediction) + '00ms - empty window 3 - LSTM ' + str(max_hidden_layer) + '~8'
#             delay = prediction / 10
#
#             if MODE == "Get_Training_Data":
#
#                 s = SerialClass("COM3", 10, width, window, robot_vel, delay)
#                 visibility_arm_delay(s, 0)
#                 calibrate(s, 5)
#                 collect_data(s, 'raw_data_5_box', 600)
#                 beep()
#
#             elif MODE == "Train_Data":
#                 # 59966 / 10 samples
#                 #
#
#                 # filt = []
#                 # smoothx = []
#                 # smoothy = []
#                 # order = 30
#                 # design_filter(filt, order, 0.1)
#                 output_param = [0] * (params_to_process*2)
#                 box_length = [26, 29]
#                 LSTM_data = []
#                 LSTM_output = []
#                 LSTM_data_train = []
#                 LSTM_data_test = []
#                 LSTM_output_train = []
#                 LSTM_output_test = []
#                 raw_data = []
#                 raw_output = []
#                 read_csv(raw_data, 'raw_data_5_box')
#
#
#                 test_data = []
#                 for x in raw_data:
#                     test_data.append(x[:])
#                 i = 0
#                 new_l = int(len(test_data) / width)
#                 while len(test_data) > new_l + width:
#                     for x in range(width - 1):
#                         test_data.pop(i + 1)
#                     i = i + 1
#                 while len(test_data) > new_l:
#                     test_data.pop(new_l)
#                 raw_data = []
#                 for x in test_data:
#                     raw_data.append(x)
#
#                 for x in range(len(raw_data)):
#                     raw_data[x].pop(2)
#
#                 for x in range(len(raw_data)):
#                     if 25 <= raw_data[x][1] < 40:
#                         if -20 <= raw_data[x][0] < -5:
#                             raw_output.append([1, 1])
#                         elif -5 <= raw_data[x][0] < 10:
#                             raw_output.append([2, 1])
#                         elif 10 <= raw_data[x][0] < 25:
#                             raw_output.append([3, 1])
#                         else:
#                             raw_output.append([0, 0])
#                     elif 40 <= raw_data[x][1] < 55:
#                         if -20 <= raw_data[x][0] < -5:
#                             raw_output.append([1, 2])
#                         elif -5 <= raw_data[x][0] < 10:
#                             raw_output.append([2, 2])
#                         elif 10 <= raw_data[x][0] < 25:
#                             raw_output.append([3, 2])
#                         else:
#                             raw_output.append([0, 0])
#                     else:
#                         raw_output.append([0, 0])
#
#
#                 split_input(raw_data, LSTM_data, window, prediction)
#                 split_output(raw_output, LSTM_output, window, prediction)
#
#                 if empty_window:
#                     for x in range(len(LSTM_data)):
#                         while len(LSTM_data[x]) > 2:
#                             LSTM_data[x].pop(1)
#
#                 split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_test, LSTM_output_test,
#                                split)
#
#                 x_prev = 0
#                 y_prev = 0
#                 include = 0
#                 output_idle = []
#                 output_changing = []
#                 input_idle = []
#                 input_changing = []
#                 for x in range(len(LSTM_output_train)):
#                     if x_prev != LSTM_output_train[x][0] or y_prev != LSTM_output_train[x][1]:
#                         input_changing.append(LSTM_data_train[x])
#                         output_changing.append(LSTM_output_train[x])
#                         include = round((prediction + window) * idle_bias)
#                     elif include > 0:
#                         input_changing.append(LSTM_data_train[x])
#                         output_changing.append(LSTM_output_train[x])
#                         include = include - 1
#                     else:
#                         input_idle.append(LSTM_data_train[x])
#                         output_idle.append(LSTM_output_train[x])
#                     x_prev = LSTM_output_train[x][0]
#                     y_prev = LSTM_output_train[x][1]
#
#                 num = [0, 0, 0, 0, 0, 0]
#                 balanced_data = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]
#                 for x in range(len(output_changing)):
#                     if output_changing[x][0] == 1 and output_changing[x][1] == 1:
#                         num[0] = num[0] + 1
#                         balanced_data[0][0].append(input_changing[x])
#                         balanced_data[0][1].append(output_changing[x])
#                     elif output_changing[x][0] == 2 and output_changing[x][1] == 1:
#                         num[1] = num[1] + 1
#                         balanced_data[1][0].append(input_changing[x])
#                         balanced_data[1][1].append(output_changing[x])
#                     elif output_changing[x][0] == 3 and output_changing[x][1] == 1:
#                         num[2] = num[2] + 1
#                         balanced_data[2][0].append(input_changing[x])
#                         balanced_data[2][1].append(output_changing[x])
#                     elif output_changing[x][0] == 1 and output_changing[x][1] == 2:
#                         num[3] = num[3] + 1
#                         balanced_data[3][0].append(input_changing[x])
#                         balanced_data[3][1].append(output_changing[x])
#                     elif output_changing[x][0] == 2 and output_changing[x][1] == 2:
#                         num[4] = num[4] + 1
#                         balanced_data[4][0].append(input_changing[x])
#                         balanced_data[4][1].append(output_changing[x])
#                     elif output_changing[x][0] == 3 and output_changing[x][1] == 2:
#                         num[5] = num[5] + 1
#                         balanced_data[5][0].append(input_changing[x])
#                         balanced_data[5][1].append(output_changing[x])
#                     else:
#                         print('Error: Output out of range')
#                         stop()
#
#                 for x in range(len(num)):
#                     if num[x] != min(num):
#                         while len(balanced_data[x][0]) > min(num):
#                             random_delete = random.randint(0, len(balanced_data[x][0])-1)
#                             balanced_data[x][0].pop(random_delete)
#                             balanced_data[x][1].pop(random_delete)
#
#                 LSTM_data_train = []
#                 LSTM_output_train = []
#                 for x in range(len(balanced_data)):
#                     for y in range(len(balanced_data[0][0])):
#                         LSTM_data_train.append(balanced_data[x][0][y])
#                         LSTM_output_train.append(balanced_data[x][1][y])
#
#                 num = [0, 0, 0, 0, 0, 0]
#                 balanced_data = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]
#                 for x in range(len(output_idle)):
#                     if output_idle[x][0] == 1 and output_idle[x][1] == 1:
#                         num[0] = num[0] + 1
#                         balanced_data[0][0].append(input_idle[x])
#                         balanced_data[0][1].append(output_idle[x])
#                     elif output_idle[x][0] == 2 and output_idle[x][1] == 1:
#                         num[1] = num[1] + 1
#                         balanced_data[1][0].append(input_idle[x])
#                         balanced_data[1][1].append(output_idle[x])
#                     elif output_idle[x][0] == 3 and output_idle[x][1] == 1:
#                         num[2] = num[2] + 1
#                         balanced_data[2][0].append(input_idle[x])
#                         balanced_data[2][1].append(output_idle[x])
#                     elif output_idle[x][0] == 1 and output_idle[x][1] == 2:
#                         num[3] = num[3] + 1
#                         balanced_data[3][0].append(input_idle[x])
#                         balanced_data[3][1].append(output_idle[x])
#                     elif output_idle[x][0] == 2 and output_idle[x][1] == 2:
#                         num[4] = num[4] + 1
#                         balanced_data[4][0].append(input_idle[x])
#                         balanced_data[4][1].append(output_idle[x])
#                     elif output_idle[x][0] == 3 and output_idle[x][1] == 2:
#                         num[5] = num[5] + 1
#                         balanced_data[5][0].append(input_idle[x])
#                         balanced_data[5][1].append(output_idle[x])
#                     else:
#                         print('Error: Output out of range')
#                         stop()
#
#                 if idle_rate < 1:
#                     total = round(len(LSTM_data_train) / (1 - idle_rate))
#                     y = 0
#                     while len(LSTM_data_train) < total:
#                         if len(balanced_data[y][0]) > 0:
#                             random_insert = random.randint(0, len(balanced_data[y][0])-1)
#                             LSTM_data_train.append(balanced_data[y][0][random_insert])
#                             LSTM_output_train.append(balanced_data[y][1][random_insert])
#                             balanced_data[y][0].pop(random_insert)
#                             balanced_data[y][1].pop(random_insert)
#                         else:
#                             a = 0
#                             for x in range(len(balanced_data)):
#                                 if len(balanced_data[x][0]) <= 0:
#                                     a = a + 1
#                             if a == len(balanced_data):
#                                 break
#                         y = y + 1
#                         if y == 6:
#                             y = 0
#                 else:
#                     for x in range(len(balanced_data)):
#                         for y in range(len(balanced_data[x][0])):
#                             LSTM_data_train.append(balanced_data[x][0][y])
#                             LSTM_output_train.append(balanced_data[x][1][y])
#
#                 get_param(LSTM_data_train, LSTM_output_train, output_param)
#                 buffer = [output_param, [0, 0, 0]]
#                 write_csv(buffer, 'Param - ' + file_name)
#
#                 preprocess_input(LSTM_data_train, output_param)
#                 preprocess_output(LSTM_output_train, output_param)
#                 preprocess_input(LSTM_data_test, output_param)
#                 preprocess_output(LSTM_output_test, output_param)
#
#
#
#                 shuffle(LSTM_data_train, LSTM_output_train)
#
#
#                 train(LSTM_data_train, LSTM_output_train, LSTM_data_test, LSTM_output_test, epoch, 'Model - ' + file_name, 0, max_hidden_layer)
#
#
#             elif MODE == "Run_Model":
#                 output_param = [0] * (params_to_process * 2)
#                 box_length = [26, 29]
#                 LSTM_data = []
#                 LSTM_output = []
#                 LSTM_data_train = []
#                 LSTM_data_test = []
#                 LSTM_output_train = []
#                 LSTM_output_test = []
#                 raw_data = []
#                 raw_output = []
#                 read_csv(raw_data, 'raw_data_5_box')
#
#                 test_data = []
#                 for x in raw_data:
#                     test_data.append(x[:])
#                 i = 0
#                 new_l = int(len(test_data) / width)
#                 while len(test_data) > new_l + width:
#                     for x in range(width - 1):
#                         test_data.pop(i + 1)
#                     i = i + 1
#                 while len(test_data) > new_l:
#                     test_data.pop(new_l)
#                 raw_data = []
#                 for x in test_data:
#                     raw_data.append(x)
#
#                 for x in range(len(raw_data)):
#                     raw_data[x].pop(2)
#
#                 for x in range(len(raw_data)):
#                     if 25 <= raw_data[x][1] < 40:
#                         if -20 <= raw_data[x][0] < -5:
#                             raw_output.append([1, 1])
#                         elif -5 <= raw_data[x][0] < 10:
#                             raw_output.append([2, 1])
#                         elif 10 <= raw_data[x][0] < 25:
#                             raw_output.append([3, 1])
#                         else:
#                             raw_output.append([0, 0])
#                     elif 40 <= raw_data[x][1] < 55:
#                         if -20 <= raw_data[x][0] < -5:
#                             raw_output.append([1, 2])
#                         elif -5 <= raw_data[x][0] < 10:
#                             raw_output.append([2, 2])
#                         elif 10 <= raw_data[x][0] < 25:
#                             raw_output.append([3, 2])
#                         else:
#                             raw_output.append([0, 0])
#                     else:
#                         raw_output.append([0, 0])
#
#                 split_input(raw_data, LSTM_data, window, prediction)
#                 split_output(raw_output, LSTM_output, window, prediction)
#
#                 if empty_window:
#                     for x in range(len(LSTM_data)):
#                         while len(LSTM_data[x]) > 2:
#                             LSTM_data[x].pop(1)
#
#
#                 split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_test, LSTM_output_test,
#                                split)
#
#                 output_param = []
#                 buffer_param = []
#                 read_csv(buffer_param, 'Param - ' + file_name)
#                 for x in range(len(buffer_param[0])):
#                     output_param.append(buffer_param[0][x])
#
#                 preprocess_input(LSTM_data_test, output_param)
#
#                 model_output = []
#                 tt = time.time()
#                 model = load_model('Model - ' + file_name)
#                 use_model(LSTM_data_test, model_output, model)
#                 the_data.append([(time.time() - tt) / len(model_output)])
#                 postprocess_output(model_output, output_param)
#                 write_csv(LSTM_output_test, 'True_output - ' + file_name)
#                 write_csv(model_output, 'Predicted - ' + file_name)
#
#             elif MODE == "Plot_Model":
#
#                 model_output = []
#                 LSTM_output_test = []
#                 read_csv(model_output, 'Predicted - ' + file_name)
#                 read_csv(LSTM_output_test, 'True_output - ' + file_name)
#
#                 test_data= []
#                 predicted_data = []
#
#                 for x in range(len(model_output)):
#                     if 0 < LSTM_output_test[x][0]:
#                         test_data.append(int(LSTM_output_test[x][0]))
#                         if LSTM_output_test[x][1] == 1:
#                             test_data[len(test_data)-1] = test_data[len(test_data)-1] + 3
#                     else:
#                         test_data.append(0)
#
#                     if 0 < round(model_output[x][0]) < 4 and 0 < round(model_output[x][1]) < 3:
#                         predicted_data.append(round(model_output[x][0]))
#                         if round(model_output[x][1]) == 1:
#                             predicted_data[len(predicted_data) - 1] = predicted_data[len(predicted_data) - 1] + 3
#                     else:
#                         predicted_data.append(0)
#
#                 acc = accuracy_score(test_data, predicted_data)
#
#
#                 predicted_data_roc = []
#
#                 for x in range(len(model_output)):
#
#                     predicted_data_roc.append([])
#                     if 0 < round(model_output[x][0]) < 4 and 0 < round(model_output[x][1]) < 3:
#                         if round(model_output[x][1]) == 1:
#                             for y in range(3):
#                                 predicted_data_roc[len(predicted_data_roc) - 1].append(0)
#                         for y in range(round(model_output[x][0]) - 1):
#                             predicted_data_roc[len(predicted_data_roc) - 1].append(0)
#                         predicted_data_roc[len(predicted_data_roc) - 1].append(1)
#                         while len(predicted_data_roc[len(predicted_data_roc) - 1]) < 6:
#                             predicted_data_roc[len(predicted_data_roc) - 1].append(0)
#                     else:
#                         for y in range(6):
#                             predicted_data_roc[len(predicted_data_roc) - 1].append(1/6)
#
#                 roc = roc_auc_score(test_data, predicted_data_roc, multi_class='ovo')
#
#                 precision = precision_score(test_data, predicted_data, average='macro')
#                 recall = recall_score(test_data, predicted_data, average='macro')
#                 f1 = f1_score(test_data, predicted_data, average='macro')
#
#                 the_data[len(the_data)-1].append(acc)
#                 the_data[len(the_data) - 1].append(roc)
#                 the_data[len(the_data) - 1].append(f1)
#                 the_data[len(the_data) - 1].append(precision)
#                 the_data[len(the_data) - 1].append(recall)
#                 the_data[len(the_data) - 1].append(the_data[len(the_data) - 1][0] * 1000)
#                 the_data[len(the_data) - 1].pop(0)
#
#
#     max_acc = -100
#     max_auc = -100
#     max_index = 0
#     for x in range(len(the_data)):
#         if the_data[x][0] > max_acc:
#             max_acc = the_data[x][0]
#             max_auc = the_data[x][1]
#             max_index = x
#         elif the_data[x][0] == max_acc:
#             if the_data[x][1] > max_auc:
#                 max_acc = the_data[x][0]
#                 max_auc = the_data[x][1]
#                 max_index = x
#
#     layer_loop = [2, 3, 4, 5, 6, 7, 8]
#     best_layer = layer_loop[max_index]
#     for x in range(len(the_data)):
#         print(the_data[x])
#     print(best_layer)
#     the_data_3 = []
#     for delay_loop in [3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]:
#         for mode_loop in ['Train_Data', 'Run_Model', 'Plot_Model']:
#
#             MODE = mode_loop
#             prediction = delay_loop
#             hidden_layer = best_layer
#             max_hidden_layer = 4
#             for x in range(hidden_layer):
#                 max_hidden_layer = max_hidden_layer * 2
#             max_hidden_layer = int(max_hidden_layer)
#             window = 3
#             width = 10
#             robot_vel = 50 #cm/s
#
#             empty_window = 1
#             idle_rate = 0.3 #how much percentage of idle data
#             idle_bias = 1 #how much should (prediction + window) multiply with
#             take_val = 10
#             take_val_acc = 10
#             split = 0.8
#             epoch = epochhhh
#             params_to_process = 4
#
#             file_name = 'Data 5 - ' + str(prediction) + '00ms - empty window 3 - LSTM ' + str(max_hidden_layer) + '~8'
#             delay = prediction / 10
#
#             if MODE == "Train_Data":
#
#                 output_param = [0] * (params_to_process*2)
#                 box_length = [26, 29]
#                 LSTM_data = []
#                 LSTM_output = []
#                 LSTM_data_train = []
#                 LSTM_data_test = []
#                 LSTM_output_train = []
#                 LSTM_output_test = []
#                 raw_data = []
#                 raw_output = []
#                 read_csv(raw_data, 'raw_data_5_box')
#
#                 test_data = []
#                 for x in raw_data:
#                     test_data.append(x[:])
#                 i = 0
#                 new_l = int(len(test_data) / width)
#                 while len(test_data) > new_l + width:
#                     for x in range(width - 1):
#                         test_data.pop(i + 1)
#                     i = i + 1
#                 while len(test_data) > new_l:
#                     test_data.pop(new_l)
#                 raw_data = []
#                 for x in test_data:
#                     raw_data.append(x)
#
#                 for x in range(len(raw_data)):
#                     raw_data[x].pop(2)
#
#                 for x in range(len(raw_data)):
#                     if 25 <= raw_data[x][1] < 40:
#                         if -20 <= raw_data[x][0] < -5:
#                             raw_output.append([1, 1])
#                         elif -5 <= raw_data[x][0] < 10:
#                             raw_output.append([2, 1])
#                         elif 10 <= raw_data[x][0] < 25:
#                             raw_output.append([3, 1])
#                         else:
#                             raw_output.append([0, 0])
#                     elif 40 <= raw_data[x][1] < 55:
#                         if -20 <= raw_data[x][0] < -5:
#                             raw_output.append([1, 2])
#                         elif -5 <= raw_data[x][0] < 10:
#                             raw_output.append([2, 2])
#                         elif 10 <= raw_data[x][0] < 25:
#                             raw_output.append([3, 2])
#                         else:
#                             raw_output.append([0, 0])
#                     else:
#                         raw_output.append([0, 0])
#
#                 split_input(raw_data, LSTM_data, window, prediction)
#                 split_output(raw_output, LSTM_output, window, prediction)
#
#                 if empty_window:
#                     for x in range(len(LSTM_data)):
#                         while len(LSTM_data[x]) > 2:
#                             LSTM_data[x].pop(1)
#
#                 split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_test, LSTM_output_test,
#                                split)
#
#                 x_prev = 0
#                 y_prev = 0
#                 include = 0
#                 output_idle = []
#                 output_changing = []
#                 input_idle = []
#                 input_changing = []
#                 for x in range(len(LSTM_output_train)):
#                     if x_prev != LSTM_output_train[x][0] or y_prev != LSTM_output_train[x][1]:
#                         input_changing.append(LSTM_data_train[x])
#                         output_changing.append(LSTM_output_train[x])
#                         include = round((prediction + window) * idle_bias)
#                     elif include > 0:
#                         input_changing.append(LSTM_data_train[x])
#                         output_changing.append(LSTM_output_train[x])
#                         include = include - 1
#                     else:
#                         input_idle.append(LSTM_data_train[x])
#                         output_idle.append(LSTM_output_train[x])
#                     x_prev = LSTM_output_train[x][0]
#                     y_prev = LSTM_output_train[x][1]
#
#                 num = [0, 0, 0, 0, 0, 0]
#                 balanced_data = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]
#                 for x in range(len(output_changing)):
#                     if output_changing[x][0] == 1 and output_changing[x][1] == 1:
#                         num[0] = num[0] + 1
#                         balanced_data[0][0].append(input_changing[x])
#                         balanced_data[0][1].append(output_changing[x])
#                     elif output_changing[x][0] == 2 and output_changing[x][1] == 1:
#                         num[1] = num[1] + 1
#                         balanced_data[1][0].append(input_changing[x])
#                         balanced_data[1][1].append(output_changing[x])
#                     elif output_changing[x][0] == 3 and output_changing[x][1] == 1:
#                         num[2] = num[2] + 1
#                         balanced_data[2][0].append(input_changing[x])
#                         balanced_data[2][1].append(output_changing[x])
#                     elif output_changing[x][0] == 1 and output_changing[x][1] == 2:
#                         num[3] = num[3] + 1
#                         balanced_data[3][0].append(input_changing[x])
#                         balanced_data[3][1].append(output_changing[x])
#                     elif output_changing[x][0] == 2 and output_changing[x][1] == 2:
#                         num[4] = num[4] + 1
#                         balanced_data[4][0].append(input_changing[x])
#                         balanced_data[4][1].append(output_changing[x])
#                     elif output_changing[x][0] == 3 and output_changing[x][1] == 2:
#                         num[5] = num[5] + 1
#                         balanced_data[5][0].append(input_changing[x])
#                         balanced_data[5][1].append(output_changing[x])
#                     else:
#                         print('Error: Output out of range')
#                         stop()
#
#                 for x in range(len(num)):
#                     if num[x] != min(num):
#                         while len(balanced_data[x][0]) > min(num):
#                             random_delete = random.randint(0, len(balanced_data[x][0])-1)
#                             balanced_data[x][0].pop(random_delete)
#                             balanced_data[x][1].pop(random_delete)
#
#                 LSTM_data_train = []
#                 LSTM_output_train = []
#                 for x in range(len(balanced_data)):
#                     for y in range(len(balanced_data[0][0])):
#                         LSTM_data_train.append(balanced_data[x][0][y])
#                         LSTM_output_train.append(balanced_data[x][1][y])
#
#                 num = [0, 0, 0, 0, 0, 0]
#                 balanced_data = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]
#                 for x in range(len(output_idle)):
#                     if output_idle[x][0] == 1 and output_idle[x][1] == 1:
#                         num[0] = num[0] + 1
#                         balanced_data[0][0].append(input_idle[x])
#                         balanced_data[0][1].append(output_idle[x])
#                     elif output_idle[x][0] == 2 and output_idle[x][1] == 1:
#                         num[1] = num[1] + 1
#                         balanced_data[1][0].append(input_idle[x])
#                         balanced_data[1][1].append(output_idle[x])
#                     elif output_idle[x][0] == 3 and output_idle[x][1] == 1:
#                         num[2] = num[2] + 1
#                         balanced_data[2][0].append(input_idle[x])
#                         balanced_data[2][1].append(output_idle[x])
#                     elif output_idle[x][0] == 1 and output_idle[x][1] == 2:
#                         num[3] = num[3] + 1
#                         balanced_data[3][0].append(input_idle[x])
#                         balanced_data[3][1].append(output_idle[x])
#                     elif output_idle[x][0] == 2 and output_idle[x][1] == 2:
#                         num[4] = num[4] + 1
#                         balanced_data[4][0].append(input_idle[x])
#                         balanced_data[4][1].append(output_idle[x])
#                     elif output_idle[x][0] == 3 and output_idle[x][1] == 2:
#                         num[5] = num[5] + 1
#                         balanced_data[5][0].append(input_idle[x])
#                         balanced_data[5][1].append(output_idle[x])
#                     else:
#                         print('Error: Output out of range')
#                         stop()
#
#                 if idle_rate < 1:
#                     total = round(len(LSTM_data_train) / (1 - idle_rate))
#                     y = 0
#                     while len(LSTM_data_train) < total:
#                         if len(balanced_data[y][0]) > 0:
#                             random_insert = random.randint(0, len(balanced_data[y][0])-1)
#                             LSTM_data_train.append(balanced_data[y][0][random_insert])
#                             LSTM_output_train.append(balanced_data[y][1][random_insert])
#                             balanced_data[y][0].pop(random_insert)
#                             balanced_data[y][1].pop(random_insert)
#                         else:
#                             a = 0
#                             for x in range(len(balanced_data)):
#                                 if len(balanced_data[x][0]) <= 0:
#                                     a = a + 1
#                             if a == len(balanced_data):
#                                 break
#                         y = y + 1
#                         if y == 6:
#                             y = 0
#                 else:
#                     for x in range(len(balanced_data)):
#                         for y in range(len(balanced_data[x][0])):
#                             LSTM_data_train.append(balanced_data[x][0][y])
#                             LSTM_output_train.append(balanced_data[x][1][y])
#
#                 get_param(LSTM_data_train, LSTM_output_train, output_param)
#                 buffer = [output_param, [0, 0, 0]]
#                 write_csv(buffer, 'Param - ' + file_name)
#
#                 preprocess_input(LSTM_data_train, output_param)
#                 preprocess_output(LSTM_output_train, output_param)
#                 preprocess_input(LSTM_data_test, output_param)
#                 preprocess_output(LSTM_output_test, output_param)
#
#
#
#                 shuffle(LSTM_data_train, LSTM_output_train)
#
#
#                 train(LSTM_data_train, LSTM_output_train, LSTM_data_test, LSTM_output_test, epoch, 'Model - ' + file_name, 0, max_hidden_layer)
#
#
#             elif MODE == "Run_Model":
#                 output_param = [0] * (params_to_process * 2)
#                 box_length = [26, 29]
#                 LSTM_data = []
#                 LSTM_output = []
#                 LSTM_data_train = []
#                 LSTM_data_test = []
#                 LSTM_output_train = []
#                 LSTM_output_test = []
#                 raw_data = []
#                 raw_output = []
#                 read_csv(raw_data, 'raw_data_5_box')
#
#                 test_data = []
#                 for x in raw_data:
#                     test_data.append(x[:])
#                 i = 0
#                 new_l = int(len(test_data) / width)
#                 while len(test_data) > new_l + width:
#                     for x in range(width - 1):
#                         test_data.pop(i + 1)
#                     i = i + 1
#                 while len(test_data) > new_l:
#                     test_data.pop(new_l)
#                 raw_data = []
#                 for x in test_data:
#                     raw_data.append(x)
#
#                 for x in range(len(raw_data)):
#                     raw_data[x].pop(2)
#
#                 for x in range(len(raw_data)):
#                     if 25 <= raw_data[x][1] < 40:
#                         if -20 <= raw_data[x][0] < -5:
#                             raw_output.append([1, 1])
#                         elif -5 <= raw_data[x][0] < 10:
#                             raw_output.append([2, 1])
#                         elif 10 <= raw_data[x][0] < 25:
#                             raw_output.append([3, 1])
#                         else:
#                             raw_output.append([0, 0])
#                     elif 40 <= raw_data[x][1] < 55:
#                         if -20 <= raw_data[x][0] < -5:
#                             raw_output.append([1, 2])
#                         elif -5 <= raw_data[x][0] < 10:
#                             raw_output.append([2, 2])
#                         elif 10 <= raw_data[x][0] < 25:
#                             raw_output.append([3, 2])
#                         else:
#                             raw_output.append([0, 0])
#                     else:
#                         raw_output.append([0, 0])
#
#                 split_input(raw_data, LSTM_data, window, prediction)
#                 split_output(raw_output, LSTM_output, window, prediction)
#
#                 if empty_window:
#                     for x in range(len(LSTM_data)):
#                         while len(LSTM_data[x]) > 2:
#                             LSTM_data[x].pop(1)
#
#                 split_training(LSTM_data, LSTM_output, LSTM_data_train, LSTM_output_train, LSTM_data_test, LSTM_output_test,
#                                split)
#
#                 output_param = []
#                 buffer_param = []
#                 read_csv(buffer_param, 'Param - ' + file_name)
#                 for x in range(len(buffer_param[0])):
#                     output_param.append(buffer_param[0][x])
#
#                 preprocess_input(LSTM_data_test, output_param)
#
#                 model_output = []
#                 tt = time.time()
#                 model = load_model('Model - ' + file_name)
#                 use_model(LSTM_data_test, model_output, model)
#                 the_data_3.append([(time.time() - tt) / len(model_output)])
#                 postprocess_output(model_output, output_param)
#                 write_csv(LSTM_output_test, 'True_output - ' + file_name)
#                 write_csv(model_output, 'Predicted - ' + file_name)
#
#             elif MODE == 'Plot_Model':
#
#                 model_output = []
#                 LSTM_output_test = []
#                 read_csv(model_output, 'Predicted - ' + file_name)
#                 read_csv(LSTM_output_test, 'True_output - ' + file_name)
#
#
#                 test_data= []
#                 predicted_data = []
#
#                 for x in range(len(model_output)):
#                     if 0 < LSTM_output_test[x][0]:
#                         test_data.append(int(LSTM_output_test[x][0]))
#                         if LSTM_output_test[x][1] == 1:
#                             test_data[len(test_data)-1] = test_data[len(test_data)-1] + 3
#                     else:
#                         test_data.append(0)
#
#                     if 0 < round(model_output[x][0]) < 4 and 0 < round(model_output[x][1]) < 3:
#                         predicted_data.append(round(model_output[x][0]))
#                         if round(model_output[x][1]) == 1:
#                             predicted_data[len(predicted_data) - 1] = predicted_data[len(predicted_data) - 1] + 3
#                     else:
#                         predicted_data.append(0)
#
#                 acc = accuracy_score(test_data, predicted_data)
#
#                 predicted_data_roc = []
#
#                 for x in range(len(model_output)):
#
#                     predicted_data_roc.append([])
#                     if 0 < round(model_output[x][0]) < 4 and 0 < round(model_output[x][1]) < 3:
#                         if round(model_output[x][1]) == 1:
#                             for y in range(3):
#                                 predicted_data_roc[len(predicted_data_roc) - 1].append(0)
#                         for y in range(round(model_output[x][0]) - 1):
#                             predicted_data_roc[len(predicted_data_roc) - 1].append(0)
#                         predicted_data_roc[len(predicted_data_roc) - 1].append(1)
#                         while len(predicted_data_roc[len(predicted_data_roc) - 1]) < 6:
#                             predicted_data_roc[len(predicted_data_roc) - 1].append(0)
#                     else:
#                         for y in range(6):
#                             predicted_data_roc[len(predicted_data_roc) - 1].append(1/6)
#
#                 roc = roc_auc_score(test_data, predicted_data_roc, multi_class='ovo')
#
#                 precision = precision_score(test_data, predicted_data, average='macro')
#                 recall = recall_score(test_data, predicted_data, average='macro')
#                 f1 = f1_score(test_data, predicted_data, average='macro')
#
#                 the_data_3[len(the_data_3) - 1].append(acc)
#                 the_data_3[len(the_data_3) - 1].append(roc)
#                 the_data_3[len(the_data_3) - 1].append(f1)
#                 the_data_3[len(the_data_3) - 1].append(precision)
#                 the_data_3[len(the_data_3) - 1].append(recall)
#                 the_data_3[len(the_data_3) - 1].append(the_data_3[len(the_data_3) - 1][0] * 1000)
#                 the_data_3[len(the_data_3) - 1].pop(0)
#
#
#     the_data_2 = []
#
#     for prediction_loop in [3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]:
#
#         MODE = 'Get_Robot_effectiveness'
#         prediction = prediction_loop
#         hidden_layer = best_layer
#         max_hidden_layer = 4
#         for x in range(hidden_layer):
#             max_hidden_layer = max_hidden_layer * 2
#         max_hidden_layer = int(max_hidden_layer)
#         window = 3
#         width = 10
#         robot_vel = 50  # cm/s
#
#         empty_window = 1
#         idle_rate = 0.3  # how much percentage of idle data
#         idle_bias = 1  # how much should (prediction + window) multiply with
#         take_val = 10
#         take_val_acc = 10
#         split = 0.8
#         epoch = 1
#         params_to_process = 4
#
#         file_name = 'Data 5 - ' + str(prediction) + '00ms - empty window 3 - LSTM ' + str(max_hidden_layer) + '~8'
#         delay = prediction / 10
#
#         if MODE == "Get_Training_Data":
#
#             pass
#
#         elif MODE == "Get_Robot_effectiveness":
#
#             # tf.keras.utils.disable_interactive_logging()
#
#             s = SerialClass_rfd(10, width, window, robot_vel, delay, 'raw_data_5_box', splittt)
#             model = load_model('Model - ' + file_name)
#             buffer_param = []
#             output_param = []
#             read_csv(buffer_param, 'Param - ' + file_name)
#             for x in range(len(buffer_param[0])):
#                 output_param.append(buffer_param[0][x])
#
#             # visibility_robot(s, 0)
#             # visibility_robot_now(s, 0)
#             # visibility_robot_delay(s, 0)
#             visibility_arm(s, 0)
#             visibility_arm_delay(s, 0)
#             # visibility_indicator(s, 0)
#
#             total_len = len(s.raw_data)
#
#             s.data_time = time.time()
#             total_time = time.time()
#             while True:
#                 # text input: delay amount
#                 # radio buttons: camera pos
#                 if len(s.raw_data) == 0:
#                     break
#                 update_serial_rfd(s, total_len)
#                 # a = time.time()
#                 predict(s, model, 0.1, output_param)
#
#             total_time = time.time() - total_time
#
#             total_len = len(s.rmse)
#             num1 = 0
#             num2 = 0
#             num3 = 0
#             num4 = 0
#             for x in range(total_len):
#                 num1 = num1 + s.rmse[x][0]
#                 num2 = num2 + s.rmse[x][1]
#                 num3 = num3 + s.rmse[x][2]
#                 num4 = num4 + s.rmse[x][3]
#
#             num1 = sqrt(num1 / total_len)
#             num2 = sqrt(num2 / total_len)
#             num3 = num3 / total_len
#             num4 = num4 / total_len
#
#             # print("Total Time: " + str(total_time))
#             # print("RMSE with AI: " + str(num1))
#             # print("RMSE without AI: " + str(num2))
#             # print("Accuracy with AI: " + str(num3))
#             # print("Accuracy without AI: " + str(num4))
#             the_data_2.append([num1, num2, num3, num4])
#
#             s.scene.delete()
#
#     best_num = 8
#     for x in range(best_layer - 1):
#         best_num = best_num * 2
#     base1 = 'Data 5 - '
#     base2 = '00ms - empty window 3 - LSTM '
#     last_index = 0
#     num3 = [0, 0]
#     for y in [3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]:
#         model_output = []
#         LSTM_output_test = []
#         read_csv(model_output, 'Predicted - ' + base1 + str(y) + base2 + str(best_num) + '~8')
#         read_csv(LSTM_output_test, 'True_output - ' + base1 + str(y) + base2 + str(best_num) + '~8')
#         time_stamp = []
#         num = 0
#         for z in range(len(LSTM_output_test)):
#             time_stamp.append(num)
#             num = num + 0.1
#
#         for x in range(len(model_output)):
#             model_output[x][0] = round(model_output[x][0])
#             model_output[x][1] = round(model_output[x][1])
#
#         prev_actual = [LSTM_output_test[0][0], LSTM_output_test[0][1]]
#         prev_predict = [model_output[0][0], model_output[0][1]]
#         num1 = 0
#         num2 = 0
#
#         for x in range(len(LSTM_output_test)):
#             if prev_actual[0] != LSTM_output_test[x][0] or prev_actual[1] != LSTM_output_test[x][1]:
#                 num1 = num1 + 1
#             if prev_predict[0] != model_output[x][0] or prev_predict[1] != model_output[x][1]:
#                 num2 = num2 + 1
#             prev_actual = [LSTM_output_test[x][0], LSTM_output_test[x][1]]
#             prev_predict = [model_output[x][0], model_output[x][1]]
#         the_data_2[last_index].append(num2 / time_stamp[len(time_stamp) - 1])
#         last_index = last_index + 1
#         num3[0] = num3[0] + num1 / time_stamp[len(time_stamp) - 1]
#         num3[1] = num3[1] + 1
#
#     last_num = num3[0] / num3[1]
#
#     for x in range(len(the_data)):
#         print(the_data[x])
#     for x in range(len(the_data_3)):
#         print(the_data_3[x])
#     for y in range(len(the_data_2[0])):
#         total_str = ''
#         for x in range(len(the_data_2)):
#             total_str = total_str + str(the_data_2[x][y]) + ', '
#         print('[', total_str, ']')
#     print(last_num)
#     print(time.time())

#LSTM with different delay
# [0.939115929941618, 0.9644653843157595, 0.9403507636563676, 0.9403981139763293, 0.9407756405262656, 5.751595286352621]
# [0.9366138448707256, 0.9640812206264648, 0.9375442742238175, 0.9352094847868008, 0.9401353677107744, 5.537941219212116]
# [0.8689482470784641, 0.9258228873816275, 0.8712785194077579, 0.8681327582360256, 0.8763714789693791, 5.199376251144282]
# [0.7938230383973289, 0.8806900756779292, 0.7965474866233229, 0.7965338567299707, 0.8011501261298822, 6.719285936307828]
# [0.7050960735171261, 0.8203670822212351, 0.7112214486940388, 0.7349464690413251, 0.7006118037020584, 5.691395666366233]
# [0.6574770258980785, 0.7926100134173321, 0.6691651899547709, 0.7070824482979119, 0.6543500223622201, 5.615730532627854]
# [0.4105351170568562, 0.6368710850607932, 0.39231778596157124, 0.5848112682304897, 0.39478514176798857, 6.352448742525235]
# [0.33612040133779264, 0.584288064113863, 0.30870800369153256, 0.47643615702906317, 0.30714677352310493, 5.7081123658247215]
# [0.19665271966527198, 0.4972159090909091, 0.09135173462794276, 0.06553391259530651, 0.16202651515151514, 5.567237024027932]
# [0.2125523012552301, 0.5000833836149101, 0.10846476118174635, 0.10921354438303592, 0.1668056393581836, 5.677158363693429]
# [0.2185929648241206, 0.5138541666666666, 0.10409272037179014, 0.07681309861930095, 0.18975694444444444, 6.083438344536914]
# [0.2185929648241206, 0.5135606060606062, 0.10411212387956574, 0.07690729180182908, 0.18926767676767678, 6.603574832679838]
