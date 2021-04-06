import pickle

import pandas as pd
import numpy as np
import os
import time
import datetime

import torch
from torch.utils.data import DataLoader

import config
from sklearn import preprocessing

from LSTM.dataset import RawDataset
from LSTM.lstm import LSTMPredictor


def str_to_time(time_str, is_sensor, plant):
    time_format = "%Y-%m-%d %H:%M:%S" if (is_sensor or plant==2) else "%d-%m-%Y %H:%M"

    return time.mktime(datetime.datetime.strptime(time_str, time_format).timetuple())


def time_to_str(target_time, is_sensor, plant):
    target_time = datetime.datetime.fromtimestamp(target_time)
    time_format = "%Y-%m-%d %H:%M:%S" if (is_sensor or plant==2) else "%d-%m-%Y %H:%M"
    return target_time.strftime(time_format)



def get_time_mapping(time_str):
    time_format = "%Y-%m-%d %H:%M:%S"
    time_tuple = datetime.datetime.strptime(time_str, time_format).timetuple()
    return (time_tuple.tm_hour * 4 + time_tuple.tm_min // 15)


def format_file(plant):
    gen_file_name = "Plant_" + str(plant) + "_Generation_Data.csv"
    sensor_file_name = "Plant_" + str(plant) + "_Weather_Sensor_Data.csv"
    gen_file_path = os.path.join("Data", "Raw", gen_file_name)
    sensor_file_path = os.path.join("Data", "Raw", sensor_file_name)

    gen_df = pd.read_csv(gen_file_path)
    sensor_df = pd.read_csv(sensor_file_path)

    # deal with time
    sensor_df['time'] = sensor_df['DATE_TIME'].apply(lambda x: str_to_time(x, True,plant))
    gen_df['time'] = gen_df['DATE_TIME'].apply(lambda x: str_to_time(x, False,plant))


    # match generation to sensor
    work_df = sensor_df[['time']]
    inverter_ids = gen_df.SOURCE_KEY.unique()

    for i, id in enumerate(inverter_ids):
        inverter_df = gen_df[gen_df['SOURCE_KEY'] == id]
        power_df = inverter_df[['time', 'DC_POWER', 'AC_POWER']]
        power_df.columns = ['time', 'DC_' + str(i), 'AC_' + str(i)]
        work_df = pd.merge(work_df, power_df, left_on=['time'], right_on=['time'], how='left')

    power_name_regex = ['AC_*', 'DC_*']
    for curr_regex in power_name_regex:
        ac = work_df.filter(regex=curr_regex, axis=1)
        ac_mean = ac.mean(axis=1)
        ac_filled = ac.apply(lambda x: x.fillna(value=ac_mean))
        sensor_df = pd.concat((sensor_df, ac_filled), axis=1)
    sensor_df = sensor_df.fillna(value=0)

    sensor_df.to_csv('./Data/Extract/plant_{}_collate_raw.csv'.format(str(plant)))
    print(sum(sensor_df['time'] != work_df['time']))


def fill_missing_data(plant):
    df = pd.read_csv('./Data/Extract/plant_{}_collate_raw.csv'.format(str(plant)), index_col=0)
    for i in range(len(df) - 1):
        if df['time'].iloc[i] + 900 == df['time'].iloc[i + 1]:
            continue
        elif df['time'].iloc[i] + 1800 == df['time'].iloc[i + 1]:
            # preserve the first 3 fields, average the reset
            df.loc[len(df)] = list(df.iloc[i][:3].to_numpy()) + list((df.iloc[i][3:] + df.iloc[i + 1][3:]).to_numpy() / 2)
            # change the time
            df.loc[len(df) - 1, 'DATE_TIME'] = time_to_str(df.iloc[len(df) - 1]['time'], True, plant)
        else:
            num_miss = (df['time'].iloc[i + 1] - df['time'].iloc[i])//900
            print(num_miss)
    df = df.sort_values('time').reset_index(drop=True)
    df.to_csv(('./Data/Extract/plant_{}_collate_raw_filled.csv'.format(str(plant))))




def collate_lstm_input(plant):

    # get inverter id
    gen_file_name = "Plant_" + str(plant) + "_Generation_Data.csv"
    gen_file_path = os.path.join("Data", "Raw", gen_file_name)
    gen_df = pd.read_csv(gen_file_path)
    inverter_ids = gen_df.SOURCE_KEY.unique()

    # get_time
    df = pd.read_csv('Data/Extract/plant_{}_collate_raw_filled.csv'.format(plant), index_col=0)
    df['time_feature'] = df['DATE_TIME'].apply(lambda x: get_time_mapping(x))
    feature_names = ["time_feature", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    inverter_output_names = ["DC_" + str(i) for i in range(config.NUM_INVERTER)]
    col_names = feature_names + inverter_output_names
    time_df = df[['time']]


    # scaling features
    fea_df = df[feature_names]
    x = fea_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[feature_names] = pd.DataFrame(x_scaled, columns=feature_names)

    # scaling DC output
    output_df = df[inverter_output_names].to_numpy()
    outputs_in_col = output_df.reshape(-1, 1)
    output_scaler = preprocessing.MinMaxScaler()
    output_scaler.fit(outputs_in_col)
    pickle.dump(output_scaler, open('./Data/Extract/scaler_plant_{}.pkl'.format(plant), 'wb'))

    # apply the scaler to every element in the output
    a = df[inverter_output_names].applymap(lambda x: output_scaler.transform([[x]])[0][0])
    df[inverter_output_names] = a

    collate = []
    for i in range(len(df) - config.WINDOW_SIZE_INT - config.NUM_OUTPUT + 1):
        if time_df['time'].at[i+config.WINDOW_SIZE_INT + config.NUM_OUTPUT-1] - time_df['time'].at[i] != 900*13:
            continue

        for inverter_id in range(config.NUM_INVERTER):
            inverter_name = inverter_output_names[inverter_id]

            flatten_feature = \
                list(df[(feature_names + [inverter_name])].loc[i:i+config.WINDOW_SIZE_INT-1,:].to_numpy().flatten()) # write features
            for output_id in range(config.NUM_OUTPUT):
                flatten_feature.append(df[inverter_name][i+config.WINDOW_SIZE_INT + output_id])

            row_list = [time_df['time'].iloc[9], inverter_id, inverter_ids[inverter_id]] + flatten_feature  # window end time
            collate.append(row_list)

    collate_df = pd.DataFrame(collate, columns=np.arange(len(collate[0])))
    collate_col_names = ['time', 'inverter_id', 'inverter_source_key'] + \
                        ['win_' + str(i) for i in range(config.WINDOW_SIZE_INT*5)] + \
                        ['out_' + str(i) for i in range(config.NUM_OUTPUT)]
    collate_df.columns = collate_col_names
    collate_df.to_csv('./Data/Extract/plant_{}_raw_features_label_fill.csv'.format(plant))



def inference(plant):
    scalar = pickle.load(open(config.SCALER_PATH.format(plant), 'rb'))

    model = LSTMPredictor()
    model.load_state_dict(torch.load(config.MODEL_PATH))


    dataset = RawDataset(plant)
    dataloader = DataLoader(dataset, batch_size=1,
                                     shuffle=False, num_workers=4)



    counter = 0
    index = 0
    with torch.no_grad():
        for data in dataloader:
            index += 1
            inputs = data['window']
            labels = data['forecast']
            outputs = model(inputs)
            # # probability_distribution = torch.nn.functional.softmax(outputs)
            prediction = np.argmax(outputs.detach().numpy())
            # print('prediction of MLP model is {}'.format(prediction))
            # print('label is {}'.format(labels.detach().numpy()[0]))
            # print('----')
            if labels.detach().numpy()[0] != prediction:
                counter += 1
                print(index)
                print('prediction of MLP model is {}'.format(prediction))
                print('label is {}'.format(labels.detach().numpy()[0]))
                print('----')
    print(counter)


    scalar.inverse_transform()


if __name__ == "__main__":
    # format_file(1)
    # format_file(2)
    # fill_missing_data(1)
    # fill_missing_data(2)

    collate_lstm_input(1)
    collate_lstm_input(2)