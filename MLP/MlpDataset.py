import re
import os
import ast
import pickle
from datetime import datetime

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset

num_of_inverter = 22


class MlpDataset(Dataset):
    def __init__(self, overwrite=False):
        plant_name = 'plant_1'
        self.csv_path = r'../Data/Extract/{}_feature_extract.csv'.format(plant_name)
        self.out_csv_path = r'../Data/Extract/{}_mlp_processed.csv'.format(plant_name)
        if not os.path.isfile(self.out_csv_path) or overwrite:
            self.preprocess()
        self.dataframe = pd.read_csv(self.out_csv_path)
        self.dataframe['features'] = self.dataframe['features'].apply(lambda x: ast.literal_eval(x))
        self.dataframe['forecasts'] = self.dataframe['forecasts'].apply(lambda x: ast.literal_eval(x))

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        unix_time = datetime.fromtimestamp(row['unix_time'])
        time_feature = (unix_time.hour * 60 + unix_time.minute) / 1440  # normalized
        features = [time_feature] + row['features']
        forecasts = row['forecasts']
        return torch.Tensor(features), torch.Tensor(forecasts)

    def __len__(self):
        return len(self.dataframe)

    def get_input_size(self):
        return len(self[0][0])

    def get_output_size(self):
        return len(self[0][1])

    def preprocess(self):
        df = pd.read_csv(self.csv_path, index_col=0)
        list_dict = {
            'date_time': [],
            'plant_id': [],
            'inverter_id': [],
            'unix_time': [],
            'features': [],
            'forecasts': [],
        }

        column_to_inverter_and_forecast_dict = {}
        column_to_inverter_feature_dict = {}
        for key in df.keys():
            if key.endswith('forecast'):
                match = re.match(r'DC_(\d+)_post_(\d+)_min_forecast', key)
                inverter_id = int(match.group(1))
                forecast_id = int(match.group(2)) // 15 - 1  # one forecast per 15 minutes, 15 ~ 60 -> 0 ~ 4
                column_to_inverter_and_forecast_dict[key] = [inverter_id, forecast_id]
            elif 'DC' in key and (key.startswith('mean') or key.startswith('min') or key.startswith('max') or key.startswith('lrc')):
                match = re.match(r'\w+ DC_(\d+)', key)
                inverter_id = int(match.group(1))
                if key.startswith('mean'):
                    feature_id = 0
                elif key.startswith('min'):
                    feature_id = 1
                elif key.startswith('max'):
                    feature_id = 2
                elif key.startswith('lrc'):
                    feature_id = 3
                column_to_inverter_feature_dict[key] = [inverter_id, feature_id]

        for i in tqdm(range(len(df))):
            date_time = df.iloc[i]['date_time']
            plant_id = df.iloc[i]['plant_id']
            unix_time = df.iloc[i]['unix time']
            features = []
            for key in df.keys():
                if key.startswith('mean') or key.startswith('min') or key.startswith('max') or key.startswith('lrc'):
                    features.append(df.iloc[i][key])

            forecasts_per_inverter = [[None for _ in range(4)] for _ in range(num_of_inverter)]
            dc_feature_per_inverter = [[None for _ in range(4)] for _ in range(num_of_inverter)]
            for key in df.keys():
                if key.endswith('forecast'):
                    inverter_id, forecast_id = column_to_inverter_and_forecast_dict[key]
                    forecasts_per_inverter[inverter_id][forecast_id] = df.iloc[i][key]
                elif 'DC' in key and (key.startswith('mean') or key.startswith('min') or key.startswith('max') or key.startswith('lrc')):
                    inverter_id, feature_id = column_to_inverter_feature_dict[key]
                    dc_feature_per_inverter[inverter_id][feature_id] = df.iloc[i][key]


            for inverter_id in range(num_of_inverter):
                list_dict['date_time'].append(date_time)
                list_dict['plant_id'].append(plant_id)
                list_dict['inverter_id'].append(inverter_id)
                list_dict['unix_time'].append(unix_time)
                list_dict['features'].append(features + dc_feature_per_inverter[inverter_id])
                # list_dict['features'].append(features)
                list_dict['forecasts'].append(forecasts_per_inverter[inverter_id])

        out_df = pd.DataFrame(list_dict, columns=list(list_dict.keys()))
        # normalize
        normalize_column(out_df, 'features')
        normalize_column(out_df, 'forecasts')

        out_df.to_csv(self.out_csv_path)


def normalize_column(df, key):
    x = df[key].to_numpy()  # returns a numpy array
    x = np.array([np.array(field) for field in x])
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x_scaled = [list(field) for field in x_scaled]
    df[key] = x_scaled
    print('column normalized: {}'.format(key))
    pickle.dump(min_max_scaler, open('./norm_data/norm_data_{}.pkl'.format(key), 'wb'))
    return df


def denormalize_column(values, key):
    min_max_scaler = pickle.load(open('./norm_data/norm_data_{}.pkl'.format(key), 'rb'))
    return min_max_scaler.inverse_transform(values)


if __name__ == '__main__':
    dataset = MlpDataset()
    print(dataset[0])