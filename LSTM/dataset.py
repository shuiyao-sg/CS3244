import pandas as pd
import numpy as np
import config
from torch.utils.data import Dataset


class RawDataset(Dataset):

    def __init__(self, plant=1):
        self.dataframe = pd.read_csv('Data/Extract/plant_{}_raw_features_label_fill.csv'.format(plant), index_col=0)



    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        time = self.dataframe['time'][idx]
        inverter_id = self.dataframe['inverter_id'][idx]
        inverter_source_key = self.dataframe['inverter_source_key'][idx]
        window = self.dataframe.filter(regex="win_*", axis=1).loc[idx].to_numpy().reshape(config.WINDOW_SIZE_INT, config.NUM_FEATURES).astype('float32')
        forecast = self.dataframe.filter(regex="out_*", axis=1).loc[idx].to_numpy().astype('float32')
        sample = {
            'time': time,
            'inverter_source_key': inverter_source_key,
            'inverter_id': inverter_id,
            'window': window,
            'forecast': forecast}
        return sample
