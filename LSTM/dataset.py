import pandas as pd
import numpy as np
import config
from torch.utils.data import Dataset


class RawDataset(Dataset):

    def __init__(self):
        self.dataframe = pd.read_csv('../Data/Extract/plant_1_raw_features_label_fill.csv', index_col=0)



    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):

        window = self.dataframe.iloc[idx, :-4].to_numpy().reshape(config.WINDOW_SIZE_INT, config.NUM_FEATURES).astype('float32')
        forecast = self.dataframe.iloc[idx, -4:].to_numpy().astype('float32')
        sample = {'window': window, 'forecast': forecast}
        return sample
