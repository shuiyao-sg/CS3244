import time
import datetime

import torch
import torch.nn as nn
from torch.utils.data import random_split
import numpy as np
import pandas as pd

import MLP.config as config
from MLP.MlpDataset import MlpDataset, denormalize_column
from MLP.MlpModel import MlpModel
from MLP.function import train, validate

train_ratio = 0.7

def main():
    dataset = MlpDataset(overwrite=False)
    train_length = int(len(dataset) * train_ratio)
    val_length = len(dataset) - train_length
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length], generator=torch.Generator().manual_seed(42))

    config.INPUT_SIZE = dataset.get_input_size()
    config.OUTPUT_SIZE = dataset.get_output_size()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MlpModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        # model.parameters(),
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    full_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_loss = float('inf')
    for epoch in range(config.TOTAL_EPOCH):
        train_loss, train_outputs = train(train_loader, model, criterion, optimizer, epoch, device)
        val_loss, val_outputs = validate(val_loader, model, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), config.BEST_MODEL_DIR)
            full_loss, full_outputs = validate(full_loader, model, criterion, device)
            full_outputs = denormalize_column(full_outputs, 'forecasts')
            save_predicted_data(dataset, full_outputs)


def save_predicted_data(dataset, outputs, plant_id=1):
    template_dataframe = pd.read_csv('../Data/Raw/Plant_{}_Generation_Data.csv'.format(plant_id))
    assert len(dataset.dataframe) == len(outputs)
    id_source_key_list = template_dataframe['SOURCE_KEY'].unique()
    output_dict = {}
    for i in range(len(dataset.dataframe)):
        for j in range(4):
            unix_time = dataset.dataframe.at[i, 'unix_time'] + (j + 1) * 900
            inverter_id = dataset.dataframe.at[i, 'inverter_id']
            # if inverter_id >= len(id_source_key_list):
            #     continue
            inverter_id = id_source_key_list[inverter_id]
            pred_str = 'pred_{}'.format(j)
            assert (unix_time, inverter_id, pred_str) not in output_dict
            # print((unix_time, inverter_id, pred_str))
            output_dict[(unix_time, inverter_id, pred_str)] = outputs[i][j]
        pass
    time_format = "%Y-%m-%d %H:%M:%S" if plant_id == 2 else "%d-%m-%Y %H:%M"
    for i in range(len(template_dataframe)):
        for j in range(4):
            time_str = template_dataframe.at[i, 'DATE_TIME']
            unix_time = time.mktime(datetime.datetime.strptime(time_str, time_format).timetuple())
            inverter_id = template_dataframe.at[i, 'SOURCE_KEY']
            pred_str = 'pred_{}'.format(j)
            if (unix_time, inverter_id, pred_str) in output_dict:
                template_dataframe.at[i, pred_str] = output_dict[(unix_time, inverter_id, pred_str)]
    template_dataframe.to_csv('../Data/Extract/mlp_revert_plant_{}.csv'.format(plant_id))
    print('prediction saved to ../Data/Extract/mlp_revert_plant_{}.csv'.format(plant_id))


if __name__ == '__main__':
    main()