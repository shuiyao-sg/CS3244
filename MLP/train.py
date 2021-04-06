import torch
import torch.nn as nn
from torch.utils.data import random_split

import MLP.config as config
from MLP.MlpDataset import MlpDataset
from MLP.MlpModel import MlpModel
from MLP.function import train, validate

train_ratio = 0.7

def main():
    dataset = MlpDataset()
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

    best_loss = float('inf')
    for epoch in range(config.TOTAL_EPOCH):
        train_loss = train(train_loader, model, criterion, optimizer, epoch, device)
        val_loss = validate(val_loader, model, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), config.BEST_MODEL_DIR)

if __name__ == '__main__':
    main()