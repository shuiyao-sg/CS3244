import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import config
from LSTM.dataset import RawDataset

device = 'cuda'

class LSTMPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :].reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


def train_model(model, dataloader, criterion, optimizer):
    model = model.to(device)
    best_loss = float('inf')
    best_acc = -1
    for epoch in range(1, config.EPOCHS+1):

        print('Epoch {}/{}'.format(epoch, config.EPOCHS))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for data in dataloader[phase]:
                inputs = data['window'].to(device)
                labels = data['forecast'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), config.MODEL_PATH)

    return model, best_loss


if __name__ == "__main__":
    dataset = RawDataset()
    val_len = int(len(dataset) * config.VAL_SPLIT)
    train_len = len(dataset) - val_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    dataloader = {}
    dataloader['train'] = DataLoader(train_set, batch_size=config.BATCH_SIZE,
                                     shuffle=True, num_workers=4)
    dataloader['val'] = DataLoader(val_set, batch_size=config.BATCH_SIZE,
                                   shuffle=True, num_workers=4)

    torch.manual_seed(1)

    model = LSTMPredictor(config.NUM_FEATURES, config.HIDDEN_DIM, config.NUM_OUTPUT)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)
    train_model(model, dataloader, loss_function, optimizer)


