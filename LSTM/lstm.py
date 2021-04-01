import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import config
from LSTM.dataset import RawDataset


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
    best_loss = float('inf')
    best_acc = -1
    for epoch in range(1, config.epochs+1):

        print('Epoch {}/{}'.format(epoch, config.epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for data in dataloader[phase]:
                inputs = data['window']
                labels = data['forecast']

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
            torch.save(model.state_dict(), config.model_path)

        if phase == 'val':
            for dataset_name in ['train', 'val']:
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in dataloader[dataset_name]:
                        inputs = data['features']
                        labels = data['action']
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = correct / total
                print(dataset_name + 'Accuracy: %.2f' % (
                        100 * accuracy))

            best_acc = max(accuracy, best_acc)

    print(model)
    return model, best_loss, best_acc


if __name__ == "__main__":
    dataset = RawDataset()
    val_len = int(len(dataset) * config.val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    dataloader = {}
    dataloader['train'] = DataLoader(train_set, batch_size=config.batch_size,
                                     shuffle=True, num_workers=4)
    dataloader['val'] = DataLoader(val_set, batch_size=config.batch_size,
                                   shuffle=True, num_workers=4)

    torch.manual_seed(1)

    model = LSTMPredictor(config.NUM_FEATURES, config.hidden_dim, config.NUM_OUTPUT)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_model(model, dataloader, loss_function, optimizer)


