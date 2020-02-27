import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.multiprocessing as mp


class LogisticDataset(Dataset):

    def __init__(self, data_path):
        super(LogisticDataset, self).__init__()
        df = pd.read_csv(data_path, names=['fea1', 'fea2', 'label'])
        self.x = df[['fea1', 'fea2']].values
        self.x = (self.x - self.x.mean()) / self.x.std()
        self.y = df['label'].values

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LogisticRegression(nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


def train_iter(model, batch, criterion, optimizer):
    model.zero_grad()
    x, y = batch
    pred = model(x.float())
    pred = torch.squeeze(pred, 1)
    loss = criterion(pred, y)
    cls1 = pred.ge(0.5)
    cls2 = y.ge(1.0)
    acc = torch.eq(cls1, cls2).int().sum().data.numpy() / x.shape[0]
    loss.backward()
    optimizer.step()
    return loss.item(), acc


def get_mean(lis):
    return sum(lis) / len(lis)


def train(model, lr, file_path, epoch, batch_size, rank):
    train_dataset = LogisticDataset(file_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for i in range(epoch):

        loss_epoch = []
        acc_epoch = []

        for batch in train_loader:
            loss, acc = train_iter(model, batch, criterion, optimizer)
            loss_epoch.append(loss)
            acc_epoch.append(acc)

        print('process:{}, epoch:{}, loss:{}, acc:{}'.format(rank, i + 1, get_mean(loss_epoch), get_mean(acc_epoch)))


if __name__ == '__main__':
    model = LogisticRegression()
    model.share_memory()
    lr = 0.01
    n_process = 4
    processes = []
    for i in range(n_process):
        p = mp.Process(target=train, args=(model, lr, 'data/logstic_data.csv', 8, 10, i,))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
