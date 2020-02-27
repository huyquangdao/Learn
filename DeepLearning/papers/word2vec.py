import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sentences = ["i like dog", "i like cat", "i like animal",
             "dog cat animal", "apple cat dog like", "dog fish milk like",
             "dog cat eyes like", "i like apple", "apple i hate",
             "apple i movie book music like", "cat dog hate", "cat dog like"]

vocab = set([word for sent in sentences for word in sent.split()])
word2idx = {word: key for key, word in enumerate(vocab)}
idx2word = {key: word for key, word in enumerate(vocab)}
window_size = 1


def create_dataset(type='skip_gram'):
    x_data = []
    y_data = []
    for sent in sentences:
        words = sent.split()
        for i in range(window_size, len(words) - window_size):
            if type =='skip_gram':
                x_data.append(word2idx[words[i]])
                x_data.append(word2idx[words[i]])
                y_data.append(word2idx[words[i - window_size]])
                y_data.append(word2idx[words[i + window_size]])
            else:
                x_data.append([word2idx[words[i-window_size]], word2idx[words[i + window_size]]])
                y_data.append(word2idx[words[i]])
    return x_data, y_data


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_dims):
        super(SkipGram, self).__init__()
        self.projection_matrix = nn.Parameter(torch.randn(size=(vocab_size, embedding_dims)))
        self.linear = nn.Linear(embedding_dims, vocab_size)

    def forward(self, input):
        x = input
        embedding_vec = torch.mm(x, self.projection_matrix)
        out = self.linear(embedding_vec)
        return out


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dims):
        super(CBOW, self).__init__()
        self.projection_matrix = nn.Parameter(torch.randn(size=(vocab_size,embedding_dims)))
        self.linear = nn.Linear(embedding_dims,vocab_size)

    def forward(self, input):
        x = input
        out_tensor = torch.zeros(size=(x.shape[0],embedding_dims))
        for i,b in enumerate(x):
            temp = torch.mm(b,self.projection_matrix)
            averaged = torch.add(temp[0],temp[1]) / window_size
            out_tensor[i] = averaged
        output = self.linear(out_tensor)
        return output


class Word2VecDataset(Dataset):

    def __init__(self,type):
        super(Word2VecDataset, self).__init__()
        self.type = type
        self.x, self.y = create_dataset(type=self.type)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.type =='skip_gram':
            onehot = [0] * len(vocab)
            onehot[self.x[idx]] = 1
            onehot = np.array(onehot)
            return onehot, self.y[idx]
        else:
            list_onehot = []
            list_word_idx = self.x[idx]
            for idx in list_word_idx:
                onehot = [0] * len(vocab)
                onehot[idx] = 1
                list_onehot.append(onehot)
            return np.array(list_onehot) , self.y[idx]


def train_iter(model, batch, criterion, optimizer):
    model.zero_grad()
    x, y = batch
    pred = model(x.float())
    loss = criterion(pred, y)
    cls1 = torch.argmax(pred, dim=1)
    acc = torch.eq(cls1, y).int().sum().data.numpy() / x.shape[0]
    loss.backward()
    optimizer.step()
    return loss.item(), acc


def get_mean(lis):
    return sum(lis) / len(lis)


def train(epoch, lr, batch_size, embedding_dims,type):
    train_dataset = Word2VecDataset(type=type)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    if type =='skip_gram':
        model = SkipGram(vocab_size=len(vocab), embedding_dims=embedding_dims)
    else:
        model = CBOW(vocab_size=len(vocab), embedding_dims=embedding_dims)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for i in range(epoch):
        loss_epoch = []
        acc_epoch = []

        for batch in train_loader:
            loss, acc = train_iter(model, batch, criterion, optimizer)
            loss_epoch.append(loss)
            acc_epoch.append(acc)

        print('epoch:{}, loss:{}, acc:{}'.format(i + 1, get_mean(loss_epoch), get_mean(acc_epoch)))

    pca = PCA(n_components=2)
    for i, param in enumerate(model.parameters()):
        if i == 0:
            for i, label in idx2word.items():
                W = param
                w_pca = pca.fit_transform(W.data.numpy())
                x = w_pca[i][0]
                y = w_pca[i][1]
                plt.scatter(x, y)
                plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        else:
            break
    plt.show()


if __name__ == '__main__':
    epoch = 30
    lr = 0.01
    batch_size = 4
    embedding_dims = 50
    type = 'CBOW'
    train(epoch, lr, batch_size, embedding_dims,type)
