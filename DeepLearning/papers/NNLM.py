import torch
import torch.nn as nn
import torch.optim as optim

sentences = ['i like dog', 'i like coffee', 'i hate milk','i hate cat','i hate fish']

vocab = set([word for sent in sentences for word in sent.split()])
word2idx = {word: key for key, word in enumerate(vocab)}
idx2word = {key: word for key, word in enumerate(vocab)}
n_gram = 3


def build_batch():
    x_data = []
    y_data = []
    for sent in sentences:
        words = sent.split()
        data = []
        for word in words[:-1]:
            data.append(word2idx[word])
        x_data.append(data)
        y_data.append(word2idx[words[-1]])

    return torch.LongTensor(x_data), torch.LongTensor(y_data)


class NNLM(nn.Module):

    def __init__(self, hidden_size, embedding_dims, vocab_size):
        super(NNLM, self).__init__()
        self.C = torch.nn.Embedding(embedding_dim=embedding_dims, num_embeddings=vocab_size)
        self.H = torch.nn.Parameter(torch.randn(size=((n_gram - 1) * embedding_dims, hidden_size)))
        self.U = torch.nn.Parameter(torch.randn(size=(hidden_size, vocab_size)))
        self.d = torch.nn.Parameter(torch.zeros(size=(hidden_size,)))
        self.b = torch.nn.Parameter(torch.zeros(size=(vocab_size,)))
        self.w = torch.nn.Parameter(torch.randn(size=((n_gram - 1) * embedding_dims, vocab_size)))

    def forward(self, input):
        x = self.C(input)
        x = x.view(-1, x.shape[1] * x.shape[2])
        y = self.b + x.mm(self.w) + (torch.tanh(self.d + x.mm(self.H))).mm(self.U)
        return y


def train(epoch, lr):
    model = NNLM(hidden_size=128, embedding_dims=50, vocab_size=len(vocab))
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    x_data, y_data = build_batch()

    model.train()

    for i in range(epoch):
        model.zero_grad()
        pred = model(x_data)
        loss = criterion(pred, y_data)
        print(loss.item())
        loss.backward()
        optimizer.step()

    predict = model(x_data).data.max(1, keepdim=True)[1].numpy().tolist()
    for i, sent in enumerate(sentences):
        train = sent.split(" ")[:-1]
        word_pred = idx2word[predict[i][0]]
        print(train, '->', word_pred)


if __name__ == '__main__':
    train(5, 0.01)
