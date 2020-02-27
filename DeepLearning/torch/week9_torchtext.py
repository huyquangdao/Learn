from torchtext.data import Field
from nltk import word_tokenize
import re
import nltk

nltk.download('punkt')

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def preprocess_reviews(line):
    line = REPLACE_NO_SPACE.sub("", line.lower())
    line = REPLACE_WITH_SPACE.sub(" ", line)
    return line


def my_tokenize(x):
    x = preprocess_reviews(x)
    words_tokens = word_tokenize(x)
    return words_tokens


import os

os.environ['KAGGLE_USERNAME'] = "huydaoquang"  # username from the json file
os.environ['KAGGLE_KEY'] = "0eb162199ef4cf635c9586a2d18aaf2f"  # key from the json file

import pandas as pd

df = pd.read_csv('/content/IMDB Dataset.csv')
df.head()

preprocess = lambda x: 1 if x == 'negative' else 0

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

import torch
import torchtext
from torchtext.data import Field, TabularDataset, BucketIterator

TEXT = Field(sequential=True, use_vocab=True, tokenize=my_tokenize, stop_words=stop_words, batch_first=True)
LABEL = Field(sequential=False, use_vocab=False,
              preprocessing=torchtext.data.Pipeline(lambda x: 1 if x == 'positive' else 0))

movie_fields = [('review', TEXT), ('sentiment', LABEL)]

dataset = TabularDataset(path='/content/IMDB Dataset.csv', format='csv', fields=movie_fields, skip_header=True)

TEXT.build_vocab(dataset)

train_dataset, val_dataset = dataset.split(split_ratio=0.8, strata_field='sentiment')

train_iter, val_iter = BucketIterator.splits(datasets=(train_dataset, val_dataset),
                                             batch_sizes=(64, 64),
                                             device=torch.device('cuda:0'),
                                             sort_key=lambda x: len(x.review),
                                             sort_within_batch=False,
                                             repeat=False,

                                             )


class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper

            y = getattr(batch, self.y_vars)
            yield (x, y)

    def __len__(self):
        return len(self.dl)


train_dl = BatchWrapper(train_iter, "review", "sentiment")
valid_dl = BatchWrapper(val_iter, "review", "sentiment")

for batch in valid_dl:
    print(batch)
    break

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report


class LSTMClassifier(nn.Module):

    def __init__(self, n_classes, vocab_size, embedding_dim, hidden_size, n_layer, drop_out):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layer)
        self.drop_out = nn.Dropout(p=drop_out)
        self.hidden = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=n_classes)

    def forward(self, input):
        input_embedding = self.embedding(input)
        input_embedding = torch.transpose(input_embedding, dim0=0, dim1=1)

        output, (_, _) = self.lstm(input_embedding)
        output = output[-1, :, :]
        output = self.drop_out(output)
        output = self.hidden(output)
        output = self.drop_out(output)
        output = self.linear(output)

        return output


n_classes = 2
vocab_size = len(TEXT.vocab)
embedding_dim = 300
hidden_size = 256
n_layer = 1
epoch = 10
lr = 0.001
lamda = 0
drop_out = 0.1

model = LSTMClassifier(n_classes, vocab_size, embedding_dim, hidden_size, n_layer, drop_out)
optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=lamda)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
model.cuda()


def get_mean(lis):
    return sum(lis) / len(lis)


for i in range(epoch):
    epoch_loss = []
    model.train()
    for batch in tqdm_notebook(train_dl):
        model.zero_grad()
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        logit = model(x)
        loss = criterion(logit, y)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    val_epoch_loss = []
    pred_list = []
    target_list = []
    print('-----------EVALUATE----------')
    for batch in tqdm_notebook(valid_dl):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        logit = model(x)
        loss = criterion(logit, y)
        val_epoch_loss.append(loss.item())
        pred_class = torch.argmax(logit, dim=-1).cpu().data.numpy().tolist()
        target_class = y.data.cpu().data.numpy().tolist()
        pred_list.extend(pred_class)
        target_list.extend(target_class)
    print('epoch: {0}, train_loss:{1:.2f}, val_loss:{2:.2f}'.format(i + 1, get_mean(epoch_loss),
                                                                    get_mean(val_epoch_loss)))
    print(classification_report(target_list, pred_list, target_names=['negative', 'positive']))
    scheduler.step()
