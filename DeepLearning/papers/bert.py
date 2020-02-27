import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook

import random
import time

from torchtext.data import Field, BucketIterator
from torchtext.datasets import IMDB
import pandas as pd

import zipfile
file = zipfile.ZipFile('/content/imdb-dataset-of-50k-movie-reviews.zip')
file.extractall()

from pytorch_transformers import BertConfig,BertTokenizer, BertPreTrainedModel, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

max_seq_length = 128
batch_size = 32
CACHE_DIR = '/content/cache'
lr = 1e-5
BERT_MODEL = 'bert-base-cased'
NUM_LABELS = 3

import pandas as pd

df = pd.read_csv('/content/IMDB Dataset.csv')

"""### Data Utils"""

from torch.utils.data import Dataset,DataLoader
import numpy as np

label2idx = {'positive':0,'negative':1}

class BertDataset(Dataset):

    def __init__(self,data_frame,max_length):
        super(BertDataset,self).__init__()
        self.data_frame = data_frame
        self.max_length = max_length
    
    def __len__(self):
        return self.data_frame.shape[0]
    
    def __getitem__(self,idx):
        row = self.data_frame.iloc[idx]
        input_ids,input_mask,segment_ids,label_id = convert_example_to_feature(row, self.max_length)
        return input_ids,input_mask,segment_ids, label_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_example_to_feature(row, max_length):
    # return example_row
    title1, label = row

    title2 = None

    # print(label_map)

    tokens_a = tokenizer.tokenize(title1)

    tokens_b = None
    if title2:
        tokens_b = tokenizer.tokenize(title2)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label2idx[label]

    return np.array(input_ids), np.array(input_mask),np.array(segment_ids), np.array(label_id)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df_train, df_valid = train_test_split(df,test_size=0.2,random_state=42)

train_dataset = BertDataset(data_frame=df_train,max_length=max_seq_length)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=2)

model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=NUM_LABELS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
gradient_accumulation_steps = 1
train_batch_size = 32
eval_batch_size = 128
train_batch_size = train_batch_size // gradient_accumulation_steps
output_dir = 'output'
bert_model = 'bert-base-chinese'
num_train_epochs = 3
num_train_optimization_steps = int(
            len(df_train) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
cache_dir = "model"
learning_rate = 5e-5
warmup_proportion = 0.1
max_seq_length = 128
label_list = ['unrelated', 'agreed', 'disagreed']

gradient_clip = 5

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

def get_mean(lis):
    return sum(lis)/len(lis)

def train_iter(model,batch,optimizer,criterion,gradient_clip):
    model.zero_grad()
    input_ids, input_mask, segment_ids, label_ids = batch
    logits = model(input_ids, segment_ids, input_mask, labels=None)[0]
    loss = criterion(logits,label_ids)
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), gradient_clip)
    optimizer.step()
    return loss.item()

def eval(model,batch,criterion):
    input_ids, input_mask, segment_ids, label_ids = batch
    logits = model(input_ids, segment_ids, input_mask, labels=None)[0]
    loss = criterion(logits,label_ids)
    pred_classes = torch.argmax(logits,dim=-1).cpu().data.numpy().tolist()
    target_classes = label_ids.cpu().data.numpy().tolist()
    return loss.item(), target_classes, pred_classes

def train(model,optimizer,criterion,epoch,batch_size,rank):

    train_dataset = BertDataset(data_frame=df_train,max_length=max_seq_length)
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=2)
    val_dataset = BertDataset(data_frame=df_valid,max_length=max_seq_length)
    val_loader = DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=2)

    for i in range(epoch):

        loss_epoch = []
        val_loss_epoch = []

        model.train()

        for batch in tqdm_notebook(train_loader):

            batch = tuple(t.to(device) for t in batch)

            loss = train_iter(model,batch,optimizer,criterion,gradient_clip)
            loss_epoch.append(loss)
        model.eval()

        target_list = []
        pred_list = []
        for batch in tqdm_notebook(val_loader):

            batch = tuple(t.to(device) for t in batch)
            loss,target_classes, pred_classes = eval(model,batch,criterion)
            val_loss_epoch.append(loss)
            target_list.extend(target_classes)
            pred_list.extend(pred_classes)

        print('Rank:{0}, Epoch:{1}, train_loss:{2:.2f}, val_loss:{3:.2f}'.format(rank,i+1,get_mean(loss_epoch),get_mean(val_loss_epoch)))
        print(classification_report(target_list,pred_list,target_names=['positive', 'negative']))

model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=NUM_LABELS)
optimizer = optim.Adam(params=model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()
model.to(device)

import torch.multiprocessing as mp

train(model,optimizer,criterion,epoch=5,batch_size=batch_size,rank=0)

