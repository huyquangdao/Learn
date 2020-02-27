
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

train_data,valid_data,test_data = Multi30k.splits(exts={'.de','.en'},
                fields=(SRC,TRG))

SRC.build_vocab(train_data,min_freq=2)
TRG.build_vocab(train_data,min_freq=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)

class Encoder(nn.Module):

    def __init__(self,vocab_size,embedding_dim,enc_hidden_size,dec_hidden_size,drop_out):

        super(Encoder,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.gru = nn.GRU(embedding_dim,enc_hidden_size,bidirectional = True)

        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

        self.drop_out = nn.Dropout(drop_out)
    
    def forward(self,input):
        
        #input = [src_len, batch_size]

        embedding_tensor = self.drop_out(self.embedding(input))

        #embedding_tensor = [src_len. batch_size, embedding_dim]

        outputs, hidden = self.gru(embedding_tensor)

        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]

        hidden = torch.tanh(self.fc(torch.cat([hidden[-2,:,:],hidden[-1,:,:]],dim=1)))

        return outputs, hidden

class Attention(nn.Module):

    def __init__(self,enc_hidden_size,dec_hidden_size):

        super(Attention,self).__init__()

        self.attn = nn.Linear((enc_hidden_size * 2) + dec_hidden_size,dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size,1,bias=False)

    def forward(self,hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        #repeat decoder hidden state src_len times

        hidden = hidden.unsqueeze(1).repeat(1,src_len,1)

        encoder_outputs = encoder_outputs.permute(1,0,2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat([hidden,encoder_outputs],dim=2)))

        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        #attention= [batch size, src len]

        return f.softmax(attention,dim=1)

class Decoder(nn.Module):

    def __init__(self,vocab_size,embedding_dim,enc_hidden_size,dec_hidden_size,drop_out,attention):

        super(Decoder,self).__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        self.gru = nn.GRU((enc_hidden_size * 2) + embedding_dim,dec_hidden_size)

        self.fc_out = nn.Linear((enc_hidden_size * 2) + dec_hidden_size + embedding_dim,vocab_size)

        self.drop_out = nn.Dropout(drop_out)

    def forward(self,input,hidden,encoder_outputs):

        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)

        #input = [1,batch_size]

        embedding_tensor = self.drop_out(self.embedding(input))

        # embedding_tensor = [1,batch_size,embeding_dim]

        a = self.attention(hidden,encoder_outputs)

        #a = [batch_size,src_len]

        a = a.unsqueeze(1)

        #a = [batch_size,1,src_len]

        encoder_outputs = encoder_outputs.permute(1,0,2)

        #encoder_outputs = [batch_size,src_len,enc_hidden_dim * 2]

        weighted = torch.bmm(a,encoder_outputs)

        #weighted = [batch_size,1, enc_hidden_dim * 2]

        weighted = weighted.permute(1,0,2)
        
        #weighted = [1,batch_size,enc_hidden_dim * 2]

        rnn_input = torch.cat([embedding_tensor,weighted],dim=2)

        #rnn_input = [1,batch_size,(enc_hidden_dim * 2) + embedding_dim]

        output, hidden = self.gru(rnn_input,hidden.unsqueeze(0))

        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]

        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()

        embedding_tensor = embedding_tensor.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat([output,weighted,embedding_tensor],dim=1))

        #prediction = [batch_size,output_dim]

        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):

    def __init__(self,encoder,decoder,device):

        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self,src,trg,teacher_forcing_ratio = 0.5):

        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75%

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size


        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer

        encoder_outputs, hidden = self.encoder(src)

        #first input to the decoder is the <sos> tokens

        input = trg[0,:]

        for t in range(1,trg_len):

            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state

            output,hidden = self.decoder(input,hidden,encoder_outputs)
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        return outputs

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

