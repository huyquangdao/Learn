

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

SRC = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
     batch_size = BATCH_SIZE,
     device = device)

class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_size,
                 n_layers,
                 n_heads,
                 pf_dim,
                 drop_out,
                 device,
                 max_length=100):

        super(Encoder,self).__init__()

        self.device = device

        self.token_embedding = nn.Embedding(input_dim,hidden_size)
        self.pos_embedding = nn.Embedding(max_length,hidden_size)

        self.layers = nn.ModuleList([EncoderLayer(hidden_size,
                                                  n_heads,
                                                  pf_dim,
                                                  drop_out,
                                                  device) for _ in range(n_layers)])
        
        self.drop_out = nn.Dropout(drop_out)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)
    
    def forward(self,src,src_mask):

        #src =[batch_size, src_len]
        #src_mask = [batch_size,src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).to(self.device)

        #pos = [batch_size, src_len]

        src = self.drop_out((self.token_embedding(src)*self.scale) + self.pos_embedding(pos))

        #src = [batch_size, src_len, hidden_size]

        for layer in self.layers:

            src = layer(src, src_mask)
        
        return src

class EncoderLayer(nn.Module):

    def __init__(self,
                 hidden_size,
                 n_heads,
                 pf_dim,
                 drop_out,
                 device):

        super(EncoderLayer,self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttentionLayer(hidden_size,
                                                      n_heads,
                                                      drop_out,
                                                      device)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_size, 
                                                                     pf_dim, 
                                                                     drop_out)
        
        self.drop_out = nn.Dropout(drop_out)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)

    
    def forward(self,src,src_mask):

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention

        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.drop_out(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.layer_norm(src + self.drop_out(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self,
                 hidden_size,
                 n_heads,
                 drop_out,
                 device):

        super(MultiHeadAttentionLayer,self).__init__()

        assert hidden_size % n_heads ==0

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.fc_q = nn.Linear(hidden_size,hidden_size)
        self.fc_k = nn.Linear(hidden_size,hidden_size)
        self.fc_v = nn.Linear(hidden_size,hidden_size)

        self.fc_o = nn.Linear(hidden_size,hidden_size)

        self.drop_out = nn.Dropout(drop_out)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    
    def forward(self,query,key,value,mask = None):
        
        batch_size = query.shape[0]

        #query =[batch_size, query_len, hidden_size]
        #key = [batch_size,key_len,hidden_size]
        #value = [batch_size,value_len,hidden_size]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q =[batch_size,query_len,hidden_size]
        # K = [batch_size, key_len,hidden_size]
        # V=[batch_size,value_len,hidden_size]

        Q = Q.view(batch_size, -1 , self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        #energy = [batch size, n heads, query_len, key_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy,dim=-1)

        #attention = [batch_size, n_heads, query_len, key_len]

        x = torch.matmul(self.drop_out(attention), V)

        #x= [batch_size, n_heads, seq_len, head_dim]

        x = x.permute(0,2,1,3).contiguous()

        #x = [batch size, seq len, n heads, head dim]

        x = x.view(batch_size, -1, self.hidden_size)

        #x = [batch size, seq len, hid dim]

        x= self.fc_o(x)

        #x = [batch_size, seq_len, hidden_size]

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):


      def __init__(self,hidden_size,pf_dim,drop_out):

          super(PositionwiseFeedforwardLayer,self).__init__()

          self.fc_1 = nn.Linear(hidden_size,pf_dim)
          self.fc_2 = nn.Linear(pf_dim,hidden_size)

          self.drop_out = nn.Dropout(drop_out)
      
      def forward(self,x):

          #x = [batch_size,seq_len,hidden_size]

          x = self.drop_out(torch.relu(self.fc_1(x)))

          # x= [batch_size, seq_len, pf_dim]

          x = self.fc_2(x)

          #x = [batch_size, seq_len, hidden_size]

          return x

class Decoder(nn.Module):

    def __init__(self,
                 output_dim,
                 hidden_size,
                 n_layers,
                 n_heads,
                 pf_dim,
                 drop_out,
                 device,
                 max_length=100):

        super().__init__()
        self.device = device
        self.token_embedding = nn.Embedding(output_dim,hidden_size)
        self.pos_embedding = nn.Embedding(max_length,hidden_size)

        self.layers = nn.ModuleList([DecoderLayer(hidden_size, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  drop_out, 
                                                  device)
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(hidden_size,output_dim)
        self.drop_out = nn.Dropout(drop_out)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)


    def forward(self,trg,enc_src,trg_mask,src_mask):

        #rtrg =[batch_size, trg_len]
        #enc_src =[batch_size, src_len, hidden_size]
        #trg_mask = [batch_size,trg_len]
        #src_mask = [batch_size, src_Len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0,trg_len).unsqueeze(0).repeat(batch_size,1).to(self.device)

        #pos = [batch_size,trg_len]

        trg = self.drop_out((self.token_embedding(trg)* self.scale)+ self.pos_embedding(pos))

        #trg= [batch_size,trg_len,hidden_size]

        for layer in self.layers:

            trg,attention = layer(trg,enc_src,trg_mask,src_mask)
        
        #trg =[batch_size,trg_len,hidden_size]
        #attention = [batch_size,n_heads,trg_len,src_len]

        output = self.fc_out(trg)

        #output = [batch_size,trg_len, output_dim]

        return output, attention

class DecoderLayer(nn.Module):

    def __init__(self,
                 hidden_size,
                 n_heads,
                 pf_dim,
                 drop_out,
                 device):

        super(DecoderLayer,self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttentionLayer(hidden_size,
                                                      n_heads,
                                                      drop_out,
                                                      device)
        self.encoder_attention = MultiHeadAttentionLayer(hidden_size,
                                                         n_heads,
                                                         drop_out,
                                                         device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_size,
                                                                     pf_dim,
                                                                     drop_out)
        
        self.drop_out = nn.Dropout(drop_out)
    
    def forward(self,trg,enc_src,trg_mask,src_mask):

        #trg =[batch_size,trg_len,hidden_size]
        #enc_src =[batch_size, src_len,hidden_size]
        #trg_mask =[batch_size,trg_len]
        #src_mask = [batch_size,src_len]

        #self attention

        _trg,_ = self.self_attention(trg,trg,trg,trg_mask)
        #drop_out, residual connection and layer norm

        trg = self.layer_norm(trg + self.drop_out(_trg))

        #trg = [batch_size, trg_len, hidden_size]

        #encoder attention

        _trg, attention = self.encoder_attention(trg,enc_src,enc_src,src_mask)

        #drop_out, residual connection and layer norm

        trg = self.layer_norm(trg + self.drop_out(_trg))

        #trg = [batch_size,trg_len,hidden_size]

        #positionwise feedforward

        trg = self.positionwise_feedforward(trg)

        #dropout, residual and layer norm

        trg = self.layer_norm(trg + self.drop_out(_trg))

        #trg = [batch_size,trg_len,hidden_size]
        #attention = [batch_size,n_heads,trg_len,src_len]

        return trg,attention

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        
        #trg_pad_mask = [batch size, 1, trg len, 1]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights);

LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
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

            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
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
        torch.save(model.state_dict(), 'tut6-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut6-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    
    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention

def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

example_idx = 8

src = vars(train_data.examples[example_idx])['src']
trg = vars(train_data.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')

translation, attention = translate_sentence(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')

display_attention(src, translation, attention)

