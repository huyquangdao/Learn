
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
            include_lengths = True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

BATCH_SIZE = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data,valid_data,test_data),
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src),
    device = device
)

class Encoder(nn.Module):

    def __init__(self,vocab_size,embedding_dim,enc_hidden_size,dec_hidden_size,drop_out):

        super(Encoder,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.gru = nn.GRU(embedding_dim,enc_hidden_size,bidirectional=True)
        self.fc = nn.Linear(enc_hidden_size *2, dec_hidden_size)
        self.drop_out = nn.Dropout(drop_out)
    
    def forward(self, src, src_len):

        # src = [src_len, batch_size]
        # src_len = [src_len]

        embedding_tensor = self.drop_out(self.embedding(src))

        #embedding_tensor = [src_len, batch_size, embedding_dim]

        packed_embedding_tensor = nn.utils.rnn.pack_padded_sequence(embedding_tensor,src_len)

        packed_outputs, hidden = self.gru(packed_embedding_tensor)

        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        #outputs is now a non-packed sequence, all hidden states obtained
        #when the input is a pad token are all zeros

        #outputs = =[src_len, batch_size, hidden_size * num_direction]
        #hidden = [n layers * num directions, batch size, hid dim]

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer

        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN

        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer

        hidden = torch.tanh(self.fc(torch.cat([hidden[-2,:,:],hidden[-1,:,:]],dim=1)))

        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]

        return outputs, hidden

class Attention(nn.Module):

    def __init__(self,enc_hidden_size,dec_hidden_size):

        super(Attention,self).__init__()
        self.atnn = nn.Linear((enc_hidden_size *2)+dec_hidden_size,dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size,1,bias=False)
    
    def forward(self,hidden,encoder_outputs,mask):

        #hidden = [batch_size,dec_hidden_size]
        #encoder_outputs = [src_len, batch_size, enc_hidden_size * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        #repat decoder hidden state srec_len times

        hidden = hidden.unsqueeze(1).repeat(1,src_len,1)
        encoder_outputs = encoder_outputs.permute(1,0,2)

        # hidden = [batch_size, src_len, dec_hidden_size]
        # encoder_outputs = [batch_size, src_len, enc_hidden_size*2]

        energy = torch.tanh(self.atnn(torch.cat([hidden,encoder_outputs],dim=2)))

        #energy = [batch_size, src_len, dec_hidden_size]

        attention = self.v(energy).squeeze(2)

        #attention = [batch_size,src_len]

        attention = attention.masked_fill(mask==0,1e-10)

        return F.softmax(attention,dim=1)

class Decoder(nn.Module):


      def __init__(self, vocab_size, embedding_dim, enc_hidden_size, dec_hidden_size, drop_out, attention):

          super(Decoder,self).__init__()
          self.vocab_size = vocab_size
          self.attention = attention

          self.embedding = nn.Embedding(vocab_size,embedding_dim)
          self.gru = nn.GRU((enc_hidden_size * 2) + embedding_dim, dec_hidden_size)

          self.fc_out = nn.Linear(((enc_hidden_size * 2) + dec_hidden_size + embedding_dim), vocab_size)

          self.drop_out = nn.Dropout(drop_out)

      def forward(self, input, hidden, encoder_outputs, mask):
          
        #input = [batch size]
        #hidden = [batch size, dec hidden size]
        #encoder_outputs = [src len, batch size, enc_hidden_size * 2]
        #mask = [batch size, src len]

        input = input.unsqueeze(0)

        #input = [1,batch_size]

        embedding_tensor = self.drop_out(self.embedding(input))

        # embedding_tensor = [1, batch_size, embedding_dim]

        a = self.attention(hidden,encoder_outputs, mask)

        # a = [batch_size, src_len]
        
        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1,0,2)

        # encoder_outputs = [batch_size, src_len, enc_hidden_size * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch_size , 1, enc_hidden_size *2]

        weighted = weighted.permute(1,0,2)

        #weighted = [1, batch_size, enc_hidden_size * 2]

        rnn_input = torch.cat((embedding_tensor,weighted), dim =2)

        #rnn_input = [1, batch_size,(enc_hidden_size)*2 + embedding_dim]

        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))

        #output = [seq len, batch size, dec hidden size * n directions]
        #hidden = [n layers * n directions, batch size, dec hid size]

        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hidden_size]
        #hidden = [1, batch size, dec hidden_size]
        #this also means that output == hidden

        assert (output == hidden).all()

        embedding_tensor = embedding_tensor.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat([output,weighted, embedding_tensor],dim=1))

        #prediction = [batch_size, vocab_size]

        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, src_pad_idx, device):

        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self,src):
        mask = (src != self.src_pad_idx).permute(1,0)
        return mask
    
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):

        #src = [src len, batch size]
        #src_len = [batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size

        #tensor to store decoder outputs

        outputs = torch.zeros(trg_len,batch_size,trg_vocab_size).to(self.device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer

        encoder_outputs, hidden = self.encoder(src, src_len)

        #first input to decoder is the <sos> tokens

        input = trg[0,:]

        mask = self.create_mask(src)

        #mask =[batch_size, src_len]

        for t in range(1,trg_len):

            #insert input token embedding, previous hidden state, all encoder hidden state
             #  and mask
            #receive output tensor (predictions) and new hidden state

            output, hidden, _ = self.decoder(input, hidden, encoder_outputs,mask)

            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
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
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

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
        
        src, src_len = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, src_len, trg)
        
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

            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, 0) #turn off teacher forcing
            
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
        torch.save(model.state_dict(), 'tut4-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut4-model.pt'))

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
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)]).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)
        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    
    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention
            
        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]

def display_attention(sentence, translation, attention):
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    attention = attention.squeeze(1).cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='bone')
   
    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                       rotation=45)
    ax.set_yticklabels(['']+translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

example_idx = 12

src = vars(train_data.examples[example_idx])['src']
trg = vars(train_data.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')

translation, attention = translate_sentence(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')

display_attention(src, translation, attention)

example_idx = 14

src = vars(valid_data.examples[example_idx])['src']
trg = vars(valid_data.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')

translation, attention = translate_sentence(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')

display_attention(src, translation, attention)

example_idx = 18

src = vars(test_data.examples[example_idx])['src']
trg = vars(test_data.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')

translation, attention = translate_sentence(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')

display_attention(src, translation, attention)

from torchtext.data.metrics import bleu_score

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)

