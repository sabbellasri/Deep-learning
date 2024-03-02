
# coding: utf-8

# In[1]:


import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import re
import json
import os
import sys
from scipy.special import expit
import random
import numpy as np
from torch.autograd import Variable





device = 0

def data_preprocess():
    filepath = 'data/'
    with open(filepath + 'training_label.json', 'r') as f:
        file = json.load(f)

    wo_co = {}
    for d in file:
        for s in d['caption']:
            wor_sen = re.sub('[.!,;?]', ' ', s).split()
            for word in wor_sen:
                word = word.replace('.', '') if '.' in word else word
                if word in wo_co:
                    wo_co[word] += 1
                else:
                    wo_co[word] = 1

    word_dict = {}
    for word in wo_co:
        if wo_co[word] > 4:
            word_dict[word] = wo_co[word]

    use_tok = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {}
    w2i = {}
    for i, w in enumerate(word_dict):
        i2w[i + len(use_tok)] = w
        w2i[w] = i + len(use_tok)

    for token, index in use_tok:
        i2w[index] = token
        w2i[token] = index

    return i2w, w2i, word_dict


def s_split(sentence, word_dict, w2i):
    sentence = [w2i[word] if word in word_dict else 3 for word in re.sub(r'[.!,;?]', ' ', sentence).split()]
    sentence.insert(0, 1)  # Adding SOS token at the beginning
    sentence.append(2)  # Adding EOS token at the end
    return sentence






def annotate(label_file, word_dict, w2i):
    lab_json = 'data/' + label_file
    caption = []
    with open(lab_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = s_split(s, word_dict, w2i)
            caption.append((d['id'], s))
    return caption




def avi(files_dir):
    avi_data = {}
    training_feats = 'data/' + files_dir
    files = os.listdir(training_feats)
    
    for i, file in enumerate(files):
        print(i)
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    
    return avi_data






def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths





class training_data(Dataset):
    def __init__(self, label_file, files_dir, word_dict, w2i):
        self.label_file = label_file
        self.word_dict = word_dict
        self.files_dir = files_dir
        self.avi = avi(label_file)
        self.w2i = w2i
        self.data_pair = annotate(files_dir, word_dict, w2i)
        
    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(sentence)





class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            if file.endswith('.npy'):
                key = file.split('.npy')[0]
                value = np.load(os.path.join(test_data_path, file))
                self.avi.append([key, value])
            
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]



class lstm_attn_decode(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.35):
        super(lstm_attn_decode, self).__init__()

        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(0.35)
        self.lstm = nn.LSTM(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_cxt = torch.zeros(decoder_current_hidden_state.size())#.to(device)

        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()#.to(device)
        seq_logProb = []
        seq_predictions = []

        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        i = 0
        while i < seq_len - 1:
            threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
            if random.uniform(0.05, 0.995) > threshold: 
                current_input_word = targets[:, i]  
            else: 
                current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, t = self.lstm(lstm_input, (decoder_current_hidden_state, decoder_cxt))
            decoder_current_hidden_state = t[0]
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]
            
            i += 1

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

        
    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_c = torch.zeros(decoder_current_hidden_state.size())
        sequence_logProb = []  
        sequence_predictions = []  
        assumption_seq_length = 28  
        
        for i in range(assumption_seq_length-1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, t = self.lstm(lstm_input, (decoder_current_hidden_state, decoder_c))
            decoder_current_hidden_state = t[0]
            logprob = self.to_final_output(lstm_output.squeeze(1))
            sequence_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        sequence_logProb = torch.cat(sequence_logProb, dim=1) 
        sequence_predictions = sequence_logProb.max(2)[1]
        return sequence_logProb, sequence_predictions


    def teacher_forcing_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85)) 
    
class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.dense1 = nn.Linear(2*hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size)
        self.dense4 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.dense1(matching_inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context





class lstm_encoder(nn.Module):
    def __init__(self):
        super(lstm_encoder, self).__init__()
        
        self.Embedding = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.35)
        self.lstm = nn.LSTM(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()    
        input = input.view(-1, feat_n)
        input = self.Embedding(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)

        output, t = self.lstm(input)
        hidden_state, context = t[0], t[1]
        return output, hidden_state


class diff_models(nn.Module):
    def __init__(self, encoder, decoder):
        super(diff_models, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, avi_feat, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feat)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(encoder_last_hidden_state = encoder_last_hidden_state, encoder_output = encoder_outputs,
                targets = target_sentences, mode = mode, tr_steps=tr_steps)
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs)
        return seq_logProb, seq_predictions




def calculate_loss(loss_fn, x, y, lengths):
    batch_size = len(x)
    pred_cat = None
    gr_cat = None
    flag_check = True

    for batch in range(batch_size):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] - 1

        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag_check:
            pred_cat = predict
            gr_cat = ground_truth
            flag_check = False
        else:
            pred_cat = torch.cat((pred_cat, predict), dim=0)
            gr_cat = torch.cat((gr_cat, ground_truth), dim=0)

    loss = loss_fn(pred_cat, gr_cat)
    avg_loss = loss / batch_size

    return loss






def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    
    for i in range(len(captions)):
        cap = captions[i]
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return avi_data, targets, lengths





def train(model, epoch, loss_fn, parameters, optimizer, train_loader):
    model.train()
    print(epoch)

    for batch_idx, batch in enumerate(train_loader):
        avi_feats, ground_truths, lengths = batch
        #avi_feats, ground_truths = Variable(avi_feats).cuda(), Variable(ground_truths).cuda()
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
        ground_truths = ground_truths[:, 1:]
        loss = calculate_loss(loss_fn, seq_logProb, ground_truths, lengths)
        
        print('Epoch - {}, Batch - {}, Loss - {}'.format(epoch, batch_idx, loss.item()))
        
        loss.backward()
        optimizer.step()

    return loss.item()




def test(test_loader, model, i2w):
    model.eval()
    ss = []
    
    for batch_idx, batch in enumerate(test_loader):
        id, avi_feats = batch
        avi_feats = avi_feats#.to(device)
        id, avi_feats = id, Variable(avi_feats).float()

        seq_logProb, seq_predictions = model(avi_feats, mode='inference')
        test_predictions = seq_predictions

        result = []
        for s in test_predictions:
            inner_result = []
            for x in s:
                if i2w[x.item()] != '<UNK>':
                    inner_result.append(i2w[x.item()])
                else:
                    inner_result.append('something')
            result.append(inner_result)

        result = [' '.join(s).split('<EOS>')[0] for s in result]
        rr = zip(id, result)
        for r in rr:
            ss.append(r)
    
    return ss





def main():
    i2w, w2i, word_dict = data_preprocess()
    with open('i2w.pickle', 'wb') as handle:
        pickle.dump(i2w, handle, protocol = pickle.HIGHEST_PROTOCOL)
    labe_fil = '/training_data/feat'
    fi_dir = 'training_label.json'
    train_dataset = training_data(labe_fil, fi_dir, word_dict, w2i)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=64, shuffle=True, num_workers=8, collate_fn=minibatch)
    
    epochs_n = 20

    encoder = lstm_encoder()
    decoder = lstm_attn_decode(512, len(i2w) +4, len(i2w) +4, 1024, 0.35)
    model = diff_models(encoder=encoder, decoder=decoder)
    
    #model = model.cuda()
    #model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.0001)
    loss_arr = []
    for epoch in range(epochs_n):
        loss = train(model, epoch+1, loss_fn, parameters, optimizer, train_dataloader) 
        loss_arr.append(loss)
    
    with open('models/loss_values.txt', 'w') as f:
        for item in loss_arr:
            f.write("%s\n" % item)
    torch.save(model, "{}/{}.h5".format('models', 'model0'))
    print("Completed the training")
    
if __name__ == "__main__":
    main()
