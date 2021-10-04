#imports
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from LSTM_model import CharModel #LSTM neural network
from functions import generate_batches, one_hot_encoder
from text_input import all_characters, encoded_text, model_name

#===================================================================================================
#text import
#with open('shakespeare.txt','r',encoding='utf8') as f:
    #text = f.read()

#all_characters = set(text) #all unique characters
#decoder = dict(enumerate(all_characters)) #decoder
#encoder = {char: ind for ind,char in decoder.items()} #encoder

#encoded_text = np.array([encoder[char] for char in text])

#===================================================================================================
#train / test split
train_percent = 0.6
train_ind = int(len(encoded_text) * train_percent)
train_data = encoded_text[:train_ind]
val_data = encoded_text[train_ind:]

#===================================================================================================
#model initialization
model = CharModel(all_chars=all_characters,
                 num_hidden=512,
                 num_layers=3,
                 drop_prob=0.5,
                 use_gpu=True)

#training parameters
epochs = 20
batch_size = 64
seq_len = 100
tracker = 0
num_char = max(encoded_text) + 1

#===================================================================================================
#model name
model_name = str(model_name + 'pt')
model.load_state_dict(torch.load(model_name))

model.train()

if model.use_gpu:
    model.cuda()


#optimizer and criterion definition
optimizer = torch.optim.Adam(model.parameters(),lr=0.00001)
criterion = nn.CrossEntropyLoss()
    
for i in range(epochs):
    
    hidden = model.hidden_state(batch_size)
    
    for x,y in generate_batches(train_data,batch_size,seq_len):
        
        tracker += 1
        
        x = one_hot_encoder(x,num_char)
        
        inputs = torch.from_numpy(x)
        targets = torch.from_numpy(y)
        
        
        
        if model.use_gpu:
            
            inputs = inputs.cuda()
            targets = targets.cuda()
        
        
        hidden = tuple([state.data for state in hidden])
        
        model.zero_grad()
        
        lstm_output,hidden = model.forward(inputs,hidden)
        
        loss = criterion(lstm_output,targets.view(batch_size*seq_len).long())
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
        
        optimizer.step()
        
        if tracker % 25 == 0:
            
            val_hidden = model.hidden_state(batch_size)
            val_losses = []
            model.eval()
            
            for x,y in generate_batches(val_data,batch_size,seq_len):
                
                x = one_hot_encoder(x,num_char)
        
                inputs = torch.from_numpy(x)
                targets = torch.from_numpy(y)
        
        
        
                if model.use_gpu:
            
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    
                val_hidden = tuple([state.data for state in val_hidden])
                
                lstm_output,val_hidden = model.forward(inputs,val_hidden)
                val_loss = criterion(lstm_output,targets.view(batch_size*seq_len).long())
                
                val_losses.append(val_loss.item())
                
            model.train()
            
            print(f'Epoch: {i} Step: {tracker} Val loss: {val_loss.item()}')

#===================================================================================================
#save the model
torch.save(model.state_dict(),model_name)            