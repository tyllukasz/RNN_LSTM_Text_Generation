import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

#one hot encoder
def one_hot_encoder(encoded_text , num_uni_chars):
    
    #encoded text - batch of encoded text
    #num_uni_chars - number of unique characters in whole text file
    
    one_hot = np.zeros((encoded_text.size , num_uni_chars)) #prepare array with correct dimensions
    
    one_hot = one_hot.astype(np.float32) #data type for PyTorch
    
    one_hot[np.arange(one_hot.shape[0]),encoded_text.flatten()] = 1.0 #put ones in the position which coresponds to encoded char value
    
    one_hot = one_hot.reshape(*encoded_text.shape,num_uni_chars)
    
    return one_hot


#batch generator
def generate_batches(encoded_text , sam_per_batch = 10 , seq_len=50):
    
    # X -> encoded text of length 'seq_len'
    # Y -> encoded text shifted by 1
    
    # how many characters per batch
    char_per_batch = sam_per_batch * seq_len
    
    # how many batches possible in entire text
    num_batches_avail = int(len(encoded_text)/char_per_batch)
    
    # cut off the end of the encoded text
    encoded_text = encoded_text[:num_batches_avail*char_per_batch]
    
    
    encoded_text = encoded_text.reshape(sam_per_batch,-1)
    
    for n in range(0,encoded_text.shape[1],seq_len):
        
        x = encoded_text[:,n:n+seq_len]
        
        y = np.zeros_like(x)
        
        try:
            
            y[:,:-1] = x[:,1:]
            y[:,-1] = encoded_text[:,n+seq_len]
            
        except:
            
            y[:,:-1] = x[:,1:]
            y[:,-1] = encoded_text[:,0]
            
        yield x,y         

#predict next character
def predict_next_char(model,char,hidden=None,k=1):
    
    encoded_text = model.encoder[char]
    
    encoded_text = np.array([[encoded_text]])
    
    encoded_text = one_hot_encoder(encoded_text,len(model.all_chars))
    
    inputs = torch.from_numpy(encoded_text)
    
    
    if model.use_gpu:
        inputs = inputs.cuda()
        
    hidden = tuple([state.data for state in hidden])
    
    lstm_out , hidden = model(inputs , hidden)
    
    probs = F.softmax(lstm_out,dim=1).data
    
    if model.use_gpu:
        
        probs = probs.cpu()
        
    
    probs, index_positions = probs.topk(k)
    
    index_positions = index_positions.numpy().squeeze()
    
    probs = probs.numpy().flatten()
    
    probs = probs/probs.sum()
    
    char = np.random.choice(index_positions,p=probs)
    
    return model.decoder[char] , hidden
    #return char , hidden
    

#text generator
def generate_text(model,size,seed='The',k=1):
    
    if  model.use_gpu:
        model.cuda()
        
    else:
        model.cpu()
        
    model.eval()
    
    output_chars = [c for c in seed]
    
    hidden = model.hidden_state(1)
    
    for char in seed:
        
        char,hidden = predict_next_char(model,char,hidden,k=k)
    
    output_chars.append(char)
    
    for i in range(size):
        
        char,hidden = predict_next_char(model,output_chars[-1],hidden,k=k)
        
        output_chars.append(char)
        
    return ''.join(output_chars)
