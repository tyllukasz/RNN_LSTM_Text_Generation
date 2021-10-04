#imports
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time

from LSTM_model import CharModel #LSTM neural network
from functions import generate_batches, one_hot_encoder, predict_next_char, generate_text
from text_input import all_characters, model_name


#===================================================================================================
#text import
#with open('shakespeare.txt','r',encoding='utf8') as f:
    #text = f.read()

#all_characters = set(text) #all unique characters

#===================================================================================================

model = CharModel(
    all_chars=all_characters,
    num_hidden=512,
    num_layers=3,
    drop_prob=0.5,
    use_gpu=True,
)


model_name = str(model_name + 'pt')

model.load_state_dict(torch.load(model_name))

model.eval()

print(generate_text(model,1000,seed='The ',k=3))