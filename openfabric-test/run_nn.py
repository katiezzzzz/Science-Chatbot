"""
train a basic nn using tdif features
"""

from code_files import *
import numpy as np
import os

PATH = os.path.dirname(os.path.realpath(__file__))
PATH += "/"

Training = True
Project_name = 'tdif'
Project_dir = PATH + 'trained/'
Proj_path = mkdr(Project_name, Project_dir, Training)

# import dataset
wiki_data = pickle_access(PATH+'code_files/wiki_data_full.pickle')
sentence_tokens, word_tokens = tokenization(wiki_data)

# define hyperparameters and architecture
ngpu = 1
lr = 0.0002
batch_size = 5
num_epochs = 10

device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")

# create networks
nn = basic_nn()

# train
if Training:
    train(PATH, Proj_path, nn, wiki_data, num_epochs, batch_size, input_dim=1024, lr=lr, device=device)