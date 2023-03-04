from code_files.util import *
from torch import nn
import numpy as np
import torch
import time

def train(parent_path, proj_path, NN, wiki_data, num_epochs, batch_size, input_dim, lr, device):

    rt = False
    iters = 100 // batch_size
    print(device, " will be used.\n")

    net = NN(input_dim).to(device)
    # if rt:
    #     nn.load_state_dict(torch.load(proj_path+'_nn.pt'))
    # else:
    #     net = nn.apply(param_init)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    print("Starting Training Loop...")

    iter_loss = 0 
    iter_acc = 0  
    net.train()

    # init
    labels, features = batch(wiki_data, parent_path, proj_path, batch_size, input_dim)

    start = time.time()
    for epoch in range(num_epochs):
        for i in range(iters):
            labels, features = batch(wiki_data, parent_path, proj_path, batch_size, input_dim, init=False)
            opt.zero_grad()                                         # zero gradients
            preds = net(features)
            loss = criterion(preds, labels)                    # compute loss
            acc = batch_accuracy(preds, labels)               # compute accuracy of predictions       
            loss.backward()                                               # find rate of change of loss wrt model weights
        
            opt.step()                                              # update the model weights
        
            iter_loss += loss.item()                                     # accumulate loss
            iter_acc += acc.item()   
            calc_eta(iters, time.time(), start, i, epoch, num_epochs)

        print(f"iteration loss: {iter_loss/iters}, iteration accuracy: {iter_acc/iters}")
        torch.save(net.state_dict(), proj_path + '_net.pt')



    
