import torch.nn.functional as F
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from torch import autograd
from tqdm import tqdm
from torch import nn
import numpy as np
import subprocess
import warnings
import torch
import wandb
import os
import nltk
import string
from nltk.stem.lancaster import LancasterStemmer
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import timeit
import pickle

PATH = os.path.dirname(os.path.realpath(__file__))

def mkdr(proj, proj_dir, Training, n=0):
    '''
    Make project directories
    Params:
        proj: string, project name
        proj_dir: string, project directory
        Training: bool
    Return:
        string of where the project information will be stored
    '''
    if Training:
        proj = proj + '_' + str(n)
        pth = proj_dir + proj
        try:
            os.makedirs(pth)
            return pth + '/' + proj
        except:
            pth = mkdr(proj[:-len('_' + str(n))], proj_dir, Training, n+1)
            return pth
    else:
        pth = proj_dir + proj
        return pth + '/' + proj

def LemTokens(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

def bag_of_empty(data):
    """
    input: dictionary of all data
    return: dictionary of bag of empty words
    """
    word_dic = {}
    all_keys = list(data)
    for key in all_keys:
        word_tokens = nltk.word_tokenize(data[key])
        for word in word_tokens:
            if word not in word_dic.keys():
                word_dic[word] = 0
    return word_dic

def one_hot(sentence, word_dic):
    """
    input: sentence tokens, empty bag of words
    return: numpy array of dimension (n_words)
    """
    all_words = list(word_dic)
    sentence_array = np.zeros(len(all_words))
    word_tokens = nltk.word_tokenize(sentence)
    # ignore repeated words in the same sentence for now
    for i in range(len(list(word_dic))):
        if all_words[i] in word_tokens:
            sentence_array[i] = 1
    return sentence_array

def tokenization(data):
    """
    input: dictionary of all wiki pages
    """
    # tokenisation
    all_pages = list(data)
    sentence_tokens = []
    word_tokens = []
    for page in all_pages:
        sentence_tokens.extend(nltk.sent_tokenize(data[page]))
        word_tokens.extend(nltk.word_tokenize(data[page]))
    return sentence_tokens, word_tokens
    
def greet(sentence):
    greet_inputs = ("hello", "hi", "hey", "whassup", "how are you?")
    greet_responses = ("hello", "hi", "Hey", "Hi there!", "Hey there!")
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)
        
def train(data, path):
    start = timeit.default_timer()
    sentence_tokens, word_tokens = tokenization(data)
    f = open(path+"sentence_tokens.pickle", 'wb')
    pickle.dump(sentence_tokens, f) 
    f.close()
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf_train = TfidfVec.fit_transform(sentence_tokens)
    # train timing
    stop = timeit.default_timer()
    print ("Training Time : ")
    print (stop - start) 
    f = open(path+"vector.pickle", 'wb')
    pickle.dump(TfidfVec, f) 
    f.close()
    f = open(path+"matrix.pickle", 'wb')
    pickle.dump(tfidf_train, f) 
    f.close()
        
def response(user_input, path, sentence_tokens, tdif=True):
    robot_response = ''
    if tdif == True:
        # load trained data
        f = open(path+'vector.pickle','rb')
        tfidf_vec = pickle.load(f)
        f = open(path+'matrix.pickle', 'rb')
        tfidf_matrix_train = pickle.load(f)

        user_tokens = nltk.sent_tokenize(user_input)
        tfidf_matrix_test = tfidf_vec.transform(user_tokens)

        # run cosine similarity between the 2 tf-idfs
        vals = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)
        # -1 is user_input
        max_val = vals.max()
        flat = vals.flatten()
    if (max_val <= 0.3):
        robot_response += "Sorry, I can't understand you."
    else:
        threshould = max_val - 0.05
        n_sentences = np.count_nonzero(flat > threshould)
        idx_list = vals.argsort()[0][-2-n_sentences:-2]
        # avoid the same answer as the question
        answer_found = False
        while not answer_found:
            idx = random.choice(idx_list)
            # add ending punctuations if necessary
            sentence = sentence_tokens[idx]
            if user_input != sentence:
                answer_found = True
        sentence_end = sentence[-3:]
        if "." not in sentence_end and "!" not in sentence_end and "?" not in sentence_end:
            sentence += "."
        robot_response += sentence
    return robot_response

# def param_init(layer):
#     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
#         torch.nn.init.normal_(layer.weight, 0.0, 0.02)
#     if isinstance(layer, nn.BatchNorm2d):
#         torch.nn.init.normal_(layer.weight, 0.0, 0.02)
#         torch.nn.init.constant_(layer.bias, 0)

# def calc_gradient_penalty(netD, real_data, fake_data, batch_size, img_length, device, gp_lambda, n_channels, lables):
#     alpha = torch.rand(batch_size, 1)
#     alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
#     alpha = alpha.view(batch_size, n_channels, img_length, img_length)
#     alpha = alpha.to(device)

#     interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
#     interpolates = interpolates.to(device)
#     interpolates.requires_grad_(True)

#     disc_interpolates = netD(interpolates, lables)
#     gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                                   grad_outputs=torch.ones(disc_interpolates.size()).to(device),
#                                   create_graph=True, only_inputs=True)[0]

#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
#     return gradient_penalty

# def test(path, labels, netG, n_classes, z_dim=64, lf=4, device='cpu', ratio=2):
#     '''
#     Generate test images for the desired label
#     Params:
#         path: string, path to directory where the generator is stored
#         labels: integer list, representing labels
#         netG: empty generator class
#         n_classes: integer
#         z_dim: integer
#         lf: integer, spatial dimension of input seed
#         device: string
#         ratio: integer, length/width ratio for generated image
#     Return:
#         tifs: numpy array of images
#         netG: imported generator class
#     '''
#     try:
#         netG.load_state_dict(torch.load(path + '_Gen.pt'))
#     except:
#         netG = nn.DataParallel(netG)
#         netG.load_state_dict(torch.load(path + '_Gen.pt'))
    
#     netG.to(device)
#     names = ['forest', 'desert', 'sea', 'star']
#     tifs, raws = [], []
#     # try to generate rectangular, instead of square images
#     random = torch.randn(1, z_dim, lf, lf*ratio-2, device=device)
#     noise = torch.zeros((1, z_dim, lf, lf*ratio)).to(device)
#     for idx0 in range(random.shape[0]):
#         for idx1 in range(random.shape[1]):
#             for idx2 in range(random.shape[2]):
#                 dim2 = random[idx0, idx1, idx2]
#                 noise[idx0, idx1, idx2] = torch.cat((dim2, dim2[:2]), -1)
#     netG.eval()
#     test_labels = gen_labels(labels, n_classes)[:, :, None, None]
#     for i in range(len(labels)):
#         lbl = test_labels[i].repeat(1, 1, lf, lf*ratio).to(device)
#         with torch.no_grad():
#             img = netG(noise, lbl, Training=False, ratio=ratio).cuda()
#             raws.append(img)
#         print('Postprocessing')
#         tif = torch.multiply(img, 255).cpu().detach().numpy()
#         try:
#             name = names[i]
#         except:
#             name = 'none'
#         tifffile.imwrite(path + '_' + name + '.tif', tif)
#         tifs.append(tif)
#     return tifs, netG

# def calc_eta(steps, time, start, i, epoch, num_epochs):
#     elap = time - start
#     progress = epoch * steps + i + 1
#     rem = num_epochs * steps - progress
#     ETA = rem / progress * elap
#     hrs = int(ETA / 3600)
#     mins = int((ETA / 3600 % 1) * 60)
#     print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
#           % (epoch, num_epochs, i, steps,
#              hrs, mins))