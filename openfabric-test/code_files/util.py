from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import warnings
import string
import random
import timeit
import pickle
import torch
import json
import nltk
import os

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

def pickle_access(filename, mode='load', data=None):
    """
    input: string (filename), data, mode ('dump' or 'load')
    """
    if mode == 'dump':
        f = open(filename, 'wb')
        pickle.dump(data, f) 
        f.close()
    elif mode == 'load':
        f = open(filename, 'rb')
        data = pickle.load(f)
        f.close()
    return data
        
def preproc_wiki(data, path):
    """
    input: dictionary of wiki page contents
    return: sentence tokens, tfidvec, tfidmatrix
    """
    start = timeit.default_timer()
    sentence_tokens, word_tokens = tokenization(data)
    pickle_access(path+"sentence_tokens.pickle", mode='dump', data=sentence_tokens)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf_train = TfidfVec.fit_transform(sentence_tokens)
    # feature extraction timing
    stop = timeit.default_timer()
    print (f"Wiki preprocessing Time: {stop-start}")
    pickle_access(path+"vector.pickle", mode='dump', data=TfidfVec)
    pickle_access(path+"matrix.pickle", mode='dump', data=tfidf_train)
    return sentence_tokens, TfidfVec, tfidf_train

def tokenize_input(text):
    merged_text = text[:-1].replace('?', ',')
    merged_text = merged_text.replace('.', ',')
    merged_text = merged_text.replace('!', ',')
    merged_text += text[-1]
    merged_text.lower()
    text_tokens = nltk.sent_tokenize(merged_text)
    return text_tokens

def preproc_sciq(parent_path, proj_path):
    start = timeit.default_timer()
    with open(parent_path+'/SciQ_dataset/train.json') as f:
        sciq_train_data = json.load(f)
    question_tokens = []
    answer_tokens = []
    choice_tokens = []
    for i in range(len(sciq_train_data)):
        answer = sciq_train_data[i]['support']
        question_token = tokenize_input(sciq_train_data[i]['question'])
        if answer != "":
            answer_token = tokenize_input(answer)
            question_tokens.extend(question_token)
            answer_tokens.extend(answer_token)
    pickle_access(proj_path+"question_tokens_train.pickle", mode='dump', data=question_tokens)
    pickle_access(proj_path+"answer_tokens_train.pickle", mode='dump', data=answer_tokens)
    stop = timeit.default_timer()
    print (f"SciQ preprocessing Time: {stop-start}")
    return question_tokens, answer_tokens
    
def batch(data, parent_path, proj_path, batch_size, input_dim, init=True):
    if init == True:
        # preprocess wiki dataset
        sentence_tokens, TfidfVec, tfidf_wiki = preproc_wiki(data, proj_path)
        # proprocess sciQ dataset
        question_tokens, answer_tokens = preproc_sciq(parent_path, proj_path)
    else:
        TfidfVec = pickle_access(proj_path+"vector.pickle")
        tfidf_wiki = pickle_access(proj_path+"matrix.pickle")
        question_tokens = pickle_access(proj_path+"question_tokens_train.pickle")
        answer_tokens = pickle_access(proj_path+"answer_tokens_train.pickle")

    # generate sample
    idx_all = list(range(len(question_tokens)))
    idx_selected = random.sample(idx_all, batch_size)
    vals_questions = []
    vals_answers = [] 
    for idx in idx_selected:
        question_token = nltk.sent_tokenize(question_tokens[idx])
        answer_token = nltk.sent_tokenize(answer_tokens[idx])

        tfidf_question = TfidfVec.transform(question_token)
        tfidf_answer = TfidfVec.transform(answer_token)
        # run cosine similarity between the 2 tf-idfs
        vals_question = cosine_similarity(tfidf_question, tfidf_wiki)
        vals_answer = cosine_similarity(tfidf_answer, tfidf_wiki)
        # extract top 300 features
        vals_idx_list = vals_question.argsort()[0][-2-input_dim:-2]

        vals_question_sel = np.array([])
        vals_answer_sel = np.array([])
        for vals_idx in vals_idx_list:
            vals_question_sel = np.append(vals_question_sel, vals_question[0][vals_idx])
            vals_answer_sel = np.append(vals_answer_sel, vals_answer[0][vals_idx])
        vals_question_sel = vals_question_sel.reshape(1, input_dim)
        vals_answer_sel = vals_answer_sel.reshape(1, input_dim)

        if vals_questions == [] or vals_answers == []:
            vals_questions = vals_question_sel
            vals_answers = vals_answer_sel
        else:
            vals_questions = np.append(vals_questions, vals_question_sel, axis=0)
            vals_answers = np.append(vals_answers, vals_answer_sel, axis=0)
    return torch.from_numpy(vals_questions).float(), torch.from_numpy(vals_answers).float()
            
def response(user_input, sentence_tokens, vals):
    robot_response = ''
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

def param_init(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.uniform_(layer.weight)
        layer.bias.data.fill_(0.01)
    if isinstance(layer, nn.BatchNorm1d):
        torch.nn.init.uniform_(layer.weight)
        torch.nn.init.constant_(layer.bias, 0)

def batch_accuracy(preds, labels):
    """
    Returns accuracy per batch
    """
    point_accs = np.array([])
    for point in range(len(preds)):
        pred_array = preds[point].detach().numpy()
        label_array = labels[point].detach().numpy()
        difference = np.absolute(label_array - pred_array).flatten()
        # considered correct prediction if difference < 0.02
        n_correct = (difference < 0.02).sum()
        point_acc = n_correct/len(difference)
        point_accs = np.append(point_accs, point_acc)
    return np.mean(point_accs)

def test(proj_path, nn, input_dim, device, question):
    NN = nn(input_dim)
    try:
        NN.load_state_dict(torch.load(proj_path+'_net.pt'))
    except:
        NN = nn.DataParallel(NN)
        NN.load_state_dict(torch.load(proj_path+'_net.pt'), False)

    NN.to(device)

    # generate test input
    TfidfVec = pickle_access(proj_path+"vector.pickle")
    tfidf_wiki = pickle_access(proj_path+"matrix.pickle")
    tfidf_question = TfidfVec.transform(nltk.sent_tokenize(question))
    vals_question = cosine_similarity(tfidf_question, tfidf_wiki)
    # feature extraction
    vals_idx_list = vals_question.argsort()[0][-2-input_dim:-2]

    vals_question_sel = np.array([])
    for vals_idx in vals_idx_list:
        vals_question_sel = np.append(vals_question_sel, vals_question[0][vals_idx])
    vals_question_sel = vals_question_sel.reshape(1, input_dim)

    NN.eval()
    with torch.no_grad():
        pred = NN(torch.from_numpy(vals_question_sel).float())

    # post processing
    pred = pred.detach().numpy()
    for i in range(len(vals_idx_list)):
        vals_question[0][vals_idx_list[i]] = pred[0][i]
    return vals_question

def calc_eta(steps, time, start, i, epoch, num_epochs):
    elap = time - start
    progress = epoch * steps + i + 1
    rem = num_epochs * steps - progress
    ETA = rem / progress * elap
    hrs = int(ETA / 3600)
    mins = int((ETA / 3600 % 1) * 60)
    print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
          % (epoch, num_epochs, i, steps,
             hrs, mins))