import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import string
import tflearn
import torch
import random
import wikipedia
import wikipediaapi
import json
import pickle
import sklearn
import os

PATH = os.path.dirname(os.path.realpath(__file__))
PATH += "/"

# load intents.json
with open('intents.json') as intents:
    data = json.load(intents)

stemmer = LancasterStemmer()

# getting information from intents.json
words = []
labels = []
x_docs = []
y_docs = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        x_docs.append(wrds)
        y_docs.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

### PREPROCESSING
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# one-hot encoding and preparing training data
training = []
output = []
out_empty = [0 for _ in range(len(labels))]

# one hot encoding, converting the words to numerals
for x, doc in enumerate(x_docs):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(y_docs[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

### Training the neural network

