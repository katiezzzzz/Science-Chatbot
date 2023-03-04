from code_files import *
import numpy as np
import os
import pickle

PATH = os.path.dirname(os.path.realpath(__file__))
PATH += "/"

print("initialising...")
Training = False
Project_name = 'tdif_0'
Project_dir = PATH + 'trained/'
Proj_path = mkdr(Project_name, Project_dir, Training)



if Training == True:
    file = open(PATH+'code_files/wiki_data_full.pickle','rb')
    wiki_data = pickle.load(file)
    train(wiki_data, Proj_path)

file = open(Proj_path+'sentence_tokens.pickle','rb')
sentence_tokens = pickle.load(file)

flag = True
print("Hi, I'm science robot. You can ask me one scientific question at a time! Please say goodbye before leaving!")
while (flag == True):
    user_input = input()
    user_input = user_input.lower()
    if (user_input != 'goodbye' and user_input != 'bye'):
        if(user_input == 'thanks' or user_input == 'thank you'):
            flag = False
            print("Bot: You are welcome!")
        else:
            if(greet(user_input) != None):
                print(f"Bot: {greet(user_input)}")
            else:
                print(f"Bot: {response(user_input, Proj_path, sentence_tokens)}")
    else:
        flag = False
        print("Bot: Bye!")
    
