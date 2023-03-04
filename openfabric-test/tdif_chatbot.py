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
    wiki_data = pickle_access(PATH+'code_files/wiki_data_full.pickle')
    train(wiki_data, Proj_path)

sentence_tokens = pickle_access(Proj_path+'sentence_tokens.pickle')
NN = basic_nn()
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")

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
                vals = test(Proj_path, NN, input_dim=1024, device=device, question=user_input)
                print(f"Bot: {response(user_input, sentence_tokens, vals)}")
    else:
        flag = False
        print("Bot: Bye!")
    
