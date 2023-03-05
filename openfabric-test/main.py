import os
import warnings
import numpy as np
from code_files import *
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    PATH = os.path.dirname(os.path.realpath(__file__))
    PATH += "/"

    ngpu = 1
    Training = False
    Project_name = 'tdif_0'
    Project_dir = PATH + 'trained/'
    Proj_path = mkdr(Project_name, Project_dir, Training)

    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    sentence_tokens = pickle_access(Proj_path+'sentence_tokens.pickle')
    NN = basic_nn()
    
    output = []
    for text in request.text:
        response = ''
        user_input = text.lower()
        if (user_input != 'goodbye' and user_input != 'bye'):
            if(user_input == 'thanks' or user_input == 'thank you'):
                response += "Bot: You are welcome!"
            else:
                if(greet(user_input) != None):
                    response += f"Bot: {greet(user_input)}"
                else:
                    vals = test(Proj_path, NN, input_dim=1024, device=device, question=user_input)
                    response += f"Bot: {response(user_input, sentence_tokens, vals)}"
        else:
            response += "bye!"
        output.append(response)

    return SimpleText(dict(text=output))
