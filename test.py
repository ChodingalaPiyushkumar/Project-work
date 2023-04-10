import numpy as np
import pandas as pd
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import wave, os, glob
import torch.utils.data
import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import shutil
import scipy.io as io
import os, sys
from scipy.io import savemat, wavfile
from hdf5storage import loadmat
from torch.autograd import Variable
from torch.utils import data
from torch.nn.init import xavier_normal_
from python_speech_features import mfcc, delta, logfbank
from sklearn.metrics import f1_score
from configparser import ConfigParser
import sys
from helpfiles import util
from helpfiles import cnn_action as cnn_1
from helpfiles import speechdata_action as sp_1
# read the config files
if len(sys.argv) < 0:
    print ("missing configuration file path")

config_path = sys.argv[1]
config = ConfigParser()
config.read(config_path)    
test_csv_path=config['variables']['test_csv_path']
device_path=config['variables']['device_path']
model_load_path=config['variables']['model_save_path']

# activate the GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# Test data preparation for all tasks
test_data_action = sp_1.speechdata_action(test_csv_path,device_path)
testloader_action = torch.utils.data.DataLoader(test_data_action, batch_size=1,shuffle=False)


# Load the CNN models
action_model=config['models']['action_model']
# Load the trained model path
path_action=model_load_path + action_model


# Loal the trained model for all tasks
model_action = cnn_1.cnn_action().to(device)
model_action.load_state_dict(torch.load(path_action))
model_action.eval()


# testing scripts
def test_action():
    model_action.eval()
    pred_action = []
    lab_action =[]
    for data, devtarget in testloader_action:
        data = data.unsqueeze_(1)
        data, target = Variable(data).float().to(device), Variable(devtarget).to(device)
        output = model_action(data).to(device)
        _,predicted = torch.max(output.data,1)
        predicted = predicted.cpu().numpy() 
        target=target.cpu().data.numpy()
        pred_action.append(predicted)
        lab_action.append(target)
    return pred_action,lab_action

# Begin the testing
pred_action,lab_action=test_action()
# Calculation of F1 scores
F1_score_action=f1_score(lab_action,pred_action,average='macro')

print('action_F1_score = {}'.formatF1_score_action)
