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
# activate the GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# Read config file
if len(sys.argv) < 0:
    print ("missing configuration file path")

config_path = sys.argv[1]
config = ConfigParser()
config.read(config_path)    
train_csv_path=config['variables']['train_csv_path']
test_csv_path=config['variables']['valid_csv_path']
device_path=config['variables']['device_path']


# training data generation for all tasks
train_data_action = sp_1.speechdata_action(train_csv_path,device_path)
trainloader_action = torch.utils.data.DataLoader(train_data_action, batch_size=32,
                                          shuffle=True) 
# validation data generation for all tasks
test_data_action = sp_1.speechdata_action(test_csv_path,device_path)
testloader_action = torch.utils.data.DataLoader(test_data_action, batch_size=1,shuffle=False)

# load the CNN models
model_action = cnn_1.cnn_action().to(device)
# activte the optimizer
optimizer_action = util.get_optimizer(model_action, lr=0.001)


criterion = nn.CrossEntropyLoss()
curr_lr = 0.0001
# load the model save path
model_save_path=config['variables']['model_save_path']
min_F1_action=0
 # Start the training for 1 epoch
for epoch in range(1,2):
    pred_action,lab_action=util.train_action(epoch,model_action,trainloader_action,criterion, testloader_action,optimizer_action)
    F1_score_action=f1_score(lab_action,pred_action,average='macro')
    if (min_F1_action<F1_score_action):
        min_F1_action=F1_score_action
        torch.save(model_action.state_dict(),model_save_path + str(epoch) + '_action.pth')

    
