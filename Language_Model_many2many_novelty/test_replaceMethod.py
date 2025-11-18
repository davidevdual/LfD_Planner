import argparse
import time
import torch
import os
import numpy as np
import torch.utils.data.dataset as dtst
import csv
import random
import utils as utils
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
from torch import default_generator
from torch._utils import _accumulate
from torch import nn, optim
from datetime import datetime
from torch.utils.data import DataLoader
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple

def test_network(dataset,model,text):
    words = np.array(text.split(' ')).reshape(-1,1);
    batch_size = 1;
    state_h, state_c = model.init_state(batch_size);
    x = utils.pad_sentence(torch.tensor(dataset.compute_word_to_index(words)).type(torch.LongTensor).cpu(),dataset);
    model.eval();

    start = time.time();
    y_pred, (state_h, state_c) = model(x,(state_h,state_c));
    row_y_pred = y_pred.shape[0];col_y_pred = y_pred.shape[1];depth_y_pred = y_pred.shape[2];
    y_pred = torch.reshape(y_pred,(row_y_pred*col_y_pred,depth_y_pred));
    y_pred = torch.argmax(y_pred,dim=1);

    # The first seven terms are part of the goal. There is no need to predict them. We can replace whatever prediction with the ground-truth
    y_pred[0]=x[0,0];y_pred[1]=x[1,0];y_pred[2]=x[2,0];y_pred[3]=x[3,0];y_pred[4]=x[4,0];y_pred[5]=x[5,0];y_pred[6]=x[6,0];

    indices = torch.reshape(y_pred,(-1,1)).transpose(1,0);
    indices = dataset.convert_matrix_to_words(indices).tolist();# Returns an imbricated list
    indices = indices[0];# Extract the imbricated list
    end = time.time();
    execution_time = (end-start)*1000;
    return indices,execution_time;

# Default arguments. They will be changed later
parser = argparse.ArgumentParser();# Parentheses are very important after ArgumentParser
parser.add_argument('--max-epochs', type=int, default=300);#Original:100,200
parser.add_argument('--batch-size', type=int, default=300);
parser.add_argument('--sequence-length', type=int, default=7);
args = parser.parse_args();

# Load the Dataset
dataset = Dataset(args,os.path.join(os.path.dirname(__file__), 'data','tasks_test_goodorder_forDebug_uniqModel.csv'));
device = torch.device('cpu');
model = torch.load("C:/Users/David/ONEDRI~1/PhD_work/Code/TAMP_H~2/VISUAL~1/TASKPL~1/EXPERI~1/lstm/RE0325~1/MAX_EP~1.0/BATCH_~1.0/SEQUEN~1.0/LEARNI~1.005/LSTM_S~1.0/EMBEDD~1.0/NUM_LA~1.0/DROPOU~1.0/TRAINE~1/TAMP_12152022_13-37-44_nbepoch1199_epochs1200_batch200_length7_rate0.005.pth").cpu();
dataset_size,train_size,test_size = utils.compute_sizes(dataset,0.8,0.2);
subsets,indices_sentences = utils.dataset_split(dataset,[train_size, test_size],300);

# Get the length of the longest sentence. Then, allow the algorithm to pad the sentences, so they all have the same length
dataset.set_to_pad(0);
dataset.set_max_length(utils.get_maxLength(dataset,subsets));

text1 = '11 cup into 31 pitcher pour to';#reversed: 26 mug into 21 pitcher pour to  
text2 = '31 cup into 11 pitcher pour to';#reversed: 25 mug into 21 pitcher pour to
text3 = '20 cup into 31 pitcher pour to';#reversed: 25 mug into 21 pitcher pour to
print("maximum length: ",dataset.get_max_length());

print("\nFirst Input: ",utils.reverse_string(text1));
actionPlan,execution_time = test_network(dataset,model,text1);
actionPlan = utils.reverse_string(utils.list_to_string(actionPlan));
print("First Action Plan:",actionPlan," execution time: ",execution_time);

print("\nSecond Input: ",utils.reverse_string(text2));
actionPlan,execution_time = test_network(dataset,model,text2);
actionPlan = utils.reverse_string(utils.list_to_string(actionPlan));
print("Second Action Plan:",actionPlan," execution time: ",execution_time);

print("\nThird Input: ",utils.reverse_string(text3));
actionPlan,execution_time = test_network(dataset,model,text3);
actionPlan = utils.reverse_string(utils.list_to_string(actionPlan));
print("Third Action Plan:",actionPlan," execution time: ",execution_time);