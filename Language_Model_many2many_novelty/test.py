import argparse
import time
import torch
import os
import numpy as np
import torch.utils.data.dataset as dtst
import csv
import random
import novelplans

import utils as utils
import utils_file_string as utils_file_string
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
from torch import default_generator
#from torch._utils import _accumulate
from torch import nn, optim
from datetime import datetime
from torch.utils.data import DataLoader
from pathlib import Path
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple

def test_network(dataset,model,text,novel_plans,target_dataset):
    words = np.array(text.split(' ')).reshape(-1,1);
    batch_size = 1;
    state_h, state_c = model.init_state(batch_size);
    x_testing = utils.pad_sentence(torch.tensor(dataset.compute_word_to_index(words)).type(torch.LongTensor).cpu(),dataset);
    model.eval();

    # Update the NovelPlans object with the potential new words' indices
    novel_plans.update_dataset(dataset);

    # Check if the goal is part of the training set
    exp_booleans = novel_plans.is_inside(torch.transpose(x_testing,0,1),target_dataset);
    print("exp_booleans = ",exp_booleans);

    # If the sentence is in the dataset, then proceed as usual
    if bool(exp_booleans[0]) == True:
        print("inside True");
        start = time.time();
        y_pred, (state_h, state_c) = model(x_testing,(state_h,state_c));
        row_y_pred = y_pred.shape[0];col_y_pred = y_pred.shape[1];depth_y_pred = y_pred.shape[2];
        y_pred = torch.reshape(y_pred,(row_y_pred*col_y_pred,depth_y_pred));
        y_pred = torch.argmax(y_pred,dim=1);

        indices = torch.reshape(y_pred,(-1,1)).transpose(1,0);
        indices = dataset.convert_matrix_to_words(indices).tolist();# Returns an imbricated list
        indices = indices[0];# Extract the imbricated list
        end = time.time();
        execution_time = (end-start)*1000;
    elif bool(exp_booleans[0]) == False:# The sentence is not inside the dataset. Therefore, proceed with the substitutions
        print("inside False");
        start = time.time();
        all_in_dataset,similar_goals_indices_inInput = novel_plans.find_probable_action_sequence(torch.transpose(x_testing,0,1),exp_booleans,target_dataset);
        x_testModified = novel_plans.replace_novelGoals_with_similarGoals(exp_booleans,torch.transpose(x_testing,0,1),similar_goals_indices_inInput,target_dataset);
        print("x_testModified: ",x_testModified);
        exit(0);
        x_testModified = torch.transpose(x_testModified,0,1);
        y_predModified,(state_h, state_c) = model(x_testModified,(state_h,state_c));
        row_y_pred = y_predModified.shape[0];col_y_pred = y_predModified.shape[1];depth_y_pred = y_predModified.shape[2];
        y_predModified = torch.reshape(y_predModified,(row_y_pred*col_y_pred,depth_y_pred));
        y_predModified = torch.from_numpy(np.transpose(np.reshape(torch.argmax(y_predModified,dim=1).cpu().numpy(),(row_y_pred,col_y_pred)),(1,0)));
        y_predModified = novel_plans.replace_similarGoals_novelGoals(exp_booleans,y_predModified,similar_goals_indices_inInput,torch.transpose(x_testing,0,1));
        indices = dataset.convert_matrix_to_words(y_predModified).tolist();
        indices = indices[0];
        end = time.time();
        execution_time = (end-start)*1000;
    else:
        raise Exception("[test.py] Wrong computation or input.");
    return indices,execution_time;

def main():
    # Default arguments. They will be changed later
    parser = argparse.ArgumentParser();# Parentheses are very important after ArgumentParser
    parser.add_argument('--max-epochs', type=int, default=300);#Original:100,200
    parser.add_argument('--batch-size', type=int, default=300);
    parser.add_argument('--sequence-length', type=int, default=7);
    args = parser.parse_args();

    # Load the Dataset
    parent_dir = Path(__file__).resolve().parent.parent;# Make it a relative path for simplicity
    dataset = Dataset(args,os.path.join('datasets','annotations_video_IRB_pass.csv'));

    device = torch.device('cpu');
    model = Model(dataset,128,128,1,0);
    model.load_state_dict(torch.load("C:\\Users\\DNCM\\Documents\\dual-arm-nus\\windows_legion7\\Language_Model_many2many_novelty\\results\\max_epochs=1.0\\batch_size=300.0\\sequence_length=7.0\\learning_rate=0.005\\lstm_size=128.0\\embedding_dim=128.0\\num_layers=1.0\\dropout=0.0\\trained_models\\TAMP_09302025_epochs1_batch300_length7_rate0.005.pth",map_location=device));
    print(model);
    dataset_size,train_size,test_size = utils.compute_sizes(dataset,0.8,0.2);
    subsets,indices_sentences = utils.dataset_split(dataset,[train_size,test_size],1);
    print("indices_sentences: ",indices_sentences);

    # Get the training and testing datasets
    train_dataset = subsets[0];
    test_dataset = subsets[1];

    # Get the length of the longest sentence. Then, allow the algorithm to pad the sentences, so they all have the same length
    max_length = utils.get_maxLength(dataset,subsets);
    dataset.set_to_pad(1);dataset.set_max_length(max_length);

    # Instantiate object that is going to find goals with objects in the neighbouring region
    novel_plans = novelplans.NovelPlans(dataset);

    x_train,y_train = novel_plans.subset_to_tensor(train_dataset);# The subset must be converted into a tensor to be processed by the NovelPlans object
    x_test,y_test = novel_plans.subset_to_tensor(test_dataset);# The subset must be converted into a tensor to be processed by the NovelPlans object
    x_train = x_train.type(torch.IntTensor).cpu();y_train = y_train.type(torch.IntTensor).cpu();# Put the matrices on the GPU. Otherwise, an error is thrown
    x_test = x_test.type(torch.IntTensor).cpu();y_test = y_test.type(torch.IntTensor).cpu();
    x = torch.cat((x_train,x_test),0);# Concatenate the training and testing datasets to get the whole dataset with all the sentences

    text1 = '14 experimenter_hand to 9 extra_large_clamp pass to';#reversed: to pass extra_large_clamp 9 to experimenter_hand 14
    #text2 = '9 cup into 22 pitcher pour to';#reversed: 25 mug into 21 pitcher pour to
    #text3 = '27 cup into 19 pitcher pour to';#reversed: 25 mug into 21 pitcher pour to
    #text4 = '25 cup into 3 pitcher pour to';#reversed: 25 mug into 21 pitcher pour to
    print("maximum length: ",dataset.get_max_length());

    
    print("\nFirst Input: ",utils_file_string.reverse_string(text1));
    actionPlan,execution_time = test_network(dataset,model,text1,novel_plans,x_train);
    actionPlan = utils_file_string.reverse_string(utils.list_to_string(actionPlan));
    print("First Action Plan:",actionPlan," execution time: ",execution_time);
    '''
    print("\nSecond Input: ",utils.reverse_string(text2));
    actionPlan,execution_time = test_network(dataset,model,text2,novel_plans,x_train);
    actionPlan = utils.reverse_string(utils.list_to_string(actionPlan));
    print("Second Action Plan:",actionPlan," execution time: ",execution_time);

    print("\nThird Input: ",utils.reverse_string(text3));
    actionPlan,execution_time = test_network(dataset,model,text3,novel_plans,x_train);
    actionPlan = utils.reverse_string(utils.list_to_string(actionPlan));
    print("Third Action Plan:",actionPlan," execution time: ",execution_time);
    

    print("\nFourth Input: ",utils_file_string.reverse_string(text4));
    actionPlan,execution_time = test_network(dataset,model,text4,novel_plans,x);
    actionPlan = utils_file_string.reverse_string(utils_file_string.list_to_string(actionPlan));
    print("Fourth Action Plan:",actionPlan," execution time: ",execution_time);
    '''

if __name__ == "__main__":
    main();