'''
    File name: novelplans.py
    Author: David Carmona
    Date created: 28/12/2022
    Date last modified: 29/12/2022
    Python Version: 3.7.6
    Description: This piece of code is to make the predictions when a new goal that has not been seen in the training dataset
                 is being processed by the action planner.
'''

import argparse
import time
import torch
import os
import math
import statistics
import numpy as np
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
import torch.utils.data.dataset as dtst
import csv
import random
import Language_Model_many2many_novelty.utils
import sys

from torch import nn, optim
from datetime import datetime
from torch.utils.data import DataLoader
from Language_Model_many2many_novelty.model import Model
from Language_Model_many2many_novelty.dataset import Dataset
from Language_Model_many2many_novelty.metrics import Metrics
from torch import default_generator
from itertools import accumulate as _accumulate
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple

T = TypeVar('T');

torch.device('cuda');
device = "cuda";

# Class dealing with all the functions to predict novel goals' action sequences
class NovelPlans:

    # Class constructor 
    # @input _dataset An object of type Dataset
    def __init__(self,_dataset):
        self.dataset = _dataset;
        self.index_to_word = self.dataset.get_index_to_word();
        self.unknown_index_to_word = self.dataset.get_unknown_index_to_word();
        self.unknown_word_to_index = self.dataset.get_unknown_word_to_index();

    # Update the Dataset object inside the class
    # @input _dataset The updated dataset
    def update_dataset(self,_dataset):
       self.dataset = _dataset;
       self.index_to_word = _dataset.get_index_to_word();

    # Insert all the data in an object of type Subset into a tensor
    # @input data_loader The Subset object that needs to be converted
    # @return the function returns two pytorch tensors. The first tensor is 
    def subset_to_tensor(self,targeted_subset):
        if type(targeted_subset)!=dtst.Subset or len(targeted_subset)<=0:
            raise Exception("[novelplans.py] Wrong input");
        first_element = targeted_subset[0];first_subelement = first_element[0];
        rows,cols = first_subelement.size();
        x = torch.zeros(len(targeted_subset),cols);y = torch.zeros(len(targeted_subset),cols);
        for k in range(0,len(targeted_subset)):
            subset_elmt = targeted_subset[k];
            x[k,:] = subset_elmt[0];
            y[k,:] = subset_elmt[1];
        row_x,col_x = x.size();row_y,col_y = y.size();
        if row_x!=row_y or col_x!=col_y or row_x<=0 or col_x<=0 or row_y<=0 or col_y<=0:
            raise Exception("[novelplans.py] Wrong computations");
        return x,y;

    # Check whether two tensors are same
    # @input tensor1 First pytorch tensor
    # @input tensor2 Second pytorch
    # @return False is both tensors are different. True is both tensors are equal
    def tensors_are_equal(self,tensor1,tensor2):
        value = False;# This value is True if two tensors are equal. Otherwise, the value is False.
        if type(tensor1)!=torch.Tensor or type(tensor2)!=torch.Tensor:
            raise Exception("[novelplans.py] Wrong input");
        if len(tensor1.size())==1 and len(tensor2.size())==1:
            tensor1 = tensor1.view(1,tensor1.size(0)); tensor2 = tensor2.view(1,tensor2.size(0));
        row_tensor1,col_tensor1 = tensor1.size(); row_tensor2,col_tensor2 = tensor2.size();
        if row_tensor1<=0 or row_tensor2<=0 or col_tensor1!=col_tensor2 or row_tensor1!=row_tensor2 or col_tensor1<=0 or col_tensor2<=0 or row_tensor1>col_tensor1 or row_tensor2>col_tensor2:
            raise Exception("[novelplans.py] Wrong input. Error in dimensions.");

        value = torch.equal(tensor1,tensor2);# Check if two tensors are equal
        return value;

    # Check whether an expression is inside a set of sentences
    # @input expression Tensor containing all the expressions that we would like to ckeck if they are in the input set. The dimensions are KxD.
    # @input input_set Tensor containing the set of sentences. The dimensions are NxD.
    # @return Tensor of boolean values informing which expressions are inside the input_set.
    def is_inside(self,expression,input_set):
        if type(expression)!=torch.Tensor or type(input_set)!=torch.Tensor:
            raise Exception("[novelplans.py] Wrong input");
        row_expression,col_expression = expression.size();row_input_set,col_input_set = input_set.size();
        if row_expression<=0 or row_input_set<=0 or col_expression!=col_input_set:
            raise Exception("[novelplans.py] Wrong input");
        equals = torch.zeros(row_expression,1,dtype=torch.bool);# Returns the elements that are identical between the two tensors
        input_set_bools = torch.zeros(row_input_set,1,dtype=torch.bool);
        for k in range(0,row_expression):
            exp = expression[k];
            for l in range(0,row_input_set):
                inputMat = input_set[l];
                equal_value = self.tensors_are_equal(exp,inputMat);
                input_set_bools[l] = equal_value;
            mat = torch.any(input_set_bools);
            equals[k] = mat;
        if len(equals)!=row_expression or len(equals)<=0:
            raise Exception("[novelplans.py] Wrong computations.");
        return equals;

    # Get the absolute the absolute position of an object based on the goal. Get any other information as well
    # @input sentence Goal to be executed by the robot. It is a tensor.
    # @input index Index in the goal that corresponds to the object's position
    # @return The absolute position of the object based. It is an Integer. It can also be a string
    def get_word(self,sentence,index):
        if type(sentence)!=torch.Tensor or type(index)!=int or type(self.unknown_index_to_word)!=dict:
            raise Exception("[novelplans.py] Wrong input");
        if len(sentence.size())==1:
            sentence = sentence.view(1,sentence.size(0));
        row_sentence,col_sentence = sentence.size();
        if index<0 or index>=col_sentence or row_sentence<1 or len(sentence.size())!=2:
            raise Exception("[novelplans.py] Wrong input. Bad dimensions");

        #index_to_word A dictionary that maps each index to a word
        word = self.index_to_word[int(sentence[0,index])];
        return word;

    # Compute the euclidean distance between two absolute positions
    # @input pst1 The first position.
    # @input pst2 The second position.
    # @return The euclidean distance.
    def compute_euclidean_distance(self,pst1,pst2):
        dist = 0;
        if type(pst1)!=int or type(pst2)!=int or pst1<0 or pst2<0:
            raise Exception("[novelplans.py] Wrong input.");
        dist = math.sqrt(math.pow(pst1-pst2,2));
        if dist<0:
            raise Exception("[novelplans.py] Error in computations");
        return dist;

    # Compute the average distances between the novel objects and those in the training set.
    # @input novel_goal The novel goal that is not in the training dataset.
    # @input input_set The training set with all the training goals.
    # @input task_name The name of the task being tested (e.g., pass, open, or pour).
    # @return A tensor containing all the average distances between the objects in the training and testing sets.
    def compute_scores(self,novel_goal,input_set,task_name):
        if type(novel_goal)!=torch.Tensor or type(input_set)!=torch.Tensor:
            raise Exception("[novelplans.py] Wrong input");
        if len(novel_goal.size())==1:
            novel_goal = novel_goal.view(1,novel_goal.size(0));
        if len(novel_goal.size())!=2 or len(input_set.size())!=2:
            raise Exception("[novelplans.py] Wrong input. Dimensions are incorrect");
        if task_name!='open' and task_name!='pass' and task_name!='pour':
            raise Exception("[novelplans.py] Wrong input. The tasks are incorrect. It is either pass, pour, or open.");
        row_novel_goal,col_novel_goal = novel_goal.size();row_input_set,col_input_set = input_set.size();
        scores = torch.zeros(row_input_set,1);

        # If the task name is not 'open', there are two symbolic locations to extract
        if task_name!='open':
            pst1_novelObject = int(self.get_word(novel_goal,0));# Position of the first object 
            typeObject1_novelObject = self.get_word(novel_goal,1);# First object's name
            pst2_novelObject = int(self.get_word(novel_goal,3));# Position of the second object 
            typeObject2_novelObject = self.get_word(novel_goal,4);# Second object's name
            task_novelObject = self.get_word(novel_goal,5);# Task being performed

            k = 0;
            for training_input in input_set:
                # Extract the position of the objects for each training input
                pst1_trainedObject = int(self.get_word(training_input,0));# Position of the first object 
                typeObject1_trainedObject = self.get_word(training_input,1);# First object's name
                pst2_trainedObject = int(self.get_word(training_input,3));# Position of the second object 
                typeObject2_trainedObject = self.get_word(training_input,4);# Second object's name
                task_trainedObject = self.get_word(training_input,5);# Task being performed

                # We must look for similar goals that have the same task and same objects. Otherwise, assign a very high distance
                if typeObject1_novelObject==typeObject1_trainedObject and typeObject2_novelObject==typeObject2_trainedObject and task_novelObject==task_trainedObject:
                    pst1_dist = self.compute_euclidean_distance(pst1_novelObject,pst1_trainedObject);# Euclidean distances for first positions
                    pst2_dist = self.compute_euclidean_distance(pst2_novelObject,pst2_trainedObject);# Euclidean distances for second positions
                else:
                    pst1_dist = sys.maxsize;# This is the maximum integer value;
                    pst2_dist = sys.maxsize;# This is the maximum integer value;
                score_pst = statistics.mean([pst1_dist,pst2_dist]);
                scores[k] = score_pst;
                k = k + 1;

        # If the task name is 'open', there is only one symbolic location to extract
        elif task_name=='open':
            pst1_novelObject = int(self.get_word(novel_goal,0));# Position of the first object. Just in the 'open' case
            typeObject_novelObject = self.get_word(novel_goal,1);# Object that is being manipulated.
            task_novelObject = self.get_word(novel_goal,2);# Task being performed

            k = 0;
            for training_input in input_set:
                # Extract the position of the object for each training input
                pst1_trainedObject = int(self.get_word(training_input,0));# Position of the first object. Just in the 'open' case
                typeObject_trainedObject = self.get_word(training_input,1);# Object that is being manipulated.
                task_trainedObject = self.get_word(training_input,2);# Task being performed
                
                # We must look for similar goals that have the same task and the same object. Otherwise, assign a very high distance
                if typeObject_trainedObject==typeObject_novelObject and task_novelObject==task_trainedObject:
                    pst1_dist = self.compute_euclidean_distance(pst1_novelObject,pst1_trainedObject);# Euclidean distances for first positions
                else:
                    pst1_dist = sys.maxsize;# This is the maximum integer value
                score_pst = pst1_dist;
                scores[k] = score_pst;
                k = k + 1;
        return scores;

    # Find a similar goal that is similar to the novel one
    # @input novel_goal The novel goal that does not exist in the training dataset. It is a tensor with only one entry.
    # @input input_set All the goals that are in the training dataset. It is a tensor wth multiple entries.
    # @input task_name The name of the task being tested (e.g., pour, open, pass).
    # @return Two outputs: 1) The position of the similar goal in the training dataset; 2) The original goal sentence
    def find_similar_goal(self,novel_goal,input_set,task_name):
        if type(novel_goal)!=torch.Tensor or type(input_set)!=torch.Tensor:
            raise Exception("[novelplans.py] Wrong input");
        if len(novel_goal.size())==1:
            novel_goal = novel_goal.view(1,novel_goal.size(0));
        if len(novel_goal.size())!=2 or len(input_set.size())!=2:
            raise Exception("[novelplans.py] Wrong input. Dimensions are incorrect");
        row_novel_goal,col_novel_goal = novel_goal.size();row_input_set,col_input_set = input_set.size();
        if row_novel_goal<=0 or col_novel_goal<=0 or row_input_set<=0 or col_input_set<0:
            raise Exception("[novelplans.py] Wrong input. Dimensions are incorrect");
        if task_name!='open' and task_name!='pass' and task_name!='pour':
            raise Exception("[novelplans.py] Wrong input. The task name must be either open, pass or pour.");

        # Get the scores for the different distances. Get the lowest score and the index of the goal in the training set
        # which objects are the closest to the current combination
        scores = self.compute_scores(novel_goal,input_set,task_name);
        highest_score, highest_score_index = torch.min(scores,0);
        if len(highest_score)==0 or len(highest_score_index)==0 or len(highest_score)!=len(highest_score_index) or highest_score_index>row_input_set or highest_score_index<0:
            raise Exception("[novelplans.py] Wrong computations");
        similar_goal = input_set[highest_score_index];
        return highest_score_index,similar_goal;

    # Find a probable asction sequence based on the training knowledge. The function computes the euclidean distances between the objects
    # and looks for the minimal distance
    # @input expression A tensor containing all the expressions
    # @input exp_booleans A tensor. Each value is a boolean indicating whether or not the corresponding expression is in the input_set
    # @input input_set A set of sentences. Those are the goal parts in the training set.
    # @input input_set_actions Action sequences for each goal in the training set.
    # @input task_name The name of the task that is being tested (e.g., 'open','pass','pour)
    # @return Several outputs: 1) A boolean that indicates whether all the testing goals are already in the training set; 2) A tensor containing the indices of the similar goals
    def find_probable_action_sequence(self,expression,exp_booleans,input_set,task_name):
        if type(expression)!=torch.Tensor or type(exp_booleans)!=torch.Tensor or type(input_set)!=torch.Tensor:
            raise Exception("[novelplans.py] Wrong input");
        elif len(expression.size())!=2 or len(exp_booleans.size())!=2 or len(input_set.size())!=2:
            raise Exception("[novelplans.py] Wrong input. Dimensions are incorrect");
        elif task_name!='open' and task_name!='pour' and task_name!='pass':
            raise Exception("[novelplans.py] Wrong input. The task can only be either open, pour, or pass");

        row_expression,col_expression=expression.size();row_exp_booleans,col_exp_booleans=exp_booleans.size();row_input_set,col_input_set=input_set.size();
        if row_expression!=row_exp_booleans or col_exp_booleans!=1:
            raise Exception("[novelplans.py] Wrong input. Dimensions are incorrect");
        
        indexes = (exp_booleans == False).nonzero(as_tuple=True)[0];# Choose the sentences that are not inside the input_set
        indexes = indexes.tolist();
        similar_goals_indices_inInput = (-1)*torch.ones(row_expression,1);# Tensor containing the indices of the similar goals in the training set
        all_in_dataset = False;# Variable that informs whether all the sentences are inside the training dataset 

        # If all the testing goals are inside the dataset, all_in_dataset becomes True. Otherwise, it stays False
        if len(indexes)==0:
            all_in_dataset = True;
        else:
            all_in_dataset = False;
            # Iterate over each novel goal to find a goal that is similar in terms of distances 
            for k in range(0,row_expression):
                if exp_booleans[k]==False:
                    novel_goal = expression[k];
                    highest_score_index,similar_goal = self.find_similar_goal(novel_goal,input_set,task_name);
                    similar_goals_indices_inInput[k] = highest_score_index;
                k = k + 1;
        if len(similar_goals_indices_inInput)!=row_expression:
            raise Exception("[novelplans.py] Wrong computations");
        return all_in_dataset,similar_goals_indices_inInput;

    # Replace all the novel goals that are not in the tranining set by similar goals computed previously
    # @input exp_booleans A tensor indicating which goals inside the testing set belong to the training set. True if the goal belongs and False if the goal does not belong
    # @input x_test Goals in the testing set. It is a tensor
    # @input similar_goals_indices_inInput The indices of the goals that are similar to the novel. It is a tensor.
    # @input x_train All the goals of the training set.
    # @return A tensor having the same dimensions of x_test. Each goal that is not in the training set has been replaced by a similar one.
    def replace_novelGoals_with_similarGoals(self,exp_booleans,x_test,similar_goals_indices_inInput,x_train):
        if type(exp_booleans)!=torch.Tensor or type(x_test)!=torch.Tensor or type(similar_goals_indices_inInput)!=torch.Tensor or type(x_train)!=torch.Tensor:
            raise Exception("[novelplans.py] Wrong input.");
        if len(exp_booleans.size())!=2 or len(x_test.size())!=2 or len(similar_goals_indices_inInput.size())!=2 or len(x_train.size())!=2:
            raise Exception("[novelplans.py] Wrong dimensions.");
        row_exp_booleans,col_exp_booleans = exp_booleans.size();row_x_test,col_x_test = x_test.size();row_similar_goals_indices_inInput,col_similar_goals_indices_inInput = similar_goals_indices_inInput.size();
        x_test_copy = torch.clone(x_test);
        if row_exp_booleans!=row_x_test or row_similar_goals_indices_inInput!=row_x_test:
            raise Exception("[novelplans.py] Wrong input");

        for k in range(0,row_exp_booleans):
            if exp_booleans[k]==False:
                similar_goal_index = int(similar_goals_indices_inInput[k]);
                x_test_copy[k] = x_train[similar_goal_index];
        return x_test_copy;

    # @TO-DO
    # Replace the similar goals with the novel goals once the action sequences have been derived by the machine learning model
    # @input exp_booleans A tensor indicating which goals inside the testing set belong to the training set. True if the goal belongs and False if the goal does not belong
    # @input y_pred The predictions for each goal in the modified testing set. The testing set incorporates now the similar goals
    # @input similar_goals_indices_inInput The indices of the goals that are similar to the novel. It is a tensor.
    # @input x_test The original testing set without the similar goals inserted
    def replace_similarGoals_novelGoals(self,exp_booleans,y_pred,similar_goals_indices_inInput,x_test):
        if type(exp_booleans)!=torch.Tensor or type(y_pred)!=torch.Tensor or type(similar_goals_indices_inInput)!=torch.Tensor or type(x_test)!=torch.Tensor:
            raise Exception("[novelplans.py] Wrong input");
        if len(exp_booleans.size())!=2 or len(y_pred.size())!=2 or len(similar_goals_indices_inInput.size())!=2 or len(x_test.size())!=2:
            raise Exception("[novelplans.py] Wrong input");
        row_exp_booleans,col_exp_booleans = exp_booleans.size();row_x_test,col_x_test = x_test.size();row_similar_goals_indices_inInput,col_similar_goals_indices_inInput = similar_goals_indices_inInput.size();
        row_x_test,col_x_test = x_test.size();
        y_pred_copy = torch.clone(y_pred);
        for k in range(0,row_exp_booleans):
            if exp_booleans[k]==False:
                similar_goal_index = int(similar_goals_indices_inInput[k]);
                y_pred_copy[k,0] = x_test[k,0];
                y_pred_copy[k,3] = x_test[k,3];
        return y_pred_copy;
        
def main():
    # Create an object of type DataLoader
    parser = argparse.ArgumentParser();
    parser.add_argument('--max-epochs', type=int, default=300);#Original:100,200
    parser.add_argument('--batch-size', type=int, default=300);
    parser.add_argument('--sequence-length', type=int, default=4);
    args = parser.parse_args();
    dataset = Dataset(args,os.path.join(os.path.dirname(__file__),'data','tasks_test_goodorder_forDebug_simple.csv'));
    dataset_size,train_size,test_size = utils.compute_sizes(dataset,0.5,0.5);
    subsets,indices_sentences = utils.dataset_split(dataset,[train_size,test_size],1);
    train_dataset = subsets[0];
    test_dataset = subsets[1];

    # Datasets for training and testing
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
    );
    testing_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        drop_last=True,
    );
    max_length = utils.get_maxLength(dataset,subsets);
    dataset.set_to_pad(1);dataset.set_max_length(max_length);
    novel_plans = NovelPlans(dataset);

    x_train,y_train = novel_plans.subset_to_tensor(train_dataset);
    x_test,y_test = novel_plans.subset_to_tensor(test_dataset);
    x_test = torch.transpose(x_test,0,1);y_test = torch.transpose(y_test,0,1);
    print("x_test: ",x_test);print("y_test: ",y_test);print("x_train: ",x_train);
    exp_booleans = novel_plans.is_inside(torch.transpose(x_test,0,1),x_train);
    index_to_word = dataset.get_index_to_word();
    all_in_dataset,similar_goals_indices_inInput = novel_plans.find_probable_action_sequence(torch.transpose(x_test,0,1),exp_booleans,x_train);
    #print("similar_goals_indices_inInput= ",similar_goals_indices_inInput);

    x_testModified = novel_plans.replace_novelGoals_with_similarGoals(exp_booleans,torch.transpose(x_test,0,1),similar_goals_indices_inInput,x_train);
    print("x_test_modified: ",x_testModified);
    y_testModified = novel_plans.replace_similarGoals_novelGoals(exp_booleans,torch.transpose(y_test,0,1),similar_goals_indices_inInput,x_testModified);
    print("y_testModified: ",y_testModified);

    # y_predModified is the prediction made by the neural network. y_pred the final prediction after replacing the goal
    y_predModified = torch.transpose(torch.tensor([[2., 0., 3., 7., 1., 4., 5., 0., 1., 6.],[2., 0., 3., 7., 1., 4., 5., 0., 1., 6.]]),0,1);
    y_pred = novel_plans.replace_similarGoals_novelGoals(exp_booleans,torch.transpose(y_predModified,0,1),similar_goals_indices_inInput,torch.transpose(x_test,0,1));
    print("y_pred: ",y_pred);
    #print("y_predModified: ",y_predModified);

if __name__ == "__main__":
    main();