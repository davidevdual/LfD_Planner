'''
    File name: performance_testing.py
    Author: David Carmona
    Date created: 01/12/2022
    Date last modified: 19/11/2025
    Python Version: 3.10 (64-bit)
    Description: Run the experiments to compare the performance of our learning-based Task Planner with Fast Downward
'''

import argparse
import time
import torch
import os
import random
import sys
import numpy as np
import pandas as pd
import torch.utils.data.dataset as dtst
import Language_Model_many2many_novelty.utils as utils
import Language_Model_many2many_novelty.utils_file_string as utils_file_string
import Language_Model_many2many_novelty.novelplans as novelplans

from torch import nn, optim
from torch.utils.data import DataLoader
from Language_Model_many2many_novelty.model import Model
from Language_Model_many2many_novelty.dataset import Dataset
from classical_task_planning.fastdownward import FastDownward
from torch import default_generator
from itertools import accumulate as _accumulate
from torch import nn, optim
from datetime import datetime
from torch.utils.data import DataLoader
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple

T = TypeVar('T');
random.seed(10);


# Function to read a csv file and insert each action plan into a list
# @input: dataset_dir: directory of the csv file
# @input: field: name of the field/column where the action plans are stored
# @output: text: list of action plans
def load_rows_list(dataset_dir,field):
    if type(dataset_dir)!=str or len(dataset_dir)<=1 or type(field)!=str or len(field)<=1:
        raise Exception("[performance_testing.py] Wrong input.");
    train_df = pd.read_csv(dataset_dir);
    text = train_df[field].str.cat(sep=' ');
    text = text.split('\\n');
    while("" in text) :
        text.remove("");
    if len(text)<=1 or type(text)!=list:
        raise Exception("[performance_testing.py] Wrong computation.");
    return text;

# Function that tests the network on a given input sentence
# @input: dataset: Dataset object containing the vocabulary and other information
# @input: model: Model object containing the trained network    
# @input: text: input sentence to be tested
# @input: novel_plans: NovelPlans object containing methods to deal with novel goals
# @input: target_dataset: Dataset object containing the training dataset
# @input: task_name: name of the task to be executed
# @output: indices: list of words predicted by the network
# @output: execution_time: time taken to infer the action plan
def test_network(dataset,model,text,novel_plans,target_dataset,task_name):
    words = np.array(text.split(' ')).reshape(-1,1);
    batch_size = 1;
    state_h, state_c = model.init_state(batch_size);
    x_testing = utils.pad_sentence(torch.tensor(dataset.compute_word_to_index(words)).type(torch.LongTensor).cpu(),dataset);
    model.eval();

    # Update the NovelPlans object with the potential new words' indices
    novel_plans.update_dataset(dataset);

    # Check if the goal is part of the training set
    print("I am here");
    exp_booleans = novel_plans.is_inside(torch.transpose(x_testing,0,1),target_dataset);
    print("exp_booleans = ",exp_booleans);

    # If the sentence is in the dataset, then proceed as usual
    if bool(exp_booleans[0]) == True:
        print("inside True. Sentence inside the dataset");
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
        print("inside False. Sentence not inside the dataset");
        start = time.time();
        all_in_dataset,similar_goals_indices_inInput = novel_plans.find_probable_action_sequence(torch.transpose(x_testing,0,1),exp_booleans,target_dataset,task_name);
        x_testModified = novel_plans.replace_novelGoals_with_similarGoals(exp_booleans,torch.transpose(x_testing,0,1),similar_goals_indices_inInput,target_dataset);
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

def test_ourTP(text,dataset_name,model_path,sequence_length,task_name):
    parser = argparse.ArgumentParser();# Parentheses are very important after ArgumentParser
    parser.add_argument('--max-epochs', type=int, default=300);#Original:100,200
    parser.add_argument('--batch-size', type=int, default=300);
    parser.add_argument('--sequence-length', type=int, default=sequence_length);# For 'open' it is 4. For the rest it is 7.
    args = parser.parse_args();

    dataset_path = os.path.join('datasets',dataset_name);
    print("dataset_path = ",dataset_path);
    dataset = Dataset(args,dataset_path);
    device = torch.device('cpu');
    print("model_path = ",model_path);
    model = Model(dataset,128,128,1,0);

    data = torch.load(model_path,map_location='cpu',weights_only=False);
    if isinstance(data,dict):
        model.load_state_dict(data);
    elif isinstance(data,nn.Module):
        model.load_state_dict(data.state_dict());
    else:
        raise TypeError("Unexpected checkpoint type: {type(data)}");
    
    #model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False));
    model.eval();
    dataset_size,train_size,test_size = utils.compute_sizes(dataset,0.8,0.2);
    subsets,indices_sentences = utils.dataset_split(dataset,[train_size,test_size],1);

    # Get the training and testing datasets
    train_dataset = subsets[0];
    test_dataset = subsets[1];

    # Get the length of the longest sentence. Then, allow the algorithm to pad the sentences, so they all have the same length
    max_length = utils.get_maxLength(dataset,subsets);    
    dataset.set_to_pad(1);dataset.set_max_length(max_length);

    # Instantiate object that is going to find goals with objects in the neighbouring region
    novel_plans = novelplans.NovelPlans(dataset);

    # Prepare the dataset
    x_train,y_train = novel_plans.subset_to_tensor(train_dataset);# The subset must be converted into a tensor to be processed by the NovelPlans object
    x_test,y_test = novel_plans.subset_to_tensor(test_dataset);# The subset must be converted into a tensor to be processed by the NovelPlans object
    x_train = x_train.type(torch.IntTensor).cpu();y_train = y_train.type(torch.IntTensor).cpu();# Put the matrices on the GPU. Otherwise, an error is thrown
    x_test = x_test.type(torch.IntTensor).cpu();y_test = y_test.type(torch.IntTensor).cpu();
    x = torch.cat((x_train,x_test),0);# Concatenate the training and testing datasets to get the whole dataset with all the sentences

    # Infer the action plan
    actionPlan,execution_time = test_network(dataset,model,text,novel_plans,x,task_name);

    return actionPlan,execution_time;

# Function that tests Fast Downward on a given input task
# @input: task: input task in PDDL format
# @input: algo: Fast Downward algorithm to be used
# @input: line_to_modify: line number in the PDDL file to be modified
# @input: problemfilename: name of the PDDL problem file
# @input: domainfilename: name of the PDDL domain file
# @output: exec_time: time taken by Fast Downward to infer the action plan
def test_FD(task,algo,line_to_modify,problemfilename,domainfilename):
    print("domainfilename = ",domainfilename);
    planner_FD = FastDownward(task,algo,line_to_modify,problemfilename,domainfilename);
    exec_time = planner_FD.get_total_taskPlanning_time();
    return exec_time;

# Function that creates a folder if it does not exist
# @input: dir_name: name of the directory to be created
# @output: None
def create_folder(dir_name):
    dirname = os.path.dirname(__file__);
    dir_to_create = os.path.join(dirname,dir_name);

    # If directories already exist, pass
    try:
        os.makedirs(dir_to_create);
    except FileExistsError:
        pass;

# Function that dumps a matrix into a csv file
# @input: mat_to_write: matrix to be written into the csv file
# @input: input_name_dir: directory of the csv file
# @output: 0 if the operation is successful
def dump_subset_file(mat_to_write,input_name_dir):
    np.savetxt(input_name_dir,mat_to_write,fmt="%10s",delimiter=",");
    return 0;

# Main function
# @input: exp_name: name of the experiment
# @input: model_name: name of the model to be used
# @input: task_name: name of the task to be executed
# @input: task_type: type of task to be executed
# @input: dataset_name: name of the dataset to be used
# @input: objects_names: list of names of the objects to be used in Experiment 2
# @input: positions_names: list of names of the positions to be used in Experiment 3
# @output: None
def main(exp_name, model_name, task_name, task_type, dataset_name, objects_names, positions_names):

    # Decide whether to run the comparisons test or the demonstration of the computational complexity
    # 0 is to not run the tests. 1 is to run the tests.
    comparisons_dir_name = "results_comparison";computational_complexity_dir_name = "results_computational_complexity_objects";
    computational_complexity_positions_dir_name = "results_computational_complexity_positions";

    # Number of times that the task planners are executed for evaluation
    nb_exec = 1;col = 1;
    if task_name=='pass' or task_name=='pour':
        sequence_length = 7;# 'open' is 4. For the rest of the tasks is 7.
    elif task_name=='open':
        sequence_length = 4;
    else:
        raise Exception("[performance_testing.py] Unrecognised task.");

    # Task to execute for Fast Downward and the learning-based planner
    input_file_name = 'input_Exp1_knownTasks.csv'
    if exp_name=='Exp1' and task_type=='knownTask':
        input_file_name = 'input_Exp1_knownTasks.csv';
    elif exp_name=='Exp1' and task_type=='unknownTask':
        input_file_name = 'input_Exp1_unknownTasks.csv';
    elif exp_name=='Exp4' and task_name=='open' and task_type=='knownTask':
        input_file_name = 'input_open_Exp4_knownTasks_updated.csv';
    elif exp_name=='Exp4' and task_name=='open' and task_type=='unknownTask':
        input_file_name = 'input_open_Exp4_unknownTasks_updated.csv';
    elif exp_name=='Exp4' and task_name=='pour' and task_type=='knownTask':
        input_file_name = 'input_pour_Exp4_knownTasks_updated.csv';
    elif exp_name=='Exp4' and task_name=='pour' and task_type=='unknownTask':
        input_file_name = 'input_pour_Exp4_unknownTasks_updated.csv';
    elif exp_name=='Exp4' and task_name=='pass' and task_type=='knownTask':
        input_file_name = 'input_pass_Exp4_knownTasks_updated.csv';
    elif exp_name=='Exp4' and task_name=='pass' and task_type=='unknownTask':
        input_file_name = 'input_pass_Exp4_unknownTasks_updated.csv';

    input_file_dir =  os.path.join(os.path.dirname(__file__),'input_comparisons',input_file_name);
    print("input_file_dir = ",input_file_dir);
    task_FD_list = load_rows_list(input_file_dir,"PDDL");
    task_ours_list = load_rows_list(input_file_dir,"Ours");
    
    if exp_name=='Exp4' and task_name=='open' and task_type=='unknownTask':
        nb_exec = 25;col = 1
    else:
        list_ids = random.sample(range(0, len(task_ours_list)),100);# Select 100 inputs randomly to speed up experiments
        task_ours_list = [task_ours_list[i] for i in list_ids];
        task_FD_list = [task_FD_list[i] for i in list_ids];
    
    list_len_ours_list = len(task_ours_list);list_len_FD_list = len(task_FD_list);
    print("list_len_ours_list = ",list_len_ours_list);

    # Both lists must have same length
    if list_len_ours_list!=list_len_FD_list:
        raise Exception("[performance_testing.py] The lists must have the same lengths.");

    # Fast Downward algorithms to execute. Set dataset name and model name for my algorithm
    algos_list = ['lazy_greedy([cg()])','lazy_greedy([cg(),ff()])','astar(cg())']; 
    model_path = os.path.join('Language_Model_many2many_novelty','models',model_name);
    
    # Variable containing all the execution times
    all_execution_times = np.zeros((1,len(algos_list)+1));

    if exp_name=="Exp1":
        problemfilename = 'task_Exp1.pddl';
        domainfilename = 'domain_pour.pddl';
        for k in range(0,list_len_ours_list):
            
            task_ours = task_ours_list[k];
            task_FD = task_FD_list[k];
            execution_times = np.zeros((nb_exec,len(algos_list)+1));

            # Execute the conventional Task Planner and ours
            for j in range(0,nb_exec):
                print("task_ours = ",task_ours);
                
                actionPlan,exec_time_ourTP = test_ourTP(utils_file_string.reverse_string(task_ours),dataset_name,model_path,sequence_length,task_name);
                actionPlan = utils_file_string.reverse_string(utils.list_to_string(actionPlan));
                print("Our action plan: ",actionPlan);
                execution_times[j,0] = exec_time_ourTP;
                
                for algo in algos_list:
                    exec_time_FD = 2;
                    exec_time_FD = test_FD(task_FD,algo,14,problemfilename,domainfilename);
                    print("Our TP execution time ",exec_time_ourTP," FD execution time: ",exec_time_FD);
                    execution_times[j,col] = exec_time_FD;
                    col = col + 1;
                col = 1;
 
            # Dump the results into a file
            create_folder(comparisons_dir_name);
            input_name_dir = os.path.join(os.path.dirname(__file__),comparisons_dir_name,'results_comparison_'+task_type+'_'+task_ours+'.csv');
            dump_subset_file(execution_times,input_name_dir);
            all_execution_times = np.vstack((all_execution_times,execution_times)); 

        all_execution_times = np.delete(all_execution_times,(0),axis=0);# Deletes the first row. It is full of zeros
        unique_file_dir = os.path.join(os.path.dirname(__file__),comparisons_dir_name,'allresults_comparison.csv');
        dump_subset_file(all_execution_times,unique_file_dir);

    elif exp_name=="Exp4":
        problemfilename = 'task_Exp4_'+task_name+'.pddl';
        domainfilename = 'domain_'+task_name+'.pddl';
        for k in range(0,list_len_ours_list):
            task_ours = task_ours_list[k];
            task_FD = task_FD_list[k];
            execution_times = np.zeros((nb_exec,len(algos_list)+1));

            # Execute the conventional Task Planner and ours
            for k in range(0,nb_exec):
                print("task_ours = ",task_ours);
                actionPlan,exec_time_ourTP = test_ourTP(utils_file_string.reverse_string(task_ours),dataset_name,model_path,sequence_length,task_name);
                actionPlan = utils_file_string.reverse_string(utils.list_to_string(actionPlan));
                print("Our action plan: ",actionPlan);
                execution_times[k,0] = exec_time_ourTP;
                
                for algo in algos_list:
                    exec_time_FD = test_FD(task_FD,algo,14,problemfilename,domainfilename);
                    print("Our TP execution time ",exec_time_ourTP," FD execution time: ",exec_time_FD);
                    execution_times[k,col] = exec_time_FD;
                    col = col + 1;
                col = 1;
 
            # Dump the results into a file
            create_folder(comparisons_dir_name);
            input_name_dir = os.path.join(os.path.dirname(__file__),comparisons_dir_name,'results_comparison_'+task_type+'_'+task_ours+'.csv');
            dump_subset_file(execution_times,input_name_dir);
            all_execution_times = np.vstack((all_execution_times,execution_times)); 

        all_execution_times = np.delete(all_execution_times,(0),axis=0);# Deletes the first row. It is full of zeros
        unique_file_dir = os.path.join(os.path.dirname(__file__),comparisons_dir_name,'allresults_comparison.csv');
        dump_subset_file(all_execution_times,unique_file_dir);

    elif exp_name=="Exp2":
        nb_exec = 100;
        problemfilename = 'task_Exp2_'+task_type+'.pddl';
        domainfilename = 'domain_pour.pddl';
        if task_type=='knownTask':
            task_ours = 'to pour pitcher 31 into cup 11';# task: (:goal (and (poured pitcher 2 cup 31)))) This is the known task
        elif task_type=='unknownTask':
            task_ours = 'to pour pitcher 15 into cup 20';# task: (:goal (and (poured pitcher 15 cup 20)))) This is the unknown task
        else:
            raise Exception("[performance_testing.py] Wrong Input. task_type does not exist.");
        objects = objects_names;#'cup pitcher cracker_box bowl gelatin_box mustard_bottle pudding_box tomato_soup_can tuna_fish_can sugar_box';
        execution_times = np.zeros((nb_exec,len(algos_list)+1));

        for k in range(0,nb_exec):
            actionPlan,exec_time_ourTP = test_ourTP(utils_file_string.reverse_string(task_ours),dataset_name,model_path,sequence_length,task_name);
            actionPlan = utils_file_string.reverse_string(utils.list_to_string(actionPlan));
            print("Our action plan: ",actionPlan);
            execution_times[k,0] = exec_time_ourTP;

            for algo in algos_list:
                objects_pddl = '(:objects '+objects+' initial_location_handleft initial_location_handright - graspable \n';
                exec_time_FD = test_FD(objects_pddl,algo,4,problemfilename,domainfilename);
                execution_times[k,col] = exec_time_FD;
                col = col + 1;
            col = 1;

        # Dump the results into a file
        create_folder(computational_complexity_dir_name);
        input_name_dir = os.path.join(os.path.dirname(__file__),computational_complexity_dir_name,'results_computational_complexity_objects'+'.csv');
        dump_subset_file(execution_times,input_name_dir);

    elif exp_name=="Exp3":
        nb_exec = 100;
        problemfilename = 'task_Exp3_'+task_type+'.pddl';
        domainfilename = 'domain_pour.pddl';
        if task_type=='knownTask':
            task_ours = 'to pour pitcher 31 into cup 11';# task: (:goal (and (poured pitcher 31 cup 11))))
        elif task_type=='unknownTask':
            task_ours = 'to pour pitcher 15 into cup 20';# task: (:goal (and (poured pitcher 15 cup 20))))
        else:
            raise Exception("[performance_testing.py] Wrong Input. task_type does not exist.");
        positions = positions_names;#'15 20 31 11 18 1 16 6 21 23 19 27 2 10';
        execution_times = np.zeros((nb_exec,len(algos_list)+1));

        for k in range(0,nb_exec):
            actionPlan,exec_time_ourTP = test_ourTP(utils_file_string.reverse_string(task_ours),dataset_name,model_path,sequence_length,task_name);
            actionPlan = utils_file_string.reverse_string(utils.list_to_string(actionPlan));
            print("Our action plan: ",actionPlan);
            execution_times[k,0] = exec_time_ourTP;

            for algo in algos_list:
                positions_pddl = positions+ ' - location \n';
                exec_time_FD = test_FD(positions_pddl,algo,6,problemfilename,domainfilename);
                execution_times[k,col] = exec_time_FD;
                col = col + 1;
            col = 1;

        # Dump the results into a file
        create_folder(computational_complexity_positions_dir_name);
        input_name_dir = os.path.join(os.path.dirname(__file__),computational_complexity_positions_dir_name,'results_computational_complexity_positions'+'.csv');
        dump_subset_file(execution_times,input_name_dir);

    else:
        raise Exception("[performance_testing.py] Wrong input. The comparisons_tests value does not exist.");

if __name__ == "__main__":

    # Experiment name. It can be 'Exp1', 'Exp2', 'Exp3', 'Exp4'
    exp_name = 'Exp4';

    # Model to run the experiments. Just put the model's name with the .pth extension
    model_name = 'Exp4_Normal/model_exp4_pour.pth';

    # Name of the task. It can be 'open', 'pour' or 'pass'
    task_name = 'pour';

    # Task type. It can be 'knownTask' or 'unknownTask'
    task_type = 'knownTask';

    # Name of the dataset to use
    dataset_name = 'Exp4_Normal/annotations_video_IRB_pour.csv';

    # This is for Experiment 2 only. The number of objects will vary
    #'cup pitcher cracker_box bowl gelatin_box mustard_bottle pudding_box tomato_soup_can tuna_fish_can sugar_box';
    objects_names = 'cup pitcher';

    # This is for Experiment 3 only. The number of positions will vary
    # '15 20 31 11 18 1 16 6 21 23 19 27 2 10';
    positions_names = '15 20 31 11';

    main(exp_name, model_name, task_name, task_type, dataset_name, objects_names, positions_names);