import argparse
import time
import torch
import numpy as np
import sys
import Language_Model_many2many_novelty.performance_testing as performance_testing
import Language_Model_many2many_novelty.utils_file_string as utils_file_string
import Language_Model_many2many_novelty.utils as utils

from torch import nn, optim
from torch.utils.data import DataLoader
from Language_Model_many2many_novelty.model import Model
from Language_Model_many2many_novelty.dataset import Dataset

# This class contains all the necessary functions to run the novel task planner which uses a learning from demonstration-based
# approach
class LfDTaskPlanner:

    model_path = '';
    dataset_path = '';
    plan = '';
    exec_time = 0;
    sequence_length = 0;
    task_name = '';

    # Class constructor
    # @input model_path String. Absolute path to the model that has been trained to predict sentences.
    # @input dataset_path String. Absolute path to the dataset file.
    def __init__(self,_model_path,_dataset_path,_task_name,_sequence_length):
        if type(_model_path)!=str or type(_dataset_path)!=str or type(_task_name)!=str or type(_sequence_length)!=int or _model_path=='' or _dataset_path=='' or _task_name=='' or _sequence_length<=0:
            raise Exception("[lfdtaskplanner.py] Wrong input");
        self.model_path = _model_path;
        self.dataset_path = _dataset_path;
        print("dataset_path = ",self.dataset_path);
        self.task_name = _task_name;
        self.sequence_length = _sequence_length;
    
    # Compute the task plan that is derived from a goal using the trained model.
    # @input goat String. The goal that the planner needs to derive the task plan from.
    # @return String. The task plan corresponding to the goal.
    def compute_plan(self,goal):
        if type(goal)!=str or goal=='':
            raise Exception("[lfdtaskplanner.py] Wrong input");

        actionPlan,exec_time_ourTP = performance_testing.test_ourTP(utils_file_string.reverse_string(goal),self.dataset_path,self.model_path,self.sequence_length,self.task_name);
        if len(actionPlan)==0 or exec_time_ourTP<0:
            raise Exception("[lfdtaskplanner.py] The action plan is empty or the execution time is negative. Error with the computation.");
        actionPlan = utils_file_string.reverse_string(utils.list_to_string(actionPlan));
        self.plan = actionPlan;
        self.exec_time = exec_time_ourTP;
        return actionPlan;

    # Get the execution time
    # @return Float. The execution time.
    def get_time(self):
        return self.exec_time;

    # Get the task plan
    # @return The action plan.
    def get_plan(self):
        return self.plan;

def main():
    model_path = 'C:/Users/David/OneDrive - National University of Singapore/PhD_work/Code/TAMP_H~2/VISUAL~1/TASKPL~1/EXPERI~1/lstm/RE56C2~1/MAX_EP~1.0/BATCH_~1.0/SEQUEN~1.0/LEARNI~1.005/LSTM_S~1.0/EMBEDD~1.0/NUM_LA~1.0/DROPOU~1.0/TRAINE~1/TAMP_02012023_epochs300_batch300_length7_rate0.005.pth';
    dataset_name = 'annotations_video_IRB.csv';
    task_planner = LfDTaskPlanner(model_path,dataset_name);
    actionPlan = task_planner.compute_plan('to pour pitcher 2 into cup 31');
    print("actionPlan: ",actionPlan);

if __name__ == "__main__":
    main();