'''
    File name: train_noNovelty.py
    Author: David Carmona
    Date created: 01/12/2022
    Date last modified: 17/11/2025
    Python Version: 3.10 (64-bit)
    Description: Train the LSTM model to predict action sequences
'''

import argparse
import time
import torch
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.utils.data.dataset as dtst
import csv
import random
import utils

from torch import nn, optim
from datetime import datetime
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
from metrics import Metrics
from torch import default_generator
from pathlib import Path
#from torch._utils import _accumulate # Not anymore valid for pytorch>v2.2.0
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple

T = TypeVar('T');

torch.device('cuda');
device = "cuda";

# Write all the content of a subset to a csv file
# @input dataset The dataset object
# @input subset The subset to write into a csv file
# @input sequence_length The length of the input sequence
# @input input_name_dir The absolute path to the csv file
# #@return 0 if everything is ok
def dump_subset_file(dataset,subset,sequence_length,input_name_dir):
    if len(subset)<=0 or sequence_length<=0 or input_name_dir=='':
        raise Exception("[train.py] Wrong input.");
    x,y = subset[0];
    rows,cols = x.size();
    concat = torch.zeros(1,cols);
    for k in range(0,len(subset)):
        x,y = subset[k];
        concat = torch.vstack((concat,x));
    concat = concat.numpy();
    concat = np.delete(concat,(0),axis=0);# First row is full of zeros
    concat = dataset.compute_index_to_word(concat);
    np.savetxt(input_name_dir,concat,fmt="%10s",delimiter="\t");
    return 0;

#Select the last column since it only contains the losses for the predicted word, which is what we are interested in.
# Then, we compute the sum of all those losses and divide by the number of plans. We must make sure that the number of plans is not 0.
# @input loss The loss tensor
# @input nb_plans The number of plans in the batch
# @return The computed loss
def compute_loss(loss,nb_plans):
    if nb_plans<=0:
        raise Exception("[train.py] The number of plans cannot be less or equal to zero.");
    return torch.div(torch.sum(loss[:]),nb_plans);

# Test the loss function
# @input model The LSTM model
# @input y
# @input y_pred_transpose The transposed predicted output
# @return 0 if everything is ok
def test_loss(model,y_pred_transpose,y):
    probas = model.softmax(y_pred_transpose);
    log_probas = -1*model.log_softmax(y_pred_transpose);
    return 0;

# Train the LSTM model
# @input resultsCSV_dir_to_Save The absolute path to the "results.csv" file where all numerical results are saved
# @input now Time
# @input results Numpy array containing all numerical results
# @input dataset The dataset object
# @input subsets List containing the training and testing subsets
# @input metrics The metrics object
# @input model The LSTM model
# @input model_dir The absolute path to the directory where the model is saved
# @input input_name_dir The absolute path to the directory where the input csv files are saved
# @input root_dir The absolute path to the root directory where all information is saved
# @input results_file_dir The absolute path to the directory where the "results.csv" file is saved
# @input parameters List containing all parameters to optimise
# @input args The arguments object
# @return current_results Numpy array containing the numerical results obtained with the current set of parameters
def train(resultsCSV_dir_to_Save,now,results,dataset,subsets,metrics,model,model_dir,input_name_dir,root_dir,results_file_dir,parameters,args):

    # Get the mapping between the numerical indices and the words
    index_to_word = dataset.get_index_to_word();

    # Open the "results.csv" file
    csvfile = open(resultsCSV_dir_to_Save,"a");

    # Parameters to optimise
    max_epochs = parameters[0];batch_size = parameters[1];sequence_length = parameters[2];learning_rate = parameters[3];lstm_size = parameters[4];
    embedding_dim = parameters[5];num_layers = parameters[6];dropout = parameters[7];

    print("sequence_length = ",sequence_length);

    # Get training and testing datasets. The first element of the subsets list is the training. The second element of the subsets list is the testing
    train_dataset = subsets[0];
    test_dataset = subsets[1];
    print("train_dataset: ",len(train_dataset)," test_dataset: ",len(test_dataset));

    if batch_size > min(len(train_dataset),len(test_dataset)):
        raise Exception("[train.py] The batch size must be equal to min(len(train_dataset),len(test_dataset)) which is: ",min(len(train_dataset),len(test_dataset)));

    # Write the training dataset into a csv file
    train_input_name_dir = os.path.join(input_name_dir,'training.csv');
    testing_input_name_dir = os.path.join(input_name_dir,'testing.csv');
    dump_subset_file(dataset,train_dataset,sequence_length,train_input_name_dir);
    dump_subset_file(dataset,test_dataset,sequence_length,testing_input_name_dir);

    print("The batch size set by the user is: ",args.batch_size);
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

    criterion = nn.CrossEntropyLoss(reduction='none');
    optimizer = optim.Adam(model.parameters(),lr=learning_rate);#Original: 0.005

    training_loss = [];
    testing_loss = [];
    training_matches = [];
    training_mismatches = [];
    training_is_equal = [];
    testing_matches = [];
    testing_mismatches = [];
    testing_is_equal = [];
    epochs = [];

    fig, axs = plt.subplots(1, 2)

    real_batch_size = dataset.get_batch_size(train_loader,testing_loader);

    for epoch in range(args.max_epochs):
        total_loss_training = [];
        total_loss_testing = [];

        total_mean_percentage_matches_training = [];
        total_mean_percentage_mismatches_training = [];
        total_mean_is_equal_training = [];

        total_mean_percentage_matches_testing = [];
        total_mean_percentage_mismatches_testing = [];
        total_mean_is_equal_testing = [];

        state_h, state_c = model.init_state(real_batch_size);

        state_h = state_h.cuda();
        state_c = state_c.cuda();

        model_name = 'TAMP_'+now+'_epochs'+str(args.max_epochs)+'_batch'+str(batch_size)+'_length'+str(sequence_length)+'_rate'+str(learning_rate);
        model_name = os.path.join(model_dir,model_name+'.pth');

        for batch, (x,y) in enumerate(train_loader):
            optimizer.zero_grad();

            # enumerate() adds a third dimension to the torch tensors. Indeed, each batch has its own dimension, which is not accepted by the LSTM.
            # Therefore, we must reshape the torch tensor, so it becomes a 2D-array.
            row_x, col_x, depth_x = x.shape;row_y, col_y, depth_y = y.shape;
            nb_plans = dataset.assign_indexes_actionPlans(y).shape[0];
            x = x.cuda();y = y.type(torch.LongTensor).cuda();# Originally, y is an Int, but criterion requires it to be in Long
            
            # Reshape and tranpose x and y so they follow the shape: [sequence_length,batch_size]
            x = torch.permute(torch.reshape(x,(row_x*col_x,depth_x)),(1,0));
            y = torch.permute(torch.reshape(y,(row_y*col_y,depth_y)),(1,0));

            print("x, size = ",x.size());
            print("y, size = ",y.size());

            # Train the model
            y_pred, (state_h, state_c) = model(x,(state_h,state_c));
            row_y_pred = y_pred.shape[0];col_y_pred = y_pred.shape[1];depth_y_pred = y_pred.shape[2];
            row_y = y.shape[0];col_y = y.shape[1];

            # Compute the loss. The input to criterion is of shape [N = batch size,C = number of classes]. Therefore, reshaping is necessary
            y_pred = torch.reshape(y_pred,(row_y_pred*col_y_pred,depth_y_pred));
            y = torch.reshape(y,(-1,));
            loss = compute_loss(criterion(y_pred,y),nb_plans);#The seven first elmts are part of the goal. The goal is fixed. We don't need to compute the loss for those terms.

            y = np.transpose(np.reshape(y.cpu().numpy(),(row_y,col_y)),(1,0));
            y_pred = np.transpose(np.reshape(torch.argmax(y_pred,dim=1).cpu().numpy(),(row_y_pred,col_y_pred)),(1,0));# Get the index with the highest probability
            
            ground_truth_name = os.path.join(root_dir,'ground_truth_training','ground_truth'+'epoch'+str(epoch)+'batch'+str(batch)+'.csv');
            training_name = os.path.join(root_dir,'predictions_training','testing'+'epoch'+str(epoch)+'batch'+str(batch)+'.csv');

            # Save the ground-truth and predicted action sequences for each batch into .csv file

            if np.array_equal(y,y_pred)==False:
                np.savetxt(ground_truth_name,dataset.convert_matrix_to_words(y),fmt="%10s",delimiter="\t");
                np.savetxt(training_name,dataset.convert_matrix_to_words(y_pred),fmt="%10s",delimiter="\t");

            # Compute the accuracy and other metrics
            mean_percentage_matches,mean_percentage_mismatches,is_equal_mean = metrics.compute_metrics(y,y_pred);
            total_mean_percentage_matches_training.append(mean_percentage_matches);
            total_mean_percentage_mismatches_training.append(mean_percentage_mismatches);
            total_mean_is_equal_training.append(is_equal_mean*100);
           
            state_h = state_h.detach();
            state_c = state_c.detach();

            loss.backward();
            optimizer.step();
            total_loss_training.append(loss.item());
            print({'epoch': epoch, 'batch': batch, 'training loss': loss.item(), 'training action plans':is_equal_mean});
        torch.save(model.state_dict(),model_name);# Save only the parameters of the model. It is more efficient. The parameters are saved on the GPU

        training_loss.append(np.mean(total_loss_training));
        training_matches.append(np.mean(total_mean_percentage_matches_training));
        training_mismatches.append(np.mean(total_mean_percentage_mismatches_training));
        training_is_equal.append(np.mean(total_mean_is_equal_training));
        x_testing_subset = 0;
        y_testing_subset = 0;
        
        for batch_idx, (x_testing,y_testing) in enumerate(testing_loader):
            x_testing_subset = x_testing;y_testing_subset = y_testing;
            row_x, col_x, depth_x = x_testing.shape;row_y, col_y, depth_y = y_testing.shape;
            nb_plans = dataset.assign_indexes_actionPlans(y_testing).shape[0];
            x_testing = x_testing.cuda();
            y_testing = y_testing.cuda();

            # Reshape and tranpose x and y so they follow the shape: [sequence_length,batch_size]
            x_testing = torch.permute(torch.reshape(x_testing,(row_x*col_x,depth_x)),(1,0));

            # Necessary permutations for the testing set
            y_testing = torch.permute(torch.reshape(y_testing,(row_y*col_y,depth_y)),(1,0));
            y_testing = y_testing.type(torch.LongTensor).cuda();# Originally, y is an Int, but criterion requires it to be in Long

            # Predict the testing sequence. The prediction is done on x_testModified and not x_testing. Therefore, afterwards, the goal must be changed to the original one
            y_pred_testing, (state_h,state_c) = model(x_testing,(state_h,state_c));
            row_y_pred = y_pred_testing.shape[0];col_y_pred = y_pred_testing.shape[1];depth_y_pred = y_pred_testing.shape[2];
            row_y = y_testing.shape[0];col_y = y_testing.shape[1];

            # Compute the loss. The input to criterion is of shape [N = batch size,C = number of classes]. Therefore, reshaping is necessary
            y_pred_testing = torch.reshape(y_pred_testing,(row_y_pred*col_y_pred,depth_y_pred));
            y_testing = torch.reshape(y_testing,(-1,));

            loss = compute_loss(criterion(y_pred_testing,y_testing),nb_plans);
            y_testing = np.transpose(np.reshape(y_testing.cpu().numpy(),(row_y,col_y)),(1,0));
            y_pred_testing = np.transpose(np.reshape(torch.argmax(y_pred_testing,dim=1).cpu().numpy(),(row_y_pred,col_y_pred)),(1,0));# Get the index with the highest probability
            total_loss_testing.append(loss.item());

            # Compute the accuracy and other metrics
            mean_percentage_matches,mean_percentage_mismatches,is_equal_mean = metrics.compute_metrics(y_testing,y_pred_testing);
            total_mean_percentage_matches_testing.append(mean_percentage_matches);
            total_mean_percentage_mismatches_testing.append(mean_percentage_mismatches);
            total_mean_is_equal_testing.append(is_equal_mean*100);
            word_matrix_testing = dataset.convert_matrix_to_words(y_testing);
            word_matrix_pred = dataset.convert_matrix_to_words(y_pred_testing);

            ground_truth_name = os.path.join(root_dir,'ground_truth_testing','ground_truth'+'epoch'+str(epoch)+'batch'+str(batch_idx)+'.csv');
            testing_name = os.path.join(root_dir,'predictions_testing','testing'+'epoch'+str(epoch)+'batch'+str(batch_idx)+'.csv');

            # Save the ground-truth and predicted action sequences for each batch into .csv file

            if np.array_equal(y_testing,y_pred_testing)==False:
                np.savetxt(ground_truth_name,dataset.convert_matrix_to_words(y_testing),fmt="%10s",delimiter="\t");
                np.savetxt(testing_name,dataset.convert_matrix_to_words(y_pred_testing),fmt="%10s",delimiter="\t");

            print({'epoch': epoch, 'batch': batch_idx, 'testing loss': loss.item(), 'testing action plans':is_equal_mean});

        testing_loss.append(np.mean(total_loss_testing));
        testing_matches.append(np.mean(total_mean_percentage_matches_testing));
        testing_mismatches.append(np.mean(total_mean_percentage_mismatches_testing));
        testing_is_equal.append(np.mean(total_mean_is_equal_testing));
        epochs.append(epoch);

        #print({'epoch': epoch, 'training loss': training_loss[-1], 'testing loss': testing_loss[-1], 'training mean matches': training_matches[-1], 'testing mean matches': testing_matches[-1], 'number training same action plans': training_is_equal[-1], 'number testing same action plans':testing_is_equal[-1]});
        current_results = np.array([epoch,max_epochs,batch_size,sequence_length,learning_rate,lstm_size,embedding_dim,num_layers,dropout,training_loss[-1],testing_loss[-1],training_matches[-1],testing_matches[-1],training_is_equal[-1],testing_is_equal[-1]]).reshape(1,15);
        np.savetxt(csvfile,current_results,delimiter=',');

        results = np.vstack((results,current_results));

        axs[0].set_xlabel('epoch',fontsize = 35.0);
        axs[0].set_ylabel('loss',fontsize = 35.0);
        axs[0].plot(epochs, training_loss, label = "train",color='red');
        axs[0].plot(epochs, testing_loss, label = "testing",color='blue');
        axs[0].legend(["train","testing"], loc ='upper right', fontsize=18.0);
        axs[1].set_xlabel('epoch',fontsize = 35.0);
        axs[1].set_ylabel('Percentage (%)',fontsize = 35.0);
        axs[1].plot(epochs, training_matches, label = "train",color='red');
        axs[1].plot(epochs, testing_matches, label = "train",color='blue');
        axs[1].legend(["train","testing"], loc ='upper right', fontsize=18.0);
        axs[1].plot(epochs, training_is_equal, label = "train action plans",color='green');
        axs[1].plot(epochs, testing_is_equal, label = "testing action plans",color='cyan');
        axs[1].legend(["train","testing","train action plans","testing action plans"], loc ='lower right', fontsize=18.0);

        # Save the metrics into one folder
        training_loss_dir_toSave = os.path.join(root_dir,'metrics', 'training_loss_epoch'+str(epoch)+'.npy');
        testing_loss_dir_toSave = os.path.join(root_dir,'metrics', 'testing_loss_epoch'+str(epoch)+'.npy');
        training_matches_dir_toSave = os.path.join(root_dir,'metrics', 'training_matches_epoch'+str(epoch)+'.npy');
        testing_matches_dir_toSave = os.path.join(root_dir,'metrics', 'testing_matches_epoch'+str(epoch)+'.npy');
        training_is_equal_dir_toSave = os.path.join(root_dir,'metrics', 'training_is_equal_epoch'+str(epoch)+'.npy');
        testing_is_equal_dir_toSave = os.path.join(root_dir,'metrics', 'testing_is_equal_epoch'+str(epoch)+'.npy');
        #print("results= ",results);
     
        #np.save(training_loss_dir_toSave, training_loss);
        #np.save(training_loss_dir_toSave, training_loss);
        #np.save(testing_loss_dir_toSave, testing_loss);
        #np.save(training_matches_dir_toSave, training_matches);
        #np.save(testing_matches_dir_toSave, testing_matches);
        #np.save(training_is_equal_dir_toSave, training_is_equal);
        #np.save(testing_is_equal_dir_toSave, testing_is_equal);

        # Save the figure
        #plt.pause(0.0005);
        figure = plt.gcf() # get current figure
        figure.set_size_inches(20, 20)
        figures_dir_toSave = os.path.join(root_dir,'figures', 'figures_epoch'+str(epoch)+'.png');
        
        if epoch==(args.max_epochs-1):
            pass;
            #plt.savefig(figures_dir_toSave,dpi=600);# Only to save figure, but it is useless
        else:
            pass;

    return current_results;

    #plt.show();

# Tune the LSTM model
# max_length The length of the longest sentence
# now: Time
# max_epochs: Maximum number of epochs
# sequence_length: Length of the input sequence
# learning_rate: Learning rate
# lstm_size: Size of the LSTM network
# embedding_dim: Dimension of the embedded dimexnsion
# num_layers: Number of layers of the LSTM network
# dropout: Dropout threshold
def tuning(resultsCSV_dir_to_Save,dataset,subsets,metrics,results,now,max_epochs,batch_size,sequence_length,learning_rate,lstm_size,embedding_dim,num_layers,dropout,path_to_save,model_dir,metrics_dir,figures_dir,results_file_dir):
    if max_epochs<=0 or batch_size<=0 or sequence_length<=0 or learning_rate<=0 or lstm_size<=0 or embedding_dim<=0 or num_layers<=0 or path_to_save=="" or model_dir=="" or metrics_dir=="" or figures_dir=="" or results_file_dir=="" or resultsCSV_dir_to_Save=="":
        raise Exception("[train.py] Error. Wrong input");
    print("sequence_length = ",sequence_length);
    model_name = 'TAMP_' + now + '_epochs'+str(max_epochs);
    dirname = os.path.dirname(__file__);
    #model_name_dir = os.path.join(path_to_save,'trained_models',model_name+'.pth');
    model_name_dir = os.path.join(path_to_save,'trained_models');
    input_name_dir = os.path.join(path_to_save,'trained_models');
    
    parser = argparse.ArgumentParser();# Parentheses are very important after ArgumentParser
    parser.add_argument('--max-epochs', type=int, default=max_epochs);#Original:100,200
    parser.add_argument('--batch-size', type=int, default=batch_size);
    parser.add_argument('--sequence-length', type=int, default=sequence_length);
    args = parser.parse_args();
    #model_name = 'TAMP_' + now + '_epochs'+str(max_epochs)+'_batch'+str(batch_size)+'_length'+str(sequence_length)+'_rate'+str(learning_rate);
    
    dataset.set_args(args);

    parameters = [max_epochs,batch_size,sequence_length,learning_rate,lstm_size,embedding_dim,num_layers,dropout];
    model = Model(dataset,lstm_size,embedding_dim,num_layers,dropout);
    model = model.cuda();
    results = train(resultsCSV_dir_to_Save,now,results,dataset,subsets,metrics,model,model_name_dir,input_name_dir,path_to_save,results_file_dir,parameters,args);
    print({'Training, Testing, and Validation completed'});

    return results;

# Create the results.csv file that contains all the numerical results
# @input results A dummy variable to initialise the "results.csv" file
# @return results_file_dir The absolute path to the "results.csv" file
def create_results_file(results):
    dirname = os.path.dirname(__file__);

    # Create the folder first
    results_file_dir = os.path.join(dirname,'results','resultsFile_folder');
    try:
        os.makedirs(results_file_dir);
    except FileExistsError:
        pass;

    # Create the file and save a dummy results variable
    resultsCSV_dir_to_Save = os.path.join(results_file_dir,'results.csv');
    np.savetxt(resultsCSV_dir_to_Save, results, delimiter=",");

    return resultsCSV_dir_to_Save;
    
# Create the folder where all the information is stored: model+figures
# to_create: 1: Can create the folder 0: Cannot create the folder
# @return path_to_save: Path of the directory where all information must be saved
# @return model_dir: Path to the directory containing the model to save
# @return metrics_dir: Path of the directory where all metrics are saved
# @return figures_dir: Path of the directory where all figures are saved
def create_folder(max_epochs,batch_size,sequence_length,learning_rate,lstm_size,embedding_dim,num_layers,dropout):
    if max_epochs<=0 or batch_size<=0 or sequence_length<=0 or learning_rate<=0 or lstm_size<=0 or embedding_dim<=0 or num_layers<=0:
        raise Exception("[train.py] Error. Wrong input");
    dirname = os.path.dirname(__file__);
    path_to_save = os.path.join(dirname,"results","max_epochs="+str(max_epochs),"batch_size="+str(batch_size),"sequence_length="+str(sequence_length),"learning_rate="+str(learning_rate),"lstm_size="+str(lstm_size),"embedding_dim="+str(embedding_dim),"num_layers="+str(num_layers),"dropout="+str(dropout));# Path to save the data
    model_dir = os.path.join(path_to_save,'trained_models');
    metrics_dir = os.path.join(path_to_save,'metrics');
    figures_dir = os.path.join(path_to_save,'figures');
    ground_truth_training_dir = os.path.join(path_to_save,'ground_truth_training');
    ground_truth_testing_dir = os.path.join(path_to_save,'ground_truth_testing');
    testing_dir = os.path.join(path_to_save,'predictions_testing');
    training_dir = os.path.join(path_to_save,'predictions_training');
    results_file_dir = os.path.join(dirname,'results','resultsFile_folder');
    # python program to check if a path exists
    #if path doesn’t exist we create a new path

    try:
        os.makedirs(ground_truth_testing_dir);
    except FileExistsError:
        pass;
    # If directories already exist, pass
    try:
        os.makedirs(ground_truth_training_dir);
    except FileExistsError:
        pass;
    try:
        os.makedirs(training_dir);
    except FileExistsError:
        pass;
    try:
        os.makedirs(testing_dir);
    except FileExistsError:
        pass;
    try:
        os.makedirs(path_to_save);
    except FileExistsError:
        pass;
    try:
        os.makedirs(model_dir);
    except FileExistsError:
        pass;
    try:
        os.makedirs(metrics_dir);
    except FileExistsError:
        pass;
    try:
        os.makedirs(figures_dir);
    except FileExistsError:
        pass;
    try:
        os.makedirs(results_file_dir);
    except FileExistsError:
        pass;

    return path_to_save,model_dir,metrics_dir,figures_dir,results_file_dir;
    
# Main function. Tune the parameters here.
def main():

    # Numpy array containing all information. It will be saved in the results folder
    results = np.zeros((1,15));
    resultsCSV_dir_to_Save = create_results_file(results);# Create the results.csv file to save all the results from the training, testing, and validation

    # Parameters to optimise
    max_epochs_list = [300];
    batch_size_list = [300];# Number of action sequences that go into one batch
    sequence_length_list = [7];# The number of words that are used as input to the neural network. The rest is padded. For 'open' it is 4. For the rest it is 7.
    learning_rate_list = [0.005];#0.005
    lstm_size_list = [128];
    embedding_dim_list = [128];#128
    num_layers_list = [1];
    dropout_list = [0];
    augmentations = [3];# Number of times each sentence must be augmented. An augmentation of 1 does not modify the dataset

    combinations = np.array(np.meshgrid(max_epochs_list,batch_size_list,sequence_length_list,learning_rate_list,lstm_size_list,num_layers_list,dropout_list)).T.reshape(-1,7);
    now = datetime.now().date();# Get only the date 
    now = now.strftime("%m%d%Y");

    # Default arguments. They will be changed later
    parser = argparse.ArgumentParser();# Parentheses are very important after ArgumentParser
    parser.add_argument('--max-epochs', type=int, default=300);#Original:100,200
    parser.add_argument('--batch-size', type=int, default=300);
    parser.add_argument('--sequence-length', type=int, default=sequence_length_list[0]);
    args = parser.parse_args();
    
    parent_dir = Path(__file__).resolve().parent.parent;# Make it a relative path for simplicity
    dataset = Dataset(args,os.path.join('datasets','Exp4_Normal','annotations_video_IRB_pour.csv'));
    metrics = Metrics(dataset);

    # Split the dataset into training and testing
    train_size = 0.8;test_size = 0.2;# Express the sizes in percentages
    subsets,indices_sentences = utils.dataset_split_wholeDataset(dataset,[train_size,test_size],augmentations[0]);

    # Get the length of the longest sentence. Then, allow the algorithm to pad the sentences, so they all have the same length
    max_length = utils.get_maxLength(dataset,subsets);
    print("max_length = ",max_length);
    dataset.set_to_pad(1);dataset.set_max_length(max_length);

    # From now on, everything is tunable
    for k in range(0,combinations.shape[0]):
        max_epochs = combinations[k,0];
        batch_size = combinations[k,1];
        sequence_length = combinations[k,2];
        learning_rate = combinations[k,3];
        lstm_size = combinations[k,4];# At present, the hidden size and the LSTM's size must have the same value as the size of the LSTM
        embedding_dim = combinations[k,4];# At present, the hidden size and the LSTM's size must have the same value as the size of the LSTM
        num_layers = combinations[k,5];
        dropout = combinations[k,6];
        augmentation = augmentations[0];
        path_to_save,model_dir,metrics_dir,figures_dir,results_file_dir = create_folder(max_epochs,batch_size,sequence_length,learning_rate,lstm_size,embedding_dim,num_layers,dropout);
        print('Selected',{'max_epochs': max_epochs,'batch_size': batch_size, 'sequence_length': sequence_length, 'learning_rate': learning_rate, 'lstm_size': lstm_size, 'embedding_dim': embedding_dim,'num_layers': num_layers,'dropout': dropout});
        results = tuning(resultsCSV_dir_to_Save,dataset,subsets,metrics,results,now,int(max_epochs),int(batch_size),int(sequence_length),learning_rate,int(lstm_size),int(embedding_dim),int(num_layers),dropout,path_to_save,model_dir,metrics_dir,figures_dir,results_file_dir);
        
        torch.cuda.empty_cache();
    print("Done");
if __name__ == "__main__":
    main();