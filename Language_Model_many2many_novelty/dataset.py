'''
    File name: dataset.py
    Author: David Carmona
    Date created: 01/12/2022
    Date last modified: 01/12/2022
    Python Version: 3.7.6
    Description: This piece of code pads all the sentences in the dataset so they all have the same length.
                 Otherwise, the Machine Learning algorithm will not accept the sequences.
'''

import torch
import Language_Model_many2many_novelty.utils_file_string as utils_file_string
import argparse
import os
import pandas as pd
import numpy as np
import numpy.ma as ma
from collections import Counter
torch.device('cuda')

class Dataset(torch.utils.data.Dataset):
    def __init__(self,args,dataset_dir):
        print("dataset_dir: ",dataset_dir);
        self.args = args;
        self.rows_list = self.load_rows_list(dataset_dir);
        self.rows_list = self.reverse_list_sentences(self.rows_list);# The model expects the sentences to be reversed. Therefore, each sentence in the list is reversed.
        self.rows_list = self.convert_actionPlan_list();
        self.words = self.load_words(dataset_dir);
        self.words.append('');# Append the empty character
        self.uniq_words = self.get_uniq_words(self.words);

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)};
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)};
        print("self.word_to_index: ",self.word_to_index);

        # Indices mapping unknown positions to unique identifiers
        self.unknown_word_index = list(self.index_to_word.keys())[-1] + 1;# The novel words have indices outside of the dictionary
        self.unknown_index_to_word = {};
        self.unknown_word_to_index = {};

        self.words_indexes = [self.word_to_index[w] for w in self.words];
        self.rows_indexes = self.assign_indexes_rows();# Contains the index of each word in the action sequences
        self.index_emptyWord = self.word_to_index[''];# Get the index of the empty word 

        self.nb_batches = 0;
        self.max_length = 0;
        self.to_pad = 0;

    # Reverse the sentences inside the list
    # @input list_sentences List of sentences. Each sentence is in the correct order. This function is to reverse each on
    # @return A list of sentences. All the sentences are reversed
    def reverse_list_sentences(self,list_sentences):
        if type(list_sentences)!=list or len(list_sentences)<=0:
            raise Exception("[dataset.py] Wrong input");
        list_reversed_sentences = [];
        for sentence in list_sentences:
            sentence = utils_file_string.reverse_string(sentence);
            list_reversed_sentences.append(sentence);
        if len(list_reversed_sentences)<=0:
            raise Exception("[dataset.py] Wrong input");
        return list_reversed_sentences;

    # Get the dictionary containing a mapping of the unknown words to their unique identifiers
    # @return A dictionary containing a mapping of the unknown words to their unique identifiers
    def get_unknown_word_to_index(self):
        return self.unknown_word_to_index;

    # Get the dictionary containing a mapping of the unknown' words' unique identifiers to their string
    # @return A dictionary containing a mapping of unique identifers to the unknown words
    def get_unknown_index_to_word(self):
        return self.unknown_index_to_word;

    # Obtain the indices from the words
    # @return A dictionary containing all indexes to words
    def get_index_to_word(self):
        return self.index_to_word;

    # Check that there are no duplicated elements across two lists 
    # @input list 1 A List
    # @input list2 A list
    # #output 1 if there are duplicates. 0 if there are no duplicates
    def is_duplicate(self,list1,list2):
        if len(list1)<=0 or len(list2)<=0:
            raise Exception("[dataset.py] Wrong input.");
        duplicate = 0;
        for item1 in list1:
            for item2 in list2:
                if item1==item2:
                    duplicate = 1;
        return duplicate;

    # Concatenate two lists, list1 and list2. list2 is concatenated at a certain position in list1
    # @input list1 The list to change
    # @input list2 The list to be inserted
    # @input index The index of list1 where the first element of list2 is inserted 
    # @output The list with all the new elements inserted
    def add(self,list1,list2,index):
        # One of the conditions to test is that there are no duplicates in list1 and list2
        if len(list1)<=0 or len(list2)<=0 or index<-1*len(list1) or index>len(list1) or type(index)!=int or type(list1)!=list or type(list2)!=list:
            raise Exception("[dataset.py] Wrong input.");
        elif self.is_duplicate(list1,list2)==1:
            raise Exception("[dataset.py] list2 contains items of list1. This is illegal.");
        # Insert the element before the last item of the list since the last one is the empty character
        new_list = list1[:index];
        for item in list2:
            new_list.append(item);
        new_list.append(list1[-1]);

        return new_list;

    # Add words to the already-existing vocabulary
    # @input words A list of words to add to the vocabulary
    # @output List of unique words with new ones inserted
    def add_words_to_vocabulary(self,words):
        if len(words)<=0 or len(self.uniq_words)<=0 or type(words)!=list:
            raise Exception("[dataset.py] Wrong input.");
        self.uniq_words = self.add(self.uniq_words,words,-1);
        return self.uniq_words;

    # Getter to retrieve the length of the longest sentence
    # @return The longest sentence's length
    def get_max_length(self):
        return self.max_length;

    # Set the padding boolean variable
    # @input _to_pad The padding boolean variable. Can be either 1 (i.e.,pad) or 0 (i.e.,not to pad)
    def set_to_pad(self,_to_pad):
        self.to_pad = _to_pad;

    # Set the maximum length of the sentences
    # @input _max_length The desired sentence's maximum length
    def set_max_length(self,_max_length):
        self.max_length = _max_length;

    # Set the number of batches
    def get_number_batches(self,nb_batches):
        self.nb_batches = nb_batches;

    # Get the sentence's length. At present, all sentences have same lengths by default. 
    # Therefore, the length of the action sentence is equal to the length of the first sentence.
    def get_sentence_length(self):
        if len(self.rows_list)<=0:
            raise Exception("[dataset.py] There seems to be no sentences in the dataset");
        return len(self.rows_list[0]);

    # Set the arguments for the Dataset
    def set_args(self,args):
        self.args = args;

    # Convert the action plan inside each row to a list 
    def convert_actionPlan_list(self):
        action_plans_list = [];
        for row in self.rows_list:
            row = row.split(" ");
            while('' in row):
               row.remove('');
            action_plans_list.append(row);
        return action_plans_list;
        
    # Assign the unique word indexes to each word in the action plan
    def assign_indexes_rows(self):
        indexes = [];
        for row in self.rows_list:
            row_indexes = [self.word_to_index[w] for w in row];
            indexes.append(row_indexes);
        return indexes;
        
    # Function to read a csv file and build a unique string out of all the action plans
    def load_words(self,dataset_dir):
        train_df = pd.read_csv(dataset_dir);
        text = train_df['Task'].str.cat(sep=' ');
        text = text.split(" ");
        while("\\n" in text):
            text.remove("\\n");
        return text;

    # Function to read a csv file and insert each action plan into a list
    def load_rows_list(self,dataset_dir):
        train_df = pd.read_csv(dataset_dir);
        text = train_df['Task'].str.cat(sep=' ');
        text = text.split('\\n');
        while("" in text) :
            text.remove("");
        return text;

    def get_uniq_words(self,str):
        word_counts = Counter(str)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def compute_index_to_word(self,indices):
        words_matrix = np.zeros((1, indices.shape[1]));
        for row in indices:
            words_row = [self.index_to_word[w] for w in row];
            words_matrix = np.vstack((words_matrix,words_row));
        words_matrix = np.delete(words_matrix, 0, axis=0);
        return words_matrix

    # Pad a list according to the sequence length. If there is a need to pad the list, the emtpy word's index is added as the padding value
    def pad_list(self,list_to_pad):
        padded_list = list_to_pad;
        if len(list_to_pad) != self.args.sequence_length:
            padded_list = list_to_pad + [self.index_emptyWord] * (self.args.sequence_length - len(list_to_pad));
        else:
            padded_list = list_to_pad;
        return padded_list;

    # Assign indices to the different action plans. This will be used later to compute the accuracies for the action plan predictions
    # action_plans: 3D tensor containing all the action plans
    def assign_indexes_actionPlans(self,action_plans):
        if action_plans.ndim!=3 and action_plans.ndim!=2:
            raise Exception('[dataset.py] The number of dimensions must be equal to three or two');
        indexes_matrix = np.zeros(action_plans.shape,dtype='int');

        if action_plans.ndim==3:
            nb_rows,nb_cols,nb_depth = action_plans.shape;
            for row in range(0,nb_rows):
                for col in range(0,nb_cols):
                    for depth in range(0,nb_depth):
                        indexes_matrix[row,col,depth] = row;
        if action_plans.ndim==2:
            nb_rows,nb_cols = action_plans.shape;
            for row in range(0,nb_rows):
                for col in range(0,nb_cols):
                    indexes_matrix[row,col] = row;
        return indexes_matrix;

    # Form an action plan based on a 3D matrix
    # nb_plans: Total number of action plans
    # x2d: 2D array containing all the action plans
    # indexes: 2D matrix containing indexes indicating which rows belong to the same action plan
    def form_action_sequences(self,nb_plans,indexes,x2d,nb_rows_first_elmt,nb_cols_first_elmt):
       # print("The number of plans is: ",nb_plans," indexes.shape[0]= ",indexes.shape[0]," indexes.shape[1]= ",indexes.shape[1]," x2d.shape[0]= ",x2d.shape[0]," x2d.shape[1]= ",x2d.shape[1]);
        if indexes.shape[0]!=nb_plans*nb_rows_first_elmt or x2d.shape[0]!=nb_plans*nb_rows_first_elmt:
            raise Exception('[dataset.py] The first dimension of the number of indexes must be equal to the number of plans multiplied by the indexes'' second dimension');
        if x2d.ndim!=2 or indexes.ndim!=2 or x2d.shape[0]!=indexes.shape[0] or x2d.shape[1]!=indexes.shape[1]:
            raise Exception('[dataset.py] The number of dimensions must be equal to three or indexes and x2d must have same dimensions');
        action_sequences = np.zeros((nb_plans,nb_rows_first_elmt-1+nb_cols_first_elmt),dtype='int');
        nb_rows = action_sequences.shape[0];nb_cols = action_sequences.shape[1];

        for plan_id in range(0,nb_rows):
            mask = (indexes == plan_id).astype(int);
            rows,cols = np.where(mask==True);
            plan = x2d[rows,cols].reshape((nb_rows_first_elmt,nb_cols_first_elmt));
            action_sequences[plan_id,0:indexes.shape[1]] = plan[0,:];
            last_elmts = plan[1:,-1];
            action_sequences[plan_id,indexes.shape[1]:(nb_cols)] = last_elmts;
                
        return action_sequences;

    # Get the words' numerical indexes
    def get_words_indexes(self):
        return self.words_indexes;

    # Replace the action plan by a series of characters to indicate the lstm to infer the words
    # @input actionPlan_words_indexes The action sequence. Each number corresponds to a unique word
    # @return The x (input) and y (ground truth) matrices for training the network
    def pad_sentence(self,actionPlan_words_indexes):

        # The total length of the sequence and the length of the input sequence
        length_seq = len(actionPlan_words_indexes);length_input_seq = self.args.sequence_length;last_index = self.words_indexes[-1];
        nb_words_toPredict = length_seq - length_input_seq;

        if nb_words_toPredict <= 0 or length_seq<=0 or length_input_seq<=0:
            raise Exception("[dataset.py] The number of words to predict cannot be less or equal to zero.");

        if self.to_pad == 1 and self.max_length>0:
            #print("length_seq: ",length_seq," length_input_seq: ",length_input_seq," last_index: ",last_index," nb_words_toPredict: ",nb_words_toPredict);
            #print("max_length= ",self.max_length);
            if self.max_length <=0 or self.max_length<length_input_seq or self.max_length<length_seq:
                raise Exception("[dataset.py] The maximum length is incorrect.");
            x_mat_init = last_index*np.ones((1,self.max_length),dtype="int");
            y_mat_init = last_index*np.ones((1,self.max_length),dtype="int");
            x_mat_init[-1:,0:length_input_seq] = np.array(actionPlan_words_indexes[0:length_input_seq]);
            y_mat_init[-1:,0:length_seq] = np.array(actionPlan_words_indexes[0:length_seq]);
            x = x_mat_init.reshape(-1).tolist();y = y_mat_init.reshape(-1).tolist();
            #print("x: ",x," y: ",y, " max_length= ",self.max_length);

            if len(x)!=len(y) or len(x)!=self.max_length or len(y)!=self.max_length:
                raise Exception("[dataset.py] The actual and expected lengths mismatch.");

        elif self.to_pad == 0:
            x = actionPlan_words_indexes;y=actionPlan_words_indexes;
        else:
            raise Exception("[dataset.py] The padding option is wrong. It should be either 0 (i.e., padding wanted) or 1 (i.e., padding unwanted) or the longest sentence's length is wrong");
        return x,y;

    def __len__(self):
        return len(self.rows_list);
        #return len(self.words_indexes) - self.args.sequence_length;

    # This function is called to retrieve an element from the dataset. For instance: 
    # train_dataset = subsets[0];elmt = train_dataset[0];
    # index is the index of the element in the action plans list
    def __getitem__(self, index):
        if index<0 or index>=len(self.rows_indexes):
            raise Exception("[dataset.py] Input error");

        #print("index = ",index);
        #actionPlans_selct = self.rows_list[index];
        #actionPlan_words_indexes = [];
        x_list = [];
        y_list = [];

        #for actionPlan_words in actionPlans_selct:
        #    index = self.word_to_index[actionPlan_words];
        #    actionPlan_words_indexes.append(index);
        
        # We don't want to pad the sentences now. Therefore, padding is 0.
        x,y = self.pad_sentence(self.rows_indexes[index]);
        #print("self.rows_indexes: ",self.rows_indexes);

        x_list.append(x);
        y_list.append(y);

        if len(x_list)!=1 or len(y_list)!=1:
            raise Exception("[dataset.py] The lengths of x_list and y_list are supposed to be one, not several.");

        return (
            torch.tensor(np.array(x_list)),
            torch.tensor(np.array(y_list)),
        );

    def compute_word_to_index(self,indexMat):
        num_indexes_mat = np.zeros(indexMat.shape,dtype='int');

        if num_indexes_mat.ndim == 3:
            nb_rows, nb_cols, nb_depth = num_indexes_mat.shape;
            #print("nb_rows: ",nb_rows," nb_cols: ",nb_cols," nb_depth: ",nb_depth);
            for row in range(0,nb_rows):
                for col in range(0,nb_cols):
                    for depth in range(0,nb_depth):
                        try:
                            num_indexes_mat[row,col,depth] = self.word_to_index[indexMat[row,col,depth]];
                        except KeyError:
                            if type(indexMat[row,col,depth]==int) and len([indexMat[row,col,depth]])==1:
                                self.add_words_to_vocabulary([indexMat[row,col,depth]]);
                                self.index_to_word[self.unknown_word_index] = indexMat[row,col,depth];
                                self.word_to_index[indexMat[row,col,depth]] = self.unknown_word_index;
                                num_indexes_mat[row,col,depth] = self.word_to_index[indexMat[row,col,depth]];
                                self.unknown_word_index = self.unknown_word_index + 1;
                            else:
                                raise Exception("[dataset.py] The unknown word is not an integer. It should be. It should be. There might be a wrong input error as well.");
                        except:
                            raise Exception("[dataset.py] Problem with the mapping of each word to a number.");

        elif num_indexes_mat.ndim == 2:
            nb_rows, nb_cols = num_indexes_mat.shape;
            for row in range(0,nb_rows):
                for col in range(0,nb_cols):
                    try:
                        num_indexes_mat[row,col] = self.word_to_index[indexMat[row,col]];
                    except KeyError:
                        if type(indexMat[row,col]==int) and len([indexMat[row,col]])==1:
                            self.add_words_to_vocabulary([indexMat[row,col]]);
                            self.index_to_word[self.unknown_word_index] = indexMat[row,col];
                            self.word_to_index[indexMat[row,col]] = self.unknown_word_index;
                            num_indexes_mat[row,col] = self.word_to_index[indexMat[row,col]];
                            self.unknown_word_index = self.unknown_word_index + 1;
                        else:
                            raise Exception("[dataset.py] The unknown word is not an integer. It should be. There might be a wrong input error as well.");
                    except:
                        raise Exception("[dataset.py] Problem with the mapping of each word to a number.");
        else:
            raise Exception("[dataset.py] Please, input a 3D or 2D matrix, not other dimension.");
        return num_indexes_mat;

    '''
    def compute_index_to_word(self,indexMat):
        num_indexes_mat = np.zeros(indexMat.shape,dtype=object);

        if num_indexes_mat.ndim == 3:
            nb_rows, nb_cols, nb_depth = num_indexes_mat.shape;
            #print("nb_rows: ",nb_rows," nb_cols: ",nb_cols," nb_depth: ",nb_depth);
            for row in range(0,nb_rows):
                for col in range(0,nb_cols):
                    for depth in range(0,nb_depth):
                        num_indexes_mat[row,col,depth] = self.index_to_word[indexMat[row,col,depth]];

        elif num_indexes_mat.ndim == 2:
            nb_rows, nb_cols = num_indexes_mat.shape;
            for row in range(0,nb_rows):
                for col in range(0,nb_cols):
                    num_indexes_mat[row,col] = self.index_to_word[indexMat[row,col]];
        return num_indexes_mat;
    '''

    def convert_matrix_to_words(self,indexMat):
        if torch.is_tensor(indexMat)==True:
            word_matrix = self.compute_index_to_word(indexMat.cpu().numpy());
        elif type(indexMat) is np.ndarray:
            word_matrix = self.compute_index_to_word(indexMat);
        else:
            raise Exception("[dataset.py] indexMat is neither a Torch Tensor nor a numpy array. It should be.");
        return word_matrix;

    # Get the number of data points inside the batch
    # @input data_loader Training data loader containing all input
    def get_batch_size(self,train_loader,testing_loader):
        batch_sizes_list = []
        for batch, (x,y) in enumerate(train_loader):
            row_x, col_x, depth_x = x.shape;row_y, col_y, depth_y = y.shape;
            #print("row_x_training: ",row_x," col_x_training: ",col_x);
            batch_sizes_list.append(row_x*col_x);
            #print("batch= ",batch);

        for batch_testing, (x_testing,y_testing) in enumerate(testing_loader): 
            row_x_testing, col_x_testing, depth_x_testing = x_testing.shape;row_y_testing, col_y_testing, depth_y_testing = y_testing.shape;
            #print("row_x_testing: ",row_x_testing," col_x_testing: ",col_x_testing);
            batch_sizes_list.append(row_x_testing*col_x_testing);

        # Checkt that all batch sizes are the same for the training set. If they are not, throw error
        is_all_same = batch_sizes_list[:-1] == batch_sizes_list[1:];
        #print("batch_sizes_list: ",batch_sizes_list);
        
        if is_all_same != True:
            raise Exception("[dataset.py] The batch sizes are not the same across different training sets. The smallest batch size is: ",np.amin(np.array(batch_sizes_list)));
        return batch_sizes_list[0];

    '''
def main():

    # Default arguments. They will be changed later
    parser = argparse.ArgumentParser();# Parentheses are very important after ArgumentParser
    parser.add_argument('--max-epochs', type=int, default=300);#Original:100,200
    parser.add_argument('--batch-size', type=int, default=300);
    parser.add_argument('--sequence-length', type=int, default=7);
    args = parser.parse_args();

    # Load the Dataset
    dataset = Dataset(args,'', 'datasets','tasks_test_goodorder_forDebug.csv'));

if __name__ == "__main__":
    main();
    '''