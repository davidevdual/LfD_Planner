import argparse
import time
import torch
import os
import numpy as np
import torch.utils.data.dataset as dtst
import csv
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from Language_Model_many2many_novelty.model import Model
from Language_Model_many2many_novelty.dataset import Dataset
from torch import default_generator
from itertools import accumulate as _accumulate
from torch import nn, optim
from datetime import datetime
from torch.utils.data import DataLoader
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple

T = TypeVar('T');

# Augment the list of indices in the dataset.
# @input indices List containing each index of each sentence in the dataset. The list has been shuffled
# @input nb Integer. Number of times each sentence should be suplicated.
# @return List of duplicated sentences' indexes.
def augment_dataset(indices,nb):
    if len(indices)<=0 or nb<=0:
        raise Exception("[train.py] Wrong input.");
    new_list_indices = [];
    for k in range(0,len(indices)):
        for p in range(0,nb):
            new_list_indices.append(indices[k]);
    return new_list_indices;

def get_lengths(subsets):
    if len(subsets) <=0:
        raise Exception("[train_padding.py] The length of the subsets input cannot be lower or equal to zero.");
    lengths = [];
    for k in range(0,len(subsets)):
        dataset = subsets[k];
        for l in range(0,len(dataset)):
            sentence_tuple = dataset[l];
            x = sentence_tuple[0];y = sentence_tuple[1];
            if x.shape[0]!=1 or y.shape[0]!=1:
                raise Exception("[train_padding.py] The number of rows of x or y cannot be different than 1.");
            lengths.append(x.shape[1]);
    if len(lengths)<=0:
        raise Exception("[train_padding.py] The lengths of the sentences cannot be lower or equal to zero.");
    return lengths;

def get_maxLength(dataset,subsets):
    lengths = get_lengths(subsets);
    last_index = dataset.get_words_indexes();
    last_index = last_index[-1];
    max_length = np.amax(np.array(lengths)); 
    return max_length;

# Compute the number of points that have to go inside the training and testing sets, respectively
# @input nb_points Total number of points
# @input train_perc Percentage (from 0 to 1) of points that are going into the training set
# @input test_perc Percentage (from 0 to 1) of points that are going into the testing set
# @return the output is three-folded: 1) The total size of the dataset; 2) The size of the training set; 3) The size of the testing set
def compute_nbPoints(nb_points,train_perc,test_perc):
    if nb_points<=0 or train_perc<=0 or test_perc<=0 or train_perc>=1 or test_perc>=1 or (train_perc+test_perc)!=1:
        raise Exception("[utils.py] Wrong input.");
    train_size = 0;
    test_size = 0;
    dataset_size = nb_points;
    if train_size>=test_size:
        train_size = int(train_perc * dataset_size);
        test_size = dataset_size - train_size;
    elif train_size<test_size:
        test_size = int(test_perc * dataset_size);
        train_size = dataset_size - test_size;
    else:
        raise Exception("[train.py] train_size and/or test_size are wrong.");

    if (train_size+test_size)>dataset_size or train_size<=0 or test_size<=0:
        raise Exception("[train.py] Wrong computation");

    return dataset_size,train_size,test_size;

# Split the dataset into a training and a testing set. However, the split is not done according to the number of unique sentences,
# but the total number of sentences after applying the augmentation
# @input nb Integer. Number of times each unique sentence should be duplicated
def dataset_split_wholeDataset(dataset: dtst.Dataset[T], lengths: Sequence[int],nb) -> List[dtst.Subset[T]]:
    if lengths[0]<=0 or lengths[1]<=0 or lengths[0]>=1 or lengths[1]>=1 or (lengths[0]+lengths[1])!=1:
        raise Exception("[utils.py] The training or testing sets cannot have a size equal to zero or negative. They cannot be higher than 1 as well.");
    if len(dataset)<=0:
        raise Exception("[utils.py] The dataset's length cannot be equal to 0 or negative");
    if nb<=0:
        raise Exception("[utils.py] The number of duplicates cannot be lower or equal to 0.");
    random.seed(0);# np.random.seed(0) makes the random numbers predictable
    indices = list(range(0,len(dataset)));# The indices of each sentence in the dataset
    split_list = [];
    indices_sentences = [];
    augmented_indices = [];

    # Augment the subsets by a xNB of times. Then, compute the sizes of the training/testing/validation sets according to the computations
    augmented_indices = augment_dataset(indices,nb);
    random.shuffle(augmented_indices);# Shuffle the augmented indices randomly
    dataset_size,train_size,test_size = compute_nbPoints(len(augmented_indices),lengths[0],lengths[1]);
    lengths[0] = train_size;lengths[1] = test_size;

    # Split the sentences into different sets (training/testing or training/testing/validation)
    for offset, length in zip(_accumulate(lengths), lengths):
        indices_sentences.append(augmented_indices[offset - length : offset]);

    # Get the subsets for each index in the dataset
    for n in range(0,len(indices_sentences)):
        subset = dtst.Subset(dataset,indices_sentences[n]);
        split_list.append(subset);
    subset_indices = split_list[0];

    if len(split_list)!=len(indices_sentences):
        raise Exception("[train.py] Computation error");
    return split_list,augmented_indices;

# Split the dataset into a training and a testing sets
# @input nb Integer. Number of times each sentence should be suplicated.
def dataset_split(dataset: dtst.Dataset[T], lengths: Sequence[int],nb) -> List[dtst.Subset[T]]:
    if lengths[0]<=0 or lengths[1]<=0:
        raise Exception("[train.py] The training or testing sets cannot have a size equal to zero or negative.");
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError("[train.py] Sum of input lengths does not equal the length of the input dataset!");
    if nb<=0:
        raise Exception("[train.py] The number of duplicates cannot be lower or equal to 0.");
    random.seed(0);# np.random.seed(0) makes the random numbers predictable
    indices = list(range(0,sum(lengths)));
    random.shuffle(indices);
    split_list = [];
    indices_sentences = [];
    augmented_indices = [];

    # Split the sentences into different sets (training/testing or training/testing/validation)
    for offset, length in zip(_accumulate(lengths), lengths):
        indices_sentences.append(indices[offset - length : offset]);

    # Augment the subsets by a xNB of times
    for k in range(0,len(indices_sentences)):
        indices = augment_dataset(indices_sentences[k],nb);
        augmented_indices.append(indices);

    # Get the subsets for each index in the dataset
    for n in range(0,len(augmented_indices)):
        subset = dtst.Subset(dataset,augmented_indices[n]);
        split_list.append(subset);
    subset_indices = split_list[0];

    if len(split_list)!=len(augmented_indices):
        raise Exception("[train.py] Computation error");

    return split_list,augmented_indices;

def compute_sizes(dataset,train_perc,test_perc):
    # Check that the input is correct
    if (train_perc+test_perc)!=1 or train_perc<=0 or test_perc<=0:
        raise Exception("[train.py] Wrong input");
    train_size = 0;
    test_size = 0;
    dataset_size = len(dataset);
    if train_size>=test_size:
        train_size = int(train_perc * dataset_size);
        test_size = dataset_size - train_size;
    elif train_size<test_size:
        test_size = int(test_perc * dataset_size);
        train_size = dataset_size - test_size;
    else:
        raise Exception("[train.py] train_size and/or test_size are wrong.");

    if (train_size+test_size)>dataset_size or train_size<=0 or test_size<=0:
        raise Exception("[train.py] Wrong computation");

    return dataset_size,train_size,test_size;

def list_to_string(s):
    str1 = " "
    return(str1.join(s))

# @input A tensor
# @return A tensor
def pad_sentence(input,dataset):
    input = np.transpose(input.numpy());
    length_seq = input.shape[1];
    words_indexes = dataset.get_words_indexes();
    max_length = dataset.get_max_length();
    last_index = words_indexes[-1];
    x_mat_init = last_index*np.ones((1,max_length),dtype="int");
    x_mat_init[-1:,0:length_seq] = np.array(input[0:length_seq]);
    x = torch.from_numpy(np.transpose(x_mat_init));
    #print("words_indexes: ",input," x: ",x," last index dataset: ",last_index," length_seq: ",length_seq);
    return x;
