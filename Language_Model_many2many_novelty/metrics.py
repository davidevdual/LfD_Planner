import torch
import argparse
import numpy.matlib
#import pandas as pd
import numpy as np
from Language_Model_many2many_novelty.dataset import Dataset
from collections import Counter

class Metrics:

    def __init__(self, _dataset):
        self.dataset = _dataset;

    # The filling is done with empty characters
    def fill_matrix(self,bigger_size,smaller_size):
        last_col_repeated = np.matlib.repmat(self.dataset.index_emptyWord+1, 1, bigger_size.shape[1]-smaller_size.shape[1]);
        smaller_size = np.hstack((smaller_size,last_col_repeated));
        return smaller_size;

    def pad(self,input_1,input_2):
        if input_1.shape[1]>input_2.shape[1]:
            input_2 = self.fill_matrix(input_1,input_2);
        elif input_1.shape[1]<input_2.shape[1]:
            input_1 = self.fill_matrix(input_2,input_1);
        elif input_1.shape[1]==input_2.shape[1]:
            print("No padding needed");
        else:
            print("Error");
        return input_1,input_2;

    # Get average of the percentages for matches and mismatches
    def get_percentage(self,nb_matches,nb_mismatches,sentence_length):
        percentage_matches = 0;
        percentage_mismatches = 0;
        if (nb_matches+nb_mismatches)!=sentence_length:
            raise Exception("[metrics.py] Error in the computation. The addition of the number of matchings and mismatchings is not equal to the size of the array");
        if sentence_length<=0:
            raise Exception("[metrics.py] The sentence length cannot be negative or zero.");
        percentage_matches = (nb_matches/sentence_length)*100;
        percentage_mismatches = (nb_mismatches/sentence_length)*100;
        if (round(percentage_matches+percentage_mismatches))!=100:
            raise Exception("[metrics.py] The sum of the percentages is not 100.");
        return percentage_matches,percentage_mismatches;

    # y and yhat are 2D numpy arrays. Get the number of matching and mistmatching words between two sentences
    # y: Numerical array representing a sentence. Each number is mapped to a word.
    # yhat: Numerical array representing a sentence. Each number is mapped to a word.
    def get_nb_words(self,s1,s2):
        nb_wrong_words = 0;nb_correct_words=0;
        if s1.ndim!=2 or s2.ndim!=2:
            raise Exception("[metrics.py] y and yhat must have two dimensions exactly.");
        if s1.shape[0]!=1 or s2.shape[0]!=1:
            raise Exception("[metrics.py] y and yhat must have one row, not multiple.");
        row_s1, col_s1 = s1.shape;row_s2, col_s2 = s2.shape;
        if col_s1!=col_s2:
            s1,s2 = self.pad(s1,s2);
        if s1.shape[1]!=s2.shape[1]:
            raise Exception("[metrics.py] Both matrices, s1 and s2, must have the same number of columns");

        nb_correct_words = np.count_nonzero(s1==s2);# Count the number of TRUE in the array
        nb_wrong_words = np.size(s1==s2) - np.count_nonzero(s1==s2);
        if (nb_correct_words+nb_wrong_words)!=np.size(s1==s2):
            raise Exception("[metrics.py] Error in the computation. The addition of the number of matchings and mismatchings is not equal to the size of the array");
        return nb_correct_words,nb_wrong_words;

    # If two action plans are the same: 1. If two action plans are different: 0.
    # For two action plans to be the same, they need to have the same words at the same places.
    # y: Ground truth
    # y_hat: Predicted action sequence
    def is_action_plan_same(self,y,y_hat):
        is_equal = 0;
        if y.shape[0]!=y_hat.shape[0] or y_hat.shape[1]!=y_hat.shape[1]:
            raise Exception("[metrics.py] The input dimensions mismatch");
        if np.array_equal(y,y_hat) == True:
            is_equal = 1;
        elif np.array_equal(y,y_hat) == False:
            is_equal = 0;
        else:
            raise Exception("[metrics.py] is_equal must be equal either to 1 or 0.");
        return is_equal;
        
    # Compute all the metrics for the ground-truth and predicted action sequence
    # y: Ground truth
    # y_hat: Predicted action sequence
    def compute_metrics(self,y,y_hat):
        if y.shape[0]!=y_hat.shape[0] or y_hat.shape[1]!=y_hat.shape[1]:
            raise Exception("[metrics.py] The input dimensions mismatch");
        total_percentage_matches = np.zeros((y.shape[0],1));total_percentage_mismatches = np.zeros((y.shape[0],1));
        is_equal_mean = np.zeros((y.shape[0],1));
        for row in range(0,y.shape[0]):
            nb_correct_words,nb_wrong_words = self.get_nb_words(y[row,:].reshape(1,-1),y_hat[row,:].reshape(1,-1));
            percentage_matches,percentage_mismatches = self.get_percentage(nb_correct_words,nb_wrong_words,y.shape[1]);
            is_equal = self.is_action_plan_same(y[row,:].reshape(1,-1),y_hat[row,:].reshape(1,-1));
            total_percentage_matches[row,:] = percentage_matches;
            total_percentage_mismatches[row,:] = percentage_mismatches;
            is_equal_mean[row,:] = is_equal;
        return np.mean(total_percentage_matches),np.mean(total_percentage_mismatches),np.mean(is_equal_mean);
'''
def main():
    parser = argparse.ArgumentParser();
    args = parser.parse_args();
    dataset_dir = 'C:/Users/David/OneDrive - National University of Singapore/PhD_work/Code/TAMP_HighLevel/Visual_Studio/taskplanningmodule/Language_Model/data/tasks_test_goodorder.csv';
    dataset = Dataset(args,dataset_dir);
    metrics = Metrics(dataset);
    y = np.array([[1,1,1,2]]);
    yhat = np.array([[1,1,2,2]]);
    y_multiple = np.array([[1,1,1,2],[0,0,0,0]]);
    yhat_multiple = np.array([[1,1,1,2],[0,0,2,0]]);
    nb_correct_words,nb_wrong_words = metrics.get_nb_words(y,yhat);
    percentage_matches,percentage_mismatches = metrics.get_percentage(nb_correct_words,nb_wrong_words,y.shape[1]);
    mean_percentage_matches,mean_percentage_mismatches = metrics.compute_metrics(y_multiple,yhat_multiple);
    print("The total percentage of matches is: ",mean_percentage_matches);
    print("The total percentage of mismatches is: ",mean_percentage_mismatches);

if __name__ == "__main__":
    main();
'''