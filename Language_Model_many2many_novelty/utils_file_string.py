import argparse
import time
import torch
import os
import numpy as np
import csv
import random

def reverse_string(string):
    if type(string)!=str or string=="":
        raise Exception("[utils_file_string.py] Wrong input");
    words = string.split();
    reversed_string = words[-1];
    for word_elmt_idx in range(len(words)-2, -1, -1):# We purposely ignore the last element since it is already in the initialization
        reversed_string = reversed_string + ' ' + words[word_elmt_idx];
    return reversed_string;

# Modify a specific line of a file
# @input filename String. The file name of the file to modify
# @input line Integer. The line to modify
# @input text String. The replacement text
def modify_text_line(filename,line,text):
    if type(filename)!=str or type(line)!=int or type(text)!=str or filename=='' or line<0 or text=='':
        raise Exception("[utils_file_string.py] Wrong input");

    with open(filename, 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    # now change the line, note that we have to add a newline
    data[line] = text;

    # Write the modification back into the file
    with open(filename, 'w') as file:
        file.writelines( data )
    return 0;

def main():
    pddlProblem_filename = 'C:/Users/David/OneDrive - National University of Singapore/PhD_work/Code/TAMP_HighLevel/Visual_Studio/taskplanningmodule/Fast_Downward/task_bimanual_2.pddl';
    modify_text_line(pddlProblem_filename,6,'1 - location');

if __name__ == "__main__":
    main();