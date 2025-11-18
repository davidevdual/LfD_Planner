import pandas as pd;
import numpy as np;

import csv;
import random

# Generate a random list of numbers
# @input _nb Number of random values inside the list to generate
# @return Two random lists of numbers between 1 and 32. No duplicates
def gen_random_lists(_nb):
	if _nb<=0:
		raise Exception("[create_unknownTasks_pass_exp4.py] Wrong input. The size of the list cannot be negative.");
	numbers_list = [];
	for k in range(1,_nb+1):
		for n in range(1,_nb+1):
			comb = (k,n);
			if k==n:
				pass
			else:
				numbers_list.append(comb);
	return numbers_list;

# Function to read a csv file and insert each row into a list
# @input dataset_dir The directory of the csv file to read
# @input field The name of the field to extract the rows from
def load_rows_list(dataset_dir,field):
	if type(dataset_dir)!=str or len(dataset_dir)<=1 or type(field)!=str or len(field)<=1:
		raise Exception("[create_unknownTasks_pour_exp4.py] Wrong input.");
	train_df = pd.read_csv(dataset_dir);
	text = train_df[field].str.cat(sep=' ');
	text = text.split('\\n');

	while("" in text) :
		text.remove("");
	if len(text)<=1 or type(text)!=list:
		raise Exception("[create_unknownTasks_pour_exp4.py] Wrong computation.");
	return text,train_df;

def main():
	# It is better to write the updated dataset into a new file to avoid moving back in case of a mistake in the writing procedure
	dataset_dir = 'C:/Users/David/OneDrive - National University of Singapore/PhD_work/Code/TAMP_HighLevel/Visual_Studio/taskplanningmodule/Language_Model_many2many_novelty/input_comparisons/input_pour_Exp4_unknownTasks.csv';
	updated_dataset_dir = 'C:/Users/David/OneDrive - National University of Singapore/PhD_work/Code/TAMP_HighLevel/Visual_Studio/taskplanningmodule/Language_Model_many2many_novelty/input_comparisons/input_pour_Exp4_unknownTasks_updated.csv';
	our_goals_list,dataFrame = load_rows_list(dataset_dir,'Ours');
	pddl_goals_list,dataFrame = load_rows_list(dataset_dir,'PDDL');
	numb_list = gen_random_lists(32);
	counter = 0;

	for k in range(0,len(our_goals_list)):
		for l,n in numb_list:
			print("k = ",k, "l = ",l," n = ",n);
			# Extract the new unknown location of the object
			unknown_location1 = l;
			unknown_location2 = n;

			# Extract the object and location from our goal
			our_goal = our_goals_list[k].split(' ');
			our_goal = list(filter(lambda s: s.strip(),our_goal));# Remove the empty strings
			obj1 = our_goal[2];# First object
			obj2 = our_goal[5];# Second object
			loc1 = unknown_location1;# Location of the first object
			loc2 = unknown_location2;# :Location of the second object
			our_goal[3] = str(unknown_location1);
			our_goal[6] = str(unknown_location2) + " \\n";
			print("our_goal = ",our_goal);

			# Replace the PDDL goal with the object and location that have been extracted
			pddl_goal = pddl_goals_list[k].split(' ');
			pddl_goal = list(filter(lambda s: s.strip(),pddl_goal));# Remove the empty strings
			pddl_goal[3] = obj1;
			pddl_goal[4] = str(unknown_location1);
			pddl_goal[5] = obj2;
			pddl_goal[6] = str(unknown_location2)+")))) \\n";
			dataFrame.loc[counter,['PDDL']] = " ".join(pddl_goal);
			dataFrame.loc[counter,['Ours']] = " ".join(our_goal);
			counter = counter + 1;

	dataFrame.to_csv(updated_dataset_dir,mode='w',index=False);

if __name__ == "__main__":
	main();