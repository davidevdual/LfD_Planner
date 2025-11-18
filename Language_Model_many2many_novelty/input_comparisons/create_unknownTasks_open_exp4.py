import pandas as pd;
import numpy as np;

import csv;
import random

# Function to read a csv file and insert each row into a list
# @input dataset_dir The directory of the csv file to read
# @input field The name of the field to extract the rows from
def load_rows_list(dataset_dir,field):
	if type(dataset_dir)!=str or len(dataset_dir)<=1 or type(field)!=str or len(field)<=1:
		raise Exception("[create_unknownTasks_open_exp4.py] Wrong input.");
	train_df = pd.read_csv(dataset_dir);
	text = train_df[field].str.cat(sep=' ');
	text = text.split('\\n');

	while("" in text) :
		text.remove("");
	if len(text)<=1 or type(text)!=list:
		raise Exception("[create_unknownTasks_open_exp4.py] Wrong computation.");
	return text,train_df;

def main():
	# It is better to write the updated dataset into a new file to avoid moving back in case of a mistake in the writing procedure
	dataset_dir = 'C:/Users/David/OneDrive - National University of Singapore/PhD_work/Code/TAMP_HighLevel/Visual_Studio/taskplanningmodule/Language_Model_many2many_novelty/input_comparisons/input_open_Exp4_unknownTasks.csv';
	updated_dataset_dir = 'C:/Users/David/OneDrive - National University of Singapore/PhD_work/Code/TAMP_HighLevel/Visual_Studio/taskplanningmodule/Language_Model_many2many_novelty/input_comparisons/input_open_Exp4_unknownTasks_updated.csv';
	our_goals_list,dataFrame = load_rows_list(dataset_dir,'Ours');
	pddl_goals_list,dataFrame = load_rows_list(dataset_dir,'PDDL');
	counter = 0;

	for k in range(0,len(our_goals_list)):
		for n in range(1,33):

			# Extract the new unknown location of the object
			unknown_location = n;

			# Extract the object and location from our goal
			our_goal = our_goals_list[k].split(' ');
			our_goal = list(filter(lambda s: s.strip(),our_goal));# Remove the empty strings
			obj = our_goal[2];# Object
			our_goal[3] = str(unknown_location) + " \\n";
			loc = unknown_location;# Location

			# Replace the PDDL goal with the object and location that have been extracted
			pddl_goal = pddl_goals_list[k].split(' ');
			pddl_goal = list(filter(lambda s: s.strip(),pddl_goal));# Remove the empty strings
			pddl_goal[3] = obj;
			pddl_goal[4] = str(loc)+")))) \\n";
			dataFrame.loc[counter,['PDDL']] = " ".join(pddl_goal);
			dataFrame.loc[counter,['Ours']] = " ".join(our_goal);\
			counter = counter + 1;

	dataFrame.to_csv(updated_dataset_dir,mode='w',index=False);

if __name__ == "__main__":
	main();