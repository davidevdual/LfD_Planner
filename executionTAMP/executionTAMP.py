import numpy as np
import numpy.matlib
import os
import time
import sys
import copy
import numpy as np
#import Fast_Downward.fastdownward as fd
import Language_Model_many2many_novelty.lfdtaskplanner as lfdtaskplanner
#import itinterpreter as itinterpreter

sys.path.append('C:\\Users\\DNCM\\Documents\\dual-arm-nus\\windows_legion7\\models');

def execute_tp_mp():

	# Objects and threshold to grasp the objects
	#object1 = [[-207.20477789,-312.875389,-270,0.03293905,-0.35213897,0.36608334,-0.86075325]];#Pst: 8
	#object2 = [[200,-312.875389,-316.98958698,0.06026358,0.12072236,-0.76464444,-0.63016926]];#Pst 1
	#object1 = [[200,-450,-270,0.03293905,-0.35213897,0.36608334,-0.86075325]];#Pst: 31
	#object2 = [[-200,-312.875389,-316.98958698,0.06026358,0.12072236,-0.76464444,-0.63016926]];#Pst 11

	## Pair of objects for pouring a pitcher 4 into a mug 14
	#object1 = [[-103,-333,-119,0.49999999,0.49999991,0.50000009,-0.50000001]];# Pitcher [[-103,-333,-119,0.49999999,0.49999991,0.50000009,-0.50000001]]
	#object2 = [[50,-500,5,0.02527894,0.01340268,-0.64181416,-0.76632625]];# Mug

	## Pair of objects for pouring a cracker_box 4 into a bowl 21
	object1 = [[-103,-333,-119,0.49999999,0.49999991,0.50000009,-0.50000001]];# cracker_box
	object2 = [[30,-500,5,0.02527894,0.01340268,-0.64181416,-0.76632625]];# bowl

	## Pair of objects for passing a masterchef can 13 to an experimenter hand 6
	#object1 = [[-103,-333,-119,0.49999999,0.49999991,0.50000009,-0.50000001]];# masterchef can
	#object2 = [[30,-500,5,0.02527894,0.01340268,-0.64181416,-0.76632625]];# experimenter_hand

	## Object to open
	#object1 = [[-103,-333,-119,0.49999999,0.49999991,0.50000009,-0.50000001]];# masterchef can

	threshold = 0.1;
	task_name = 'open';# Type of task being done. It is important for executing the task planner

	if task_name=='pass' or task_name=='pour':
		sequence_length = 7;# 'open' is 4. For the rest of the tasks is 7.
	elif task_name=='open':
		sequence_length = 4;
	else:
		raise Exception("[performance_testing.py] Unrecognised task.");

	# Compute the task plan
	model_path = 'models\TAMP_09202023_epochs300_batch300_length4_rate0.005.pth';
	dataset_name = 'annotations_video_IRB_open.csv';
	task_planner = lfdtaskplanner.LfDTaskPlanner(model_path,dataset_name,task_name,sequence_length);
	#actionPlan = task_planner.compute_plan('to pass master_chef_can 13 to experimenter_hand 6');
	actionPlan = task_planner.compute_plan('to open bleach_cleanser 6');
	print("actionPlan: ",actionPlan);
	exit(0);
	#actionPlan = "approach handleft bleach_cleanser and enclose handleft bleach_cleanser and approach handright bleach_cleanser to open bleach_cleanser 6";
	#print("new actionPlan: ",actionPlan);

	# Compute the motion plan
	#is_simulation = 0;#1 if simulation desired. Otherwise, 0 if simulation not desired.
	#is_vision = 0;#1 if the computer is connected to the RGB-D camera. 0 if the camera is not connected to the RGB-D camera.
	#motion_planner = itinterpreter.ItInterpreter(is_simulation,is_vision,threshold,tuple((object1,object2)));# This entry is valid for 'pass' and 'open' tasks
	#motion_planner = itinterpreter.ItInterpreter(is_simulation,is_vision,threshold,tuple((object1)));
	#motion_planner.compute_motion_plan(actionPlan);
	#motion_planner.execute_commands();
	return 0;

if __name__ == "__main__":
	task_plan = execute_tp_mp();
