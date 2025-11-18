import numpy as np
import numpy.matlib
import os
import time
import sys
import copy
import numpy as np
import motionplanmodule as mp
import Fast_Downward.fastdownward as fd
import Language_Model.humanTAMP as human_tamp
from itinterpreter import ItInterpreter

def compute_task_plan():
	task_planner = human_tamp.HumanTAMP("C:/Users/David/OneDrive - National University of Singapore/PhD_work/Code/TAMP_HighLevel/Visual_Studio/taskplanningmodule/Language_Model/trained_models/model_TAMP_22_02_2020.pth");
	task_planner.compute_plan("pouring pitcher_base into mug");
	task_plan = task_planner.get_plan();
	is_simulation = 1;#1 if simulation desired. Otherwise, 0 if simulation not desired.
	is_vision = 0;#1 if the computer is connected to the RGB-D camera. 0 if the camera is not connected to the RGB-D camera.
	interpreter = ItInterpreter();
	interpreter.compute_motion_plan("approach handleft cup and approach handright bottle and enclose handright bottle and enclose handleft cup and approach handright cup to pour");
	return task_plan;

if __name__ == "__main__":
	task_plan = compute_task_plan();