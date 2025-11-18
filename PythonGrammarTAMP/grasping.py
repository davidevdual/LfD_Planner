import numpy as np
import motionplanmodule as mp
import ikmodule as ik
import utils as uts
import trajectorygeneration as tg
from abc import ABC, ABCMeta, abstractmethod

## Motion Planner Grammar. It inherits from the MotionPlanner class since it is a type of motion planner.
## MotionPlannerGrammar is called by the parser to execute the appropriate commands.
##
#
class GraspingGrammar(object):
	
    ## The end effector does not execute any action.
	#	@param pose The end-effector's pose.
	#	@param hand. Left hand: 0 and Right hand: 1
	#	@return Numpy array of angles for either the left or right end-effector.
	def no_action(self,angles,hand):
		if np.size(angles,1)!=6 or np.size(angles,1)!=6:
			raise Exception("[grasping.py] Initial or final angles are not 6-value numpy arrays.");		
		if hand!=0 and hand!=1:
			raise Exception("[grasping.py] Hand value must be either 1 or 0. It is either 1 for left or 0 for right.");
		else:
			angles = np.array(angles);
			#print("The angles are: ",angles);
		if angles.size == 0:
			raise Exception("[motionplanner.py] angles is empty. It is not supposed to be.");
		return angles;
