import numpy as np
import motionplanmodule as mp
import ikmodule as ik
import utils as uts
import trajectorygeneration as tg
from abc import ABC, ABCMeta, abstractmethod

## Interface dealing with all aspects of motion planning for a robotic system. In this case, the motion planner,
## generates a set of waypoints in the task space. Then, the waypoints are fed into the trajectory planner to generate
#  the trajectories in the joint space.
#
#
class MotionPlannerInterface(metaclass=ABCMeta):

	@classmethod
	def __subclasshook__(cls, subclass):
		return (hasattr(subclass, 'generate_trajectory_jointSpace') and
				hasattr(subclass, 'append_list') and
				hasattr(subclass, 'generate_waypoints') and
				hasattr(subclass, 'generate_waypoints_listStates') and
				hasattr(subclass, 'convert_jointSpace') and
				hasattr(subclass, 'synchronize_arms') and
				hasattr(subclass, 'fill_matrix') and
				callable(subclass.append_list) and
				callable(subclass.generate_trajectory_jointSpace) or
				callable(subclass.generate_waypoints) or
				callable(subclass.generate_waypoints_listStates) or
				callable(subclass.convert_jointSpace) or
				callable(subclass.synchronize_arms) or
				callable(subclass.fill_matrix) or
				NotImplemented)

	##	Generate a list of lists using a list of ensemble of lists.
	#	@param list_input List of ensemble of lists.
	#	@param list_output List of lists.
	#	@return List of lists.
	@abstractmethod
	def append_list(self,list_input,list_output):
		raise NotImplementedError;

	##	Generate the trajectories for the different waypoints in the joint space.
	#	@param waypoints List of waypoints expressed in the joint space.
	#	@param times Times to achieve the different trajectories. 	
	#	@param samples_nb Number of samples between two waypoints.
	#	@return Entire trajectory in the joint space.
	@abstractmethod
	def generate_trajectory_jointSpace(self,waypoints,times,samples_nb):
		raise NotImplementedError;

	##	Generate the different waypoints in the robot's task space.
	#	@param initialState End effector's initial state: (x,y,z,w,qX,qY,qZ). Translation in millimeters and orientation under the quaternion representation.
	#	@param finalState Desired end effector's final state: (x,y,z,w,qX,qY,qZ). Translation in millimeters and orientation under the quaternion representation. 	
	#	@param arm Generate waypoints either for left or right end-effectors. arm = 1 for left end-effector. arm = 0 for right end-effector.
	#	@return List of waypoints under the (x,y,z,w,qX,qY,qZ) form. Each row is an end-effector's waypoint.
	@abstractmethod
	def generate_waypoints(self,initialState,finalState,arm):
		raise NotImplementedError;

	##	Generate the different waypoints for a list of initial and final states.
	#	@param initialStatesList list of end effector's initial states: (x,y,z,w,qX,qY,qZ). Translation in millimeters and orientation under the quaternion representation.
	#	@param finalStatesList List of desired end effector's final state: (x,y,z,w,qX,qY,qZ). Translation in millimeters and orientation under the quaternion representation. 	
	#	@param arm Generate waypoints either for left or right end-effectors. arm = 1 for left end-effector. arm = 0 for right end-effector.
	#	@return List of waypoints under the (x,y,z,w,qX,qY,qZ) form. Each row is an end-effector's waypoint.
	@abstractmethod
	def generate_waypoints_listStates(self,initialStatesList,finalStatesList,arm):
		raise NotImplementedError;

	##	Convert all the waypoints to the joint space
	#	@param waypoints End-effector's waypoints following the representation: (x,y,z,w,qX,qY,qZ). Translation in millimeters and orientation under the quaternion representation.
	#	@param arm_option arm = 1 for left end-effector. arm = 0 for right end-effector.
	#	@return numpy multidimensional array of joint angles where n is the number of waypoints. Actually, the array's size is nx5 where n is the number of waypoints.
	@abstractmethod
	def convert_jointSpace(self,waypoints,arm_option):
		raise NotImplementedError;

	##	Synchronize trajectories if the number of trajectory samples is different across arms.
	#	@param traj_points_left Left arm's trajectory. Numpy array of dimensions nx11. n is the number of trajectory samples.
	#	@param traj_points_right Right arm's trajectory. Numpy array of dimensions nx11. n is the number of trajectory samples.
	#	@param times_left Time samples for left arm. Numpy array of dimensions nx1. n is the number of trajectory samples.
	#	@param times_right Time samples for right arm. Numpy array of dimensions nx1. n is the number of trajectory samples.
	#	@return One numpy multidimensional arrays of sizes pxnb_joints. p is the new number of samples after synchronizing both trajectories.
	@abstractmethod
	def synchronize_arms(self,traj_points_left,traj_points_right,times_left,times_right):
		raise NotImplementedError;

	##	Fill the totatal trajectories matrix with the synchronized left and right trajectories.
	#	@param trajectories Final trajectories matrix.
	#	@param times Merged array of all times.
	#	@param traj_points_left Trajectories for left arm.
	#	@param traj_points_right Trajectories for right arm.
	#	@param times_left Times for left arm.
	#	@param times_right Times for right arm.
	#	@return The final trajectories matrix.
	@abstractmethod
	def fill_matrix(self,trajectories,times,traj_points_left,traj_points_right,times_left,times_right):
		raise NotImplementedError;

## Motion Planner. It inherits from the MotionPlannerInterface.
##
#
class MotionPlanner(MotionPlannerInterface):

	# Arm indices
	arm_left = 1;
	arm_right = 0;

	def append_list(self,list_input,list_output):
		for list_elmt in list_input:
			list_output.append(list_elmt.tolist());
		return list_output

	# Old code
	def generate_trajectory_jointSpace(self,waypoints,times_init,times_final,sampling_rate,num_joints):
		if len(waypoints)<1 or len(times_init)<1 or len(times_final)<1:
			raise Exception("[motionplanner.py] The input list is zero, which is incorrect.");
		elif len(waypoints)<2:
			raise Exception("[motionplanner.py] There must be at least two elements in waypoints. Here it does not.");
		linear_gen = tg.LinearTrajectoryGenerator(num_joints);
		final_trajectory = [];# Trajectory in the joint space to return. Initialized with first waypoint
		final_times = [];
		psts = np.arange(0,len(waypoints)-1,1);
		last_elmt_time = 0;

		# Compute the intermediate trajectories between different waypoints
		for k in psts:

			if k>0:# If it is not the first element, eliminate the previous position as it will be repeated
				del final_trajectory[-1];
				del final_times[-1];
			inter_trajectory,inter_times = linear_gen.compute_trajectories(waypoints[k],waypoints[k+1],times_init[k],times_final[k],sampling_rate);# Intermediate trajectory
			#print("inter_trajectory: ",inter_trajectory);
			inter_times = np.transpose(np.add(inter_times, last_elmt_time));
			last_elmt_time = inter_times[-1,:];# Then result is a numpy multidimensional array
			last_elmt_time = last_elmt_time[0];# The last element is a float now
			final_trajectory = self.append_list(inter_trajectory,final_trajectory);
			final_times = self.append_list(inter_times,final_times);

		final_times = np.array(final_times).flatten();# Make the times array one-dimensional. Otherwisie, it has two dimensions, which raises errors.
		final_trajectory = np.array(final_trajectory);
		return final_trajectory,final_times;

	def generate_waypoints(self,initialState,finalState,arm):
		#print("initialState type: ",type(initialState));
		if arm != 1 & arm != 0:
			raise Exception("[motionplanner.py] Arm is wrong. 1 is for the left arm. 0 is for the right arm.");
		if len(initialState)!=0 & len(finalState)!=0:
			raise Exception("[motionplanner.py] initialState and finalState must have same number of elements.");
		# If the initial and final states are the same; then return the initial pose
		waypoints = np.zeros((1,22));
		if np.array_equal(np.array(initialState),np.array(finalState)) == True:
			waypoints = np.array(initialState);
			waypoints =  waypoints.reshape(-1,1);
			waypoints =  waypoints.reshape(1,7);
		else:
			waypoints = mp.runRRT(initialState,finalState,arm);
		#print("waypoints= ",waypoints);
		return waypoints;

	def generate_waypoints_listStates(self,initialStatesList,finalStatesList,arm):
		waypoints_list = [];
		if len(initialStatesList)==1 and len(finalStatesList)==1 and set(initialStatesList[0])==set(finalStatesList[0]):
			waypoints_list.append(np.array(initialStatesList[0]))
		if type(initialStatesList)!= list or type(finalStatesList)!= list:
			raise Exception("[motionplanner.py] initialStatesList or finalStatesList must be lists.");
		if arm != 1 & arm != 0:
			raise Exception("[motionplanner.py] Arm is wrong. 1 is for the left arm. 0 is for the right arm.");
		for k in range(0, len(initialStatesList)):
			waypoints = self.generate_waypoints(initialStatesList[k],finalStatesList[k],arm);
			for row in range(0,np.size(waypoints,0)):
				waypoints_list.append(waypoints[row,:]);
				#print("waypoints: ",waypoints[row,:]);
		return waypoints_list #List of numpy arrays

	# The order of the angles is: [LSR,LSA,LEFE,LSFE,LWR]
	def convert_jointSpace(self,waypoints,arm_option):
		if len(waypoints) <1:
			raise Exception("[motionplanner.py] The number of waypoints cannot be smaller than 1.");
		elif len(waypoints[0]) != 7:
			raise Exception("[motionplanner.py] A waypoint must be described by seven values. Here it is not.");
		angles = np.zeros((len(waypoints),5));# There are 5 joints per arm

		row = 0;
		# Convert each waypoint into a set of 5 angles using the IK algorithm
		for waypoint in waypoints:
			# There is no wrist flexion/extension. Therefore, the fifth element is eliminated
			angles_t = np.delete(ik.runIK(waypoint[0],waypoint[1],waypoint[2],waypoint[3],waypoint[4],waypoint[5],waypoint[6],100,0.01,arm_option),5);
			angles_t = uts.anglesTransform_to_real(angles_t[0],angles_t[1],angles_t[2],angles_t[3],angles_t[4],arm_option);
			angles[row,:] = np.array([angles_t[0],angles_t[1],angles_t[2],angles_t[3],angles_t[4]]);
			#print("angles_t: ",angles_t);
			# The shoulder and elbow are swapped in the robot's control
			#print("For arm_option before: ",arm_option," the angles are: ",angles_t);
			#angles[row,:] = np.array([angles_t[0],angles_t[1],angles_t[3],angles_t[2],angles_t[4]]);
			#print("For arm_option after: ",arm_option," the angles are: ",angles);
			#Insert a new row at each iteration
			row = row + 1;# TODO: Change the orders so the angles correspond to each real joint (create utils file)
		return angles;

	def fill_matrix(self,trajectories,times,traj_points_left,traj_points_right,times_left,times_right):
		# Check that the times and trajectories have the same number of samples and dimensionalities. Otherwise, raise error
		if np.size(times_left,0)!=np.size(traj_points_left,0) or np.size(times_right,0)!=np.size(traj_points_right,0):
			raise Exception("[motionplanner.py] The number of samples in the time and trajectory matrices must match. Here it does not.");
		elif np.size(traj_points_left,1)!=np.size(traj_points_right,1):
			raise Exception("[motionplanner.py] The trajectories' dimensionalities must match. Here they do not.");
		elif np.size(trajectories,0)!=np.size(times,0):
			raise Exception("[motionplanner.py] The number of samples in the trajectories matrix must match the number of samples in the times vector. Here they do not.");
		elif np.size(trajectories,1)!=(np.size(traj_points_right,1) + np.size(traj_points_left,1)):
			raise Exception("[motionplanner.py] The trajectories matrix's dimensionality must match the sum of both left and right trajectory matrices' dimensionalities. Here it is not.");

		idx_traj = 0;last_nonNull_left = 0;last_nonNull_right=0;
		for t in times:
			idx_left = np.where(times_left == t);# This is a tuple
			idx_right = np.where(times_right == t);# This is a tuple
			if idx_left[0].size == 0:
				idx_left = last_nonNull_left;
			else:
				idx_left = idx_left[0];#idx_left[0] is a multidimensional numpy array
				idx_left = idx_left[0].astype(int);# Convert the numpy array into an integer
				last_nonNull_left = idx_left;
			if idx_right[0].size == 0:
				idx_right = last_nonNull_right;
			else:
				idx_right = idx_right[0];#idx_right[0] is a multidimensional numpy array
				idx_right = idx_right[0].astype(int);# Convert the numpy array into an integer
				last_nonNull_right = idx_right;
			trajectories[idx_traj,0:11] = traj_points_left[idx_left,:];
			trajectories[idx_traj,11:22] = traj_points_right[idx_right,:];
			idx_traj = idx_traj + 1;
		return trajectories;

	def synchronize_arms(self,traj_points_left,traj_points_right,times_left,times_right):
		if traj_points_left.size<1 or traj_points_right.size<1:
			raise Exception("[motionplanner.py] traj_points_left or traj_points_right is empty. It is not supposed to be.");
		# Check that the times vectors' dimensionalities are equal to one
		elif np.ndim(times_left)!=1 or np.ndim(times_right)!=1:
			raise Exception("[motionplanner.py] The times' dimensionalities must be equal to 1. Here it does not.");
		# Check that each value in the times vector is unique. Otherwise, the synchronization will not work
		elif np.size(np.unique(times_left),0)!=np.size(times_left,0) or np.size(np.unique(times_right),0)!=np.size(times_right,0):
			raise Exception("[motionplanner.py] The times vectors do not contain unique values. They are supposed to.");
		# Check that the times and trajectories have the same number of samples and dimensionalities. Otherwise, raise error
		elif np.size(times_left,0)!=np.size(traj_points_left,0) or np.size(times_right,0)!=np.size(traj_points_right,0):
			raise Exception("[motionplanner.py] The number of samples in the time and trajectory matrices must match. Here it does not.");
		elif np.size(traj_points_left,1)!=np.size(traj_points_right,1):
			raise Exception("[motionplanner.py] The trajectories' dimensionalities must match. Here they do not.");
		
		times = np.array(uts.merge_lists(list(times_left),list(times_right)));
		trajectories = np.zeros((np.size(times,0),np.size(traj_points_left,1)+np.size(traj_points_right,1)));
		trajectories = self.fill_matrix(trajectories,times,traj_points_left,traj_points_right,times_left,times_right);
		return trajectories;

## Motion Planner Grammar. It inherits from the MotionPlanner class since it is a type of motion planner.
## MotionPlannerGrammar is called by the parser to execute the appropriate commands.
##
#
class MotionPlannerGrammar(MotionPlanner):

	##	Make the one of the two end-effectors execute an 'approach' action.
	#	@param init 6x1 Numpy array. Initial 6D pose.
	#	@param final Numpy array. Final 6D pose.
	#	@param hand. Left hand: 0 and Right hand: 1
	#	@return Integer. Numpy array containing the angles that the left arm must achieve to accomplish the action.
	def approach(self,init,final,hand):
		#print("init: ",init[0]," final: ",final[0]);
		if np.size(init,1)!=7 or np.size(final,1)!=7:
			raise Exception("[motionplanner.py] Initial or final poses are not 7-value numpy arrays.");
		if hand==1:
			#print("init: ",init, " final: ",final);
			angles = self.approach_left(init,final);
		elif hand==0:
			angles = self.approach_right(init,final);
		else:
			raise Exception("[motionplanner.py] Hand identifier is wrong. It is either 1 for left or 0 for right.");
		if angles.size == 0:
			raise Exception("[motionplanner.py] angles is empty. It is not supposed to be.");
		return angles;

	##	Make the left end-effector execute an 'approach' action.
	#	@param init 6x1 Numpy array. Initial 6D pose.
	#	@param final Numpy array. Final 6D pose.
	#	@return Numpy array containing the angles that the left arm must achieve to accomplish the action.
	def approach_left(self,init,final):
		if np.size(init,1)!=7 or np.size(final,1)!=7:
			raise Exception("[motionplanner.py] Initial or final poses are not 7-value numpy arrays.");			
		else:
			waypoints_left = self.generate_waypoints_listStates(init,final,self.arm_left);
			if len(waypoints_left) == 0:
				raise Exception("[motionplanner.py] waypoints_left is null.");
			else:
				waypoints_left = np.delete(waypoints_left,0,0);
				angles_left = self.convert_jointSpace(waypoints_left,self.arm_left);
		if angles_left.size == 0:
			raise Exception("[motionplanner.py] angles_left is empty. It is not supposed to be.");
		return angles_left;

	##	Make the right end-effector execute an 'approach' action.
	#	@param init 6x1 Numpy array. Initial 6D pose.
	#	@param final Numpy array. Final 6D pose.
	#	@return Integer. 1 if the approach action has succeeded. 0 if the approach action has failed.
	def approach_right(self,init,final):
		if np.size(init,1)!=7 or np.size(final,1)!=7:
			raise Exception("[motionplanner.py] Initial or final poses are not 7-value numpy arrays.");			
		else:
			waypoints_right = self.generate_waypoints_listStates(init,final,self.arm_right);
			if len(waypoints_right) == 0:
				raise Exception("[motionplanner.py] waypoints_right is null.");
			else:
				waypoints_right = np.delete(waypoints_right,0,0);
				angles_right = self.convert_jointSpace(waypoints_right,self.arm_right);
		if angles_right.size == 0:
			raise Exception("[motionplanner.py] angles_right is empty. It is not supposed to be.");
		return angles_right;

	##	Make one of the hands close its fingers to grasp an object.
	#	@param hand. Left hand: 0 and Right hand: 1
	#	@return Integer. 1 if the grasp has succeeded. 0 if the grasp has failed.
	def enclose(self,hand):
		print('IN APPROACH MOTIONPLANNERGRAMMAR');
		if hand==0:
			success = self.enclose_left();
		elif hand==1:
			success = self.enclose_right();
		else:
			raise Exception("[motionplanner.py] Hand identifier is wrong. It is either 1 for right or 0 for left.");
		return success;
	
	##	Make the left hand close to grasp an object.
	#	@return Integer. 1 if the grasp has succeeded. 0 if the grasp has failed.
	def enclose_left(self):
		return 1;

	##	Make the right hand close to grasp an object.
	#	@return Integer. 1 if the grasp has succeeded. 0 if the grasp has failed.
	def enclose_right(self):
		return 1;

	## The end effector does not execute any action.
	#	@param pose The end-effector's pose.
	#	@param hand. Left hand: 0 and Right hand: 1
	#	@return Numpy array of angles for either the left or right end-effector.
	def no_action(self,pose,hand):
		if np.size(pose,1)!=7 or np.size(pose,1)!=7:
			raise Exception("[motionplanner.py] Initial or final poses are not 7-value numpy arrays.");		
		if hand!=0 and hand!=1:
			raise Exception("[motionplanner.py] Hand value must be either 1 or 0. It is either 1 for left or 0 for right.");
		else:
			angles = self.convert_jointSpace(pose,hand);
			#print("The angles are: ",angles);
		if angles.size == 0:
			raise Exception("[motionplanner.py] angles is empty. It is not supposed to be.");
		return angles;

def main():
    
	# Test the vision 
	motion_planner = MotionPlannerGrammar();
	initialStatesList_left = np.array([[1.03770000e+02,-3.69816847e+02,-2.99816847e+02,2.70598050e-01,6.53281482e-01,-6.53281482e-01,2.70598050e-01]]);
	finalStatesList_left = np.array([[-6.09758697,-420.4836115,-316.98958698,0.06026358,0.12072236,-0.76464444,-0.63016926]]);
	angles_left = motion_planner.approach_left(initialStatesList_left,finalStatesList_left);
	#print(angles_left)

if __name__=="__main__":
    main();

'''
traj_points_left = np.array([[1,2,3,4,5,6,7,8,9,10,11],[12,13,14,15,16,17,18,19,20,21,22],[23,24,25,26,27,28,29,30,31,32,33]],ndmin = 2);
traj_points_right = np.array([[120,130,140,150,160,170,180,190,200,210,220],[121,131,141,151,161,171,181,191,201,211,221]],ndmin = 2);
times_left = np.array([[0,1,2]]);times_right = np.array([[0,0.5]]);
print(np.ndimndim(times_left));
times_left = times_left.flatten();times_right = times_right.flatten();
motion_planner = MotionPlanner();
motion_planner.synchronize_arms(traj_points_left,traj_points_right,times_left,times_right);
'''

'''
waypoint1 = np.zeros(22)
times12_init = 0

waypoint2 = [35,25,90,45,0,35,25,90,45,0,0,0,0,0,0,0,0,0,0,0,0,0]
times12_final = 5
times23_init = 0

waypoint3 = np.zeros(22)
times23_final = 10
times34_init = 0

waypoint4 = [35,25,90,45,0,35,25,90,45,0,0,0,0,0,0,0,0,0,0,0,0,0]
times34_final = 5
times45_init = 0

waypoint5 = np.zeros(22)
times45_final = 60

waypoints = [waypoint1,waypoint2]
times_init = [times12_init]
times_final = [times12_final]

motion_planner = MotionPlanner() 
final_trajectory = motion_planner.generate_trajectory_jointSpace(waypoints,times_init,times_final,0.01)
print(final_trajectory)
'''
