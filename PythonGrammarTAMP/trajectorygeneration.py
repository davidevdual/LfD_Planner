import numpy as np
import utils as utls;
from abc import ABC, ABCMeta, abstractmethod

## Interface dealing with all aspects of trajectory generation for a robotic system. It is an interface. Therefore, methods
## are only defined, but not implemented.
#
#  
class TrajectoryGeneratorInterface(metaclass=ABCMeta):

	@classmethod
	def __subclasshook__(cls, subclass):
		return (hasattr(subclass, 'set_nb_joints') and 
				callable(subclass.set_nb_joints) and 
				hasattr(subclass, 'get_nb_joints') and 
				callable(subclass.get_nb_joints) and
				hasattr(subclass, 'set_velocities') and
				callable(subclass.set_velocities) and
				hasattr(subclass, 'get_velocities') and
				callable(subclass.get_velocities) and
				hasattr(subclass, 'compute_trajectory') and
				callable(subclass.compute_trajectory) and
				hasattr(subclass, 'set_times') and
				callable(subclass.set_times) and
				hasattr(subclass, 'get_times') and
				callable(subclass.get_times) or 
				NotImplemented)

	erase_sciNotation_vectorized = np.vectorize(utls.erase_sciNotation);

	## Number of joints to compute the trajectory for
	nb_joints = 22;

	##	Array of velocities for the 22 joints (5 left, 5 right, 6 fingers left, 6 fingers right)
	velocities = np.zeros(nb_joints);

	## Times for executing a trajectory
	times = np.zeros(nb_joints);

	# Number of samples
	nb_samples = 0;

	##	Constructor that sets the velocity for all the robot's joints
	#	@param _velocities Array of 22 values setting the motors' velocities (in deg/s)
	#	@param _nb_joints Number of joints to compute the trajectory for
	#	@param _times Times for executing a trajectory
	@abstractmethod
	def __init__(self,_nb_joints):
		raise NotImplementedError;

	#	Define the time scaling function
	#	@param	_current_times Current times of execution for each joint. Numpy array
	#	@param _desired_times Desired times for execution of each joint. Numpy array
	#	@return Time scale at the current time
	@abstractmethod
	def time_scaling_function(self,_current_times,_desired_times):
		raise NotImplementedError;

	## Set the number of joints to compute the trajectory for
	#	@param _nb_joints Number of joints to compute the trajectory for
	@abstractmethod
	def set_nb_joints(self,_nb_joints):
		raise NotImplementedError;

	## Get the number of joints 
	#	@return Number of joints
	@abstractmethod
	def get_nb_joints(self):
		raise NotImplementedError;

	##	Set the velocities for the 22 joints
	#	@param _velocities Array of 22 values setting the motors' velocities (in deg/s)
	@abstractmethod
	def set_velocities(self,_velocities):
		raise NotImplementedError;

	## Get the velocities
	#	@return velocities of the 22 motors
	@abstractmethod
	def get_velocities(self):
		raise NotImplementedError;

	## Compute linear trajectory for each motor in Joint Space for a set of velocities (in deg/s)
	#	@param	init_psts Initial position
	#	@param	final_psts Final position
	#	@param	init_times
	#	@param	final_times
	#	@param	sampling_rate
	#	@return Array of size Nx22. Angles in degrees.
	@abstractmethod
	def compute_trajectories(self,init_psts,final_psts,init_times,final_times,sampling_rate):
		raise NotImplementedError;

	## Get the controller's sampling rate 
	#	@return Sampling rate
	@abstractmethod
	def get_times(self):
		raise NotImplementedError;

	## Set the controller's sampling rate
	#	@param Controller's sampling rate
	@abstractmethod
	def set_times(self,_times):
		raise NotImplementedError;

## Linear Trajectory Generator. It inherits from the TrajectoryGeneratorInterface.
##
#
class LinearTrajectoryGenerator(TrajectoryGeneratorInterface):

	nb_joints = 0;

	velocities = np.zeros(nb_joints);

	times = np.zeros(nb_joints);

	nb_samples = 0;

	def __init__(self,_nb_joints):
		self.nb_joints = _nb_joints;

	def time_scaling_function(self,_current_times,_desired_times):
		return np.divide(_current_times,_desired_times);

	def get_velocities(self):
		return self.velocities;

	def set_velocities(self,_velocities):
		self.velocities = _velocities;

	def get_nb_joints(self):
		return self.nb_joints;

	def set_nb_joints(self,_nb_joints):
		self.nb_joints = _nb_joints;

	def compute_trajectories(self,init_psts,final_psts,init_times,final_times,sampling_rate):
		nb_samples = int(final_times/sampling_rate);
		times = np.transpose(np.reshape(np.linspace(init_times, final_times, nb_samples, endpoint=True),(nb_samples,1)));
		delta_positions = final_psts - init_psts;
		init_psts = np.tile(init_psts, (nb_samples, 1));
		product = np.multiply(np.transpose(self.time_scaling_function(times,final_times)), delta_positions);
		trajectory_points = init_psts + np.multiply(np.transpose(self.time_scaling_function(times,final_times)), delta_positions);
		return trajectory_points,times;

	'''
	def compute_trajectories(self,init_psts,final_psts,init_times,final_times,samples_nb):
		traj_points = np.zeros((self.nb_joints, samples_nb));# Points of the trajectory path. It is an array of cols=size nb_joints x rows=samples_nb
		times = np.transpose(np.reshape(np.linspace(init_times[0], final_times[0], samples_nb, endpoint=True),(samples_nb,1)))# Each time step comprised in the interval [final_times;init_times]
		#print("times: ",times)
		vel = np.reshape(self.compute_velocities(init_psts,final_psts,init_times,final_times),(self.nb_joints,1))
		#print("vel: ",vel)
		traj_points = vel * times
		#print("traj_points: ",traj_points)
		return np.transpose(traj_points)# dimensions: samples_nb x nb_joints
	'''
	
	def compute_velocities(self,init_psts,final_psts,init_times,final_times):
		return (final_psts-init_psts)/(final_times-init_times);

	def get_times(self):
		return self.times;

	def set_times(self,_times):
		self.times = _times;

'''
nb_joints = 22;
samples_nb = 20;# Number of intermediate angles till reaching the final one
init_psts = np.zeros(nb_joints);
final_psts = [10,20,30,40,50,67,34,23,12,3,-9,-90,-180,3,4,5,1,3,76,-98,12,45];
init_times = np.zeros(nb_joints);
final_times = 10*np.ones(22);# All joints must achieve their final position within 10 seconds

linear_gen = LinearTrajectoryGenerator(nb_joints);
traj_points = linear_gen.compute_trajectories(init_psts,final_psts,init_times,final_times,samples_nb);
print(traj_points.shape)
print(traj_points)
'''
