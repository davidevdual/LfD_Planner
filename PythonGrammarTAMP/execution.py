import sys
import copy
import socket
import time
import numpy as np
import simulator as sim
import motionplanner as mp
import utils as utls

from math import pi

## Interface between the entire robot's low-level controller and the Task and Motion Planner
class TAMPExecInterface:

	# Class constructor
	# @input _threshold Float. The threshold down below the object is considered attached to the end-effector. This only works for simulation. In metres.
	def __init__(self,_is_simulation,_is_vision,_threshold,_objects):
		if type(_objects)!=tuple or len(_objects)<1 or len(_objects)>2 or type(_threshold)!=float:
			raise Exception("[execution.py] Wrong input");
		if _threshold<=0:
			raise Exception("[execution.py] The threshold cannot be negative or equal to zero.");
		self.commands = np.ones((1,22));# This numpy array contains all the commands
		self.commands_left_fingers = [];
		self.commands_right_fingers = [];
		self.commands_left_hand = [];
		self.commands_right_hand = [];
		self.nb_arm_motors = 5;
		self.nb_fingers_motors = 6;
		self.trajectory = Trajectory();
		self.is_simulation = _is_simulation;
		self.is_vision = _is_vision;
		self.threshold = _threshold;
		self.simulator = sim.Simulator();
		self.objects = _objects;

		if len(self.objects)==1:
			self.object1 = self.objects[0];
		elif len(self.objects)==2:
			self.object1 = self.objects[0];
			self.object2 = self.objects[1];
		else:
			raise Exception("[execution.py] Wrong input");

	## Add a command that needs to be executed by the low-level controller. It only accepts end-effector or fingers' angles.
	#  @param left_hand   Numpy array containing the desired angles for the left arm's actuators.
	#  @param right_hand  Numpy array containing the desired angles for the right arm's actuators.
	#  @param left_fingers  Numpy array containing the desireed angles for the left fingers' motors.
	#  @param right_fingers  Numpy array containing the desireed angles for the right fingers' motors.
	def add_command(self,left_hand,right_hand,left_fingers,right_fingers):
		# If there is no action to do, the function's arguments are integers and the handedness is -1 since there is no hand to control.
		if type(left_hand)==int or type(right_hand)==int or type(left_fingers)==int or type(right_fingers)==int:
			print("[execution.py] No command to add.");
		elif np.size(left_hand,1)!=self.nb_arm_motors or np.size(right_hand,1)!=self.nb_arm_motors or np.size(left_fingers,1)!=self.nb_fingers_motors or np.size(right_fingers,1)!=self.nb_fingers_motors:
			raise Exception("[execution.py] angles must have 5 or 6 values. 5 for the arm motors. 6 for the finger motors.");
		else:
			h = np.hstack((left_hand,left_fingers,right_hand,right_fingers));
			self.commands = np.vstack((self.commands,h));

	## Send all the commands to the robot's controller.
	def send_commands(self):
		# Delete the first row since it contains only ones
		if np.size(self.commands,0)>1 or self.commands.size==0:
			self.commands = np.delete(self.commands,(0),axis=0);
			traj_points = self.trajectory.compute_trajectories(self.commands);
			traj_point_mat = np.array(traj_points);

			# If the simulation option is activated, run the simulation and exit the program
			if self.is_simulation==1:
				############################################ Run the simulator ###############################################
				angles_left = traj_point_mat[:,0:5];
				angles_right = traj_point_mat[:,11:16];

				if len(self.objects)==1:
					raise Exception("[execution.py] The one-object option for the simulator has not been implemented yet. Therefore, the simulator cannot handle two objects.");
				elif len(self.objects)==2:
					self.simulator.runSimulator(angles_left,angles_right,np.transpose(np.array(self.object1,ndmin=2)),np.transpose(np.array(self.object2,ndmin=2)),self.threshold);
				exit(-1);
				##############################################################################################################
			elif self.is_simulation==0:
				try:
					udp_sender = UDPSender();
					udp_sender.send_commands_udp(traj_point_mat);
				except:
					raise Exception("[execution.py] The network cable is not connected to the computer.");
			else:
				raise Exception('[execution.py] is_simulation or is_vision are equal either to 1 or 0. Here it is not');
		else:
			print("[execution.py] The commands array must not be empty or its number of rows must be greater than 1. Nothing to execute. The program will exit.")
			exit(-1); 

class Trajectory:

	# Class constructor
	def __init__(self):
		self.motion_planner = mp.MotionPlanner();
	
	## Compute the final and initial times 
	#  @param angles   Waypoints in the form of angles
	#  @return Final and initial times for the trajectories
	def compute_times(self,angles):
		times_init = (np.zeros(len(angles)-1)).tolist();
		times_final = (2*(np.ones(len(angles)))).tolist();# Each trajectory is executed in two seconds
		return times_final,times_init;

	## Compute all the trajectories that the end-effector must execute
	#  @param commands   Numpy array of 22 elements containing all the angles for each waypoint.
	#  @return Numpy array of angles with all the trajectory for the arm actuators and fingers.
	def compute_trajectories(self,commands):
		if np.size(commands,0)<1 or commands.size==0:
			raise Exception("[execution.py] The commands array must not be empty or its number of rows must be greater than 1.");
		elif np.size(commands,1)!=22:
			raise Exception("[execution.py] The commands array must have 22 columns.");
		
		# Add the initial commands that bring the end-effectors above the table before manipulating
		commands1 = np.array([[0,0,0,0,0,0,0,0,0,0,0,    0,0,0,0,0,0,0,0,0,0,0]]);
		commands2 = np.array([[0,45,0,0,0,0,0,0,0,0,0,    0,45,0,0,0,0,0,0,0,0,0]]);# This line is temporary. TO REMOVE
		commands3 = np.array([[0,45,0,90,0,0,0,0,0,0,0,    0,45,0,90,0,0,0,0,0,0,0]]);# This line is temporary. TO REMOVE
		commands4 = np.array([[0,0,0,90,0,0,0,0,0,0,0,   0,0,0,90,0,0,0,0,0,0,0]]);# Flexion of the elbow only. TO REMOVE
		
		# We stack several commands4 vectors so the robot maintains the 90-degree elbow position for a while so the experimenter can setup the table
		commands = np.vstack((commands1,commands2,commands3,commands4,commands));

		# Make the arms stay in the same position for some time. Otherwise, the movement is too fast.
		commands = np.vstack((commands,commands[-1,:],commands[-1,:],commands[-1,:],commands[-1,:],commands[-1,:],commands[-1,:],commands[-1,:],commands[-1,:],commands[-1,:],commands[-1,:],commands[-1,:]));

		# Compute the times and trajectories
		times_final,times_init = self.compute_times(commands);
		angles_left = commands[:,0:11];# The left angles
		angles_right = commands[:,11:22];# The right angles
		traj_points_left,times_left = self.motion_planner.generate_trajectory_jointSpace(angles_left,times_init,times_final,0.01,len(angles_left[0]));
		traj_points_right,times_right = self.motion_planner.generate_trajectory_jointSpace(angles_right,times_init,times_final,0.01,len(angles_right[0]));

		# Synchronize both trajectories for the left and right arms if they have different number of points
		traj_points = self.motion_planner.synchronize_arms(traj_points_left,traj_points_right,times_left,times_right);

		# Do not move the left and right wrists at all. This decision is only when the robot is in real-life mode
		#traj_points[:,4] = 0;
		#traj_points[:,15] = 0;

		# Flip the array vertically so the robot can go back to its original position
		traj_points = np.vstack((traj_points,np.flip(traj_points, axis=0)));
		np.savetxt('data.csv', traj_points.tolist(), delimiter=',');# Optional since it saves the trajectories into a file
		return traj_points;

## Interface between the entire robot's low-level controller and the Task and Motion Planner
class UDPSender:

	# Class constructor. Initialize all server parameters
	def __init__(self):
		self.localIP     = "10.0.0.1";
		self.localPort   = 8080;			
		self.bufferSize  = 1024;
		self.arm_left = 1;
		self.arm_right = 0;

		# Create a datagram socket
		self.UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM);

		# Bind to address and ip
		print('Before the bind');
		self.UDPServerSocket.bind((self.localIP, self.localPort));
		print('After the bind');

		# Tuple containing the address and port of UDP client to listen to
		self.clientAddress = ("10.0.0.2",8080);

		# Stops until it receives the message from the client
		print('Before addresspair');
		self.bytesAddressPair = self.UDPServerSocket.recvfrom(self.bufferSize);
		print('After addresspair');

		self.address = self.bytesAddressPair[1];

		print('before client ip');
		self.clientIP  = "Client IP Address:{}".format(self.address);
		print('after client ip');
		time.sleep(10);
		print("UDP server up and listening");

		time.sleep(10);

	## Convert a list into a String
	#  @param list_to_convert   The list to be converted
	#  @param delimitor Type of delimitor in the list
	#  @return A String
	def listToString(self,list_to_convert,delimitor):

		# Empty string for initialization
		final_string = "";

		# Go through the list and concatenate to the initial variable
		for elmt in list_to_convert:

			# Get rid of the scientific notation that is not understood by the robot's low-level controller
			elmt = '{:f}'.format(elmt); 
			elmt = str(elmt);
			final_string += (str(elmt) + delimitor);
		return final_string;

	## Send the trajectories to the robot's microcontroller for execution
	#  @param traj_points   Numpy array containing the angles to execute.
	def send_commands_udp(self,traj_points):

		# Convert input to list
		traj_points = traj_points.tolist();

		# For the robot's controller, LSFE and LEFE positions in the array of angles are swapped. Therefore, we must swap them here
		traj_points = utls.swap_elmts(traj_points,2,3);

		# Listen for incoming datagrams
		for angles in traj_points:
			output_str = self.listToString(angles,',')[:-1];
			msgFromServer       = output_str;
			bytesToSend         = str.encode(msgFromServer);
			serverMsgPrint = "Message from Server:{}".format(msgFromServer);
			print(bytesToSend);

			# Sending a reply to client
			self.UDPServerSocket.sendto(bytesToSend,self.address);
			time.sleep(0.01);

def main():
	object1 = [[-207.20477789,-312.875389,-270,0.03293905,-0.35213897,0.36608334,-0.86075325]];
	object2 = [[200,-312.875389,-316.98958698,0.06026358,0.12072236,-0.76464444,-0.63016926]];
	exec = TAMPExecInterface(1,0,0.1,tuple((object1,object2)));#_is_simulation,_is_vision

	# At time t=0
	left_hand1 = np.array([[0,0,0,0,0]]);
	right_hand1 = np.array([[0,0,0,0,0]]);
	left_fingers1 = np.array([[0,0,0,0,0,0]]);
	right_fingers1 = np.array([[0,0,0,0,0,0]]);

	# At time t=1
	left_hand2 = np.array([[0,0,90,0,0]]);
	right_hand2 = np.array([[0,0,0,0,0]]);
	left_fingers2 = np.array([[0,0,0,0,0,0]]);
	right_fingers2 = np.array([[0,0,0,0,0,0]]);

	# At time t=2
	left_hand3 = np.array([[0,0,90,0,0]]);
	right_hand3 = np.array([[0,0,90,0,0]]);
	left_fingers3 = np.array([[0,0,0,0,0,0]]);
	right_fingers3 = np.array([[0,0,0,0,0,0]]);

	# At time t=3
	left_hand4 = np.array([[0,0,90,0,0]]);
	right_hand4 = np.array([[0,0,90,0,0]]);
	left_fingers4 = np.array([[0,0,0,0,0,0]]);
	right_fingers4 = np.array([[0,0,0,0,0,0]]);
	
	# Execution
	exec.add_command(left_hand1,right_hand1,left_fingers1,right_fingers1);
	exec.add_command(left_hand2,right_hand2,left_fingers2,right_fingers2);
	exec.add_command(left_hand3,right_hand3,left_fingers3,right_fingers3);
	exec.add_command(left_hand4,right_hand4,left_fingers4,right_fingers4);
	exec.send_commands();

if __name__=="__main__":
	main();