################# MOTION PLANNING FOR RIGHT ARM #################
import socket
import time
import motionplanner as mp
import trajectorygeneration as tg
import numpy as np

#list_to_convert is a list of integers
def listToString(list_to_convert,delimitor):

	# Empty string for initialization
	final_string = ""

	# Go through the list and concatenate to the initial variable
	for elmt in list_to_convert:
		final_string += (str(elmt) + delimitor)   

	return final_string

# Important variables
localIP     = "10.0.0.1"
localPort   = 8080
bufferSize  = 1024
arm_left = 1;
arm_right = 0;

# Compute the waypoints for each initial and final states
motion_planner = mp.MotionPlanner() 
initialStatesList = [[100,-300,-200,0.2929,-0.1774,0.6746,-0.6539],[0,-300,-200,0.2563,-0.1022,-0.6838,-0.6755],[-100,-300,-300,0.1415,0.3917,-0.7524,0.5104],[0,-300,-300,0.1887,-0.0859,0.8348,-0.5080]];
finalStatesList = [[0,-300,-200,0.2563,-0.1022,0.6838,-0.6755],[-100,-300,-300,0.1415,0.3917,-0.7524,0.5104],[0,-300,-300,0.1887,-0.0859,0.8348,-0.5080],[100,-300,-200,0.3556,-0.1020,0.6890,-0.6233]];
arm_left = 1;
arm_right = 0;
waypoints_right = motion_planner.generate_waypoints_listStates(initialStatesList,finalStatesList,arm_right);

# Compute the angles corresponding to each waypoint
angles_right = motion_planner.convert_jointSpace(waypoints_right,arm_right);
angles_left = np.zeros([np.size(angles_right,0),5]);
angles_left_hand = np.zeros([np.size(angles_left,0),6]);
angles_right_hand = np.zeros([np.size(angles_left,0),6]);
angles = np.hstack((angles_left,angles_left_hand,angles_right,angles_right_hand));
angles = np.vstack((np.zeros((1,22)),angles));
angles = np.concatenate((angles,np.zeros((1,22))));
waypoints = list(angles);# Create a list of waypoints

# Compute the trajectory
times_init = (np.zeros(len(angles)-1)).tolist();
times_final = (2*(np.ones(len(angles)))).tolist();# Each trajectory is executed in two seconds
traj_points = motion_planner.generate_trajectory_jointSpace(angles,times_init,times_final,0.01)

# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
print('Before the bind')
UDPServerSocket.bind((localIP, localPort))
print('After the bind')

# Tuple containing the address and port of UDP client to listen to
clientAddress = ("10.0.0.2",8080)

# Stops until it receives the message from the client
print('Before addresspair')
bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
print('After addresspair')

address = bytesAddressPair[1]

print('before client ip')
clientIP  = "Client IP Address:{}".format(address)
print('after client ip')
time.sleep(10)

print("UDP server up and listening")

# Listen for incoming datagrams
for angles in traj_points:

	output_str = listToString(angles,',')[:-1]
	msgFromServer       = output_str 
	bytesToSend         = str.encode(msgFromServer)

	serverMsgPrint = "Message from Server:{}".format(msgFromServer)
	print(bytesToSend)

	# Sending a reply to client
	UDPServerSocket.sendto(bytesToSend, address)

	time.sleep(0.01)