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

localIP     = "10.0.0.1"

localPort   = 8080

bufferSize  = 1024

'''
nb_joints = 22;
samples_nb = 1000;# Number of intermediate angles till reaching the final one
init_psts = np.zeros(nb_joints);
#final_psts = [35,25,90,45,0,35,25,90,45,0,0,0,0,0,0,0,0,0,0,0,0,0];
final_psts = [0,0,0,0,0,0,0,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
init_times = np.zeros(nb_joints);
final_times = 60*np.ones(22);# All joints must achieve their final position within 60 seconds

linear_gen = tg.LinearTrajectoryGenerator(nb_joints);
traj_points = linear_gen.compute_trajectories(init_psts,final_psts,init_times,final_times,samples_nb);
'''

waypoint1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])# Initialization array. Do not alter this array
times12_init = 0

waypoint2 = np.array([0,0,90,0,0,0,0,0,0,0,0,0,0,90,0,0,0,0,0,0,0,0])# Order: LSR, LSA, LEFE, LSFE, LWR, RSR, RSA, REFE, RSFE, RWR
times12_final = 2
times23_init = 0

waypoint3 = np.array([0,0,90,0,0,0,270,270,270,270,270,0,0,90,0,0,0,270,270,270,270,270])
times23_final = 2
times34_init = 0

waypoint4 = np.array([0,0,90,0,0,0,270,270,270,270,270,0,0,90,0,0,0,270,270,270,270,270])
times34_final = 2
times45_init = 0

waypoint5 = np.array([25,0,60,10,0,0,270,270,270,270,270,0,0,90,15,0,0,270,270,270,270,270])
times45_final = 2
times56_init = 0

waypoint6 = np.array([25,0,60,10,0,0,270,270,270,270,270,0,0,90,15,85,0,270,270,270,270,270])
times56_final = 2

waypoints_forward = np.vstack((waypoint1,waypoint2,waypoint3,waypoint4,waypoint5,waypoint6));#waypoint1 must be included in the waypoints
waypoints_backward = np.vstack((waypoint5,waypoint4,waypoint3,waypoint2,waypoint1));
waypoints = np.vstack((waypoints_forward,waypoints_backward));
print(len(waypoints))

times_init_forward = [times12_init,times23_init,times34_init,times45_init,times56_init]#times12_init must be included in times_init
times_init_backward = [times56_init,times45_init,times34_init,times23_init,times12_init]
times_init = times_init_forward + times_init_backward
print(len(times_init))

times_final_forward = [times12_final,times23_final,times34_final,times45_final,times56_final]#times12_final must be included in times_final
times_final_backward = [times56_final,times45_final,times34_final,times23_final,times12_final]
times_final = times_final_forward + times_final_backward
print(len(times_final))

motion_planner = mp.MotionPlanner() 
angles_left = waypoints[:,0:11];# The left angles
angles_right = waypoints[:,11:22];# The right angles
traj_points_left,times_left = motion_planner.generate_trajectory_jointSpace(angles_left,times_init,times_final,0.01,len(angles_left[0]));
traj_points_right,times_right = motion_planner.generate_trajectory_jointSpace(angles_right,times_init,times_final,0.01,len(angles_right[0]));

# Concatenate trajectories horizontally. First it is the left arm then the right
traj_points = np.hstack((traj_points_left,traj_points_right));
print(len(traj_points))

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
