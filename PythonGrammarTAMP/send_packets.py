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

waypoint2 = np.array([80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
times12_final = 2
times23_init = 0

waypoint3 = np.array([80,0,110,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
times23_final = 2
times34_init = 0

waypoint4 = np.array([80,0,110,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
times34_final = 2
times45_init = 0

waypoint5 = np.array([0,0,110,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
times45_final = 2
times56_init = 0

waypoint6 = np.array([0,0,110,0,95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
times56_final = 2
times67_init = 0

waypoint7 = np.array([0,0,90,0,95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
times67_final = 2 
times78_init = 0

waypoint8 = np.array([0,0,90,0,95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
times78_final = 2 
times89_init = 0 

waypoint9 = np.array([0,0,90,0,95,0,215,110,270,100,0,0,0,0,0,0,0,0,0,0,0,0])
times89_final = 2
times910_init = 0

waypoint10 = np.array([0,0,110,0,95,0,215,110,270,100,0,0,0,0,0,0,0,0,0,0,0,0])
times910_final = 2
times1011_init = 0

waypoint11 = np.array([-20,0,110,0,95,0,215,110,270,100,0,0,0,0,0,0,0,0,0,0,0,0])
times1011_final = 2
times1112_init = 0

waypoint12 = np.array([-20,0,110,0,95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
times1112_final = 2

waypoints_forward = [waypoint1,waypoint2,waypoint3,waypoint4,waypoint5,waypoint6,waypoint7,waypoint8,waypoint9,waypoint10,waypoint11,waypoint12]#waypoint1 must be included in the waypoints
waypoints_backward = [waypoint11,waypoint10,waypoint9,waypoint8,waypoint7,waypoint6,waypoint5,waypoint4,waypoint3,waypoint2,waypoint1]
waypoints = waypoints_forward + waypoints_backward
print(len(waypoints))

times_init_forward = [times12_init,times23_init,times34_init,times45_init,times56_init,times67_init,times78_init,times89_init,times910_init,times1011_init,times1112_init]#times12_init must be included in times_init
times_init_backward = [times1112_init,times1011_init,times910_init,times89_init,times78_init,times67_init,times56_init,times45_init,times34_init,times23_init,times12_init]
times_init = times_init_forward + times_init_backward
print(len(times_init))

times_final_forward = [times12_final,times23_final,times34_final,times45_final,times56_final,times67_final,times78_final,times89_final,times910_final,times1011_final,times1112_final]#times12_final must be included in times_final
times_final_backward = [times1112_final,times1011_final,times910_final,times89_final,times78_final,times67_final,times56_final,times45_final,times34_final,times23_final,times12_final]
times_final = times_final_forward + times_final_backward
print(len(times_final))

motion_planner = mp.MotionPlanner() 
traj_points = motion_planner.generate_trajectory_jointSpace(waypoints,times_init,times_final,0.01)# Code with new way of computing the trajectories
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
