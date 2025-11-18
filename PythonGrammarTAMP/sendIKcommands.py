import socket
import time
import ikmodule as dual_ik

import motionplanner as mp
import trajectorygeneration as tg
import numpy as np

#list_to_convert is a list of integers
def listToString(list_to_convert,delimitor):

	# Empty string for initialization
	final_string = ""

	# Go through the list and concatenate to the initial variable
	for elmt in list_to_convert:
		final_string += (str(f"{elmt:8f}") + delimitor) # If  f"{elmt:8f} is not given, the list will contain scientific notations that are not handled by the robot controller

	return final_string

# Network configuration
localIP     = "10.0.0.1"
localPort   = 8080
bufferSize  = 1024

# Arm configuration
arm_left = 1
arm_right = 0

# TODO: Waypoints computation from task space to joint space
waypoint1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])# Initialization array. Do not alter this array
times12_init = 0

left = dual_ik.runIK(103.77,-45.58245346,-451.0098271,-90.,-5.,180.,arm_left)# Order: RSR, RSA, REFE, RSFE, RWR
right = dual_ik.runIK(-103.77,-45.58245346,-451.0098271,90.,5.,-180.,arm_right)# Order: LSR, LSA, LEFE, LSFE, LWR
waypoint2 = np.array([left[0],left[1],left[3],left[2],0,0,0,0,0,0,0,right[0],right[1],right[3],right[2],0,0,0,0,0,0,0])# Order: LSR, LSA, LEFE, LSFE, LWR, RSR, RSA, REFE, RSFE, RWR
times12_final = 2

waypoints_forward = [waypoint1,waypoint2]#waypoint1 must be included in the waypoints
waypoints_backward = [waypoint1]
waypoints = waypoints_forward + waypoints_backward
print(len(waypoints))

times_init_forward = [times12_init]#times12_init must be included in times_init
times_init_backward = [times12_init]
times_init = times_init_forward + times_init_backward
print(len(times_init))

times_final_forward = [times12_final]#times12_final must be included in times_final
times_final_backward = [times12_final]
times_final = times_final_forward + times_final_backward
print(len(times_final))

motion_planner = mp.MotionPlanner() 
#traj_points = motion_planner.generate_trajectory_jointSpace(waypoints,times_init,times_final,1000)# Code with old way of computing the trajectories
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
