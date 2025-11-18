import socket
import time
import motionplanner as mp
import simulator as sim
import trajectorygeneration as tg
import simulator as sim
import numpy as np
import utils as utls

# 0 to not visualize the simulation. # 1 to visualize the simulation
is_simulation = 1;

#list_to_convert is a list of integers
def listToString(list_to_convert,delimitor):

	# Empty string for initialization
	final_string = "";

	# Go through the list and concatenate to the initial variable
	for elmt in list_to_convert:
		elmt = '{:f}'.format(elmt); 
		elmt = str(elmt);
		final_string += (elmt + delimitor);
	return final_string;

# Important variables
localIP     = "10.0.0.1";
localPort   = 8080;			
bufferSize  = 1024;
arm_left = 1;
arm_right = 0;

# Initial pose for the left arm. The order is: LSR,LSA,LSFE,LEFE,LWR
pose_left = utls.runFK(0,0,90,0,0,arm_left);
print("pose 1: ",pose_left);
agls = utls.runIK(pose_left[0],pose_left[1],pose_left[2],pose_left[3],pose_left[4],pose_left[5],pose_left[6],100,1,arm_left);
print("agls= ",agls);
pose_left  = utls.runFK(agls[0],agls[1],agls[2],agls[3],agls[4],arm_left);
print("pose 2: ",pose_left);

# Initial pose for the right arm
pose_right = utls.runFK(0,0,0,90,0,arm_right);
print("pose 1: ",pose_right);
agls = utls.runIK(pose_right[0],pose_right[1],pose_right[2],pose_right[3],pose_right[4],pose_right[5],pose_right[6],100,1,arm_right);
print("agls= ",agls);
pose_right = utls.runFK(agls[0],agls[1],agls[2],agls[3],agls[4],arm_right);
print("pose 2: ",pose_right);

# 6D poses of the objects recognized by the vision system
object1 = [[-207.20477789,-312.875389,-270,0.03293905,-0.35213897,0.36608334,-0.86075325]];
object2 = [[-6.09758697,-420.4836115,-316.98958698,0.06026358,0.12072236,-0.76464444,-0.63016926]];

# Compute the waypoints for each initial and final states
motion_planner = mp.MotionPlanner(); 
initialStatesList_right = [[pose_right[0],pose_right[1],pose_right[2],pose_right[3],pose_right[4],pose_right[5],pose_right[6]]];
initialStatesList_left = [[pose_left[0],pose_left[1],pose_left[2],pose_left[3],pose_left[4],pose_left[5],pose_left[6]]];
finalStatesList_right = [[pose_right[0],pose_right[1],pose_right[2],pose_right[3],pose_right[4],pose_right[5],pose_right[6]]];
finalStatesList_left = [[pose_left[0],pose_left[1],pose_left[2],pose_left[3],pose_left[4],pose_left[5],pose_left[6]]];

arm_left = 1;
arm_right = 0;
waypoints_left = motion_planner.generate_waypoints_listStates(initialStatesList_left,finalStatesList_left,arm_left);
#print("waypoints_left before delete: ",waypoints_left);
#waypoints_left = np.delete(waypoints_left,0,0);
#print("waypoints_left after delete: ",waypoints_left);
waypoints_right = motion_planner.generate_waypoints_listStates(initialStatesList_right,finalStatesList_right,arm_right);
#waypoints_right = np.delete(waypoints_right,0,0);

# Compute the angles corresponding to each waypoint
waypoint2 = np.array([[0,30,0,0,0,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,0,0]]);
waypoint3 = np.array([[0,30,0,90,0,0,0,0,0,0,0,0,30,0,90,0,0,0,0,0,0,0]]);
waypoint4 = np.array([[0,0,0,90,0,0,0,0,0,0,0,0,0,0,90,0,0,0,0,0,0,0]]);

#print("length: ", waypoints_left[0]);
angles_left = motion_planner.convert_jointSpace(waypoints_left,arm_left);
angles_right = motion_planner.convert_jointSpace(waypoints_right,arm_right);
angles_left_hand = np.zeros([np.size(angles_left,0),6]);

angles_right_hand = np.zeros([np.size(angles_left,0),6]);
angles = np.hstack((angles_left,angles_left_hand,angles_right,angles_right_hand));
angles = np.vstack((np.zeros((1,22)),waypoint2,waypoint3,waypoint4,angles,angles,angles,angles,angles));

# Last point must include grasping
waypoint5 = angles[-1,:];
waypoint5[6] = 0;
waypoint5[7] = 0;
waypoint5[8] = 0;
waypoint5[9] = 0;
waypoint5[10] = 0;
angles = np.vstack((angles,waypoint5));
waypoints = list(angles);# Create a list of waypoints

# Compute the trajectory
times_init = (np.zeros(len(angles)-1)).tolist();
times_final = (2*(np.ones(len(angles)))).tolist();# Each trajectory is executed in two seconds
angles_left = angles[:,0:11];
angles_right = angles[:,11:22];

traj_points_left,times_left = motion_planner.generate_trajectory_jointSpace(angles_left,times_init,times_final,0.01,len(angles_left[0]));
traj_points_right,times_right = motion_planner.generate_trajectory_jointSpace(angles_right,times_init,times_final,0.01,len(angles_right[0]));

# Synchronize both trajectories for the left and right arms if they have different number of points
traj_points = motion_planner.synchronize_arms(traj_points_left,traj_points_right,times_left,times_right);

# Do not move the left and right wrists at all
traj_points[:,4] = 0;
traj_points[:,15] = 0;
traj_points = np.vstack((traj_points,np.flip(traj_points, axis=0)));# Flip the array vertically

traj_points = traj_points.tolist();
np.savetxt('data.csv', traj_points, delimiter=',');

if is_simulation==1:
	############################################ Run the simulator ###############################################
	traj_point_mat = np.array(traj_points);
	angles_left = traj_point_mat[:,0:5];
	angles_right = traj_point_mat[:,11:16];
	sim.runSimulator(angles_left,angles_right,np.transpose(np.array(object1,ndmin=2)),np.transpose(np.array(object2,ndmin=2)));
	exit(-1);
	##############################################################################################################

# For the robot's controller, LSFE and LEFE positions in the array of angles are swapped. Therefore, we must swap them here
traj_points = utls.swap_elmts(traj_points,2,3);

# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM);

# Bind to address and ip
print('Before the bind');
UDPServerSocket.bind((localIP, localPort));
print('After the bind');

# Tuple containing the address and port of UDP client to listen to
clientAddress = ("10.0.0.2",8080);

# Stops until it receives the message from the client
print('Before addresspair');
bytesAddressPair = UDPServerSocket.recvfrom(bufferSize);
print('After addresspair');

address = bytesAddressPair[1];

print('before client ip');
clientIP  = "Client IP Address:{}".format(address);
print('after client ip');
time.sleep(10);

print("UDP server up and listening");

time.sleep(10);

# Listen for incoming datagrams
for angles in traj_points:

	output_str = listToString(angles,',')[:-1];
	msgFromServer       = output_str;
	bytesToSend         = str.encode(msgFromServer);

	serverMsgPrint = "Message from Server:{}".format(msgFromServer);
	print(bytesToSend);

	# Sending a reply to client
	UDPServerSocket.sendto(bytesToSend, address);

	time.sleep(0.01);