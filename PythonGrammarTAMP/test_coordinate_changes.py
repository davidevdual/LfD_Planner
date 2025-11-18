from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed

import numpy as np
import numpy.matlib
import math 
import pybullet
import pybullet_data
import time

pi = 3.1416;

# Compute the sinus of a value
# @input value Float. Radians
# @return Float. Sinus of value 
def sin(value):
	return math.sin(value);

# Compute the cosinus of a value
# @input value Float. Radians
# @return Float. Cosinus of value
def cos(value):
	return math.cos(value);

# Transformation matrix between the right eef's frame to the World coordinate
def Tw6_right(q0,q1,q2,q3,q4,q5,d0,l0,l1,l2,l3,l4):
	Tw6 = np.zeros((4,4));
	Tw6[0,0] = cos((pi*q5)/180)*(cos((pi*q4)/180)*(cos((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*sin((pi*(q0 + 180))/180) + cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180)) + sin((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*sin((pi*(q0 + 180))/180) - sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180))) - sin((pi*q4)/180)*cos((pi*(q0 + 180))/180)*sin((pi*(q1 + 90))/180)) + sin((pi*q5)/180)*(cos((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*sin((pi*(q0 + 180))/180) - sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180)) - sin((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*sin((pi*(q0 + 180))/180) + cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180)));
	Tw6[1,0] = - sin((pi*q5)/180)*(cos((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*cos((pi*(q0 + 180))/180) + sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180)) - sin((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*cos((pi*(q0 + 180))/180) - cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180))) - cos((pi*q5)/180)*(cos((pi*q4)/180)*(cos((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*cos((pi*(q0 + 180))/180) - cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180)) + sin((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*cos((pi*(q0 + 180))/180) + sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180))) + sin((pi*q4)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180));
	Tw6[2,0] = sin((pi*q5)/180)*(cos((pi*q2)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q3 + 90))/180) + sin((pi*q2)/180)*cos((pi*(q3 + 90))/180)*sin((pi*(q1 + 90))/180)) - cos((pi*q5)/180)*(sin((pi*q4)/180)*cos((pi*(q1 + 90))/180) + cos((pi*q4)/180)*(cos((pi*q2)/180)*cos((pi*(q3 + 90))/180)*sin((pi*(q1 + 90))/180) - sin((pi*q2)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q3 + 90))/180)));
	Tw6[3,0] = 0;

	Tw6[0,1] = - sin((pi*q4)/180)*(cos((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*sin((pi*(q0 + 180))/180) + cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180)) + sin((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*sin((pi*(q0 + 180))/180) - sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180))) - cos((pi*q4)/180)*cos((pi*(q0 + 180))/180)*sin((pi*(q1 + 90))/180);
	Tw6[1,1] = sin((pi*q4)/180)*(cos((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*cos((pi*(q0 + 180))/180) - cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180)) + sin((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*cos((pi*(q0 + 180))/180) + sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180))) - cos((pi*q4)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180);
	Tw6[2,1] = sin((pi*q4)/180)*(cos((pi*q2)/180)*cos((pi*(q3 + 90))/180)*sin((pi*(q1 + 90))/180) - sin((pi*q2)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q3 + 90))/180)) - cos((pi*q4)/180)*cos((pi*(q1 + 90))/180);
	Tw6[3,1] = 0;

	Tw6[0,2] = sin((pi*q5)/180)*(cos((pi*q4)/180)*(cos((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*sin((pi*(q0 + 180))/180) + cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180)) + sin((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*sin((pi*(q0 + 180))/180) - sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180))) - sin((pi*q4)/180)*cos((pi*(q0 + 180))/180)*sin((pi*(q1 + 90))/180)) - cos((pi*q5)/180)*(cos((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*sin((pi*(q0 + 180))/180) - sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180)) - sin((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*sin((pi*(q0 + 180))/180) + cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180)));
	Tw6[1,2] = cos((pi*q5)/180)*(cos((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*cos((pi*(q0 + 180))/180) + sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180)) - sin((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*cos((pi*(q0 + 180))/180) - cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180))) - sin((pi*q5)/180)*(cos((pi*q4)/180)*(cos((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*cos((pi*(q0 + 180))/180) - cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180)) + sin((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*cos((pi*(q0 + 180))/180) + sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180))) + sin((pi*q4)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180));
	Tw6[2,2] = - sin((pi*q5)/180)*(sin((pi*q4)/180)*cos((pi*(q1 + 90))/180) + cos((pi*q4)/180)*(cos((pi*q2)/180)*cos((pi*(q3 + 90))/180)*sin((pi*(q1 + 90))/180) - sin((pi*q2)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q3 + 90))/180))) - cos((pi*q5)/180)*(cos((pi*q2)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q3 + 90))/180) + sin((pi*q2)/180)*cos((pi*(q3 + 90))/180)*sin((pi*(q1 + 90))/180));
	Tw6[3,2] = 0;

	Tw6[0,3] = d0 - l3*(cos((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*sin((pi*(q0 + 180))/180) - sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180)) - sin((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*sin((pi*(q0 + 180))/180) + cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180))) + l2*(sin((pi*q2)/180)*sin((pi*(q0 + 180))/180) + cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180)) - l4*(cos((pi*q5)/180)*(cos((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*sin((pi*(q0 + 180))/180) - sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180)) - sin((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*sin((pi*(q0 + 180))/180) + cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180))) - sin((pi*q5)/180)*(cos((pi*q4)/180)*(cos((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*sin((pi*(q0 + 180))/180) + cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180)) + sin((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*sin((pi*(q0 + 180))/180) - sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*cos((pi*(q0 + 180))/180))) - sin((pi*q4)/180)*cos((pi*(q0 + 180))/180)*sin((pi*(q1 + 90))/180))) - l1*cos((pi*(q0 + 180))/180)*sin((pi*(q1 + 90))/180);
	Tw6[1,3] = l3*(cos((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*cos((pi*(q0 + 180))/180) + sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180)) - sin((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*cos((pi*(q0 + 180))/180) - cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180))) - l2*(sin((pi*q2)/180)*cos((pi*(q0 + 180))/180) - cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180)) + l4*(cos((pi*q5)/180)*(cos((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*cos((pi*(q0 + 180))/180) + sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180)) - sin((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*cos((pi*(q0 + 180))/180) - cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180))) - sin((pi*q5)/180)*(cos((pi*q4)/180)*(cos((pi*(q3 + 90))/180)*(sin((pi*q2)/180)*cos((pi*(q0 + 180))/180) - cos((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180)) + sin((pi*(q3 + 90))/180)*(cos((pi*q2)/180)*cos((pi*(q0 + 180))/180) + sin((pi*q2)/180)*cos((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180))) + sin((pi*q4)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180))) - l1*sin((pi*(q1 + 90))/180)*sin((pi*(q0 + 180))/180);
	Tw6[2,3] = l0 - l4*(sin((pi*q5)/180)*(sin((pi*q4)/180)*cos((pi*(q1 + 90))/180) + cos((pi*q4)/180)*(cos((pi*q2)/180)*cos((pi*(q3 + 90))/180)*sin((pi*(q1 + 90))/180) - sin((pi*q2)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q3 + 90))/180))) + cos((pi*q5)/180)*(cos((pi*q2)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q3 + 90))/180) + sin((pi*q2)/180)*cos((pi*(q3 + 90))/180)*sin((pi*(q1 + 90))/180))) - l1*cos((pi*(q1 + 90))/180) - l3*(cos((pi*q2)/180)*sin((pi*(q1 + 90))/180)*sin((pi*(q3 + 90))/180) + sin((pi*q2)/180)*cos((pi*(q3 + 90))/180)*sin((pi*(q1 + 90))/180)) - l2*cos((pi*q2)/180)*sin((pi*(q1 + 90))/180);
	Tw6[3,3] = 1;
	return Tw6;

##	Load and execute the pybullet simulator with the set of joint positions
#	@param in_degrees Numpy matrix of dimensions 1x5 containing the joint positions (in radiians) for the left arm's joints
#   @return numpy matrix containing values in degrees.
def toDegrees(in_radians):
	if type(in_radians)!=list:
		raise Exception("[test_coordinate_changes.py] Wrong input");
	in_radians = np.multiply(np.array(in_radians),180/math.pi);
	return in_radians.tolist();

# Transform the position and orientation of a point expressed in the robot's Frame 6 coordinates to simulation
# @input coords List. Coordinates expressed according to the robot's frame 6
# @input angles_list List. List of the current robot's angles in degrees
# @return Coordinates expressed according to simulation's world frame
def convert_Frame6_to_FrameSim(coords,angles_list,l0):
	if type(coords)!=list or len(coords)!=3 or type(angles_list)!=list or len(angles_list)!=6 or l0<0:
		raise Exception("[test_coordinate_changes] Wrong input");
	q0=angles_list[0];q1=angles_list[1];q2=angles_list[2];q3=angles_list[3];q4=angles_list[4];q5=angles_list[5];
	d0=-55;l0=70;l1=-48.77;l2=300;l3=153;l4=180;
	coords = np.array(coords);
	coords = np.vstack((coords.reshape(-1,1),np.array([1])));
	Tw6 = Tw6_right(q0,q1,q2,q3,q4,q5,d0,l0,l1,l2,l3,l4);
	T = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,l0],[0,0,0,1]]);# From robot's world frame to simulation's frame
	coords = np.matmul(T,np.matmul(Tw6,coords));
	return coords[0:3,:];

def main():
	# Position of the object at all times with respect to the right end-effector's frame
	object_pst_right_eef = [0,0,70];# In mm
	angles_list = toDegrees([0,0,0,0,0,0]);# Angles in radians to convert to degrees
	l0 = 1120;# In mm

	# Convert the object's position wrt right end-effector to Simulation frame
	object_pst_sim = convert_Frame6_to_FrameSim(object_pst_right_eef,angles_list,l0);
	print("object_pst_sim: ",object_pst_sim/1000);
	print("original (in metres): ",[-0.03864336305430193, -0.14428531648698195, 0.627961823357212]);

if __name__=="__main__":
	main();
