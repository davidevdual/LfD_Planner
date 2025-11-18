#################################################################
## This code contains all the additional functions that the 
## other classes may need to perform their operations correctly.
#################################################################
#################################################################
## Author: David Carmona-Moreno
## Copyright: Copyright 2020, Dual-Arm project
## Version: v1.1
## Maintainer: David Carmona-Moreno
## Email: e0348847@u.nus.edu
## Status: First stable release
#################################################################

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

M_PI = 3.14159265358979323846264338327950288

## @class Helper
#  @brief Class containing extra functions necessary to the well-execution of the code
class Helper:

	## Global method transforming degrees into radians
	#  @param degrees      An integer value 
	#  @return             An integer value
	def to_radians(self, degrees):
		return degrees * M_PI/180.0
	
	## Global method transforming degrees into radians
	#  @param degrees      An integer value 
	#  @return             An integer value
	def to_radians(self, degrees):
		return degrees * M_PI/180.0

	## Global method transforming radians into radians.
	#  @param radians      An integer value 
	#  @return             An integer value 
	def to_degrees(self, radians):
		return radians * 180.0/M_PI

	## Convenience global method for testing if a list of values are within a tolerance of their counterparts in another list
	#  @param goal         A list of floats, a Pose or a PoseStamped
	#  @param actual       A list of floats, a Pose or a PoseStamped
	#  @param tolerance    A float
	#  @return             A bool
	def all_close(self, goal, actual, tolerance):

	 	all_equal = True
		if type(goal) is list:
			for index in range(len(goal)):
				if abs(actual[index] - goal[index]) > tolerance:
					return False

				elif type(goal) is geometry_msgs.msg.PoseStamped:
					return self.all_close(goal.pose, actual.pose, tolerance)

		elif type(goal) is geometry_msgs.msg.Pose:
			return self.all_close(pose_to_list(goal), pose_to_list(actual), tolerance)
			
	  	return True