"""
This demo script demonstrates the various functionalities of each controller available within robosuite.

For a given controller, runs through each dimension and executes a perturbation "test_value" from its
neutral (stationary) value for a certain amount of time "steps_per_action", and then returns to all neutral values
for time "steps_per_rest" before proceeding with the next action dim.

	E.g.: Given that the expected action space of the Pos / Ori (OSC_POSE) controller (without a gripper) is
	(dx, dy, dz, droll, dpitch, dyaw), the testing sequence of actions over time will be:

		***START OF DEMO***
		( dx,  0,  0,  0,  0,  0, grip)     <-- Translation in x-direction      for 'steps_per_action' steps
		(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
		(  0, dy,  0,  0,  0,  0, grip)     <-- Translation in y-direction      for 'steps_per_action' steps
		(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
		(  0,  0, dz,  0,  0,  0, grip)     <-- Translation in z-direction      for 'steps_per_action' steps
		(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
		(  0,  0,  0, dr,  0,  0, grip)     <-- Rotation in roll (x) axis       for 'steps_per_action' steps
		(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
		(  0,  0,  0,  0, dp,  0, grip)     <-- Rotation in pitch (y) axis      for 'steps_per_action' steps
		(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
		(  0,  0,  0,  0,  0, dy, grip)     <-- Rotation in yaw (z) axis        for 'steps_per_action' steps
		(  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
		***END OF DEMO***

	Thus the OSC_POSE controller should be expected to sequentially move linearly in the x direction first,
		then the y direction, then the z direction, and then begin sequentially rotating about its x-axis,
		then y-axis, then z-axis.

Please reference the documentation of Controllers in the Modules section for an overview of each controller.
Controllers are expected to behave in a generally controlled manner, according to their control space. The expected
sequential qualitative behavior during the test is described below for each controller:

* OSC_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
			z-axis, relative to the global coordinate frame
* OSC_POSITION: Gripper moves sequentially and linearly in x, y, z direction, relative to the global coordinate frame
* IK_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
			z-axis, relative to the local robot end effector frame
* JOINT_POSITION: Robot Joints move sequentially in a controlled fashion
* JOINT_VELOCITY: Robot Joints move sequentially in a controlled fashion
* JOINT_TORQUE: Unlike other controllers, joint torque controller is expected to act rather lethargic, as the
			"controller" is really just a wrapper for direct torque control of the mujoco actuators. Therefore, a
			"neutral" value of 0 torque will not guarantee a stable robot when it has non-zero velocity!

"""

import pybullet as p
import pybullet_data
import robosuite as suite
import numpy as np
import numpy.matlib
import os
import Motion_Planning.RigidBodyPlanning as rbp
import time
import Motion_Planning.plotting as plot
import Motion_Planning.ourmotionplanning as our_motion_planner
import utils.common as utils
import Task_Planning.Fast_Downward.fastdownward as fd
import Task_Planning.Language_Model.humanTAMP as human_tamp

from robosuite.controllers.ik import *
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.robots import Bimanual
from robosuite.controllers.interpolators.linear_interpolator import LinearInterpolator
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
from scipy.spatial.transform import Rotation as R

def to_radians(degrees):
	return degrees*0.01745329251

def initialize_robot(bodyUniqueId,sim,viewer,endEffectorLinkIndex_left,endEffectorLinkIndex_right):
	pose_left = [[0.514992892742157, 0.21485956013202667, 1.9788315296173096]]
	orientation_left = [[0.02527622, -0.01184546, 0.70701492, 0.70664752]]# No rotation
	pose_right = [[0.5149956941604614, -0.30806973576545715, 1.978783011436462]]
	orientation_right = [[0.01108189, 0.02130469, 0.99705648, -0.07281214]]# No rotation
	joint_angles_left = get_positions_eef(bodyUniqueId,endEffectorLinkIndex_left,pose_left,orientation_left)
	joint_angles_right = get_positions_eef(bodyUniqueId,endEffectorLinkIndex_right,pose_right,orientation_right)
	joints_angles = np.add(joint_angles_left,joint_angles_right)

	initialized = False
	if initialized == False:
		joint_angles = joints_angles[0]
		sim_state = sim.get_state()

		#reset the joint state (ignoring all dynamics, not recommended to use during simulation)
		numJoints = p.getNumJoints(bodyUniqueId)
		for i in range(numJoints):
			jointInfo = p.getJointInfo(bodyUniqueId, i)
			joint_name = str(jointInfo[1])
			joint_name = joint_name.replace('b', '')
			joint_name = joint_name.replace("'", "")
			qIndex = jointInfo[3]
			if qIndex > -1:
				sim_state.qpos[sim.model.get_joint_qpos_addr(joint_name)] = joint_angles[qIndex - 7]
				p.resetJointState(bodyUniqueId, i, joint_angles[qIndex - 7])

		sim.set_state(sim_state)
		sim.forward()
		viewer.render()
		p.stepSimulation()
		initialized = True

def get_joint_ranges(robot_id):
	"""
	Return a default set of values for the arguments to IK
	with nullspace turned on. Returns joint ranges from the
	URDF and the current value of each joint angle for the
	rest poses.

	Returns:
		4-element tuple containing:

		- list: list of lower limits for each joint (shape: :math:`[DOF]`).
		- list: list of upper limits for each joint (shape: :math:`[DOF]`).
		- list: list of joint ranges for each joint (shape: :math:`[DOF]`).
		- list: list of rest poses (shape: :math:`[DOF]`).
	"""
	n = p.getNumJoints(robot_id)
	rest = []
	lower = []
	upper = []
	ranges = []
	for i in range(n):
		info = p.getJointInfo(robot_id, i)
		# Retrieve lower and upper ranges for each relevant joint
		if info[3] > -1:
			rest.append(p.getJointState(robot_id, i)[0])
			lower.append(info[8])
			upper.append(info[9])
			ranges.append(info[9] - info[8])

	return lower, upper, ranges, rest

def fill_matrix(bigger_size,smaller_size):
	last_row = smaller_size[-1]
	last_row_repeated = np.matlib.repmat(last_row, len(bigger_size)-len(smaller_size), 1)
	smaller_size = np.vstack((smaller_size,last_row_repeated))
	return smaller_size

def pad(input_1,input_2):
	if len(input_1)>len(input_2):
		input_2 = fill_matrix(input_1,input_2)
	elif len(input_1)<len(input_2):
		input_1 = fill_matrix(input_2,input_1)
	elif len(input_1)==len(input_2):
		print("No padding needed")
	else:
		print("Error")
	return input_1,input_2

def oversample(states, nb_samples):
	oversampled_states = np.zeros(len(states[0]))
	for k in range(0,len(states)):
		if k!=len(states)-1:
			init_state = states[k]
			end_state = states[k+1]
			oversampled_states = np.vstack((oversampled_states, np.linspace(init_state, end_state, num=nb_samples)))

	oversampled_states = np.delete(oversampled_states, (0), axis=0)
	return oversampled_states

def extract_pose_trajectories(geometric_path):
	state_positions = np.zeros(3)
	state_orientations = np.zeros(4)

	for k in range(0,geometric_path.getStateCount()):
		state = geometric_path.getState(k)
		state_positions = np.vstack((state_positions,[state.getX(),state.getY(),state.getZ()]))
		state_orientations = np.vstack((state_orientations,[state.rotation().x,state.rotation().y,state.rotation().z,state.rotation().w]))

	state_positions = np.delete(state_positions, (0), axis=0)
	state_orientations = np.delete(state_orientations, (0), axis=0)

	# Oversample the positions to obtain a smoother trajectory
	state_positions = oversample(state_positions,300)
	state_orientations = oversample(state_orientations,300)

	return state_positions,state_orientations

def get_ee_pose(robot_id,ee_link_id):
	"""
	Return the end effector pose.

	Returns:
		4-element tuple containing

		- np.ndarray: x, y, z position of the EE (shape: :math:`[3,]`).
		- np.ndarray: quaternion representation of the
		  EE orientation (shape: :math:`[4,]`).
		- np.ndarray: rotation matrix representation of the
		  EE orientation (shape: :math:`[3, 3]`).
		- np.ndarray: euler angle representation of the
		  EE orientation (roll, pitch, yaw with
		  static reference frame) (shape: :math:`[3,]`).
	"""
	info = p.getLinkState(robot_id,ee_link_id)
	pos = info[4]
	quat = info[5]

	return np.array(pos), np.array(quat)

def euler2quat(euler, axes='xyz'):
	"""
	Convert euler angles to quaternion.
	Args:
		euler (list or np.ndarray): euler angles (shape: :math:`[3,]`).
		axes (str): Specifies sequence of axes for rotations.
			3 characters belonging to the set {'X', 'Y', 'Z'}
			for intrinsic rotations (rotation about the axes of a
			coordinate system XYZ attached to a moving body),
			or {'x', 'y', 'z'} for extrinsic rotations (rotation about
			the axes of the fixed coordinate system).
	Returns:
		np.ndarray: quaternion [x,y,z,w] (shape: :math:`[4,]`).
	"""
	r = R.from_euler(axes, euler)
	return r.as_quat()

def quat2euler(quat, axes='xyz'):
	"""
	Convert quaternion to euler angles.
	Args:
		quat (list or np.ndarray): quaternion [x,y,z,w] (shape: :math:`[4,]`).
		axes (str): Specifies sequence of axes for rotations.
			3 characters belonging to the set {'X', 'Y', 'Z'}
			for intrinsic rotations (rotation about the axes of a
			coordinate system XYZ attached to a moving body),
			or {'x', 'y', 'z'} for extrinsic rotations (rotation about
			the axes of the fixed coordinate system).
	Returns:
		np.ndarray: euler angles (shape: :math:`[3,]`).
	"""
	r = R.from_quat(quat)
	return r.as_euler(axes)

def quat2rot(quat):
	"""
	Convert quaternion to rotation matrix.
	Args:
		quat (list or np.ndarray): quaternion [x,y,z,w] (shape: :math:`[4,]`).
	Returns:
		np.ndarray: rotation matrix (shape: :math:`[3, 3]`).
	"""
	r = R.from_quat(quat)
	return r.as_dcm()

def to_quat(ori):
	"""
	Convert orientation in any form (rotation matrix,
	quaternion, or euler angles) to quaternion.
	Args:
		ori (list or np.ndarray): orientation in any following form:
			rotation matrix (shape: :math:`[3, 3]`)
			quaternion (shape: :math:`[4]`)
			euler angles (shape: :math:`[3]`).
	Returns:
		np.ndarray: quaternion [x, y, z, w](shape: :math:`[4, ]`).
	"""
	ori = np.array(ori)
	if ori.size == 3:
		# [roll, pitch, yaw]
		ori = euler2quat(ori)
	elif ori.shape == (3, 3):
		ori = rot2quat(ori)
	elif ori.size != 4:
		raise ValueError('Orientation should be rotation matrix, '
						 'euler angles or quaternion')
	return ori

def toRadians(degrees):
	return degrees*0.01745329251

def to_ppi_int(angle):
	return ((angle + np.pi) % (2 * np.pi) - np.pi)

def to_ppi(angle):
	"""
	Convert the angle to the range [-pi, pi).
	Args:
		angle (float): angle in radians.
	Returns:
		float: equivalent angle in [-pi, pi).
	"""
	return list(map(lambda x: (x + np.pi) % (2 * np.pi) - np.pi, angle))

def get_urdf_data(urdf_path):
	physicsClient = p.connect(p.DIRECT)# or p.DIRECT for non-graphical version
	p.setRealTimeSimulation(1) # Simulation will update as fast as it can in real time, instead of waiting for step commands like in the non-realtime case.
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	p.setGravity(0,0,-10)
	planeId = p.loadURDF("plane.urdf")
	robotStartPos = [0,0,1]
	robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
	bodyUniqueId = p.loadURDF(urdf_path)
	p.resetBasePositionAndOrientation(bodyUniqueId,robotStartPos,robotStartOrientation)

	endEffectorLinkIndex_left = p.getJointInfo(bodyUniqueId,6)[0]
	endEffectorLinkIndex_right = p.getJointInfo(bodyUniqueId,29)[0]

	return bodyUniqueId,endEffectorLinkIndex_left,endEffectorLinkIndex_right

def get_positions_eef(bodyUniqueId,endEffectorLinkIndex,trajectory_positions,trajectory_orientations):
	joint_angles = np.zeros(p.getNumJoints(bodyUniqueId)-1)
	for k in range(0,len(trajectory_positions)):   
		joint_angles = np.vstack((joint_angles,compute_joint_angles(bodyUniqueId,endEffectorLinkIndex,trajectory_positions[k],trajectory_orientations[k])))

	joint_angles = np.delete(joint_angles, (0), axis=0)
	return joint_angles

def compute_task_plan():
	task_planner = human_tamp.HumanTAMP("/home/yulab/Documents/PhD_Work/mujoco_dualarm_simulation/src/Task_Planning/Language_Model/model_TAMP.pth")
	task_planner.compute_plan("pouring pitcher_base into mug")
	task_plan = task_planner.get_plan()
	task_planning_time = task_planner.get_planning_time()
	return task_plan,task_planning_time

def compute_joint_angles(bodyUniqueId, endEffectorLinkIndex, targetPosition, targetOrientation):
	ikSolver = 0
	joint_angles = p.calculateInverseKinematics(bodyUniqueId,endEffectorLinkIndex,targetPosition,targetOrientation,
												solver=ikSolver,maxNumIterations=100)# Orientation not considered at the present moment
	return joint_angles

def compute_motion_plan(task_plan,bodyUniqueId,endEffectorLinkIndex_left,endEffectorLinkIndex_right,init_position_left,init_position_right,init_orientation_left,init_orientation_right):
	motion_planner = our_motion_planner.OurMotionPlanning(bodyUniqueId,endEffectorLinkIndex_left,endEffectorLinkIndex_right,init_position_left,init_position_right,init_orientation_left,init_orientation_right)
	motion_planner.compute_plan(task_plan)
	motion_plan_left,motion_plan_right = motion_planner.get_motion_plan()

	if len(motion_plan_left) == 0:
		motion_plan_left = np.array(compute_joint_angles(bodyUniqueId, endEffectorLinkIndex_left, init_position_left, init_orientation_left)).reshape(-1, 44)
	elif len(motion_plan_right)==0:
		motion_plan_right = np.array(compute_joint_angles(bodyUniqueId, endEffectorLinkIndex_right, init_position_right, init_orientation_right)).reshape(-1, 44)

	return motion_plan_left,motion_plan_right,motion_planner.get_total_motion_time()

if __name__ == "__main__":
	rigid_body_planning = rbp.RigidBodyPlanning()

	urdf_path = "/home/yulab/Documents/PhD_Work/mujoco_dualarm_simulation/robotic_arm_description/urdf/robotic_arm_description.urdf"
	model = load_model_from_path("../xmls/Sample_body_2_withJointLimits.xml")
	sim = MjSim(model)
	viewer = MjViewer(sim)
	bodyUniqueId,endEffectorLinkIndex_left,endEffectorLinkIndex_right = get_urdf_data(urdf_path)
	initialize_robot(bodyUniqueId,sim,viewer,endEffectorLinkIndex_left,endEffectorLinkIndex_right)
	init_position_left,init_orientation_left = get_ee_pose(bodyUniqueId,endEffectorLinkIndex_left)
	init_position_right,init_orientation_right = get_ee_pose(bodyUniqueId,endEffectorLinkIndex_right)

	results_taskPlan_time = []
	results_motionPlan_time = []
	results_totalPlan_time = []

	for k in range(0,100):
		task_plan,task_planning_time = compute_task_plan()
		motion_plan_left,motion_plan_right,motion_planning_time = compute_motion_plan(task_plan,bodyUniqueId,endEffectorLinkIndex_left,endEffectorLinkIndex_right,
																						init_position_left,init_position_right,init_orientation_left,init_orientation_right)# The motion plan is in the configuration space
		results_taskPlan_time.append(task_planning_time)
		results_motionPlan_time.append(motion_planning_time)
		results_totalPlan_time.append(task_planning_time+motion_planning_time)

		results_taskPlan_matrix = np.asarray(results_taskPlan_time)
		results_motionPlan_matrix = np.asarray(results_motionPlan_time)
		results_totalPlan_matrix = np.asarray(results_totalPlan_time)

		np.save('results_taskPlanOurs_matrix.npy', results_taskPlan_matrix)
		np.save('results_motionPlanOurs_matrix.npy', results_motionPlan_matrix)
		np.save('results_totalPlanOurs_matrix.npy', results_totalPlan_matrix)
		print("task plan time is: ",task_planning_time," and motion plan is: ",motion_planning_time, " together: ",task_planning_time+motion_planning_time)