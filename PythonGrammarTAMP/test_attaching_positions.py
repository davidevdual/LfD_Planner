from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed

import numpy as np
import numpy.matlib
import math 
import pybullet
import pybullet_data
import time

# The Transformation matrices

# Transform the position and orientation of a point expressed in the robot's Frame 6 coordinates to simulation
# @input coords Coordinates expressed according to the robot's frame 6
# @return Coordinates expressed according to simulation's world frame
def convert_Frame6_to_FrameSim(coords):
    # Check that the coords vector is a column vector
    if np.ndim(coords)!=2 or np.size(coords,1)!=1 or np.size(coords,0)!=3 or type(coords)!=np.array:
        raise Exception("[test_attaching.py] Wrong input");

    return 0;

def main():
    urdf_path = 'C:\\Users\\David\\OneDrive - National University of Singapore\\PhD_work\\Code\\TAMP_HighLevel\\Visual_Studio\\source';

    #can also connect using different modes, GUI, SHARED_MEMORY, TCP, UDP, SHARED_MEMORY_SERVER, GUI_SERVER
    physicsClient = pybullet.connect(pybullet.GUI);#or p.DIRECT for non-graphical version
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath());
    pybullet.setGravity(0,0,-10);
    planeId = pybullet.loadURDF("plane.urdf");
    boxId3 = pybullet.loadURDF(urdf_path+"\\Assem1_forURDF\\urdf\\Assem1_forURDF.urdf",[0,0,1.12],pybullet.getQuaternionFromEuler([0,0,0]));# Object1
    boxId4 = pybullet.loadURDF(urdf_path+"\\object1_forURDF\\urdf\\object1.urdf",[1,0,1],pybullet.getQuaternionFromEuler([0,0,0]));# Object2

    ed0 = ed.UrdfEditor()
    ed0.initializeFromBulletBody(boxId3,physicsClient)
    ed1 = ed.UrdfEditor()
    ed1.initializeFromBulletBody(boxId4,physicsClient)

    pybullet.setTimeStep(0.004);

    pybullet.setRealTimeSimulation(1);

    # Get the position of the object
    link_data = pybullet.getBasePositionAndOrientation(boxId4);
    print("link_data: ",link_data);#position and orientation

    # Get the position of the right end-effector
    right_eef_link_index = 5;
    jointPst0 = pybullet.getJointState(boxId3,0);
    jointPst1 = pybullet.getJointState(boxId3,1);
    jointPst2 = pybullet.getJointState(boxId3,2);
    jointPst3 = pybullet.getJointState(boxId3,3);
    jointPst4 = pybullet.getJointState(boxId3,4);
    print("jointPst0: ",jointPst0[0]," jointPst1: ",jointPst1[0]," jointPst2: ",jointPst2[0]," jointPst3: ",jointPst3[0]," jointPst4: ",jointPst4[0]);

    # Get the position of the left end-effector
    left_eef_link_index = 10;
    link_info = pybullet.getLinkState(boxId3,left_eef_link_index);
    left_eef_link_WorldPosition = link_info[0];
    left_eef_link_WorldOrientation = link_info[1];
    print("left_eef_link_WorldPosition: ",left_eef_link_WorldPosition," left_eef_link_WorldOrientation: ",left_eef_link_WorldOrientation);

    while(pybullet.isConnected()):
        pybullet.getCameraImage(320, 200, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL);

        link_info = pybullet.getLinkState(boxId3,right_eef_link_index);
        right_eef_link_WorldPosition = link_info[0];
        right_eef_link_WorldOrientation = link_info[1];
        print("right_eef_link_WorldPosition: ",right_eef_link_WorldPosition," right_eef_link_WorldOrientation: ",right_eef_link_WorldOrientation);
        # Set the position and orientation of the object when it is take by the eef
        object_WorldPosition = [right_eef_link_WorldPosition[0],right_eef_link_WorldPosition[1],right_eef_link_WorldPosition[2]-0.07];
        print("object_WorldPosition: ",object_WorldPosition);
        pybullet.resetBasePositionAndOrientation(boxId4,object_WorldPosition,right_eef_link_WorldOrientation);# Set the object to a new position
        pybullet.stepSimulation();

if __name__=="__main__":
	main();