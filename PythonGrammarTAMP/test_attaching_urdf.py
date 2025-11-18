#####################################################################
## This file contains the code to test the attachment of an object
## to the robot's end-effectors by using the createMultiBody command
#####################################################################
#####################################################################
## Author: David Carmona-Moreno
## Copyright: Copyright 2023, Dual-Arm project
## Version: v1.1
## Maintainer: David Carmona-Moreno
## Email: e0348847@u.nus.edu
## Status: First stable release
#####################################################################

from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed
from simulator import Attachment

import numpy as np
import pybullet
import pybullet_data
import time
import math

urdf_path = 'C:\\Users\\David\\OneDrive - National University of Singapore\\PhD_work\\Code\\TAMP_HighLevel\\Visual_Studio\\source';
urdf_path2 = 'C:\\Users\\David\\OneDrive - National University of Singapore\\PhD_work\\Code\\TAMP_HighLevel\\Visual_Studio\\PythonGrammarTAMP';

# Check if one of the robot's end-effectors is close to an object; hence, we must attach it to the end-effector
# @input side String. End-effector that we are tracking. Two choices: "left_eef" or "right_eef"
# @input robotId Int. Robot's unique identifier
# @input mapping Dictionary. Dictionary mapping the robot's joint names to their unique identifiers.
# @input availabe_objects List. Names of the objects that are available in the scene, ready for manipulation.
# @input threshold Float (in metres). Number indicating how far away the object must be close to the end-effector so it is considered attached.
# @input physicsClient Physics Client
# @return eef_is_attached,object_editor,joints_links_info,available_objects,attached_object
def checkeefisAttached(side,robotId,mapping,available_objects,threshold,physicsClient):
    if type(side)!=str or type(robotId)!=int or type(mapping)!=dict or type(available_objects)!=list or type(threshold)!=float or type(physicsClient)!=int:
        raise Exception("[test_attaching_urdf.py] Wrong input.");
    if (side!="left_eef" and side!="right_eef") or robotId<0 or len(mapping)<=0 or len(available_objects)<=0 or threshold<0 or physicsClient<0:
        raise Exception("[test_attaching_urdf.py] Wrong input.");
    
    eef_is_attached = 0;
    attached_object = 0;
    object_editor = 0;
    joints_links_info = getLinkJointStates(robotId,mapping);

    for k in range(0,len(available_objects)):
        selected_object = available_objects[k];
        selected_object_translation = pybullet.getBasePositionAndOrientation(selected_object);
        selected_object_translation = np.array(selected_object_translation[0]);
        if side=="left_eef":
            eef_is_attached,dist = is_close(joints_links_info[10],selected_object_translation,threshold);
        elif side=="right_eef":
            eef_is_attached,dist = is_close(joints_links_info[11],selected_object_translation,threshold);

        if eef_is_attached == 1:
            counter = k;
            break;

    if eef_is_attached == 1:
        attached_object = available_objects.pop(counter);# Remove the object that is close to the end-effector
        object_editor = ed.UrdfEditor();
        object_editor.initializeFromBulletBody(attached_object,physicsClient);

    return eef_is_attached,object_editor,joints_links_info,available_objects,attached_object;

# Attach the end-effector to the object which is the closest to the end-effector
# @return bodyId The robot's unique identifier
# @return mapping The unique mapping between the 
def attachEEF(bodyId,ed0,joint_name,attached_object,mapping,object_editor,parentLinkIndex,jointPivotXYZInParent,jointPivotRPYInParent,jointPivotXYZInChild,jointPivotRPYInChild,joints_links_info,filename):
    pybullet.removeBody(bodyId);
    pybullet.removeBody(attached_object);
    newjoint = ed0.joinUrdf(object_editor,parentLinkIndex,jointPivotXYZInParent,jointPivotRPYInParent,
                    jointPivotXYZInChild,jointPivotRPYInChild);
    newjoint.joint_type = pybullet.JOINT_FIXED;
    newjoint.joint_name = joint_name;#'joint_dummy1'

    orn = [0,0,0,1];
    bodyId = ed0.createMultiBody([0,0,1.12],orn);
    ed0.saveUrdf(filename);#birobot_copy1.urdf
    pybullet.removeBody(bodyId);
    bodyId = pybullet.loadURDF(urdf_path2+"\\"+filename,[0,0,1.12],pybullet.getQuaternionFromEuler([0,0,0]));
    pybullet.changeVisualShape(bodyId,1,rgbaColor=[0.79216,0.81961,0.93333,1.00000]);# The RTA link is loaded in a black colour for an unknown reason. Therefore, make it the same colour as all the other links.
    mapping = map_ids_names(bodyId);
    if b'obj2_chassis_joint' in mapping:
        pybullet.changeVisualShape(bodyId,mapping[b'obj2_chassis_joint'],rgbaColor=[0,0,0,1]);
    if b'obj1_chassis_joint' in mapping:
        pybullet.changeVisualShape(bodyId,mapping[b'obj1_chassis_joint'],rgbaColor=[0.2,0.2,1,1]);

    pybullet.resetJointState(bodyId,mapping[b'Joint_RTA'],joints_links_info[0]);
    pybullet.resetJointState(bodyId,mapping[b'Joint_RPA'],joints_links_info[1]);
    pybullet.resetJointState(bodyId,mapping[b'Joint_RSA'],joints_links_info[2]);
    pybullet.resetJointState(bodyId,mapping[b'Joint_REA'],joints_links_info[3]);
    pybullet.resetJointState(bodyId,mapping[b'Joint_RWA'],joints_links_info[4]);
    pybullet.resetJointState(bodyId,mapping[b'Joint_LTA'],joints_links_info[5]);
    pybullet.resetJointState(bodyId,mapping[b'Joint_LPA'],joints_links_info[6]);
    pybullet.resetJointState(bodyId,mapping[b'Joint_LSA'],joints_links_info[7]);
    pybullet.resetJointState(bodyId,mapping[b'Joint_LEA'],joints_links_info[8]);
    pybullet.resetJointState(bodyId,mapping[b'Joint_LWA'],joints_links_info[9]);
    return bodyId,mapping;

# Get the Link and Joint states
# @input bodyId Int. The robot's unique identifier
# @input mapping Dictionary. A dictioonary mapping the joint identifiers to their names
# @return List with all the joints and links information
def getLinkJointStates(bodyId,mapping):
    if bodyId<0 or type(mapping)!=dict or len(mapping)<=0:
        raise Excpetion("[test_attaching_urdf] Wronf input.");
    angle1 = pybullet.getJointState(bodyId,mapping[b'Joint_RTA']);angle1 = angle1[0];
    angle2 = pybullet.getJointState(bodyId,mapping[b'Joint_RPA']);angle2 = angle2[0];
    angle3 = pybullet.getJointState(bodyId,mapping[b'Joint_RSA']);angle3 = angle3[0];
    angle4 = pybullet.getJointState(bodyId,mapping[b'Joint_REA']);angle4 = angle4[0];
    angle5 = pybullet.getJointState(bodyId,mapping[b'Joint_RWA']);angle5 = angle5[0];
    right_linkState = pybullet.getLinkState(bodyId,mapping[b'Joint_RWA']);
    right_linkState = np.array(right_linkState[0]);
    angle6 = pybullet.getJointState(bodyId,mapping[b'Joint_LTA']);angle6 = angle6[0];
    angle7 = pybullet.getJointState(bodyId,mapping[b'Joint_LPA']);angle7 = angle7[0];
    angle8 = pybullet.getJointState(bodyId,mapping[b'Joint_LSA']);angle8 = angle8[0];
    angle9 = pybullet.getJointState(bodyId,mapping[b'Joint_LEA']);angle9 = angle9[0];
    angle10 = pybullet.getJointState(bodyId,mapping[b'Joint_LWA']);angle10 = angle10[0];
    left_linkState = pybullet.getLinkState(bodyId,mapping[b'Joint_LWA']);
    left_linkState = np.array(left_linkState[0]);
    return [angle1,angle2,angle3,angle4,angle5,angle6,angle7,angle8,angle9,angle10,left_linkState,right_linkState];

# Create a dictionary that maps names to ids. The key entries are the names
# @input robot_id The robot unique identifier
# @return A dictionary that maps all the Joint Names to their Unique Identifiers
def map_ids_names(robot_id):
    if type(robot_id)!=int or robot_id<0:
        raise Exception("[test_attaching_urdf.py] Wrong input.");
    nb_joints = pybullet.getNumJoints(robot_id);
    #print("Number of joints: ",nb_joints);
    if nb_joints<=0:
        raise Exception("[test_attaching_rudf.py] The number of joints is zero or negative. It is abnormal.");
    mapping = {};# Initialisation of the empty dictionary mapping joint names to unique ids
    for joint_index in range(0,nb_joints):
        joint_info = pybullet.getJointInfo(robot_id,joint_index);
        joint_name = joint_info[1];
        mapping[joint_name] = joint_index;
    return mapping;

# Check if two positions are close
# @input pst1 3d Numpy array. First position. In meters
# @input pst2 3d Numpy array. Second position. In meters
# @input threshold Float. Threshold of the euclidean distance below which we consider two objects to be close
# @return Integer. 1 if both positions are close. 0 if both positions are not distant
# @return Integer. Float. Distance between the positions
def is_close(pst1,pst2,threshold):
    if type(pst1)!=np.ndarray or type(pst2)!=np.ndarray or len(pst1)!=3 or len(pst2)!=3 or threshold<0:
        raise Exception("[test_attaching_urdf.py] Wrong input");
    # Compute the euclidean distance between pst1 and pst2
    dist = math.sqrt(math.pow(pst1[0]-pst2[0],2)+math.pow(pst1[1]-pst2[1],2)+math.pow(pst1[2]-pst2[2],2));
    #print("distance is: ",dist);
    are_positions_close = 0;# 0 if no and 1 if yes
    if dist<0:
        raise Exception("[novelplans.py] Error in computations");
    if dist<=threshold:
        are_positions_close = 1;
    else:
        are_positions_close = 0;
    return are_positions_close,dist;

'''
def main():
    physicsClient = pybullet.connect(pybullet.GUI);#or p.DIRECT for non-graphical version
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath());
    #pybullet.setGravity(0,0,-10);

    birobot = pybullet.loadURDF(urdf_path+"\\Assem1_forURDF\\urdf\\Assem1_forURDF.urdf",[0,0,1.12],pybullet.getQuaternionFromEuler([0,0,0]));
    #boxId4 = pybullet.loadURDF(urdf_path+"\\object1_forURDF\\urdf\\object1.urdf",[0.2,0.2,1],pybullet.getQuaternionFromEuler([0,0,0]));
    #boxId5 = pybullet.loadURDF(urdf_path+"\\object2_forURDF\\urdf\\object2.urdf",[0.2,-0.2,1],pybullet.getQuaternionFromEuler([0,0,0]));
    mug = pybullet.loadURDF(urdf_path+"\\mug_forURDF\\model_mug.urdf",[0.4,0.2,1],pybullet.getQuaternionFromEuler([0,0,0]));
    pitcher = pybullet.loadURDF(urdf_path+"\\pitcher_forURDF\\model_pitcher.urdf",[0.4,-0.2,1],pybullet.getQuaternionFromEuler([0,0,0]));

    # List of objects present in the scene that can be grasped by the end-effectors
    available_objects = [mug,pitcher];

    ed0 = ed.UrdfEditor();
    ed0.initializeFromBulletBody(birobot,physicsClient);

    parentLinkIndex_right = 6;
    jointPivotXYZInParent_right = [0,0,-0.15];
    jointPivotRPYInParent_right = [0,0,0];
    jointPivotXYZInChild_right = [0,0,0];
    jointPivotRPYInChild_right = [0,-1.57,0];

    parentLinkIndex_left = 10;
    jointPivotXYZInParent_left = [0,0,-0.25];
    jointPivotRPYInParent_left = [0,0,0];
    jointPivotXYZInChild_left = [0,0,0];
    jointPivotRPYInChild_left = [0,-1.57,0];

    pybullet.setRealTimeSimulation(1);
    pybullet.setTimeStep(0.000001);#In seconds
    counter = 0;
    position = 0;
    righteef_is_attached = 0;# Variable that checks if the right end-effector is attached to any object
    lefteef_is_attached = 0;# Variable that checks if the left end-effector is attached to any object
    righteef_is_done = 0;# Variable that checks if the object attachment has been done on the right end-effector. Therefore, there is no need to do it again.
    lefteef_is_done = 0;# Variable that checks if the object attachment has been done on the left end-effector. Therefore, there is no need to do
    counter = 0;# Index of the object that is close to one of the robot's end-effectors
    mapping = map_ids_names(birobot);

    while (pybullet.isConnected()):

        if lefteef_is_attached == 0:
            lefteef_is_attached,object_editor_left,joints_links_info,available_objects,attached_object_left = checkeefisAttached("left_eef",birobot,mapping,available_objects,0.1,physicsClient);

        if righteef_is_attached == 0:
            righteef_is_attached,object_editor_right,joints_links_info,available_objects,attached_object_right = checkeefisAttached("right_eef",birobot,mapping,available_objects,0.1,physicsClient);

        if righteef_is_attached == 1 and righteef_is_done == 0:
            birobot,mapping = attachEEF(birobot,ed0,'joint_dummy1',attached_object_right,mapping,object_editor_right,parentLinkIndex_right,
                                jointPivotXYZInParent_right,jointPivotRPYInParent_right,jointPivotXYZInChild_right,jointPivotRPYInChild_right,joints_links_info,"birobot_copy2.urdf");
            righteef_is_done = 1;

        if lefteef_is_attached == 1 and lefteef_is_done == 0:
            birobot,mapping = attachEEF(birobot,ed0,'joint_dummy2',attached_object_left,mapping,object_editor_left,parentLinkIndex_left,
                                jointPivotXYZInParent_left,jointPivotRPYInParent_left,jointPivotXYZInChild_left,jointPivotRPYInChild_left,joints_links_info,"birobot_copy1.urdf");
            lefteef_is_done = 1;

        pybullet.stepSimulation();
        counter = counter + 1;

    pybullet.disconnect();
'''

def main():
    attachment = Attachment();
    physicsClient = pybullet.connect(pybullet.GUI);#or p.DIRECT for non-graphical version
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath());
    #pybullet.setGravity(0,0,-10);

    birobot = pybullet.loadURDF(urdf_path+"\\Assem1_forURDF\\urdf\\Assem1_forURDF.urdf",[0,0,1.12],pybullet.getQuaternionFromEuler([0,0,0]));
    #boxId4 = pybullet.loadURDF(urdf_path+"\\object1_forURDF\\urdf\\object1.urdf",[0.2,0.2,1],pybullet.getQuaternionFromEuler([0,0,0]));
    #boxId5 = pybullet.loadURDF(urdf_path+"\\object2_forURDF\\urdf\\object2.urdf",[0.2,-0.2,1],pybullet.getQuaternionFromEuler([0,0,0]));
    mug = pybullet.loadURDF(urdf_path+"\\mug_forURDF\\model_mug.urdf",[0.4,0.2,1],pybullet.getQuaternionFromEuler([0,0,0]));
    pitcher = pybullet.loadURDF(urdf_path+"\\pitcher_forURDF\\model_pitcher.urdf",[0.4,-0.2,1],pybullet.getQuaternionFromEuler([0,0,0]));

    # List of objects present in the scene that can be grasped by the end-effectors
    available_objects = [mug,pitcher];

    ed0 = ed.UrdfEditor();
    ed0.initializeFromBulletBody(birobot,physicsClient);

    parentLinkIndex_right = 6;
    jointPivotXYZInParent_right = [0,0,-0.15];
    jointPivotRPYInParent_right = [0,0,0];
    jointPivotXYZInChild_right = [0,0,0];
    jointPivotRPYInChild_right = [0,-1.57,0];

    parentLinkIndex_left = 10;
    jointPivotXYZInParent_left = [0,0,-0.25];
    jointPivotRPYInParent_left = [0,0,0];
    jointPivotXYZInChild_left = [0,0,0];
    jointPivotRPYInChild_left = [0,-1.57,0];

    pybullet.setRealTimeSimulation(1);
    pybullet.setTimeStep(0.000001);#In seconds
    counter = 0;
    position = 0;
    righteef_is_attached = 0;# Variable that checks if the right end-effector is attached to any object
    lefteef_is_attached = 0;# Variable that checks if the left end-effector is attached to any object
    righteef_is_done = 0;# Variable that checks if the object attachment has been done on the right end-effector. Therefore, there is no need to do it again.
    lefteef_is_done = 0;# Variable that checks if the object attachment has been done on the left end-effector. Therefore, there is no need to do
    counter = 0;# Index of the object that is close to one of the robot's end-effectors
    mapping = map_ids_names(birobot);

    while (pybullet.isConnected()):

        if lefteef_is_attached == 0:
            lefteef_is_attached,object_editor_left,joints_links_info,available_objects,attached_object_left = attachment.checkeefisAttached("left_eef",birobot,mapping,available_objects,0.1,physicsClient);

        if righteef_is_attached == 0:
            righteef_is_attached,object_editor_right,joints_links_info,available_objects,attached_object_right = attachment.checkeefisAttached("right_eef",birobot,mapping,available_objects,0.1,physicsClient);

        if righteef_is_attached == 1 and righteef_is_done == 0:
            birobot,mapping = attachment.attachEEF(birobot,ed0,'joint_dummy1',attached_object_right,mapping,object_editor_right,parentLinkIndex_right,
                                    jointPivotXYZInParent_right,jointPivotRPYInParent_right,jointPivotXYZInChild_right,jointPivotRPYInChild_right,joints_links_info,"birobot_copy2.urdf");
            righteef_is_done = 1;

        if lefteef_is_attached == 1 and lefteef_is_done == 0:
            birobot,mapping = attachment.attachEEF(birobot,ed0,'joint_dummy2',attached_object_left,mapping,object_editor_left,parentLinkIndex_left,
                                    jointPivotXYZInParent_left,jointPivotRPYInParent_left,jointPivotXYZInChild_left,jointPivotRPYInChild_left,joints_links_info,"birobot_copy1.urdf");
            lefteef_is_done = 1;

        pybullet.stepSimulation();
        counter = counter + 1;

    pybullet.disconnect();

if __name__ == "__main__":
    main();