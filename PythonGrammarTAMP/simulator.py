import pybullet 
import pybullet_data
import numpy as np
import numpy.matlib
import math 
import pybullet_utils
import time

from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed

urdf_path = 'C:\\Users\\David\\OneDrive - National University of Singapore\\PhD_work\\Code\\TAMP_HighLevel\\Visual_Studio\\source';
urdf_path2 = 'C:\\Users\\David\\OneDrive - National University of Singapore\\PhD_work\\Code\\TAMP_HighLevel\\Visual_Studio\\PythonGrammarTAMP';

# Class that manages attaching objects to the robot's end-effectors.
class Attachment:

    # Class constructor to define two member variables. Both variables check that either the left or right end-effectors
    # are attached to an object
    def __init__(self,_object1,_object2):
        self.lefteef_is_attached = 0# becomes 1 if the left end-effector is attached to an object
        self.righteef_is_attached = 0;# becomes 1 if the right end-effector is attached to an object
        self.object1 = _object1;
        self.object2 = _object2;
        
    # Check if one of the robot's end-effectors is close to an object; hence, we must attach it to the end-effector
    # @input side String. End-effector that we are tracking. Two choices: "left_eef" or "right_eef"
    # @input robotId Int. Robot's unique identifier
    # @input mapping Dictionary. Dictionary mapping the robot's joint names to their unique identifiers.
    # @input availabe_objects List. Names of the objects that are available in the scene, ready for manipulation.
    # @input threshold Float (in metres). Number indicating how far away the object must be close to the end-effector so it is considered attached.
    # @input physicsClient Physics Client
    # @return eef_is_attached,object_editor,joints_links_info,available_objects,attached_object
    def checkeefisAttached(self,side,robotId,mapping,available_objects,threshold,physicsClient):
        if type(side)!=str or type(robotId)!=int or type(mapping)!=dict or type(available_objects)!=list or type(threshold)!=float or type(physicsClient)!=int:
            raise Exception("[test_attaching_urdf.py] Wrong input.");
        if (side!="left_eef" and side!="right_eef") or robotId<0 or len(mapping)<=0 or len(available_objects)<=0 or threshold<0 or physicsClient<0:
            raise Exception("[test_attaching_urdf.py] Wrong input.");

        eef_is_attached = 0;
        attached_object = 0;
        object_editor = 0;
        joints_links_info = self.getLinkJointStates(robotId,mapping);

        for k in range(0,len(available_objects)):
            selected_object = available_objects[k];
            selected_object_translation = pybullet.getBasePositionAndOrientation(selected_object);
            selected_object_translation = np.array(selected_object_translation[0]);
            if side=="left_eef":
                eef_is_attached,dist = self.is_close(joints_links_info[10],selected_object_translation,threshold);
            elif side=="right_eef":
                eef_is_attached,dist = self.is_close(joints_links_info[11],selected_object_translation,threshold);
            if eef_is_attached == 1:
                counter = k;
                break;

        if eef_is_attached == 1:
            attached_object = available_objects.pop(counter);# Remove the object that is close to the end-effector
            object_editor = ed.UrdfEditor();
            object_editor.initializeFromBulletBody(attached_object,physicsClient);
        #print("Info: ",joints_links_info[10]);
        return eef_is_attached,object_editor,joints_links_info,available_objects,attached_object;

    # Attach the end-effector to the object which is the closest to the end-effector
    # @input bodyId Integer. The robot's unique identifier
    # @input ed0 Urdf editor. The URDF editor
    # @input joint_name String. The name of the joint to which the object is going to be attached
    # @input attached_object Integer. The identifier of the object that is going to be attached to the end-effector
    # @input mapping Dictionary. The mapping between the robot's joint names and identifiers
    # @input object_editor Object editor. The object editor
    # @input objects_attachments Dictionary. The coordinates of the point where the attachment with the object must happen
    # @input joints_links_info List. Information about the links.
    # @input filename String. Name of the urdf file containing the robot's structural information.
    # @input eef_side String. "right_eef" if the object is attached to the right end-effector. Otherwise, it is the left one.
    # @return bodyId,mapping The robot's unique identifier and new mapping between the joint names and identifiers
    def attachEEF(self,bodyId,ed0,joint_name,attached_object,mapping,object_editor,objects_attachments,joints_links_info,filename,eef_side):
        if type(objects_attachments)!=dict or type(bodyId)!=int or type(ed0)!=pybullet_utils.urdfEditor.UrdfEditor or type(joint_name)!=str or type(attached_object)!=int or type(mapping)!=dict or type(object_editor)!=pybullet_utils.urdfEditor.UrdfEditor:
            raise Exception("[simulator.py] Wrong input.");
        elif type(filename)!=str or type(joints_links_info)!=list or type(eef_side)!=str or (eef_side!="right_eef" and eef_side!="left_eef"):
            raise Exception("[simulator.py] Wrong input.");
        elif len(objects_attachments)!=4 or bodyId<0 or joint_name=="" or attached_object<0 or filename=="" or len(mapping)<= len(joints_links_info)<=0:
            raise Exception("[simulator.py] Wrong input.");
        
        # Attachment point coordinates
        parentLinkIndex=[];jointPivotXYZInParent=[];jointPivotRPYInParent=[];jointPivotXYZInChild=[];jointPivotRPYInChild=[];
        print("attached_object = ",attached_object," object1: ",self.object1," object2: ",self.object2);

        # Choose the point to attach the object according to the eef's side and object category
        if eef_side=="right_eef":
            if attached_object==self.object1:
                attachment = objects_attachments['object1_righteef'];
            if attached_object==self.object2:
                attachment = objects_attachments['object2_righteef'];
        if eef_side=="left_eef":
            if attached_object==self.object1:
                attachment = objects_attachments['object1_lefteef'];
            if attached_object==self.object2:
                attachment = objects_attachments['object2_lefteef'];
        parentLinkIndex = attachment[0];
        jointPivotXYZInParent = attachment[1];
        jointPivotRPYInParent = attachment[2];
        jointPivotXYZInChild = attachment[3];
        jointPivotRPYInChild = attachment[4];
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
        mapping = self.map_ids_names(bodyId);
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
    def getLinkJointStates(self,bodyId,mapping):
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
    def map_ids_names(self,robot_id):
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
    def is_close(self,pst1,pst2,threshold):
        if type(pst1)!=np.ndarray or type(pst2)!=np.ndarray or len(pst1)!=3 or len(pst2)!=3 or threshold<=0:
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

# Simulate the robot, the environment, and the interactions
class Simulator:

    # Class constructor
    def __init__(self):
        self.object1 = [[]];
        self.object2 = [[]];
        self.available_objects = [];
        self.attached_objects = [];
        self.pitcher = 0;
        self.mug = 0;

    ##  Convert numpy values from millimetres to metres.
    #   @param in_metres Array of values to convert in millimitres.
    #   @return Numpy array of same dimensions as in_metres with all values in metres
    def toMilli(self,in_metres):
        # Check that the coords vector is a column vector
        if np.ndim(in_metres)!=2 or np.size(in_metres,1)!=1 or np.size(in_metres,0)!=3:
            raise Exception("[simulator.py] The vector of coordinates is supposed to be a column vector with three rows (x,y,z). Here it is not.");
        # Check that the length between the simulation and robot's world coordinates origins is not negative or equal to zero
        return np.multiply(in_metres,0.001);#element-wise multiplication

    ##	Load and execute the pybullet simulator with the set of joint positions
    #	@param in_degrees Numpy matrix of dimensions 1x5 containing the joint positions (in radiians) for the left arm's joints
    #   @return numpy matrix containing values in degrees.
    def toDegrees(self,in_radians):
        return np.multiply(in_radians,180/math.pi);

    ##	Load and execute the pybullet simulator with the set of joint positions
    #	@param in_degrees Numpy matrix of dimensions 1x5 containing the joint positions (in degrees) for the left arm's joints
    #   @return numpy matrix containing values in radians.
    def toRadians(self,in_degrees):
        return np.multiply(in_degrees,math.pi/180);

    ##	Transform the coordinates from robot's world coordinates into simulation's world coordinates
    #	@param l0 Normal distance between the simulation and robot's world coordinates origins. In millimetres.
    #   @param coords Coordinates of the point in the robot's world frame. Numpy array. In millimetres.
    #   @return numpy array containing three coordinates: (x,y,z) expressed in the simulation world frame
    def transform2Sim(self,l0,coords):
        # Check that the coords vector is a column vector
        if np.ndim(coords)!=2 or np.size(coords,1)!=1 or np.size(coords,0)!=3:
            raise Exception("[simulator.py] The vector of coordinates is supposed to be a column vector with three rows (x,y,z). Here it is not.");
        # Check that the length between the simulation and robot's world coordinates origins is not negative or equal to zero
        if l0<=0:
            raise Exception("[simulator.py] The length cannot be negative or equal to zero. Here it is.");
        # Transformation matrix between the simulation and robot's world frames and multiplication.
        T = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,l0],[0,0,0,1]]);
        # Append one to the end of the coordinates vector. Otherwise, the arrays multiplication will raise an error.
        coords = np.matmul(T,np.vstack((coords,np.array([1]))));
        return coords[0:3,:];

    ## TO-DO: TO IMPLEMENT ONE DAY. STILL NOT FUNCTIONAL
    # Get the object's pose according to whether or not it is attached to the end-effector.
    # @return List of two lists. Each list contains the 7-value poses of the object1 and object2
    def getObjectsPoses(self):
        if type(self.attached_objects)!=list or type(self.attached_objects)!=list:
            raise Exception("[simulator.py] Wrong input.");
        if len(self.attached_objects)<0 or len(self.available_objects)<0:
            raise Exception("[simulator.py] Wrong computation.");
        object1_pst = 0;object2_pst = 1;
        return [self.object1,self.object2];

    ##	Load and execute the pybullet simulator with the set of joint positions. Be aware that the two arms will move to the back for the initialization
    #	@input target_joint_positions Python list containing the joint positions (in radians)
    #   @input threshold Float. The threshold down below the object is considered attached to the end-effector (in metres)
    def loadSimulator(self,target_joint_positions,object1,object2,threshold):

        object1 = self.toMilli(self.transform2Sim(1120,object1[0:3,:]));# Position of object1 is given with respect to the robot's frame. Then, it is converted into the World coordinates
        object2 = self.toMilli(self.transform2Sim(1120,object2[0:3,:]));# Position of object2 is given with respect to the robot's frame. Then, it is converted into the World coordinates
        physicsClient = pybullet.connect(pybullet.GUI);#or pybullet.DIRECT for non-graphical version
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath()); #used by loadURDF
        #pybullet.setGravity(0,0,-10);
        pybullet.setRealTimeSimulation(1);
        pybullet.setTimeStep(0.000001);#In seconds
        planeId = pybullet.loadURDF("plane.urdf");
        birobot = pybullet.loadURDF(urdf_path+"\\Assem1_forURDF\\urdf\\Assem1_forURDF.urdf",[0,0,1.12],pybullet.getQuaternionFromEuler([0,0,0]));# First ensemble is position. Second ensemble is orientation.
        boxId2 = pybullet.loadURDF(urdf_path+"\\table_forURDF\\urdf\\table.urdf",[0.63,0,0.29],pybullet.getQuaternionFromEuler([0,0,0]));# Table position: [0.47,0,0.29]
        self.pitcher = pybullet.loadURDF(urdf_path+"\\pitcher_forURDF\\model_pitcher.urdf",object1,pybullet.getQuaternionFromEuler([0,0,0]));# Object1 -1.5708
        self.mug = pybullet.loadURDF(urdf_path+"\\mug_forURDF\\model_mug.urdf",object2,pybullet.getQuaternionFromEuler([0,0,0]));# Object2
        jointIndices = np.arange(1,pybullet.getNumJoints(birobot),1,dtype=int);

        # Create the class that is going to attach the objects to the end-effectors
        attachment = Attachment(self.pitcher,self.mug);

        # List of objects present in the scene that can be grasped by the end-effectors
        self.available_objects = [self.mug,self.pitcher];

        ed0 = ed.UrdfEditor();
        ed0.initializeFromBulletBody(birobot,physicsClient);

        #object1_leftteef = [10,[-0.05,0,-0.3],[0,0,0],[0,0,0],[0,-1.57,0]];# 1st way of attaching
        object1_lefteef = [11,[-0.05,0,-0.12],[0,0,0],[0,0,0],[0,-1.57,1.57]];# 2nd way of attaching
        object2_righteef = [6,[0,0,-0.1],[0,0,0],[0,0,0],[0,-1.57,0]];
        object1_righteef = [6,[0,0,-0.15],[0,0,0],[0,0,0],[0,-1.57,0]];
        object2_lefteef = [11,[0,0,-0.1],[0,0,0],[0,0,0],[0,-1.57,0]];
        objects_attachments = {
                'object1_lefteef': object1_lefteef,
                'object2_righteef': object2_righteef,
                'object1_righteef': object1_righteef,
                'object2_lefteef': object2_lefteef,
        };

        righteef_is_attached = 0;# Variable that checks if the right end-effector is attached to any object
        lefteef_is_attached = 0;# Variable that checks if the left end-effector is attached to any object
        righteef_is_done = 0;# Variable that checks if the object attachment has been done on the right end-effector. Therefore, there is no need to do it again.
        lefteef_is_done = 0;# Variable that checks if the object attachment has been done on the left end-effector. Therefore, there is no need to do
        counter = 0;# Index of the object that is close to one of the robot's end-effectors
        mapping = attachment.map_ids_names(birobot);
        counter_position = 0;
        time.sleep(10);

        while 1:
            if counter<len(target_joint_positions):

                for position in target_joint_positions:
                    pybullet.resetJointState(birobot,mapping[b'Joint_RTA'],position[0]);
                    pybullet.resetJointState(birobot,mapping[b'Joint_RPA'],position[1]);
                    pybullet.resetJointState(birobot,mapping[b'Joint_RSA'],position[2]);
                    pybullet.resetJointState(birobot,mapping[b'Joint_REA'],position[3]);
                    pybullet.resetJointState(birobot,mapping[b'Joint_RWA'],position[4]);
                    pybullet.resetJointState(birobot,mapping[b'Joint_LTA'],position[5]);
                    pybullet.resetJointState(birobot,mapping[b'Joint_LPA'],position[6]);
                    pybullet.resetJointState(birobot,mapping[b'Joint_LSA'],position[7]);
                    pybullet.resetJointState(birobot,mapping[b'Joint_LEA'],position[8]);
                    pybullet.resetJointState(birobot,mapping[b'Joint_LWA'],position[9]);

                    '''
                    pybullet.resetJointState(birobot,1,position[0]);
                    pybullet.resetJointState(birobot,2,position[1]);
                    pybullet.resetJointState(birobot,3,position[2]);
                    pybullet.resetJointState(birobot,4,position[3]);
                    pybullet.resetJointState(birobot,5,position[4]);
                    pybullet.resetJointState(birobot,6,position[5]);
                    pybullet.resetJointState(birobot,7,position[6]);
                    pybullet.resetJointState(birobot,8,position[7]);
                    pybullet.resetJointState(birobot,9,position[8]);
                    pybullet.resetJointState(birobot,10,position[9]);
                    '''
                    if lefteef_is_attached == 0:
                        lefteef_is_attached,object_editor_left,joints_links_info,self.available_objects,attached_object_left = attachment.checkeefisAttached("left_eef",birobot,mapping,self.available_objects,threshold,physicsClient);
                        self.attached_objects.append(attached_object_left);

                    if righteef_is_attached == 0:
                        righteef_is_attached,object_editor_right,joints_links_info,self.available_objects,attached_object_right = attachment.checkeefisAttached("right_eef",birobot,mapping,self.available_objects,threshold,physicsClient);
                        self.attached_objects.append(attached_object_right);

                    if righteef_is_attached == 1 and righteef_is_done == 0:# Attach to the right end-effector
                        birobot,mapping = attachment.attachEEF(birobot,ed0,'joint_dummy1',attached_object_right,mapping,object_editor_right,objects_attachments,joints_links_info,"birobot_copy2.urdf",'right_eef');
                        righteef_is_done = 1;

                    if lefteef_is_attached == 1 and lefteef_is_done == 0:# Attach to the left end-effector
                        birobot,mapping = attachment.attachEEF(birobot,ed0,'joint_dummy2',attached_object_left,mapping,object_editor_left,objects_attachments,joints_links_info,"birobot_copy1.urdf",'left_eef');
                        lefteef_is_done = 1;

                    # Those lines are only for debugging to know if the LWA joint works properly
                    '''
                    if lefteef_is_attached == 1 and lefteef_is_done == 1:
                        print("counter_position = ",counter_position);
                        pybullet.resetJointState(birobot,mapping[b'Joint_LWA'],counter_position);
                        counter_position = counter_position + 1;
                    '''

                    counter = counter + 1;
                    pybullet.stepSimulation();

        pybullet.disconnect();

    ##	Run the pybullet simulator
    #	@input angles_left Numpy matrix of dimensions 1x5 containing the joint positions (in degrees) for the left arm's joints
    #	@input angles_right Numpy matrix of dimensions 1x5 containing the joint positions (in degrees) for the right arm's joints
    #   @input threshold The threshold down below the object is considered to be attached to the end-effector
    def runSimulator(self,angles_left,angles_right,object1,object2,threshold):
        angles_left = self.toRadians(angles_left);
        angles_right = self.toRadians(angles_right);
        angles_left_sim = np.zeros((np.size(angles_left,0),np.size(angles_left,1)));
        angles_right_sim = np.zeros((np.size(angles_right,0),np.size(angles_right,1)));

        # The angles passed to runSimulator are all positive. However, the simulator's directions are different. Therefore, a mapping must take place.
        angles_left = np.multiply(angles_left, np.matlib.repmat(np.array([-1,-1,-1,-1,-1]),np.size(angles_left,0),1));
        angles_right = np.multiply(angles_right, np.matlib.repmat(np.array([1,-1,1,-1,1]),np.size(angles_right,0),1));
        angles_left_sim[:,0] = angles_left[:,0];
        angles_left_sim[:,1] = angles_left[:,1];
        angles_left_sim[:,2] = angles_left[:,2];
        angles_left_sim[:,3] = angles_left[:,3];
        angles_left_sim[:,4] = angles_left[:,4];
        angles_right_sim[:,0] = angles_right[:,0];
        angles_right_sim[:,1] = angles_right[:,1];
        angles_right_sim[:,2] = angles_right[:,2];
        angles_right_sim[:,3] = angles_right[:,3];
        angles_right_sim[:,4] = angles_right[:,4];
        target_joint_positions = list(np.hstack((angles_right_sim,angles_left_sim)));
        self.loadSimulator(target_joint_positions,object1,object2,threshold);

if __name__ == "__main__":
    #angles_left = np.matlib.repmat(np.array([0,0,0,90,0]),100000,1);
    #angles_right = np.matlib.repmat(np.array([0,0,0,90,0]),100000,1);
    sim = Simulator();
    angles_left = np.matlib.repmat(np.array([0,0,0,0,0]),100000,1);
    angles_right = np.matlib.repmat(np.array([0,0,0,0,0]),100000,1);
    #object1 = [[-207.20477789,-312.875389,-270,0.03293905,-0.35213897,0.36608334,-0.86075325]];#Pst: 8
    #object2 = [[200,-312.875389,-316.98958698,0.06026358,0.12072236,-0.76464444,-0.63016926]];#Pst 1
    object1 = [[181.0709334,-542.0914752,-260,0.02527894,0.01340268,-0.64181416,-0.76632625]];#pitcher at Pst: 5. Be careful the translation vector must be in mm!
    object2 = [[50.4252055,-545.6029044,-340,0.00091483,-0.04028703,0.63988639,0.76741223]];# cup at Pst: 6. Be careful the translation vector must be in mm!
    object1 = np.transpose(np.array(object1,ndmin=2));
    object2 = np.transpose(np.array(object2,ndmin=2));
    sim.runSimulator(angles_left,angles_right,object1,object2,0.1);