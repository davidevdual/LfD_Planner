#################################################################
## This code executes the low-level instructions by parsing the
## sentence derived from the user's high-level instruction.
#################################################################
#################################################################
## Author: David Carmona-Moreno
## Copyright: Copyright 2020, Dual-Arm project
## Version: v1.1
## Maintainer: David Carmona-Moreno
## Email: e0348847@u.nus.edu
## Status: First stable release
#################################################################

from abc import ABC, ABCMeta, abstractmethod
import vision as vs
import motionplanner as mp
import grasping as gp
import utils as utls
import ply.yacc as yacc
import numpy as np
import dispatching as dp
import execution as ex
import copy

## Class maintaining the end-effector's state (i.e., pose)
#
class State:

    ## Class constructor
    #  @param _state Object's state. Can be either a 6D pose or an angle.
    def __init__(self, _state):
        if type(_state) == list:
            self.state = _state;
        else:
            raise Exception("[dispatching.py] The state must be a list. It is not.");
    
    ## Get the end-effector's pose
    #  @return the end-effector's state. The state can be either a pose or an numpy array of angles.
    def get_state(self):
        return self.state;

    ## Set the end-effector's pose
    #  @param _pose the new state.
    def set_state(self,_state):
        if type(_state) == list:
            self.state = copy.deepcopy(_state);# Must do a deepcopy. Otherwise, the States share the same memory address
        else:
            raise Exception("[dispatching.py] The state must be a list. It is not.");

## @class Command
#  @brief Send the low-level instructions to the robot according to the output of the parser. The commands can be vision, grasping, or joint movement
class Command:   

    def __init__(self,_is_simulation,_is_vision,_objects):
        if type(_objects)!=tuple or len(_objects)<1 or len(_objects)>2:
            raise Exception("[command.py] Wrong input.");
        self.is_vision = _is_vision;
        self.is_simulation = _is_simulation;
        self.objects_tuple = _objects;
        
        # Initialize the aruco marker id for the objects and the ids for both hands
        self.cup_id = 8;
        self.bottle_id = 9;
        self.left_hand_id = 1;
        self.right_hand_id = 0;
        self.approach_handleft_done = 0;
        self.approach_handright_done = 0;

        # Initial pose for the left arm. The order is: LSR,LSA,LSFE,LEFE,LWR
        pose_left = utls.runFK(0,0,0,90,0,self.left_hand_id);
        agls = utls.runIK(pose_left[0],pose_left[1],pose_left[2],pose_left[3],pose_left[4],pose_left[5],pose_left[6],100,0.01,self.left_hand_id);
        pose_left  = utls.runFK(agls[0],agls[1],agls[2],agls[3],agls[4],self.left_hand_id);

        # Initial pose for the right arm
        pose_right = utls.runFK(0,0,0,90,0,self.right_hand_id);
        agls = utls.runIK(pose_right[0],pose_right[1],pose_right[2],pose_right[3],pose_right[4],pose_right[5],pose_right[6],100,0.01,self.right_hand_id);
        pose_right = utls.runFK(agls[0],agls[1],agls[2],agls[3],agls[4],self.right_hand_id);

        # Initialise all states
        self.lefthandState = State([[pose_left[0],pose_left[1],pose_left[2],pose_left[3],pose_left[4],pose_left[5],pose_left[6]]]);
        self.righthandState = State([[pose_right[0],pose_right[1],pose_right[2],pose_right[3],pose_right[4],pose_right[5],pose_right[6]]]);
        self.leftfingersState = State([[0,0,0,0,0,0]]);
        self.rightfingersState = State([[0,0,0,0,0,0]]);
        self.visionModule = vs.VisionGrammar();
        self.motionPlanner = mp.MotionPlannerGrammar();
        self.grasping = gp.GraspingGrammar();

        # Initialise the member variables storing the right and left arm joints angles as well as the fingers
        self.angles_righthand = [[]];
        self.angles_lefthand = [[]];
        self.angles_rightfingers = [[]];
        self.angles_leftfingers = [[]];

        # Hold a dictionary of objects with their identifier and state (i.e., 6D pose)
        #self.objects = {"cup" : (1,State([[-207.20477789,-312.875389,-270,0.03293905,-0.35213897,0.36608334,-0.86075325]])), 
        #                "bottle" : (2,State([[pose_right[0],pose_right[1],pose_right[2],pose_right[3],pose_right[4],pose_right[5],pose_right[6]]])), 
        #               };
        #print("length of tuple of objects = ",len(self.objects_tuple));
        if len(self.objects_tuple)==1:
            self.object1 = self.objects_tuple[0];
            object1_state = State(self.object1);
            self.objects = {"bleach_cleanser" : (1,object1_state),
            };
        elif len(self.objects_tuple)==2:
            self.object1 = self.objects_tuple[0];
            self.object2 = self.objects_tuple[1];
            object1_state = State(self.object1);
            object2_state = State(self.object2);
            self.objects = {"mustard_bottle" : (1,object1_state),
                "experimenter_hand" : (2,object2_state),
            };
        else:
            raise Exception("[command.py] Wrong input");
        #self.objects = {"cracker_box" : (1,object1_state),
        #                "bowl" : (2,object2_state),
        #               };
        
    def approach(self,hand,object):
        if type(hand)!=str or type(object)!=str or hand=="" or object=="":
            raise Exception("[command.py] Wrong input");
        print("approach ",hand," ",object);
        objectState = self.check(object);

        if hand=='handleft' and self.approach_handleft_done==0:
            final = copy.deepcopy(self.lefthandState.get_state());
            final_list = final[0];
            final_list[0] = final_list[0] - 60;
            final_list[1] = final_list[1] - 150;
            final_list[2] = final_list[2];
            final = [final_list];
            self.angles_lefthand = self.motionPlanner.approach(self.lefthandState.get_state(),final,self.left_hand_id);
            self.angles_righthand = self.motionPlanner.no_action(self.righthandState.get_state(),self.right_hand_id);
            self.angles_lefthand = self.angles_lefthand[0];
            self.angles_lefthand[3] = 80;
            self.angles_lefthand[4] = 0;
            updated_state = utls.runFK(self.angles_lefthand[0],self.angles_lefthand[1],self.angles_lefthand[2],self.angles_lefthand[3],self.angles_lefthand[4],self.left_hand_id);
            self.angles_lefthand = [self.angles_lefthand];
            self.lefthandState.set_state(objectState.get_state());# The target pose of the object is the final end-effector's pose
            self.angles_rightfingers = self.rightfingersState.get_state();
            self.angles_leftfingers = self.leftfingersState.get_state();
            handedness = self.left_hand_id;
            self.approach_handleft_done = 1;

        elif hand=='handright' and self.approach_handright_done==0:
            final = copy.deepcopy(self.righthandState.get_state());
            final_list = final[0];
            final_list[0] = final_list[0] + 180;
            final_list[1] = final_list[1] - 150;
            final_list[2] = final_list[2];
            final = [final_list];
            self.angles_righthand = self.motionPlanner.approach(self.righthandState.get_state(),final,self.right_hand_id);
            self.angles_lefthand = self.motionPlanner.no_action(self.lefthandState.get_state(),self.left_hand_id);
            self.righthandState.set_state(objectState.get_state());# The target pose of the object is the final end-effector's pose
            self.angles_rightfingers = self.rightfingersState.get_state();
            self.angles_leftfingers = self.leftfingersState.get_state();
            self.angles_righthand = self.angles_righthand[0];
            self.angles_righthand[3] = 130;
            self.angles_righthand[4] = 90;
            updated_state = utls.runFK(self.angles_righthand[0],self.angles_righthand[1],self.angles_righthand[2],self.angles_righthand[3],self.angles_righthand[4],self.right_hand_id);
            self.angles_righthand = [self.angles_righthand];
            self.righthandState.set_state([updated_state]);
            handedness = self.right_hand_id;
            self.approach_handright_done = 1;
        else:
            raise Exception("[command.py] Wrong input.");
        return self.angles_lefthand,self.angles_righthand,self.angles_leftfingers,self.angles_rightfingers;

    def pour(self,hand):
        if hand=='handleft':
            self.angles_rightfingers = self.rightfingersState.get_state();
            self.angles_leftfingers = self.leftfingersState.get_state();
            self.angles_lefthand = self.angles_lefthand[0];
            #self.angles_lefthand[4] = 45; # For simulation pouring pitcher 31 into mug 11
            self.angles_lefthand = [self.angles_lefthand];
            handedness = self.left_hand_id;
        elif hand=='handright':
            self.angles_rightfingers = self.rightfingersState.get_state();
            self.angles_leftfingers = self.leftfingersState.get_state();
            self.angles_righthand = self.angles_righthand[0];
            self.angles_righthand[4] = 100; 
            self.angles_righthand = [self.angles_righthand];
            handedness = self.right_hand_id;
        else:
            raise Exception("[command.py] Wrong input.");
        return self.angles_lefthand,self.angles_righthand,self.angles_leftfingers,self.angles_rightfingers;

    def passing(self,manipulating_hand,receiving_hand):
        if type(manipulating_hand)!=str or type(receiving_hand)!=str or manipulating_hand=="" or (manipulating_hand!="handleft" and manipulating_hand!="handright") or receiving_hand=="" or receiving_hand!="experimenter_hand":
            raise Exception("[command.py] Wrong input");

        final_left = copy.deepcopy(self.lefthandState.get_state());
        final_right = copy.deepcopy(self.righthandState.get_state());
        final_list_left = final_left[0];
        final_list_left[0] = final_list_left[0];
        final_list_left[1] = final_list_left[1] - 130;
        final_list_left[2] = final_list_left[2] + 63;
        final_left = [final_list_left];
        final_list_right = final_right[0];
        final_list_right[0] = final_list_right[0];
        final_list_right[1] = final_list_right[1] - 130;
        final_list_right[2] = final_list_right[2] + 63;
        final_right = [final_list_right];

        self.angles_righthand = self.motionPlanner.approach(self.righthandState.get_state(),final_right,self.right_hand_id);
        self.angles_lefthand = self.motionPlanner.approach(self.lefthandState.get_state(),final_left,self.left_hand_id);
        self.angles_righthand = self.angles_righthand[0];
        self.angles_lefthand = self.angles_lefthand[0];

        self.angles_righthand[4] = 0;
        self.angles_lefthand[4] = 0;
        self.angles_righthand[3] = 90;
        self.angles_lefthand[3] = 90;
        self.angles_righthand = [self.angles_righthand];
        self.angles_lefthand = [self.angles_lefthand];

        return self.angles_lefthand,self.angles_righthand,self.angles_leftfingers,self.angles_rightfingers;

    # Get the 6D poses of the objects using the vision system's object detection algorithm
    # @input object String. The object to detect in the scene
    # @return State of the object that is in the scene
    def check(self,object):
        print("check ",object);
        objectState = ();
        if self.is_vision!=0 and self.is_vision!=1:
            raise Exception("[command.py] is_vision must be either 1 or 0. Here it is not.");
        elif self.is_vision==0:# Computer not connected to RGBD camera. Add a random object to test the entire approach
            tuple_state = self.objects[object];
            objectState = tuple_state[1];
        elif self.is_vision==1:
            pass;
        elif len(objectState)==0:
            raise Exception("[command.py] The tuple is empty.");
        else:
            raise Exception("[command.py] Wrong input.");
        return objectState;

    def enclose(self,hand,object):
        print("enclose ",hand," ",object);
        objectState = self.check(object);
        if hand=='handleft':
            self.angles_lefthand = self.angles_lefthand[0];
            self.angles_lefthand[0] = 18;
            self.angles_lefthand[4] = -15;
            updated_state = utls.runFK(self.angles_lefthand[0],self.angles_lefthand[1],self.angles_lefthand[2],self.angles_lefthand[3],self.angles_lefthand[4],self.left_hand_id);
            
            self.angles_rightfingers = self.grasping.no_action(self.rightfingersState.get_state(),self.right_hand_id);
            self.angles_leftfingers = [[270,80,80,80,80,80]];
            self.leftfingersState.set_state(self.angles_leftfingers);
            self.lefthandState.set_state([updated_state]);
            self.angles_lefthand = [self.angles_lefthand];
            handedness = self.left_hand_id;

        elif hand=='handright':
            final = self.righthandState.get_state();
            final_list = final[0];# Uncomment this line only for pouring pitcher into mug in simulation
            final = [final_list];# Uncomment this line only for pouring pitcher 31 into mug 11 in simulation
            self.angles_righthand = self.motionPlanner.approach(self.righthandState.get_state(),final,self.right_hand_id);
            self.angles_righthand = self.angles_righthand[0];
            self.angles_righthand[4] = -15;
            updated_state = utls.runFK(self.angles_righthand[0],self.angles_righthand[1],self.angles_righthand[2],self.angles_righthand[3],self.angles_righthand[4],self.right_hand_id);
            self.angles_righthand = [self.angles_righthand];

            self.angles_lefthand = self.motionPlanner.no_action(self.lefthandState.get_state(),self.left_hand_id);
            self.angles_rightfingers = [[0,270,270,270,270,270]];
            self.angles_leftfingers = self.grasping.no_action(self.leftfingersState.get_state(),self.left_hand_id);
            self.rightfingersState.set_state(self.angles_rightfingers);
            self.righthandState.set_state([updated_state]);
            handedness = self.right_hand_id;
        else:
            raise Exception("[command.py] Wrong input.");
        return self.angles_lefthand,self.angles_righthand,self.angles_leftfingers,self.angles_rightfingers;

    def raise_hand(self,hand,object):
        print("raise ",hand," ",object);
        if hand=='handleft':
            pass;
        elif hand=='handright':
            pass;
        else:
            raise Exception("[command.py] Wrong input.");
        return 0;

    def open(self,hand):
        if hand=='handleft':
            print("We are in handleft");
        elif hand=='handright':
            self.angles_righthand = self.angles_righthand[0];
            self.angles_righthand[3] = 110;
            updated_state = utls.runFK(self.angles_righthand[0],self.angles_righthand[1],self.angles_righthand[2],self.angles_righthand[3],self.angles_righthand[4],self.right_hand_id);
            self.lefthandState.set_state([updated_state]);
            self.angles_righthand = [self.angles_righthand];
            handedness = self.right_hand_id;
        else:
            raise Exception("[command.py] Wrong input.");
        return self.angles_lefthand,self.angles_righthand,self.angles_leftfingers,self.angles_rightfingers;

def main():
	print("Hello");

if __name__=="__main__":
    main();