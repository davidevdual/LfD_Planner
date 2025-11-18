from abc import ABC, ABCMeta, abstractmethod
import numpy as np
import vision as vs
import motionplanner as mp
import utils as utls

'''
## Class maintaining the end-effector's state (i.e., pose)
#
class State(object):

    ## Class constructor
    #  @param _state Object's state. Can be either a 6D pose or an angle.
    def __init__(self, _state):
        if type(_state) is np.ndarray:
            self.state = _state;
        else:
            raise Exception("[dispatching.py] The state must be an numpy array. It is not.");
    
    ## Get the end-effector's pose
    #  @return the end-effector's state. The state can be either a pose or an numpy array of angles.
    def get_state(self):
        return self.state;

    ## Set the end-effector's pose
    #  @param _pose the new state.
    def set_state(self,_state):
        if type(_state) is np.ndarray:
            self.state = _state;
        else:
            raise Exception("[dispatching.py] The state must be an numpy array. It is not.");

## Class interface for the dispatching the high-level commands to the appropriate commands.
#
class Dispatcher(metaclass=ABCMeta):

    ## Class constructor
    def __init__(self,_is_vision):

        self.is_vision = _is_vision;
        
        # Initialize the aruco marker id for the objects and the ids for both hands
        self.cup_id = 8;
        self.left_hand_id = 1;
        self.right_hand_id = 0;

        # Initial pose for the left arm. The order is: LSR,LSA,LSFE,LEFE,LWR
        pose_left = utls.runFK(0,0,0,90,0,self.left_hand_id);
        #print("The pose_left is: ",pose_left);
        agls = utls.runIK(pose_left[0],pose_left[1],pose_left[2],pose_left[3],pose_left[4],pose_left[5],pose_left[6],100,0.01,self.left_hand_id);
        pose_left  = utls.runFK(agls[0],agls[1],agls[2],agls[3],agls[4],self.left_hand_id);

        # Initial pose for the right arm
        pose_right = utls.runFK(0,0,0,90,0,self.right_hand_id);
        agls = utls.runIK(pose_right[0],pose_right[1],pose_right[2],pose_right[3],pose_right[4],pose_right[5],pose_right[6],100,0.01,self.right_hand_id);
        pose_right = utls.runFK(agls[0],agls[1],agls[2],agls[3],agls[4],self.right_hand_id);
        
        # Initialize all states
        self.cupState = State(np.array([[0,0,0,0,0,0,0]]));
        self.lefthandState = State(np.array([[pose_left[0],pose_left[1],pose_left[2],pose_left[3],pose_left[4],pose_left[5],pose_left[6]]]));
        self.righthandState = State(np.array([[pose_right[0],pose_right[1],pose_right[2],pose_right[3],pose_right[4],pose_right[5],pose_right[6]]]));
        self.leftfingersState = State(np.array([[0,0,0,0,0,0]]));
        self.rightfingersState = State(np.array([[0,0,0,0,0,0]]));
        self.visionModule = vs.VisionGrammar();
        self.motionModule = mp.MotionPlannerGrammar();

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'dispatch') and
            callable(subclass.dispatch) or
            NotImplemented)

    ##	Dispatch the symbolic action to the vision or motion planning module accordingly.
    ##  It is a polymorphic function
    @abstractmethod
    def dispatch(self,request,object):
        raise NotImplementedError;

class DispatcherVision(Dispatcher):

    ## Class constructor
    def __init__(self,_is_vision):
        self.is_vision = _is_vision;
        super().__init__(self.is_vision);
    
    ##	Dispatch a vision-related action
    #	@param request A String. Type of vision request.
    #	@param object Object that the vision system must retrieve information from.
    #	@return Numpy array containing the information needed.
    def dispatch(self,request,object):
        output = np.array([0]);
        if self.is_vision!=0 and self.is_vision!=1:
            raise Exception("[dispatcher.py] is_vision must be either 1 or 0. Here it is not.");
        if request=='getPose' and object=='cup' and self.is_vision==1:
            tvec,quat = self.visionModule.get_object(self.cup_id);
            output = np.hstack((tvec,quat));
            self.cupState.set_state(output);
        elif request=='getPose' and object=='cup' and self.is_vision==0:
           output = np.array([[-207.20477789,-312.875389,-270,0.03293905,-0.35213897,0.36608334,-0.86075325]]);
           self.cupState.set_state(output);
        else:
            raise Exception("[dispatcher.py] Input is wrong.");
        if np.array_equal(output,np.array([0])) == True:
            raise Exception("[dispatcher.py] output is equal to zero. No value has been assigned.");
        return output;

class DispatcherAction(Dispatcher):

    ## Class constructor
    def __init__(self,_is_vision):
        self.is_vision = _is_vision;

        # Actions include vision requests. Therefore, a DispatchVision object must be instantiated
        self.dispatch_vision = DispatcherVision(self.is_vision);
        super().__init__(self.is_vision);

    ##	Dispatch a motion-related action
    #	@param request A String. Type of motion planning request.
    #	@param *arg Variable array of objects that are affected by the actions.
    #	@return Either numpy array or Integer. 0 means that there is nothing to do. If it is a numpy array, it is an action that the robot must achieve.
    def dispatch(self,request,*arg):
        angles_lefthand = 0;
        angles_righthand = 0;
        angles_leftfingers = 0;
        angles_rightfingers = 0;
        handedness = -1;# Left hand is 1 and right hand is 0. -1 is when there is no action to do
        if len(arg)==1:
            if arg[0]=='lefthand' or arg[0]=='righthand':
                print("Nothing to do");# In the future: get the left hand's 6D pose doing an FK on the joint values
                angles = 0;
                handedness = -1;
        elif len(arg)==2:
            if request=='approach' and arg[0]=='lefthand' and arg[1]=='cup':
                pose = self.dispatch_vision.dispatch('getPose','cup');
                self.cupState.set_state(pose);
                angles_lefthand = self.motionModule.approach(self.lefthandState.get_state(),self.cupState.get_state(),self.left_hand_id);
                angles_righthand = self.motionModule.no_action(self.righthandState.get_state(),self.right_hand_id);
                print("angles_righthand: ",angles_righthand);
                print("angles_lefthand: ",angles_lefthand);
                angles_rightfingers = self.rightfingersState.get_state();
                angles_leftfingers = self.leftfingersState.get_state();
                handedness = self.left_hand_id;
        else:
            raise Exception("[dispatcher.py] Input is wrong.");
        return angles_lefthand,angles_righthand,angles_leftfingers,angles_rightfingers,handedness;

def main():
    
    # Test the vision 
    dispatch_vision = DispatcherVision();
    dispatch_action = DispatcherAction();
    #output = dispatch_vision.dispatch('getPose','cup');
    #print(output);

    # Test the motion planning dispatcher
    lefthand,righthand,leftfingers,rightfingers,handedness = dispatch_action.dispatch('approach','lefthand','cup');
    print("lefthand is: ",lefthand);
    print("righthand is: ",righthand);
    print("leftfingers is: ",leftfingers);
    print("rightfingers is: ",rightfingers);
    print("handedness is: ",handedness);

if __name__=="__main__":
    main();

'''

