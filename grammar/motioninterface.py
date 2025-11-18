import numpy as np
import dispatching as dp
import execution as ex
import command as cmd

class MotionInterface(object):

    def __init__(self,_is_simulation,_is_vision,_threshold,_objects):
        if type(_objects)!=tuple or len(_objects)<1 or len(_objects)>2 or type(_threshold)!=float:
            raise Exception("[execution.py] Wrong input");
        if _threshold<=0:
            raise Exception("[execution.py] The threshold cannot be nagative or equal to zero.");
        self.is_simulation = _is_simulation;
        self.is_vision = _is_vision;
        self.threshold = _threshold
        self.objects = _objects;

        self.command = cmd.Command(self.is_simulation,self.is_vision,self.objects);
        self.executionLayer = ex.TAMPExecInterface(self.is_simulation,self.is_vision,self.threshold,self.objects);

    ## Add the robot's initial angles to the commands to be sent. Otherwise, it will create unstability
    def init_command(self):
        lefthand = np.zeros((1,5));
        righthand = np.zeros((1,5));
        leftfingers = np.zeros((1,6));
        rightfingers = np.zeros((1,6));
        self.executionLayer.add_command(lefthand,righthand,leftfingers,rightfingers);

    def approach(self,hand,object):
        lefthand,righthand,leftfingers,rightfingers = self.command.approach(hand,object);
        self.executionLayer.add_command(lefthand,righthand,leftfingers,rightfingers);
        return 0;

    def pour(self,hand):
        lefthand,righthand,leftfingers,rightfingers = self.command.pour(hand);
        self.executionLayer.add_command(lefthand,righthand,leftfingers,rightfingers);
        return 0;

    def check(self,object):
        self.command.check(object);
        return 0;

    def enclose(self,hand,object):
        lefthand,righthand,leftfingers,rightfingers = self.command.enclose(hand,object);
        self.executionLayer.add_command(lefthand,righthand,leftfingers,rightfingers);
        return 0;

    def raise_hand(self,hand,object):
        self.command.raise_hand(hand,object);
        return 0;

    def passing(self,hand,object):
        lefthand,righthand,leftfingers,rightfingers = self.command.passing(hand,object);
        self.executionLayer.add_command(lefthand,righthand,leftfingers,rightfingers);
        return 0;

    def open(self,hand):
        lefthand,righthand,leftfingers,rightfingers = self.command.open(hand);
        self.executionLayer.add_command(lefthand,righthand,leftfingers,rightfingers);
        return 0;

    ## Function that sends all the commands to the robot (simulated or real) for execution
    def execute_commands(self):
        self.executionLayer.send_commands();

def main():
    object1 = [[-207.20477789,-312.875389,-270,0.03293905,-0.35213897,0.36608334,-0.86075325]];
    object2 = [[200,-312.875389,-316.98958698,0.06026358,0.12072236,-0.76464444,-0.63016926]];
    motion_interface = MotionInterface(1,0,0.1,tuple((object1,object2)));
    angles = motion_interface.approach('handleft','mustard_bottle');
    #p = pour('handright');

if __name__=="__main__":
    main();
    