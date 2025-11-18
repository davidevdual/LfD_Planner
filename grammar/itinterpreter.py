import AST
from AST import addToClass
from motionparser import MotionParser
from motionthreader import MotionThreader
from functools import reduce
import sys
import motioninterface as mp

# Iterative Interpreter Class
class ItInterpreter(object):

    def __init__(self,_is_simulation,_is_vision,_threshold,_objects):
        if type(_objects)!=tuple or len(_objects)<1 or len(_objects)>2 or type(_threshold)!=float:
            raise Exception("[execution.py] Wrong input");
        if _threshold<=0:
            raise Exception("[execution.py] The threshold cannot be nagative or equal to zero.");

        self.threshold = _threshold;
        self.objects = _objects;
        
        self.motnInter = mp.MotionInterface(_is_simulation,_is_vision,self.threshold,self.objects);
        self.operations = {
            'approach' : lambda hand,object: self.motnInter.approach(hand,object),
            'pour'     : lambda hand: self.motnInter.pour(hand),
            'check'    : lambda object: self.motnInter.check(object),
            'enclose'  : lambda hand,object: self.motnInter.enclose(hand,object),
            'raise'    : lambda hand,object: self.motnInter.raise_hand(hand,object),
            'pass'     : lambda hand,object: self.motnInter.passing(hand,object),
            'open'     : lambda hand: self.motnInter.open(hand),
        };
        self.stack = [];
        self.vars = {};
        self.manipulating_hand = "";

    def valueOfToken(self, t):
        return t;

    def execute(self, node, manipulated_object):
        while node:
            print("node = ",node, " type = ",type(node));
            if node.__class__ in [AST.EntryNode]:
                pass;
            elif node.__class__ in [AST.PlanNode]:
                print("plan node = ",node);
                arg2 = self.valueOfToken(self.stack.pop());
                if node.nbargs == 2:
                    arg1 = self.valueOfToken(self.stack.pop());
                else:
                    arg1 = 0;
                if self.manipulating_hand == "":
                    raise Exception("[itinterpreter.py] The manipulating hand variable is empty. This is an error.");
                if node.last_action == "pass":
                    self.stack.append(self.operations[node.last_action](self.manipulating_hand,"experimenter_hand"));
                elif node.last_action == "open" or node.last_action == "pour":
                    self.stack.append(self.operations[node.last_action](self.manipulating_hand));
                else: 
                    raise Exception("[itinterpreter.py] The final action is not pass, pour, or open. This is an error.")
            elif node.__class__ == AST.TokenNode:
                self.stack.append(node.tok);
            elif node.__class__ == AST.ConjNode:
                pass;
            elif node.__class__ == AST.ActionNode:
                print("In ActionNode");
                arg2 = self.valueOfToken(self.stack.pop());
                print("node: ",node," argument: ",arg2);
                if node.nbargs == 2:
                    arg1 = self.valueOfToken(self.stack.pop());
                    print("arg2: ",arg2," arg1: ",arg1);
                    # Save the hand that is currently manipulating the target object.
                    # This information will be used later to know which hand executes the final action
                    if arg2 == manipulated_object:
                        self.manipulating_hand = arg1;
                        print("manipulating_hand = ",self.manipulating_hand);
                        exit(0);
                    self.stack.append(self.operations[node.action](arg1,arg2));
                elif node.nbargs == 1:
                    self.stack.append(self.operations[node.action](arg2));
                else:
                    arg1 = 0;
            if node.next:
                node = node.next[0];
            else:
                node = None;
            print("self.stack: ",self.stack);
        #print("self.vars: ",self.vars);

    ## Execute the low-level commands for the robot to execute
    def execute_commands(self):
        self.motnInter.execute_commands();

    ## Compute the motion plan 
    #  @param task_plan the task plan to execute.
    def compute_motion_plan(self,task_plan):
        if type(task_plan)!=str or task_plan=="":
            raise Exception("[itinterpreter.py] Wrong input.");
        motionparser = MotionParser(); 
        motionthreader = MotionThreader();
        ast,manipulated_object = motionparser.parse(task_plan);
        entry = motionthreader.thread(ast);
        self.execute(entry,manipulated_object);

def main():
    is_simulation = 1;#1 if simulation desired. Otherwise, 0 if simulation not desired.
    is_vision = 0;#1 if the computer is connected to the RGB-D camera. 0 if the camera is not connected to the RGB-D camera.
    #object1 = [[-207.20477789,-312.875389,-270,0.03293905,-0.35213897,0.36608334,-0.86075325]];
    #object2 = [[200,-312.875389,-316.98958698,0.06026358,0.12072236,-0.76464444,-0.63016926]];
    object1 = [[200,-450,-270,0.03293905,-0.35213897,0.36608334,-0.86075325]];#Pst: 31
    object2 = [[-200,-312.875389,-320,0.06026358,0.12072236,-0.76464444,-0.63016926]];#Pst 11
    threshold = 0.1;
    #motion_plan = ItInterpreter(is_simulation,is_vision,0.1,tuple((object1,object2)));
    motion_plan = ItInterpreter(is_simulation,is_vision,0.1,tuple((object1)));
    #motion_plan.compute_motion_plan("approach handright cup and approach handleft pitcher and enclose handright cup and enclose handleft pitcher and approach handleft cup to pour pitcher 31 into cup 11");
    motion_plan.compute_motion_plan("approach handleft bleach_cleanser and enclose handleft bleach_cleanser and approach handright bleach_cleanser to open bleach_cleanser 5");
    #motion_plan.execute_commands();

if __name__=="__main__":
    main();