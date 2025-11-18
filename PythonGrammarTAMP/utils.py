# Motion Planning code taken from https://github.com/caelan/motion-planners

import ikmodule
import motionplanmodule
import numpy as np

arm_left = 1;
arm_right = 0;

# Transform the joint angles from the simulated robot's coordinates to the real coordinates
def anglesTransform_to_real(q0,q1,q2,q3,q4,arm_option):
    result = [0,0,0,0,0]
    if arm_option == 1:
        result = to_real_left(q0,q1,q2,q3,q4);
    elif arm_option == 0:
        result = to_real_right(q0,q1,q2,q3,q4);
    else:
        print("Arm is wrong. 1 is for the left arm. 0 is for the right arm.");
    return result;

def anglesTransform_to_sim(q0,q1,q2,q3,q4,arm_option):
    result = [0,0,0,0,0];
    if arm_option == 1:
        result = to_sim_left(q0,q1,q2,q3,q4);
    elif arm_option == 0:
        result = to_sim_right(q0,q1,q2,q3,q4);
    else:
        print("Arm is wrong. 1 is for the left arm. 0 is for the right arm.");
    return result;

def to_real_left(q0,q1,q2,q3,q4):
    return [abs(q0),abs(q1),abs(q2),abs(q3),q4];

def to_real_right(q0,q1,q2,q3,q4):
    return [abs(q0),abs(q1),abs(q2),abs(q3),-q4];

def to_sim_left(q0,q1,q2,q3,q4):
    return [-q0,q1,-q2,-q3,q4];

def to_sim_right(q0,q1,q2,q3,q4):
    return [q0,-q1,-q2,-q3,-q4];

def runFK(q0,q1,q2,q3,q4,arm_option):
    agls = anglesTransform_to_sim(q0,q1,q2,q3,q4,arm_option)# Always before running the FK
    q0=agls[0];q1=agls[1];q2=agls[2];q3=agls[3];q4=agls[4];
    fk = ikmodule.runFK(q0,q1,q2,q3,q4,0,arm_option);# There is no wrist flexion/extension. Therefore, the sixth element is zero
    return fk;

def runIK(poseX,poseY,poseZ,mu,rotX,rotY,rotZ,max_iter,thresh,arm_option):
    ik = np.delete(ikmodule.runIK(poseX,poseY,poseZ,mu,rotX,rotY,rotZ,max_iter,thresh,arm_option),5);# There is no wrist flexion/extension. Therefore, the fifth element is eliminated
    ik = anglesTransform_to_real(ik[0],ik[1],ik[2],ik[3],ik[4],arm_option);
    return ik;

# Merge lists. Erase the duplicates. Return sorted list of values.
def merge_lists(list1,list2):
    merged_list = list1 + list(set(list2) - set(list1));
    merged_list.sort();
    return merged_list;

##	Swap two elements in a list
#	@param toSwap List with elements to swap
#	@param idx_origin Index of the column to swap
#	@param idx_final Targeted index
#	@return List with all elements swapped
def swap_elmts(toSwap,idx_origin,idx_final):
    if len(toSwap)<=0 or idx_origin<1 or idx_final<1 or len(toSwap[0])!=22:
        raise Exception("[utils.py] Wrong input.")
    else:
        # Convert the list into a numpy array since it is easier for manipulation.
        # Then, extract the LEFE values for both the left and right arms.
        toSwap = np.array(toSwap);
        toSwap_2 = np.copy(toSwap);

        # Insert the LSFE values into column where LEFE values are
        toSwap[:,idx_final] = toSwap[:,idx_origin];
        toSwap[:,idx_final+11] = toSwap[:,idx_origin+11];

        # Insert the LEFE values at the position where the LSFE were
        toSwap[:,idx_origin] = toSwap_2[:,idx_final];
        toSwap[:,idx_origin+11] = toSwap_2[:,idx_final+11];
        toSwap = toSwap.tolist();

    return toSwap;

##	Erase the scientific notation for all values in the list. Otherwise, the robot's controller will not understand
#	@param toErase List to erase the scientific notation from
#	@return List with all the scientific notations changed to decimal points
def erase_sciNotation(elmt):
    elmt = f"{elmt:.9f}";    
    return elmt;

if __name__ == "__main__":
    #Pose_left = runFK(0,0,0,0,0,arm_left)
    #print("Pose_left= ",Pose_left)
    #Pose_right = runFK(0,0,0,0,0,arm_right)
    #print("Pose_right= ",Pose_right)

    #X_left = runIK(Pose_left[0],Pose_left[1],Pose_left[2],Pose_left[3],Pose_left[4],Pose_left[5],arm_left)
    #print("X_left= ",X_left)
    #X_right = runIK(Pose_right[0],Pose_right[1],Pose_right[2],Pose_right[3],Pose_right[4],Pose_right[5],arm_right)
    #print("X_right= ",X_right)

    print(to_real_left(0,45,0,0,180))
    print(to_real_left(-90,0,-90,-90,-90))
    print(to_real_right(90,0,0,0,90))
    print(to_real_right(0,-45,-90,-90,-180))

    print(to_sim_left(0,45,0,0,180))
    print(to_sim_left(90,0,90,90,-90))
    print(to_sim_right(90,0,0,0,-90))
    print(to_sim_right(0,45,90,90,180))

    print("New line")

    print(anglesTransform_to_real(0,45,0,0,180,arm_left))
    print(anglesTransform_to_real(-90,0,-90,-90,-90,arm_left))
    print(anglesTransform_to_real(90,0,0,0,90,arm_right))
    print(anglesTransform_to_real(0,-45,-90,-90,-180,arm_right))

    print("New line")

    print(anglesTransform_to_sim(0,45,0,0,180,arm_left))
    print(anglesTransform_to_sim(90,0,90,90,-90,arm_left))
    print(anglesTransform_to_sim(90,0,0,0,-90,arm_right))
    print(anglesTransform_to_sim(0,45,90,90,180,arm_right))

    ## IK validation for the left arm ##
    pt1 = [100,-300,-200,0.4279,0.3251,-0.6029,0.5897]
    pt1_agls = runIK(pt1[0],pt1[1],pt1[2],pt1[3],pt1[4],pt1[5],pt1[6],100,0.01,arm_left)
    pt1_fk = runFK(pt1_agls[0],pt1_agls[1],pt1_agls[2],pt1_agls[3],pt1_agls[4],arm_left)
    
    pt2 = [0,-300,-200,0.5240,0.2169,-0.6533,0.5014]
    pt2_agls = runIK(pt2[0],pt2[1],pt2[2],pt2[3],pt2[4],pt2[5],pt2[6],100,0.01,arm_left)
    pt2_fk = runFK(pt2_agls[0],pt2_agls[1],pt2_agls[2],pt2_agls[3],pt2_agls[4],arm_left)
    pt3 = [-100,-300,-300,0.4141,0.2051,-0.8388,0.2879]
    pt3_agls = runIK(pt3[0],pt3[1],pt3[2],pt3[3],pt3[4],pt3[5],pt3[6],100,0.01,arm_left)
    pt3_fk = runFK(pt3_agls[0],pt3_agls[1],pt3_agls[2],pt3_agls[3],pt3_agls[4],arm_left)
    pt4 = [0,-300,-300,0.3747,0.2668,-0.7946,0.3963]
    pt4_agls = runIK(pt4[0],pt4[1],pt4[2],pt4[3],pt4[4],pt4[5],pt4[6],100,0.01,arm_left)
    pt4_fk = runFK(pt4_agls[0],pt4_agls[1],pt4_agls[2],pt4_agls[3],pt4_agls[4],arm_left)

    #print("pt1_agls = ",pt1_agls)
    #print("pt1_fk = ",pt1_fk)
    #print("pt2_agls = ",pt2_agls)
    #print("pt2_fk = ",pt2_fk)
    #print("pt3_agls = ",pt3_agls)
    #print("pt3_fk = ",pt3_fk)
    #print("pt4_agls = ",pt4_agls)
    #print("pt4_fk = ",pt4_fk)

    ## IK validation for the right arm ##
    pt1 = [100,-300,-200,0.2929,-0.1774,0.6746,-0.6539]
    pt1_agls = runIK(pt1[0],pt1[1],pt1[2],pt1[3],pt1[4],pt1[5],pt1[6],100,0.01,arm_right)
    pt1_fk = runFK(pt1_agls[0],pt1_agls[1],pt1_agls[2],pt1_agls[3],pt1_agls[4],arm_right)
    
    pt2 = [0,-300,-200,0.2563,-0.1022,0.6838,-0.6755]
    pt2_agls = runIK(pt2[0],pt2[1],pt2[2],pt2[3],pt2[4],pt2[5],pt2[6],100,0.01,arm_right)
    pt2_fk = runFK(pt2_agls[0],pt2_agls[1],pt2_agls[2],pt2_agls[3],pt2_agls[4],arm_right)
    pt3 = [-100,-300,-300,0.1415,0.3917,-0.7524,0.5104]
    pt3_agls = runIK(pt3[0],pt3[1],pt3[2],pt3[3],pt3[4],pt3[5],pt3[6],100,0.01,arm_right)
    pt3_fk = runFK(pt3_agls[0],pt3_agls[1],pt3_agls[2],pt3_agls[3],pt3_agls[4],arm_right)
    pt4 = [0,-300,-300,0.1817,0.10112,0.8348,-0.5098]
    pt4_agls = runIK(pt4[0],pt4[1],pt4[2],pt4[3],pt4[4],pt4[5],pt4[6],100,0.01,arm_right)
    pt4_fk = runFK(pt4_agls[0],pt4_agls[1],pt4_agls[2],pt4_agls[3],pt4_agls[4],arm_right)

    print("pt1_agls = ",pt1_agls)
    print("pt1_fk = ",pt1_fk)
    print("pt2_agls = ",pt2_agls)
    print("pt2_fk = ",pt2_fk)
    print("pt3_agls = ",pt3_agls)
    print("pt3_fk = ",pt3_fk)
    print("pt4_agls = ",pt4_agls)
    print("pt4_fk = ",pt4_fk)

    ## Test the RRT-Connect Motion Planner
    initialState = [100,-300,-200,0.4279,0.3251,-0.6029,0.5897];
    finalState = [0,-300,-200,0.5240,0.2169,-0.6533,0.5014];
    arm_left = 1;
    arm_right = 0;
    pt1_agls = runIK(0,-300,-200,0.5240,0.2169,-0.6533,0.5014,100,0.01,arm_left)
    print(pt1_agls)

    ## Test the swapping operation
    toSwap = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],[111,211,311,411,511,611,711,811,911,1011,1111,1211,1311,1411,1511,1611,1711,1811,1911,2011,2111,2211]];
    toSwap = swap_elmts(toSwap,2,3);
    print(toSwap);
    #print(motionplanmodule.runRRT(initialState,finalState,arm_left));