from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed
import numpy as np
import pybullet
import pybullet_data
import time
import math

def main():
    urdf_path = 'C:\\Users\\David\\OneDrive - National University of Singapore\\PhD_work\\Code\\TAMP_HighLevel\\Visual_Studio\\source';
    urdf_path2 = 'C:\\Users\\David\\OneDrive - National University of Singapore\\PhD_work\\Code\\TAMP_HighLevel\\Visual_Studio\\PythonGrammarTAMP';
    physicsClient = pybullet.connect(pybullet.GUI);#or p.DIRECT for non-graphical version
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath());
    pybullet.setGravity(0,0,-10);

    birobot = pybullet.loadURDF(urdf_path2+"\\birobot_copy1.urdf",[0,0,1.12],pybullet.getQuaternionFromEuler([0,0,0]));
    #birobot = pybullet.loadURDF(urdf_path2+"\\birobot.urdf",[0,0,1.12],pybullet.getQuaternionFromEuler([0,0,0]));
    #pybullet.changeVisualShape(birobot,1,rgbaColor=[0.79216,0.81961,0.93333,1.00000]);# The RTA link is loaded in a black colour for an unknown reason. Therefore, make it the same colour as all the other links.

    pybullet.setRealTimeSimulation(1);
    pybullet.setTimeStep(0.000001);#In seconds

    while (pybullet.isConnected()):
        pybullet.stepSimulation();
    pybullet.disconnect();

if __name__ == "__main__":
    main();
