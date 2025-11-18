import socket
import time
import motionplanner as mp
import matplotlib.pyplot as plt
import trajectorygeneration as tg
import numpy as np
import utils as utls

# load array
data = np.loadtxt('data.csv', delimiter=',');

# Get the matrix's number of rows
rows, columns = data.shape;
num = np.linspace(0,rows,num=rows,dtype=int,axis=0)

# Get the data for the left arm
lsr_left = data[:,0];
lsa_left = data[:,1];
lefe_left = data[:,2];
lsfe_left = data[:,3];
lwr_left = data[:,4];

# Get the data for the right arm
lsr_right = data[:,11];
lsa_right = data[:,12];
lefe_right = data[:,13];
lsfe_right = data[:,14];
lwr_right = data[:,15];

# Plot the data for the left arm
fig = plt.figure(figsize=(12,8))

ax1 = plt.subplot(1,5,1)
ax1.plot(num,lsr_left,color='blue',label='lsr',marker='x')
plt.xlabel('time (/)')
plt.ylabel('LSR - abad angle (degrees)')

ax2 = plt.subplot(1,5,2)
ax2.plot(num,lsa_left,color='blue',label='lsa',marker='x')
plt.xlabel('time (/)')
plt.ylabel('LSA - abd angle (degrees)')

ax3 = plt.subplot(1,5,3)
ax3.plot(num,lefe_left,color='blue',label='lefe',marker='x')
plt.xlabel('time (/)')
plt.ylabel('LEFE - hip angle (degrees)')

ax4 = plt.subplot(1,5,4)
ax4.plot(num,lsfe_left,color='blue',label='lsfe',marker='x')
plt.xlabel('time (/)')
plt.ylabel('LSFE - knee angle (degrees)')

ax5 = plt.subplot(1,5,5)
ax5.plot(num,lwr_left,color='blue',label='lwr',marker='x')
plt.ylabel('LWR - rot angle (degrees)')
plt.xlabel('time (/)')
plt.suptitle('Bimanipulator Left Arm',fontsize=20)
plt.show()

# Plot the data for the right arm
fig = plt.figure(figsize=(12,8))

ax1 = plt.subplot(1,5,1)
ax1.plot(num,lsr_right,color='blue',label='lsr',marker='x')
plt.xlabel('time (/)')
plt.ylabel('LSR - abad angle (degrees)')

ax2 = plt.subplot(1,5,2)
ax2.plot(num,lsa_right,color='blue',label='lsa',marker='x')
plt.xlabel('time (/)')
plt.ylabel('LSA - abd angle (degrees)')

ax3 = plt.subplot(1,5,3)
ax3.plot(num,lefe_right,color='blue',label='lefe',marker='x')
plt.xlabel('time (/)')
plt.ylabel('LEFE - hip angle (degrees)')

ax4 = plt.subplot(1,5,4)
ax4.plot(num,lsfe_right,color='blue',label='lsfe',marker='x')
plt.xlabel('time (/)')
plt.ylabel('LSFE - knee angle (degrees)')

ax5 = plt.subplot(1,5,5)
ax5.plot(num,lwr_right,color='blue',label='lwr',marker='x')
plt.ylabel('LWR - rot angle (degrees)')
plt.xlabel('time (/)')
plt.suptitle('Bimanipulator Right Arm',fontsize=20)
plt.show()