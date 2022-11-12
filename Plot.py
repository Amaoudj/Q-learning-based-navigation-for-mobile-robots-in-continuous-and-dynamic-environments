# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:40:31 2021
@authors: Aberraouf MAOUDJ <abma@mmmi.sdu.dk>, Anders Lyhne Christensen <andc@mmmi.sdu.dk>

"""
import sys
import configuration
import os
import logging
from   agent   import  agent_
import matplotlib.pyplot as plt
from   VRep_Env import V_Rep_Env
import numpy as        np
import math

from matplotlib import style
import time

style.use("ggplot")


def plot(v1):
    
    x = np.arange(0, 5000, 30)
    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, v1, '.-', label='yolo method')    
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200) 
    
    
def file_read(fname):
        content_array = []
        rate= []
        with open(fname) as f:
                #Content_list is the list that contains the read lines.     
                for line in f:
                        content_array.append(line)
                        x=str(line)
                        rate.append(x.split(","))
                
        return(rate)

rate1 = file_read('Q_tables/AVRreward.txt')
rate2 = file_read('Q_tables/AVRreward1.txt')


raward=[]
raward2=[]
steps=[]
steps2=[]
for i in rate1:    
      #raward.append(float(i[1]))
      steps.append(float(i[2])) #i[0]

for i in rate2:    
      #raward2.append(float(i[1]))
      steps2.append(float(i[2]))
      
#print(math.cos(math.radians(70)) )
#print(steps)
directory  = os.path.dirname(__file__)
results_dir = os.path.join(directory, 'Results_plot/')
file_name = "Agerage valeus"
fig = plt.figure(1)


"""     
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()                   
"""
 
#plt.subplot(211) #122)
plt.axis([1,500  , 0, max(steps) +8])
plt.plot(steps, 'r')
plt.plot(steps2, 'b')           
plt.ylabel('Average Steps')
plt.xlabel('Episodes')

plt.legend(('IQL with heuristic algorithms', 'IQL without using heuristic algorithms'),loc='upper right')                    
plt.show()           
fig.savefig(results_dir + file_name + '.png', dpi=150, bbox_inches='tight') 