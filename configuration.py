# -*- coding: utf-8 -*-
"""@authors: Aberraouf MAOUDJ <abma@mmmi.sdu.dk>, Anders Lyhne Christensen <andc@mmmi.sdu.dk>
"""

import random
import numpy as np


# MiR100 robot Velocityto define Action space
Vr_l=np.array([-1, -0.8, 0, 0.5, 0.8, 2])  # don't use Move bacK 
"""
Actions Space (Vr_V_l): 
                  (0)[-1.4, -1.4], (1)[-1.4, -0.3], (2)[-1.4, 0.0], (3)[-1.4, 0.3], (4)[-1.4, 0.8], (5)[-1.4, 2.0], (6)[-0.3, -1.4], 
                  (7)[-0.3, -0.3], (8)[-0.3, 0.0], (9)[-0.3, 0.3]], (10)[-0.3, 0.8], (11)[-0.3, 2.0], (12)[0.0, -1.4], 
                  (13)[0.0, -0.3], (14)[0.0, 0.0]], (15)[0.0, 0.3], (16)[0.0, 0.8], (17)[0.0, 2.0], (18)[0.3, -1.4], 
                  (19)[0.3, -0.3], (20)[0.3, 0.0], (21)[0.3, 0.3], (22)[0.3, 0.8], (23)[0.3, 2.0], (24)[0.8, -1.4], (25)[0.8, -0.3],
                  (26)[0.8, 0.0], (27)[0.8, 0.3], (28)[0.8, 0.8], (29)[0.8, 2.0], (30)[2.0, -1.4], (31)[2.0, -0.3], 
                  (32)[2.0, 0.0]], (33)[2.0, 0.3], (34)[2.0, 0.8], (35)[2.0, 2.0]   
"""

# Rewards
Reward_Failleur      = -15
Reward_GOAL_ACHIEVED =  15

Reward_SR_NSR        = -4
Reward_SR_NSR_rech_T = -1   # use Target_reg and last_reg_T to define this
Reward_SR_SR         = -1   # to motivate the robot to go towar the right sens
Reward_SR_SR_enh     =  4   # use Target_reg and last_reg_T to define this
Reward_SR_Blocking_R = -6

Reward_NSR_SR        =  5
Reward_NSR_SR_enh    =  6
Reward_NSR_NSR_enh   = -1   #-1 use Target_reg and last_reg_T to define this
Reward_NSR_NSR       = -2

Reward_Blocking_R_SR =  6
Reward_Blocking_R_NSR= -1  # it wase '0' 
Reward_Blocking_R    = -5
Reward_Blocking_R_Blocking_R = -5


Reward_DEFAULT       = -1

# Number of episodes to run
NUM_EPISODES        = 20000
MIN_EPISODES_TO_RUN = 10000
# Number of max actions to execute an episode, so terminate episode when exceed this number
NUM_MAX_ACTIONS     = 300
# Least actions needed to achieve the goal
MIN_ACTIONS_EXPECTED = 5

# Initialisation of the Q value by 0 (in futur work, we will propose GA to init these value)
Q_INIT_VAL = 0.0


# The discount rate in equation of Q-Value
DISCOUNT         = 0.9 # 0.9
# Epsilon value 
EPSILON    = 0.5
# The learn rate in equation of Q-Value
LEARN_RATE       = 0.3  #  from 0.1 to 0.1
LEARN_RATE_DECAY = LEARN_RATE * 0.0001

# The amount by which epsilon will be reduced every episode
# to reduce exploration as we start executing more episodes
EPSILON_DECAY    = EPSILON * 0.0001

Q_TABLE_Path = 'Q_tables'
# File where Q table will be saved, loaded from. It's a numpy array.
Q_TABLE_FILE = 'Q_tables/qtableF.npy' #qtable_new, qtable_F.npy'
# File where per-episodic data will be saved: episode id, is success?, total rewards, total actions.
PLOT_FILE     = 'Q_tables/episodes.txt'
PLOT_AVR_R    = 'Q_tables/AVRreward.txt'
# Some artificial delays to let V-REP stabilize after an action
SLEEP_VAL     = 0.2
SLEEP_VAL_MIN = 0.4

# The step size to use to discretize the environment
UNIT_STEP_SIZE = 0.1 # 300ms loop func

Obs_threshold_Distance = 80 # 80 cm
Obs_threshold_Distance_side = 70 # 70 cm

#calculate the target region
target_region_d = 70

# A value to judge wither the robot reachs its target or no
TOLERANCE = 30  
# A finer value to judge nearness
TOLERANCE_FINER = 10 #cm



