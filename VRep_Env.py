# -*- coding: utf-8 -*-
"""@authors: Aberraouf MAOUDJ <abma@mmmi.sdu.dk>, Anders Lyhne Christensen <andc@mmmi.sdu.dk>
"""
import math
import random
import numpy as np
import copy
from   robot import MiR100
import configuration
import usedFunction
import datetime
from usedFunction import log_and_display
from numpy.linalg import inv
import ctypes


class V_Rep_Env(object):

    def __init__(self, vrep_ip: str, vrep_port: int):
        """Prepares the actions, states and other environment variables
        """     
        self.robot = MiR100(vrep_ip, vrep_port)
                                     
        """ Actions Space whithout move back: """         
        #for Vr in configuration.Vr_l:
            #for Vl in configuration.Vr_l:               
                #self.actions.append([Vr, Vl])
                
        self.actions = [[-0.8, 0.8], [0.8, -0.8], [0.5, 0.8], [0.8, 0.8], [0.8, 0.5], [2.0, 2.0], [0.5, 1.0], [1.0, 0.5]]
        self.total_actions= len(self.actions)       
        """ Definitin of the state Space """   
        self.states = []           
        for Obs_r1 in [True, False]: #ob_F
            for Obs_r2 in [True, False]:#ob_R
                for Obs_r3 in [True, False]:#ob_L
                    for Obs_r4 in [True, False]:#ob_R_Side
                        for Obs_r5 in [True, False]:#ob_L_Side
                                for Targ_reg in range(1,11):
                                    for is_targ_in_saf_reg in [True, False]: #Trg_dist_min in [-1, 1] -1 = there is minimisation 
                                        for Last_Action in range(0,len(self.actions)):
                                         #for last_Targ_reg in range(1,11):
                                            #for get_closer_T in [True, False]: #*******
                                                  self.states.append([Obs_r1, Obs_r2, Obs_r3, Obs_r4, Obs_r5, Targ_reg, is_targ_in_saf_reg,Last_Action ]) #last_Targ_reg, get_closer_T            
        # Used Variables
        self.total_states = len(self.states)
               
                 
        log_and_display("There are {0} actions.".format(self.total_actions))
        log_and_display("There are {0} states.".format(self.total_states))

        self.current_state = []
        self.newState      = []
        self.Last_state    = []
        self.Last_Action_id   = 0   # stop
        self.Last_target_reg  = 1  
        self.Target_last_Dist = 0
        self.target_reg       = 1
        self.robot_collided   = False
        self.is_success       = False

        #self.Non_safe_region   = True
 
    def reset_environment(self):
        """Prepares the Environment for a new episode.
        """
        #self.robot.Stop_robot()
        self.is_robot_collided = False
        self.is_success = False      
        self.robot.restart_sim_newEpisode()   # restart with new position of the robot and target
        #
        self.current_state =self.get_current_state()
    
    def set_robot_pos(self, x,y):
        """Prepares the Environment for a new episode.
        """
        #self.robot.Stop_robot()
        self.is_robot_collided = False
        self.is_success = False      
        self.robot.set_newEpisode(x,y)   # restart with new position of the robot and target
        #
        self.current_state =self.get_current_state()
        
    def reset_environment2(self):
        """Prepares the Environment for executing a task.
        """
        self.robot.restart_sim() 
        self.robot.Stop_robot()
        self.is_robot_collided = False
        self.is_success = False
        self.current_state =self.get_current_state()
        
    def get_obs_Distances(self):
        
        ob=self.robot.get_Obs_State()
        o=self.robot.get_Obs_Risky_R()
        
        dist = np.array([o, ob[0], ob[1], ob[2], ob[3]])
        return dist
        
    def is_Target_reached_(self):        
        x=self.robot.is_Target_reached()
        return x
    def is_robot_collided_(self):
        Y=self.robot.is_robot_collided()
        return Y
  
    def is_previous_current_state_same(self):
       if self.newState == self.Last_state :
          return True
       else:
          return False
    
    def startSimulation(self):
        self.robot.start_sim()
   
    def is_safe_region(self, Flag_F,Flag_R,Flag_L,R_side, L_side, targ_region,targ_dist, obs_F,obs_R,obs_L, obs_Rside, obs_Lside): 
            # Non safe region defined by tha fact that the robot and target are in the same derection       
            safe_reg =True                       
            
            if targ_region == 1 or targ_region == 5 or targ_region == 6 or targ_region == 7:
                if Flag_F  and targ_dist > obs_F  and obs_F >= 0 :                                   
                      safe_reg = False                          
        
            elif targ_region == 2 or targ_region == 8 or targ_region == 10:
                  if Flag_R and  targ_dist > obs_R and obs_R >= 0 :              
                     safe_reg = False
                  """elif R_side and  targ_dist > obs_Rside and obs_Rside >= 0 :
                      safe_reg = False"""
            
            elif targ_region == 3 or targ_region == 4 or targ_region == 9:
                  if Flag_L and targ_dist > obs_L and obs_L >= 0 : 
                       safe_reg = False 
                  """elif L_side and  targ_dist > obs_Lside and obs_Lside >= 0 :
                      safe_reg = False"""
                 
            """elif targ_region == 10:
                if R_side  and targ_dist > obs_Rside and obs_Rside > 0 :              
                   safe_reg = False  
                   
            elif targ_region == 9:
                if L_side  and targ_dist > obs_Lside and obs_Lside > 0 :              
                   safe_reg = False"""
            
                           
            return safe_reg
    
   
    def get_Target_Angle(self):
        
        # 1- get the target state and also the robot state
        target_pos = self.robot.get_position(self.robot.target_handle)
        robot_pos  = self.robot.get_position(self.robot.robot)
        thit_rob   = self.robot.get_oriention_rad(self.robot.robot)
        # to get the target region we have to calculate the transformation matrix and
        """ Define the rotation matrix from the robotic base frame (frame r)
         to the V-rep frame (frame v-rep).
        """
        rot_angle = thit_rob
        rot_mat_r_Vrep = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
                                 [np.sin(rot_angle), np.cos(rot_angle) ,   0 ],
                                 [0,                 0 ,                   1]])
        """ Define the displacement vector from frame Robot to frame V-rep
        """ 
        disp_vec_r_Vrep = np.array([[robot_pos[0]],
                                  [robot_pos[1]], 
                                  [0.0]])
        # Row vector for bottom of homogeneous transformation matrix
        bottom_row_homgen = np.array([[0, 0, 0, 1]])
 
        # Create the homogeneous transformation matrix from frame 0 to frame c
        homgen_r_Vrep = np.concatenate((rot_mat_r_Vrep, disp_vec_r_Vrep),  axis=1)     # side by side
        homgen_r_Vrep = np.concatenate((homgen_r_Vrep, bottom_row_homgen), axis=0) # one above the other
      
        coord_targ_Vrep = np.array([[target_pos[0]],
                                [target_pos[1]],
                                [0.0],
                                [1]])
        # Coordinates of the object in base reference frame
        coord_Target_robot = inv(homgen_r_Vrep) @ coord_targ_Vrep
  
        thit_target  = - math.atan2((coord_Target_robot[1]),(coord_Target_robot[0])) *(180/math.pi)
        
        return thit_target
        
    def get_current_state(self):   # returns the state id
        """Fetches position of the Robot MiR100, the location of all detected Obstacles and the last Action and target region
           Then calculates the state id that their values correspond to, and returns the state id
        """                
        Flag_F,Flag_R,Flag_L,Flag_R_Side,Flag_L_Side = False,False,False,False,False
        getCloser2T=False
        # 1- get the target state and also the robot state
        target_pos = self.robot.get_position(self.robot.target_handle)
        robot_pos  = self.robot.get_position(self.robot.robot)
        thit_rob   = self.robot.get_oriention_rad(self.robot.robot)
                
        Obs_f, Obs_r,Obs_l,obs_R_S,obs_L_S = self.robot.get_Obs_State()
        #riskZone_dist= self.robot.get_Obs_Risky_R()
        Dist = self.robot.get_Dist_Target()
        if Dist <= self.Target_last_Dist:
            getCloser2T = True
        
        self.Target_last_Dist = Dist
        
        if self.robot.is_robot_collided() :           
            self.robot_collided = True
        
     
        if Obs_f  <= configuration.Obs_threshold_Distance:
           Flag_F=True
        if Obs_r  <= configuration.Obs_threshold_Distance:
           Flag_R=True
        if Obs_l  <= configuration.Obs_threshold_Distance:
           Flag_L=True
        if obs_R_S <= configuration.Obs_threshold_Distance_side:
           Flag_R_Side =True
        if obs_L_S <= configuration.Obs_threshold_Distance_side:
           Flag_L_Side =True
           
        # to get the target region we have to calculate the transformation matrix and
        """ Define the rotation matrix from the robotic base frame (frame r)
         to the V-rep frame (frame v-rep).
        """
        rot_angle = thit_rob
        rot_mat_r_Vrep = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
                                 [np.sin(rot_angle), np.cos(rot_angle) ,   0 ],
                                 [0,                 0 ,                   1]])
        """ Define the displacement vector from frame Robot to frame V-rep
        """ 
        disp_vec_r_Vrep = np.array([[robot_pos[0]],
                                  [robot_pos[1]], 
                                  [0.0]])
        # Row vector for bottom of homogeneous transformation matrix
        bottom_row_homgen = np.array([[0, 0, 0, 1]])
 
        # Create the homogeneous transformation matrix from frame 0 to frame c
        homgen_r_Vrep = np.concatenate((rot_mat_r_Vrep, disp_vec_r_Vrep),  axis=1)     # side by side
        homgen_r_Vrep = np.concatenate((homgen_r_Vrep, bottom_row_homgen), axis=0) # one above the other
      
        coord_targ_Vrep = np.array([[target_pos[0]],
                                [target_pos[1]],
                                [0.0],
                                [1]])
        # Coordinates of the object in base reference frame
        coord_Target_robot = inv(homgen_r_Vrep) @ coord_targ_Vrep
  
        thit_target  = - math.atan2((coord_Target_robot[1]),(coord_Target_robot[0])) *(180/math.pi)
        
        
        print("************* dist target " ,Dist)
        #calculate the target region now
        if Dist <= configuration.target_region_d :
           if thit_target <= 10 and thit_target >= -10: #10
              self.target_reg = 6 
           if thit_target <= 20 and thit_target > 10:   #30, 10
              self.target_reg = 7
           if thit_target <= 90 and thit_target > 20:   #90, 30
              self.target_reg = 8
              
           if thit_target < -10 and thit_target >= -20: #30
              self.target_reg = 5
           if thit_target <-20 and thit_target >= -90:  #30
              self.target_reg = 4  
                       
        else : # if target > 80
           if thit_target <= 20 and thit_target >= -20: #30
              self.target_reg = 1
           if thit_target <= 90 and thit_target > 20:   #30
              self.target_reg = 2                   
           if thit_target < -20 and thit_target >= -90:  #30
              self.target_reg = 3  
         
        if thit_target <= 180 and thit_target > 90:
             self.target_reg = 10  
        if thit_target < -90 and thit_target >= -180:
             self.target_reg = 9  
             
    
        """if thit_target < -10 and thit_target >= -30:
              self.target_reg = 5
        if thit_target <= 30 and thit_target > 10:
              self.target_reg = 7 """     
                    
        is_Targ_in_saf_reg=self.is_safe_region(Flag_F,Flag_R,Flag_L,Flag_R_Side ,Flag_L_Side , self.target_reg, Dist, Obs_f,Obs_r,Obs_l,obs_R_S,obs_L_S)
        
        state_calc=[Flag_F,Flag_R,Flag_L,Flag_R_Side,Flag_L_Side,self.target_reg,is_Targ_in_saf_reg,self.Last_Action_id]#self.Last_target_reg, getCloser2T
        self.Last_target_reg=self.target_reg
        
        return  state_calc  #self.current_state
    
        """
           To check either the target region is safe or not to calculate the reward fuction
        """
        
    def get_index_state(self, st):        
        return self.states.index(st)

    def get_MiR100_position(self):
        return self.robot.get_position(self.robot.robot)
    
    def get_Target_position(self):
        return self.robot.get_position(self.robot.target_handle)

    def calculate_reward_new(self, lastState, newstae, lastDistTarget, newdist, lastTarget_Ang):
        
        log_and_display('S(t): ' + str(lastState)+' --> S(t+1): ' + str(newstae))
        Reg_LastState=""
        Reg_NewState=""
        reward=-1
        
        robot_coll   =  self.is_robot_collided_()
        targetReached = self.is_Target_reached_()
        if robot_coll:
           log_and_display('Penalty: Collision with the robot , terminating')
           reward= -50 #configuration.Reward_Failleur

        if targetReached:
           log_and_display('Reward: Target achieved Thanks !!!!!!!!')
           reward = 50 # configuration.Reward_GOAL_ACHIEVED
        
        if not robot_coll and not targetReached: # 
           target_Ang = self.get_Target_Angle()
           
           if target_Ang  >  89:
               target_Ang =  89
           if target_Ang  < -89:
               target_Ang = -89
               
           safe_new = newstae[6]
           x = 0
           num_obs_reg = 0
           Dist= lastDistTarget - newdist
           if safe_new:# safe 
              reward = math.cos(math.radians(target_Ang)) * (Dist) 
              if Dist == 0 :
                 if lastTarget_Ang - target_Ang <= lastTarget_Ang : # rotate to target direction
                    reward = math.cos(math.radians(target_Ang)) 
                 else :
                    reward = -math.cos(math.radians(target_Ang))  
                    
           else: # note safe         
              x = -1
              y = 1
              if newstae[0]:
                     y=0
              for i in range(0,5):
                 if newstae[i]:
                    num_obs_reg += 1
              reward = - num_obs_reg + y + (math.cos(math.radians(target_Ang))*(Dist) / (num_obs_reg +1))
              if Dist == 0 :
                 if lastTarget_Ang - target_Ang <= lastTarget_Ang: # rotate to target direction
                    reward = -num_obs_reg + y + (math.cos(math.radians(target_Ang))/ (num_obs_reg +1)) 
                 else :
                    reward = - num_obs_reg + y - math.cos(math.radians(target_Ang))           
           print('---------------------------')
           print('Reward:  ',reward)   
           print('dist:  ',Dist)             
           print('Thitha last:  ',lastState[5])
           print('Thitha new:  ', newstae[5])
           print('number Of region with obs:  ',num_obs_reg)
           print('---------------------------')
           log_and_display('Reward: ' + str(reward))
        
        return reward ,robot_coll, targetReached     
            
    def calculate_new_reward(self, lastState, newstae, lastDistTarget, newdist, lastTarget_Ang, LAst_robot_ori ):
        
        log_and_display('S(t): ' + str(lastState)+' --> S(t+1): ' + str(newstae))
        Reg_LastState=""
        Reg_NewState=""
        reward = -1
                
        R_goal      =  100
        R_collision = -100
        R_pos = 0.1 # use big value if we want the robot to minimize theta then go
        R_neg = -0.2
        
        fact        = R_neg
        rward_UNS   = -5
        R_SF_front  = 0
        
        Titha_minim=False
        still_in_same_reg=False
        
        target_Reg = newstae[5]
        last_target_Reg = lastState[5]
        
        if last_target_Reg == 1 or last_target_Reg == 2 or last_target_Reg == 6 or last_target_Reg == 7 or last_target_Reg == 8 or last_target_Reg == 10:
           if target_Reg == 1 or target_Reg == 2 or target_Reg == 6 or target_Reg == 7 or target_Reg == 8 or target_Reg == 10:
              still_in_same_reg = True
        
        if last_target_Reg == 1 or last_target_Reg == 6 or last_target_Reg == 3 or last_target_Reg == 4 or last_target_Reg == 5 or last_target_Reg == 9:
           if target_Reg == 1 or target_Reg == 6 or target_Reg == 3 or target_Reg == 4 or target_Reg == 5 or target_Reg == 9:
              still_in_same_reg = True
              
              
        if target_Reg == 1 or target_Reg == 5 or target_Reg == 6 or target_Reg == 7:
           fact = R_pos
           Titha_minim=True
        
        if not newstae[0] :
           R_SF_front = 1
           
        robot_coll   =  self.is_robot_collided_()
        targetReached = self.is_Target_reached_()
        
        
        
        target_Ang = self.get_Target_Angle()
        robot_ori  = self.robot.get_robot_oriention()
        
        err_Ang = math.fabs(robot_ori) - math.fabs(LAst_robot_ori)
        safe_new  = newstae[6]
        safe_last = lastState[6]
        Dist= lastDistTarget - newdist
        
        
        if safe_last: # safe_new and            
            if target_Reg == 1 :#or target_Reg == 5 or target_Reg == 6 or target_Reg == 7:#if newstae[8]: # there is a minimisation         
               reward =  math.cos(math.radians(target_Ang)) + 1
            elif target_Reg == 5 or target_Reg == 6 or target_Reg == 7: 
                if Dist > 0 : # minimation
                   reward =  math.cos(math.radians(target_Ang)) + 1
                else: #no minimization
                   reward = math.cos(math.radians(target_Ang)) - 1  
            else:
               reward = math.cos(math.radians(target_Ang)) -  1
                 
        else: # note safe   ==> robot should move away from the font obstacles and then minimize thitha               
            num_obs_reg=0
            """for i in range(0,5):
                 if newstae[i]:
                     num_obs_reg +=1"""
            if lastState[0]:
                    num_obs_reg += 2
            #if target_Reg == 2 or target_Reg == 7 or target_Reg == 8 or target_Reg == 10:
            if last_target_Reg == 2 or last_target_Reg == 7 or last_target_Reg == 8 or last_target_Reg == 10:
                 if lastState[1]:#newstae
                    num_obs_reg += 1
                 if lastState[3]:
                    num_obs_reg += 1  
            #if target_Reg == 5 or target_Reg == 4 or target_Reg == 3 or target_Reg == 9:
            if last_target_Reg == 3 or last_target_Reg == 4 or last_target_Reg == 5 or last_target_Reg == 9:
                 if lastState[2]:
                    num_obs_reg += 1
                 if lastState[4]:
                    num_obs_reg += 1
                    
            """if still_in_same_reg :#lastTarget_Ang - target_Ang <= lastTarget_Ang: # rotate to target direction
                  reward = - num_obs_reg - math.cos(math.radians(err_Ang)) 
            else :
                  reward = - num_obs_reg - 3 """# + math.cos(math.radians(err_Ang)) # punishement for tourning   
            reward = - num_obs_reg - math.cos(math.radians(err_Ang)) + math.cos(math.radians(target_Ang))
        
        if robot_coll:
           log_and_display('Penalty: Collision with the robot , terminating')
           reward= R_collision #configuration.Reward_Failleur

        if targetReached:
           log_and_display('Reward: Target achieved Thanks !!!!!!!!')
           reward = R_goal # configuration.Reward_GOAL_ACHIEVED
        
           
        print('---------------------------')
        print('Reward:  ',reward)   
        print('dist:  ',Dist)             
        #print('number Of region with obs:  ',num_obs_reg)
        print('---------------------------')
        log_and_display('Reward: ' + str(reward))
        
        return reward ,robot_coll, targetReached     
              
        
    def calculate_reward(self, lastState,newstae, lastDistTarget):
        """
         Implements the reward strategy, returns reward, is_success or failed
                 
        """ 
        log_and_display('S(t): ' + str(lastState)+' --> S(t+1): ' + str(newstae))
        
        #obs= self.robot.get_Obs_State()
        #dist_targ=self.robot.get_Dist_Target()
        
        T_lastState = lastState[5]
        #Last_reg_T_lastState = lastState[6]
        
        T_newstae  = newstae[5]
        #Last_reg_T_newstae  = newstae[6]
        
        reward = configuration.Reward_DEFAULT  
        
        robot_coll   =  self.is_robot_collided_()
        targetReached = self.is_Target_reached_()
        
        # check if the same state 
        state_= ([lastState[i] == newstae[i] for i in range(len(newstae))])
        is_the_same_state = True      
        for x in state_ :
            if not  x :
                is_the_same_state = False
                       
        if is_the_same_state:   # we should encorage the robot to enhence its behevior
           reward = -3          #configuration.Reward_DEFAULT
           log_and_display('Penalty: still with the same State Reward_DEFAULT : -3  ')          
        
   
        Reg_LastState=""
        Reg_NewState=""
 
        safe     = lastState[6]
        safe_new = newstae[6]
  
        if safe:
            Reg_LastState="SR"
        else:
            Reg_LastState="NSR"        
        if  safe_new:
            Reg_NewState="SR"
        else:
            Reg_NewState="NSR"        
        
        if lastState[0] and lastState[1] and lastState[2] and lastState[3] and lastState[4]:         
           if lastState[5] == 1 or lastState[5] == 2 or  lastState[5] == 3 or  lastState[5] == 4 or lastState[5] == 5 or lastState[5] == 6 or  lastState[5] == 7 or lastState[5] == 8 :   
            Reg_LastState="Blocking_R"           
  
        if  newstae[0] and newstae[1] and newstae[2] and newstae[3] and newstae[4]:
             if newstae[5] == 1 or newstae[5] == 2 or  newstae[5] == 3 or  newstae[5] == 4 or newstae[5] == 5 or newstae[5] == 6 or  newstae[5] == 7 or newstae[5] == 8 : 
               Reg_NewState="Blocking_R"
           
        #SR ----> NSR
        if Reg_LastState == "SR" and Reg_NewState == "NSR" :
           #if T_lastState== 1 or T_lastState==2 or T_lastState ==3:
            #  if T_newstae == 4 or T_newstae == 5 or T_newstae == 6 or T_newstae == 7 or T_newstae == 8 :             
           if newstae[8]:     
                 reward= configuration.Reward_SR_NSR_rech_T
                 log_and_display('Penalty : Moved from SR to NSR while approaching target: '+ str(reward))
           elif T_lastState  == 10 or T_lastState == 9 :
                if T_newstae != 10 and T_newstae  != 9 :
                    reward= configuration.Reward_SR_NSR_rech_T
                    log_and_display('Small Penalty : Moved from SR to NSR approaching target: '+ str(reward))
           else :
              reward= configuration.Reward_SR_NSR
              log_and_display('Penalty Big: Moved from SR to NSR: '+ str(reward))        
        # SR --->Blocking_R or  
        elif Reg_LastState == "SR" and Reg_NewState == "Blocking_R" : ############## redefine this
           if T_newstae==5 or T_newstae==6 or T_newstae== 7 :
               reward= configuration.Reward_SR_SR
               log_and_display('Penalty : Moved from SR to Blocking_R but apraching to target: '+ str(reward))
           else :
               reward= configuration.Reward_SR_Blocking_R
               log_and_display('Penalty hight: Moved from SR to Blocking_R: '+ str(reward))
        ## SR---->SR    
        elif Reg_LastState == "SR" and Reg_NewState == "SR" :
           
           if T_lastState  == 10 or T_lastState == 9 :
               if T_newstae == 1 or T_newstae == 5 or T_newstae == 6 or T_newstae == 7 :              
                   reward= configuration.Reward_SR_SR_enh + 2
                   log_and_display('reward Big: Moved from back pos to SR in front of pos while approaching target: '+ str(reward))
           elif newstae[8]: 
                reward= configuration.Reward_SR_SR_enh
                log_and_display('reward : Moved from SR to SR while approaching target: '+ str(reward))
          
           else:## mouve in safe region without minimization 
                  reward= configuration.Reward_SR_SR
                  log_and_display('reward: Moved from SR to SR without distance minimization: '+ str(reward))                 
              
                                                   
        #NSR_SR or NSR_NSR or NSR_Bloking_R
        if Reg_LastState == "NSR" :           
           if Reg_NewState == "NSR" :
              if newstae[8]:              
                    reward= configuration.Reward_NSR_NSR_enh
                    log_and_display('Small Penalty : Moved from NSR to NSR while approaching target: '+ str(reward))
              
              else :
                     reward= configuration.Reward_NSR_NSR 
                     log_and_display('Penalty : Moved from NSR to NSR: '+ str(reward))   
                     
           elif Reg_NewState == "SR" :              
                 if newstae[8]:              
                    reward= configuration.Reward_NSR_SR_enh
                    log_and_display('Big reward : Moved from NSR to SR while approaching target: '+ str(reward))                 
                 else :
                    reward= configuration.Reward_NSR_SR
                    log_and_display('reward : Moved from NSR to SR: '+ str(reward))           
           elif Reg_NewState == "Blocking_R" :
              reward= configuration.Reward_Blocking_R
              log_and_display('Hight Penalty : Moved from NSR to Blocking_R: '+ str(reward))
                     
        #NSR_Bloking_R_SR or NSR_Bloking_R_NSR or NSR_Bloking_R to NSR_Bloking_R
        if Reg_LastState == "Blocking_R" :
           if Reg_NewState == "SR" :
              if T_newstae == 5 or T_newstae == 6 or T_newstae == 7 :
                  reward= configuration.Reward_Blocking_R_SR
                  log_and_display('reward : Moved from Blocking_R to SR while approaching target: '+ str(reward))
              if T_newstae == 4 or T_newstae == 8 :    
                  reward= configuration.Reward_Blocking_R_SR-1
                  log_and_display('reward : Moved from Blocking_R to SR: '+ str(reward))
              if T_newstae == 1 or T_newstae == 2 or T_newstae == 3 :    
                  reward= configuration.Reward_Blocking_R_SR-2
                  log_and_display('reward : Moved from Blocking_R to SR with enh: '+ str(reward))
              else :
                  reward= configuration.Reward_Blocking_R_SR-4
                  log_and_display('reward : Moved from Blocking_R to SR: '+ str(reward))
                  
           if Reg_NewState == "NSR" :
              if T_newstae == 5 or T_newstae == 6 or T_newstae == 7 :
                  reward= configuration.Reward_Blocking_R_NSR +2
                  log_and_display('reward : Moved from Blocking_R to NSR: '+ str(reward))
              else :
                  reward= configuration.Reward_Blocking_R_NSR
                  log_and_display('reward : Moved from Blocking_R to NSR: '+ str(reward))  
           
           if Reg_NewState == "Blocking_R" :
              reward= configuration.Reward_Blocking_R_Blocking_R
              log_and_display('reward : Moved from Blocking_R to Blocking_R: '+ str(reward)) 
 
    
        if robot_coll:
           log_and_display('Penalty: Collision with the robot , terminating')
           reward=configuration.Reward_Failleur

        if targetReached:
           log_and_display('Reward: Target achieved Thanks !!!!!!!!')
           reward = configuration.Reward_GOAL_ACHIEVED
       
        
        return reward ,robot_coll, targetReached 
    
    #  return the reward for an action_id and the new state
    def move_MiR_robot(self, action_id_):  
        lastDist = self.Target_last_Dist    
        action_ = self.actions[action_id_]   
        target_Ang = self.get_Target_Angle()
        self.Last_Action_id = action_id_
        self.robot.Move_robot(action_[0],action_[1])  # its duration is 100 ms                            
        self.newState = self.get_current_state()  # this fuction will update 'Last_state' and 'current_state'         
        #reward, is_coll, is_success = self.calculate_reward_new(self.current_state, self.newState, lastDist, self.Target_last_Dist,target_Ang)
        robot_ori  = self.robot.get_robot_oriention()
        reward, is_coll, is_success = self.calculate_new_reward(self.current_state, self.newState, lastDist, self.Target_last_Dist,target_Ang,robot_ori)
       
        self.robot_collided = is_coll
        self.is_success    = is_success
        
        self.current_state = self.newState
        
        return reward, self.newState
