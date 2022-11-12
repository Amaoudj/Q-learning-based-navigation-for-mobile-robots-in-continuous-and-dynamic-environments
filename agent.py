# -*- coding: utf-8 -*-
"""@authors: Aberraouf MAOUDJ <abma@mmmi.sdu.dk>, Anders Lyhne Christensen <andc@mmmi.sdu.dk>
"""
import numpy as np
import os
import time
import configuration
from VRep_Env import V_Rep_Env
import math

class agent_(object):
    def __init__(self,env: V_Rep_Env,agent_id,training,continue_training, epsilon, q_init_val, discount, learn_rate):
        print ('Environment created')
        self.env = env
        self.agent_ID = agent_id      # we use this for multi-robot system
        self.learn_rate = learn_rate
        self.discount = discount
        self.epsilon = epsilon                      
        #self.q_table = np.array([(env.total_states, env.total_actions)])  
        
        self.q_table =np.zeros((self.env.total_states, self.env.total_actions))        
                 
        print(self.q_table)
        self.current_state = []  # currente state
        self.total_explorations = 0
    
    def init_qtable_propos_Alg(self):         
        self.q_table =np.zeros((self.env.total_states, self.env.total_actions))
        f = configuration.Q_TABLE_FILE  
        """if os.path.exists(f): 
            os.remove(f) """    
        for s in self.env.states:           
              act_id=self.select_action_H_ALG(s)
              sateID=self.env.get_index_state(s)            
              self.q_table[sateID,act_id] = 20.0  ## max value of the reward function               
        np.save(f, self.q_table)
        
    def init_qtable_propos_Alg_state(self,state):    
        f = configuration.Q_TABLE_FILE
        q_table_ = np.load(f)       
        act_id=self.select_action_H_ALG(state)
        sateID=self.env.get_index_state(state)            
        q_table_[sateID,act_id] += 51.15482  # max value of the reward function                 
        np.save(f, q_table_)    
   
    
    def load_qtable_execution(self):
        f = configuration.Q_TABLE_FILE 
        if os.path.exists(f):
            self.q_table = np.load(f)
        else :
            print("Q-tqble not founded")
    def load_qtable_training(self):
        f = configuration.Q_TABLE_FILE 
        self.q_table = np.load(f)

    def save_qtable(self):
        """The Q table is saved after every episode
        """
        f = configuration.Q_TABLE_FILE
        if os.path.exists(f):     
            os.remove(f)          #remove the existing file and remplace with new Q_tqble
        np.save(f, self.q_table)

    def reset(self):
        """Prepares the Agent for a new episode.
        """
        print('Initializing episode')
        self.env.reset_environment()       
        self.total_explorations = 0       
        self.current_state = self.env.get_current_state()
    
    def reset2(self):
         """
          Prepares the Agent for the task execution ...
         """
         print('Initialization ... ')
         self.env.reset_environment2()        
         self.total_explorations = 0
         self.current_state = self.env.get_current_state()
    
    def softmax(x):
         """ applies softmax to an input x"""
         e_x = np.exp(x - np.max(x))
         return (e_x/e_x.sum())
    
    def select_action_Free_space(self, Current_state):
        #if not Current_state[0] and  not Current_state[1] and not Current_state[2] and  not Current_state[3] :
         print(' Free Env: Give some knowldge to the robot')
         targ_reg=Current_state[5]
         if  targ_reg==6:
                  action_id_= 3# Forward V1
         elif targ_reg ==1:
                  action_id_= 5 # Forward Vmax
         elif targ_reg ==7:    
                   action_id_= 0 #2
         elif targ_reg ==5:    
                   action_id_= 1 #4   
        
         elif targ_reg ==10 or Current_state[5] ==8:
              action_id_= 0       
         elif targ_reg ==9 or Current_state[5] ==4:
              action_id_= 1      
        
         elif targ_reg == 2 :#or Current_state[5] ==8 : 
              action_id_= 6 #2#6    
             
         elif targ_reg ==3 : #or Current_state[5] ==4 :
              action_id_= 7 #4#7
              
               
         return action_id_
        
    def select_action_H_ALG(self, Current_state):
        """ 
        my new function to select an action.
        """        
        current_state_ID     = self.env.get_index_state(Current_state)
        #actionsNews = np.argwhere(self.q_table[current_state_ID]== 0)                
        
        ii=1
        # define the state of target in safe or no safe region
        safe_regF,safe_regR, safe_regL,safe_R_side, safe_L_side=True,True,True, True,True
        
        T_s_reg = Current_state[6]
        Targ_reg= Current_state[5]
        
        if not T_s_reg:
            if Targ_reg  ==1 or Targ_reg  ==5 or Targ_reg==6 or Targ_reg==7:
                 safe_regF = False
            elif Targ_reg==2 or Targ_reg==8 or Targ_reg==10 :  
                 safe_regR = False
            elif Targ_reg==3 or Targ_reg==4 or Targ_reg==9:  
                 safe_regL = False

        if ii==1:#len(actionsNews) == self.env.total_actions :                    
           #1) in free space ture to the direction of the target and then go stright forward to target
           if T_s_reg :
               print(' Free space')
               action_id_= self.select_action_Free_space(Current_state)
           #2)There are obstacles only in front
           else:       
              print(' There are Obstacles:')
              
              lsatAction = Current_state[7]
   
              # 1) target in font and there are obstacles in front
              if Targ_reg  ==1 or Targ_reg  ==5 or Targ_reg==6 or Targ_reg==7 : # and flag unsafe_reg == True
                  if not Current_state[1] and not Current_state[2]:
                      if  Targ_reg==7:   # not Current_state[3] 
                          action_id_ = 0 #
                      elif  Targ_reg ==5: # not Current_state[4]
                          action_id_ = 1 #
                          
                          
                      else: # target in reg 1 or 6
                          if not Current_state[3] :
                              action_id_= 0 # default chose  
                          elif not Current_state[4]:
                              action_id_= 1 #
                          else:
                              if   lsatAction == 1 : #
                                   action_id_=  1
                              else :
                                  action_id_=  0 # 
                 
                  elif not Current_state[1] : 
                           action_id_=  0 #             
                  elif not Current_state[2] : 
                           action_id_=  1 # 
                  #----------------------------------------------------------------------
                  elif Current_state[1] and Current_state[2]:   # no free space
                      if  Targ_reg==7:  
                          if not Current_state[3] and lsatAction  != 1 : ###
                               action_id_ = 0   
                          elif not Current_state[4] and lsatAction != 0:  ###
                              action_id_ = 1 #  
                          else :
                               action_id_ = 0 
                      elif Targ_reg==5:
                           if not Current_state[4] and lsatAction  != 0:   ###
                              action_id_ = 1 #
                           elif   not Current_state[3] and lsatAction  != 1:###
                               action_id_ = 0 
                           else:
                               action_id_ = 1
                      elif  Targ_reg==6 or Targ_reg==1:  # 6 ou 1                          
                           if  lsatAction == 1 : #
                               action_id_ =  1                               
                           else :
                               action_id_ =  0  #
                  ############################################         
            
              # 2) target on Right and there are obstacles in Right  (2 or 8)            
              elif Targ_reg==2 or Targ_reg==8 or Targ_reg==10 :#
                  if not Current_state[0] :
                         action_id_= 3 
                         
                  elif not Current_state[2]: 
                          action_id_=  1
                  elif not Current_state[3]and lsatAction  != 1:  #+++++++++      :                        
                         action_id_=   0     
                         
                  elif not Current_state[4] and lsatAction  != 0:                        
                         action_id_=   1                
 
                  else :
                    if lsatAction  == 1 :
                        action_id_=  1 
                    else :
                        action_id_=  0 #
              # 3) target on Left  and there are obstacles in Left
              elif Targ_reg==3 or Targ_reg==4 or Targ_reg==9 :# and Unsafe_flag is True
                   if not Current_state[0]:                   
                         action_id_=  3                                                    
                   elif not Current_state[1] :                                                 
                             action_id_= 0
                   
                   elif not Current_state[4] and lsatAction  != 0:  #+++++++++                 
                         action_id_=   1 # 
                         
                   elif not Current_state[3] and lsatAction  != 1:                        
                        action_id_=   0 # 
                   else :
                     if lsatAction == 0 :
                         action_id_=  0 
                     else :
                         action_id_= 1 # 
                           
        print(Current_state)    
        print('action choosen', action_id_) 
        return action_id_

    def Init_Table_withou_lastAction(self, Current_state):
        """ 
        my new function to select an action.
        """        
        current_state_ID     = self.env.get_index_state(Current_state)
        #actionsNews = np.argwhere(self.q_table[current_state_ID]== 0)                
        
        ii=1
        # define the state of target in safe or no safe region
        safe_regF,safe_regR, safe_regL,safe_R_side, safe_L_side=True,True,True, True,True
        
        T_s_reg = Current_state[6]
        Targ_reg= Current_state[5]
        
        if not T_s_reg:
            if Targ_reg  ==1 or Targ_reg  ==5 or Targ_reg==6 or Targ_reg==7:
                 safe_regF = False
            elif Targ_reg==2 or Targ_reg==8 or Targ_reg==10 :  
                 safe_regR = False
            elif Targ_reg==3 or Targ_reg==4 or Targ_reg==9:  
                 safe_regL = False

        if ii==1:#len(actionsNews) == self.env.total_actions :                    
           #1) in free space ture to the direction of the target and then go stright forward to target
           if T_s_reg :
               print(' Free space')
               action_id_= self.select_action_Free_space(Current_state)
           #2)There are obstacles only in front
           else:       
              print(' There are Obstacles:')
                 
              # 1) target in font and there are obstacles in front
              if Targ_reg  ==1 or Targ_reg  ==5 or Targ_reg==6 or Targ_reg==7 : # and flag unsafe_reg == True
                  if not Current_state[1] and not Current_state[2]:
                      if  Targ_reg==7:   # not Current_state[3] 
                          action_id_ = 0 #
                      elif  Targ_reg ==5: # not Current_state[4]
                          action_id_ = 1 #
                          
                          
                      else: # target in reg 1 or 6
                          if not Current_state[3] :
                              action_id_= 0 # default chose  
                          elif not Current_state[4]:
                              action_id_= 1 #
                          else:
                              action_id_=  0 # 
                 
                  elif not Current_state[1] : 
                           action_id_=  0 #             
                  elif not Current_state[2] : 
                           action_id_=  1 # 
                  #----------------------------------------------------------------------
                  elif Current_state[1] and Current_state[2]:   # no free space
                      if  Targ_reg==7:                            
                            action_id_ = 0 
                      elif Targ_reg==5:                           
                            action_id_ = 1
                      elif  Targ_reg==6 or Targ_reg==1:  # 6 ou 1                                                     
                            action_id_ =  0  #
                  ############################################         
            
              # 2) target on Right and there are obstacles in Right  (2 or 8)            
              elif Targ_reg==2 or Targ_reg==8 or Targ_reg==10 :#
                  if not Current_state[0] :
                         action_id_= 3 
                         
                  elif not Current_state[2]: 
                          action_id_=  1
                  else :
                       action_id_=  0 #
              # 3) target on Left  and there are obstacles in Left
              elif Targ_reg==3 or Targ_reg==4 or Targ_reg==9 :# and Unsafe_flag is True
                   if not Current_state[0]:                   
                         action_id_=  3                                                    
                   elif not Current_state[1] :                                                 
                             action_id_= 0                                      
                   else :                     
                        action_id_= 1 # 
 
        return action_id_

    def select_action(self, Current_state_):
        """ 
            mix of exploratory and exploitative actions based on epsilon value.
        """
        current_state_ID     = self.env.get_index_state(Current_state_)
        
        if np.random.uniform() < self.epsilon:  
            print('Exploring ...') 
            self.total_explorations += 1
            # get new action that we haven't explore befor
            actionsNews = np.argwhere(self.q_table[current_state_ID]== 0) # choose random A where Q_table is equale to 0
            if len(actionsNews) > 0 :
                v = np.random.choice(len(actionsNews))
                action_id_=actionsNews[v][0]               
            else :
                action_id_ = np.random.choice(self.env.total_actions)
        else:
           print('Exploiting ...')                   
           """if np.random.uniform() < self.epsilon: 
               #action_id_ = self.select_action_H_ALG(Current_state_)  
               action_id_ = np.argmax(self.q_table[current_state_ID])
           else: """             
              
           action_id_ = np.argmax(self.q_table[current_state_ID])           
        return action_id_

    def select_action_Execution(self, Current_staten):       
        # get the index of the max value in the Q-tqble that coresponding to Current_state
        current_state_ID     = self.env.get_index_state(Current_staten) 
        print('Current state : ',Current_staten)
        print('Q-Table : ', self.q_table[current_state_ID])
        action_id_ = np.argmax(self.q_table[current_state_ID])          
        print('The best ation : ', action_id_)
        return action_id_
   
    def update_q_table(self, state_, action, reward, state_new_):
        """
        Equation is :       
        Q(s, a) += alpha * ( reward(s,a) + gamma * max(Q(s') - Q(s,a) )                    
       
        q_current = q_table[state, action]
        next_max = np.max(q_table[next_state])       
        new_value = (1 - alpha) * q_current + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
    
        """
        # we work with indexes       
        state     = self.env.get_index_state(state_)
        state_new = self.env.get_index_state(state_new_)
        
        q_current = self.q_table[state, action]
        next_max = np.max(self.q_table[state_new])                
        #new_value = q_current + self.learn_rate * (reward + self.discount * next_max - q_current)  
        
        new_value= (1 - self.learn_rate) * q_current + self.learn_rate * (reward + self.discount * next_max)        
        self.q_table[state, action] = new_value
        msg = "Q-Value: S:{}, A:{}, R:{}, New_State:{}, Q_Val:{}, new_Q_Val:{}".format(state, action, reward, state_new, q_current, new_value)
        #print(msg)
      
 # -------------------------------  execution parts-------------------------------------
    def execute_action(self, action_id):         #        
        action = self.env.actions[action_id]
        print('Action chosed : ' + str(action_id) + ' = ' +str(action))       
        return self.env.move_MiR_robot(action_id) # return the reward value of this action 
   
    def execute_action2(self, action_id):         #         
        action = self.env.actions[action_id]           
        return self.env.move_MiR_robot(action_id) # return the reward value of this action 
    
    def execute_episode_qlearn(self, max_steps: int):
        """ execute one learning episode.
        """
        total_reward = 0
        total_steps = 0
        
        robotColl   = self.env.is_robot_collided_()
        trgetreched = self.env.is_Target_reached_()
       
        while max_steps > 0 and not robotColl and not trgetreched:
                           
              action_id         =self.select_action(self.current_state)
              reward, new_state = self.execute_action(action_id)   # and get the newState
              self.update_q_table(self.current_state, action_id, reward, new_state)
              self.current_state = new_state
                   
              max_steps    -= 1
              total_reward += reward
              total_steps  += 1
              
              trgetreched   = self.env.is_Target_reached_() 
              robotColl     = self.env.is_robot_collided_()
              if robotColl or trgetreched or max_steps == 0:
                  self.env.robot.restart_sim() 
              
        return total_reward, total_steps,trgetreched ,robotColl, self.total_explorations

    """
       ================< You run your Task once you have a robust Q-Table>============================
    """
    def execute_Task(self, max_steps: int):
        """Here, assuming that we have a robust qtable
        """
        total_reward = 0
        total_steps = 0
        self.epsilon = 0.0
        is_target_reched = self.env.is_Target_reached_()
        robot_coll       = self.env.is_robot_collided_()
        max_steps += 100 
        #Last_state=self.current_state
        self.load_qtable_execution()
        
        # set the inital position of the robot using uniform distribution
        """ x = np.uniform(-6, 7, size=1) 
        y = np.uniform(-7, 7, size=1)       
         self.env.set_robot_pos(x[0],y[0])"""
        while not is_target_reched and not robot_coll:#max_steps > 0 and
           #
            action_id = self.select_action_Execution(self.current_state)
       
            reward,new_state  = self.execute_action2(action_id)          
            self.current_state= new_state
            is_target_reched  =self.env.is_Target_reached_()
            robot_coll        =self.env.is_robot_collided_()
            max_steps    -= 1
            total_reward += reward
            total_steps  += 1
            
            if robot_coll or is_target_reched :
               self.env.robot.stop_sim() 
              
        return total_reward, total_steps, is_target_reched, robot_coll, self.total_explorations
