# -*- coding: utf-8 -*-
"""@authors: Aberraouf MAOUDJ <abma@mmmi.sdu.dk>, Anders Lyhne Christensen <andc@mmmi.sdu.dk>
"""
import sys
import configuration
import os
import logging
from   agent   import  agent_
import matplotlib.pyplot as plt
#from  matplotlib.animation import FuncAnimation
from   VRep_Env import V_Rep_Env
import numpy as        np


"""from matplotlib import style
import time
style.use("ggplot")
"""
x_vals = []
AVRrewar = []
AVRStep  =[]

    
    
def plot_results(self, V1,V2):
   
   f, (ax1, ax2) = plt.subplot(nrows=1, ncols=2) #122) 
   #plt.axis([1,400  , 0, max(steps) +8])
   ax1.plot(np.range(len(V1)), V1, 'b')
   ax1.set_xlabel('Episodes')
   ax1.set_ylabel('V1')
   ax1.set_title('Episode Via Steps')
   #Example of two plot on the same axe
   #plt.axis([1,400  , 0, max(steps) +8])
   ax2.plot(np.range(len(V2)), V2, 'r')
   ax2.plot(np.range(len(V1)), V1, 'b')
   ax2.set_xlabel('Episodes')
   ax2.set_ylabel('Average Steps')
   ax2.set_title('Episode Via Reward')
   ax2.legend(('V1', 'V2'),loc='upper right')
   #
   plt.tight_layout()
   #
   #plt.axis([1,400  , 0, max(steps) +8])
   plt.figure()
   plt.plot(np.range(len(V1)), V1, 'b')
   plt.title('Episode Via Steps')
   plt.ylabel('V1')
   plt.xlabel('Episodes')
   
   plt.figure()
   plt.plot(np.range(len(V1)), V1, 'b')
   plt.plot(np.range(len(V2)), V2, 'r')
   plt.plot(V1, 'r')
   plt.plot(V2, 'b')    
   plt.title('Episode Via Reward')    
   plt.ylabel('Average Steps')
   plt.xlabel('Episodes')

                       
   plt.show()
       
#Please choose the execution mode :
"""if   you choose: 'RunTask', so make train_mode = False
   elif you choose 'Train', make  train_mode      = True
"""

#==================================
#==================================
train_mode        = False#True  
continue_training = True  : Means, it's the first runing => init Q-table by Algo
#==================================
#==================================

"""print ("Please choose the running Mode  : put True for Training mode , Fales for execution Mode ? " )
train_mode = input()

if train_mode:
    print ("Please choose if you want continue training or it's first time : Yes for continue raining ?" )
    continue_training = input()
    print ("You are choosed Training Mode with continuation = ", continue_training )
else: 
    print ("You are choosed  the execution Mode" )
"""

vrep_ip = '127.0.0.1'
vrep_port = 19997
#add this in V-Rep scene
#simExtRemoteApiStart(19997)

env = V_Rep_Env(vrep_ip, vrep_port)

# Load thf fil configuration them ceate the agent
if not os.path.exists(configuration.Q_TABLE_Path):
    os.makedirs(configuration.Q_TABLE_Path)

""" We use the variable 'Agent_id' when we use multi-robot system"""
Agent_id=1  
agent = agent_(env,Agent_id,train_mode,continue_training, epsilon=configuration.EPSILON, q_init_val=configuration.Q_INIT_VAL,discount=configuration.DISCOUNT, learn_rate=configuration.LEARN_RATE)
episodes = configuration.NUM_EPISODES


def status(title):
    print("")
    print(title)
    print("Is Target reached > ", env.robot.is_Target_reached())
    print("------------------------> ")
    
print("====================< Main fuction: starts >==================")
print("Num_state : " + str(env.total_states))
print("Num_actions : " + str(env.total_actions))
print("Max Episodes: " + str(configuration.NUM_EPISODES))
print("Max Actions/Episodes: " + str(configuration.NUM_MAX_ACTIONS))
print("Discount: " + str(configuration.DISCOUNT))
print("Learning Rate: " + str(configuration.LEARN_RATE))
print("Epsilon: " + str(configuration.EPSILON))

print("Epsilon Decay: " + str(configuration.EPSILON_DECAY))


if train_mode:
    # In this mode, we train the agent and generate the qtable that will be dumped in qtables/qtable.txt.npy
    """
    # use this if you want to limite training time 
    t = time.time()
    while (time.time()-t) < 3600: #  --> 60 min (1h)   
    """
    steps = []
    cost  = []
    succes= []
    plot_var=0  
    cont=0   
    ind=0
    print('====================< Training the robot==================>')
    avr=[]
    avstep=[]
    x=0
    
    if continue_training:
        agent.load_qtable_training()
    #else: # new training phase 
        #agent.init_qtable_propos_Alg() # initialize Q-Table by using the proposed Algorithms
       
    while episodes > 0:
        
        agent.reset()
        print('============================================================>')
        print('====================< Episode : ' + str(cont)+'==========================>')
        print('============================================================>')
                
        total_reward, total_steps, success,robot_collision, total_explorations = agent.execute_episode_qlearn(configuration.NUM_MAX_ACTIONS)
        #if success:
        steps.append(total_steps)
        cost.append(total_reward)
       
        if success:
           succes.append(1)
        else:
           succes.append(0) 
        
        episode_num    = configuration.NUM_EPISODES - episodes + 1        
        # Reduce exploration rate with each episode
        agent.epsilon   = np.maximum(0.0, configuration.EPSILON - episode_num * configuration.EPSILON_DECAY)
        if agent.learn_rate > 0.1:
            agent.learn_rate= np.maximum(0.0, configuration.LEARN_RATE - episode_num * configuration.LEARN_RATE_DECAY)
        else:
            agent.learn_rate = 0.1    
       
        print('Learning Rate = ', agent.learn_rate)
        print('epsilon = ',       agent.epsilon )
            
        stat_file = open(configuration.PLOT_FILE, 'a')       
        stat_file.write("{0},{1},{2},{3},{4},{5}\n".format(episode_num, success, total_reward,total_steps, total_explorations, agent.epsilon))
        stat_file.close()
        agent.save_qtable()        
        
        echal=20 # to calculate the average value of reward and steps (each 30 episodes)
        if success:
            ind += 1
            avr.append(total_reward)
            avstep.append(total_steps)
        
        if ind==echal :           
            x += echal
            x_vals.append(x)           
            ind =0
            val =0
            v =0
            
            for i in avr:
               val += i
            avr=[]
            for j in avstep:
                v += j
            
            avstep=[]
            r =val/echal
            s = v/echal
            
            AVRrewar.append(r)
            AVRStep.append(s)
            f2=open(configuration.PLOT_AVR_R, 'a')
            f2.write("{0},{1},{2}\n".format(x,r,s)) # save index x
            f2.close()
            """
            # ------------------
            ani = FuncAnimation(plt.gcf(), animate, interval=1000)

            plt.legend()
            plt.tight_layout()
            plt.show()"""
         
        episodes -= 1
        cont     += 1
              
        if robot_collision :
            print('Simulation failled. Terminating ....')
        if success and total_steps == configuration.MIN_ACTIONS_EXPECTED \
                and total_explorations == 0 and episode_num > configuration.MIN_EPISODES_TO_RUN:
            print('MiR100 Optimal moves learnt. Terminating training. Now run agent with epsilon as 0.')
            break
       
        if cont - plot_var == 500: #plot results: total_reward and total_steps per episode each 50 episides
           plot_var=cont   
           

           directory  = os.path.dirname(__file__)
           results_dir = os.path.join(directory, 'Results_plot/')
           file_name = str(cont)  
           
           fig = plt.figure(1)
           
           
           plt.subplot(211) #121)           
           plt.axis([1, cont + 10 , 0, max(steps) +10])
           plt.plot(steps, 'b')
           plt.title('Plot: Steps VS Episode, Cos VS Episode')
           plt.ylabel('steps per episode')
           #plt.xlabel('episode')
                     
           plt.tight_layout()  # make distance between figures           
           
           plt.subplot(212) #122)
           plt.axis([1,cont + 10 , min(cost), max(cost) +10])
           plt.plot(cost, 'r')
           #plt.title('Reward VS Episode')
           plt.ylabel('Cost per episode')
           plt.xlabel('episode')
                     
           plt.show()                        
           fig.savefig(results_dir + file_name + '.png', dpi=150, bbox_inches='tight') #save this in 'results_dir'  
else:
    print('====================>  Executing Task ....')
    # Here, exucute the task
    agent.reset2()
    
    total_reward, total_steps, success,robot_coll, total_explorations = agent.execute_Task(configuration.NUM_MAX_ACTIONS)
    msg = "The agent was tested with a learnt Q-Table with the following results:\n"
    msg += "Success: {}\n".format(success)
    msg += "Total Reward: {}\n".format(total_reward)
    msg += "Total Steps: {}\n".format(total_steps)
    
    
    print(msg)
