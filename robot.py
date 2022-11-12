# -*- coding: utf-8 -*-
"""@authors: Aberraouf MAOUDJ <abma@mmmi.sdu.dk>, Anders Lyhne Christensen <andc@mmmi.sdu.dk>
"""
import numpy as np
import os
import math
import time
import usedFunction
import configuration
import numpy as np
import sim
import ctypes
import matplotlib.pyplot as plt

class MiR100(object):
   
    def __init__(self, ip, port):
        """ Initializes connectivity to V-REP environment, and environmental variables.
        """
        
        
       # sim=remApi('remoteApi')       
        sim.simxFinish(-1)
        self.clientID = sim.simxStart(ip, port, True, True, 5000, 5)
        if self.clientID != -1:
            print('OK: Connected to remote API of V-REP')
        else:
            raise RuntimeError('Could not connect to V-REP')

        # Some artificial delays to let V-REP stabilize after an action
        self.sleep_sec     = configuration.UNIT_STEP_SIZE
        self.sleep_sec_min = configuration.SLEEP_VAL_MIN

        # id of the MiR100 mobile Robot in V-REP environment
        self.main_object = 'MiR'
        # Initial state of the Gripper
        self.path=[]
        self.pos = 0
        self.n_episode=0
        # ============= <define objects >=======================
        #====================================================================================#0
        err_code,self.left_motor=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_leftMotor',sim.simx_opmode_blocking)       
        if err_code != sim.simx_return_ok:
            raise RuntimeError("Could not get handle to the Pioneer_p3dx_leftMotor")

        err_code,self.right_motor=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_rightMotor',sim.simx_opmode_blocking)       
        if err_code != sim.simx_return_ok:
            raise RuntimeError("Could not get handle to the Pioneer_p3dx_rightMotor")
       
        #robot objet handler to get its position         
        err_code, self.robot        = sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',sim.simx_opmode_blocking)
        err_code, self.rob_pos_vrep = sim.simxGetObjectPosition(self.clientID,self.robot,-1,sim.simx_opmode_streaming)
        err_code, self.rob_ori      = sim.simxGetObjectOrientation(self.clientID,self.robot,-1,sim.simx_opmode_streaming)
        
        
        #collision handler 
        #collision object
       
        res, self.robot_collisionID = sim.simxGetCollisionHandle(self.clientID, "Pioneer_p3dx", sim.simx_opmode_blocking) 
        res, status = sim.simxReadCollision(self.clientID, self.robot_collisionID, sim.simx_opmode_streaming)


        #Targets handler 
        err_code, self.target_handle   = sim.simxGetObjectHandle(self.clientID,'Node1',sim.simx_opmode_blocking)
        err_code, self.target_pos_vrep = sim.simxGetObjectPosition(self.clientID,self.target_handle,-1,sim.simx_opmode_streaming)
 
        #Init Handler for all US sensor of the robot 
        #retrieve sensor arrays and initiate sensors
        self.sensor_h=[]                  #list for handles
        self.sensor_data_val=np.array([]) #array for sensor measurements
        ## US sensores handlers
        for x in range(1,14):
            errorCode,sensor_handle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),sim.simx_opmode_oneshot_wait) #+'#0'
            self.sensor_h.append(sensor_handle) #keep list of handles        
            errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=sim.simxReadProximitySensor(self.clientID,sensor_handle,sim.simx_opmode_streaming)                
            self.sensor_data_val=np.append(self.sensor_data_val,np.linalg.norm(detectedPoint)) #get list of values
 
       # self.backUS=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_ultrasonicSensor12',vrep.simx_opmode_blocking)
       # errorCode,detectionState,detBackObs,detectedObjectHandle,detectedSurfaceNormalVector= vrep.simxReadProximitySensor(self.clientID,self.backUS,vrep.simx_opmode_streaming)
        
        #Initiating LASER
        sim.simxGetStringSignal(self.clientID,'measuredDataAtThisTime',sim.simx_opmode_streaming)
 
        # this values can be obtained after the simulation starts        
        self.Target_position = None
        self.objects = None
          
    def __del__(self):
        print('Disconnecting from V-REP')
        self.disconnect()
 
    #read Back sensor
    def get_Obs_State(self):
        sensor_val = np.array([])  
        US_F, US_R, US_L,U_RightSide,U_leftSide=0,0,0,0,0
        
        for x in range(1,14):          
            errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=sim.simxReadProximitySensor(self.clientID,self.sensor_h[x-1],sim.simx_opmode_buffer)                
            sensor_val= np.append(sensor_val,np.linalg.norm(detectedPoint))
               
        US_F  = min(sensor_val[3:5])   # US4, US5
        US_R  = min(sensor_val[5:7])   # US6, US7
        US_L  = min(sensor_val[1:3])   # US2, US3
        U_RightSide= sensor_val[7]     # US8
        U_leftSide = sensor_val[0]     # US1
        
        if round(U_RightSide*100) == round(US_R*100): # wrong mesure
            U_RightSide = 4
            
            
        sensor=np.array([ round( US_F*100), round(US_R*100), round(US_L*100),round(U_RightSide*100),round(U_leftSide*100)])                     
 
    
        return sensor
    
    def get_Obs_Risky_R(self):
        sensor_val=np.array([])
        US_F = 0
        riskDis=False
        
        for x in range(1,14):          
            errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=sim.simxReadProximitySensor(self.clientID,self.sensor_h[x-1],sim.simx_opmode_buffer)                
            sensor_val= np.append(sensor_val,np.linalg.norm(detectedPoint))
               
        US_F  = min(sensor_val[3:5])   # US4, US5
        
        US  = min(round(US_F*100),round(100*sensor_val[2]), round(100*sensor_val[5]))

        return US
  
    """ To get the LMS scan values 
    """           
    def get_LaserScan(self):
        #get LMS Scan on the robot
        US_F, US_R, US_L=0,0,0
        result,self.ScanData = sim.simxGetStringSignal(self.clientID,'measuredDataAtThisTime',sim.simx_opmode_buffer);   
 
        if result == sim.simx_return_ok :
           
           laserData  = sim.simxUnpackFloats(self.ScanData)
           laserDataX=[None] * int (len(laserData)/2)
           laserDataY= [None] * int (len(laserData)/2)
           theta     = [None] * len(laserDataX)
           #print("len DaTa : " ,len(laserData))  # len= 1368
           ind=  0
           for i in range(0, len(laserData)-1):
               if i%2 == 0:
                  laserDataX[ind] = laserData[i]  #laserData[1:2:end-1] 
                  ind=ind+1
           ind=  0
           for i in range(0, len(laserData)):
               if i%2 != 0:
                  laserDataY[ind] = laserData[i]   
                  ind=ind+1          
          # print("len X : " ,len(laserDataX))
          # print("len Y : " ,len(laserDataY))
          # print("size : " , np.size(laserDataY))
                     
           for i in range(len(laserDataY)):
               self.theta[i] = math.atan2(laserDataY[i], laserDataX[i])        #  
               self.ObsDistances[i] = laserDataX[i] / math.cos(self.theta[i])  #    
           
           inRangeIdx  = self.theta[self.theta *(180/math.pi) <= 28] 
           theta_ront  = self.theta(inRangeIdx);
           dist_front  = self.ObsDistances(inRangeIdx)              
       
           inRangeIdx  = self.theta_ront[theta_ront*(180/math.pi) >= -28]
           theta_ront  = theta_ront(inRangeIdx)
           dist_front  = dist_front(inRangeIdx)             
           US_F        =np.minimum(dist_front)
        
           inRangeIdx   = self.theta[self.theta*(180/math.pi) < -40]       
           theta_Right  = self.theta(inRangeIdx)
           dist_Right   = self.ObsDistances(inRangeIdx)              
       
           inRangeIdx   = self.theta_Right[theta_Right *(180/math.pi) >= -65]
           theta_Right  = theta_Right(inRangeIdx)
           dist_Right   = dist_Right(inRangeIdx)            
           US_R         = np.minimum(dist_Right)
               
           inRangeIdx  = self.theta[self.theta*(180/math.pi) < 65] 
           theta_Left  = self.theta(inRangeIdx)
           dist_Left   = self.ObsDistances(inRangeIdx) 
                
           inRangeIdx  = self.theta_Left[theta_Left*(180/math.pi) > 40]
           theta_Left  = theta_Left(inRangeIdx)
           dist_Left   = dist_Left(inRangeIdx)           
           US_L        = np.minimum(dist_Left)                     
        
        
           sensor_data=np.array([round( US_F*100), round(US_R*100), round(US_L*100)])  # en cm arounded
          
           x=[]
           y=[] 
           global is_plot
           while is_plot:
               plt.figure(1)
               plt.cla()
               plt.ylim(-9000,9000)
               plt.xlim(-9000,9000)
               plt.scatter(x,y,c='r',s=8)
               plt.pause(0.001)
               plt.close("all")
    
                
               is_plot = True
               
           for _ in range(360):
              x.append(0)
              y.append(0)
          
           t=time.time()          
           while (time.time() - t) < 30: #scan for 30 seconds
               #data = next(gen)
               for angle in range(0,360):
                 if(laserData[angle]>1000):
                    x[angle] = laserData[angle] * math.cos(math.radians(angle))
                    y[angle] = laserData[angle] * math.sin(math.radians(angle))
           is_plot = False
        
        return sensor_data
        
          
    def disconnect(self):
        self.stop_sim()
        sim.simxGetPingTime(self.clientID)
        sim.simxFinish(self.clientID)
   
    def send_Point_drawing(self):     
        sleep_time = 0.07
        for i in self.path:
           packedData=sim.simxPackFloats(i.flatten())
           raw_bytes = (ctypes.c_ubyte * len(packedData)).from_buffer_copy(packedData) 
           returnCode= sim.simxWriteStringStream(self.clientID, "path_coord", raw_bytes, sim.simx_opmode_oneshot)
           time.sleep(sleep_time)
           
        
    def get_position(self, handle_):
        """return position of any given object handle
        """
        er,pos = sim.simxGetObjectPosition(self.clientID, handle_,-1,sim.simx_opmode_buffer)            
        
        return pos
   
    def get_oriention(self, handle_):
        """return orientation of any given object handle
        """
        err_code, robot_ori = sim.simxGetObjectOrientation(self.clientID, handle_,-1,sim.simx_opmode_buffer )
        
        return robot_ori[2]     # around 'Z' Axis

    def get_robot_oriention(self):
        """return orientation of robot
        """
        robot_p    = self.get_oriention(self.robot)
        thit_rob     = robot_p*(180/math.pi)
        
        return thit_rob 
    def get_Thita_robot(self): 
        robot_p    = self.get_oriention(self.robot)
        thit_rob     = robot_p*(180/math.pi)
        return thit_rob
    def get_oriention_rad(self, hend):     
        err_code, robot_ori = sim.simxGetObjectOrientation(self.clientID, hend,-1,sim.simx_opmode_buffer )      
        return robot_ori[2]
    
    def get_Thita_Target(self, handle_):  # to be verified        
        target_pos = self.get_position(self.target_handle)
        robot_p    = self.get_position(self.robot)
        thit_targ       =math.atan2((target_pos[1]-robot_p[1]),(target_pos[0]-robot_p[0]))*(180/math.pi) 
        return thit_targ

    def get_env_dimensions(self):
        # Dimensions of the virtual space within which the MiR mobile robot Move 
        # X min, X max; Y min, Y max
        dim = np.array([configuration.xmin,configuration.xmax,configuration.ymin, configuration.ymax])
        return dim

    def Move_robot(self, vr, vl):
        """Maneuvers the mobile base with given speeds.
        """
       # vrep.simxPauseCommunication(self.clientID,1);         
        sim.simxSetJointTargetVelocity(self.clientID,self.left_motor,vl,sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(self.clientID,self.right_motor,vr,sim.simx_opmode_streaming)
       #vrep.simxPauseCommunication(clientID,0);   
        time.sleep(self.sleep_sec )  # suspende 0.1*2 s (200ms)
 
   
    def Move_robot_V_W(self, v_des,omega_des):
        d = 0.331 #wheel axis distance
        r_w = 0.09751 #wheel radius    
        v_r = (v_des+d*omega_des)
        v_l = (v_des-d*omega_des)
        omega_right = v_r/r_w
        omega_left = v_l/r_w
       
        #time.sleep(self.sleep_sec)
        sim.simxSetJointTargetVelocity(self.clientID,self.left_motor,omega_left,sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(self.clientID,self.right_motor,omega_right,sim.simx_opmode_streaming)    
        
        time.sleep(self.sleep_sec)  # suspende 0.1*2 s (200ms)
        
    def Stop_robot(self):
        #time.sleep(self.configuration.SLEEP_VAL_MIN)
        sim.simxSetJointTargetVelocity(self.clientID,self.left_motor,0, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(self.clientID,self.right_motor,0, sim.simx_opmode_streaming)
    

    def start_sim(self):
        time.sleep(0.8) #self.sleep_sec_min
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
        
    #fuction to get distance between two points   
    def get_distance(points1, points2):
        return np.sqrt(np.sum(np.square(points1 - points2), axis=1))

    def stop_sim(self):
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)

    def restart_sim(self):
        self.stop_sim()
        self.start_sim()
    
    # restar with new position of robot and target
    def restart_sim_newEpisode(self):
        #self.stop_sim()     
        #sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot) 
        #self.Stop_robot()   
       
        #self.pos += 1    
        #n_episode += 1
        """newTarget_pos_vrep=np.append(-2.34,3.28)
        
        if self.pos == 1 :#or self.pos== 4 or self.pos==10 or self.pos==12 or self.pos==8:
            newTarget_pos_vrep=np.append(1.58,-5.69)
            
            
        elif self.pos==2:# or self.pos==5 or self.pos==7 or self.pos==9:
            newTarget_pos_vrep=np.append(2.05,-2.34)
            
        elif self.pos==3:
            newTarget_pos_vrep=np.append(-1.46,-1.59)  
        elif self.pos==4:
            newTarget_pos_vrep=np.append(-4.46,5.49)  
        elif self.pos==5:
            newTarget_pos_vrep=np.append(-1.94,-1.74)        
        elif self.pos==6:
            newTarget_pos_vrep=np.append(-6.74,3.73)  
        elif self.pos==7:
            newTarget_pos_vrep=np.append(-2.19,3.48)              
        elif self.pos==8:
            newTarget_pos_vrep=np.append(2.25,3.28)           
        elif self.pos==9:
            newTarget_pos_vrep=np.append(5.51,6.3)     
        elif self.pos==10:
            newTarget_pos_vrep=np.append(1.35,5.18) 
        elif self.pos==11:
            newTarget_pos_vrep=np.append(-1.34,1.15)  
        
        self.pos = self.pos + 1     
        if  self.pos==12:
            self.pos==0"""
            
        xmin = -6.8 
        xmax = -5.6
        if self.pos > 1000 :
            xmin = -5.65
            xmax = -4.65
        if np.random.uniform() < 0.5:
           xmin = 5.16
           xmax = 6.7
           
           
        ymin = -5.8
        ymax =  5.8   
               
        ##############  
        
        """xmin = -2
        xmax = 2
        ymin = -2.5
        ymax = 0.5"""
        ##############       

          
        N = 1
                   
        max_val, min_val = xmax,xmin 
        range_size = (max_val - min_val)  
        x=np.random.rand(N) * range_size + min_val       
        
        max_val_, min_val_ = ymax, ymin 
        range_size = (max_val_ - min_val_)  
        y=np.random.rand(N) * range_size + min_val_
  
        
        newTarget_pos_vrep=np.append(x[0],y[0])
        
        
        
        sim.simxSetObjectPosition(self.clientID,self.target_handle,-1, newTarget_pos_vrep, sim.simx_opmode_oneshot);        
        
 
        time.sleep(0.1) #self.sleep_sec_min
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
        #self.start_sim()
    
    
    
    def set_newEpisode(self,x,y):
                
        newTarget_pos_vrep=np.append(x,y)       
        sim.simxSetObjectPosition(self.clientID,self.robot,-1, newTarget_pos_vrep, sim.simx_opmode_oneshot);        
        
        time.sleep(0.1) #self.sleep_sec_min
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
        #self.start_sim()
        
    def get_Dist_Target(self):
        target_pos = self.get_position(self.target_handle)
        robot_pos  = self.get_position(self.robot)
        
        pos_robot=np.array([robot_pos[0],robot_pos[1]])
        self.path.append(pos_robot)
        #self.send_Point_drawing()
        #distance to target
        d = math.sqrt((target_pos[0]-robot_pos[0])**2+(target_pos[1]-robot_pos[1])**2) # m 

        return round(d*100)  # en cm 
            
        
    def is_Target_reached(self):
        """
            Based on position of the MiR100 robot and target, calculates if the distance is zero.
        """
        target_pos = self.get_position(self.target_handle)
        robot_pos  = self.get_position(self.robot) 

        d = math.sqrt((target_pos[0]-robot_pos[0])**2 + (target_pos[1]-robot_pos[1])**2 ) # m 

        if round(d*100) <= configuration.TOLERANCE:
            return True
            
        return False
    
    def is_robot_collided(self):                
        
        colled = False
        obss=self.get_Obs_State()
        obs=obss[0:3]
        print ("")
        print (" Obstacle states :" ,obs)
        for i in obs :
            if i <= 10 and i > 5: 
              colled=True      
              
        return colled
