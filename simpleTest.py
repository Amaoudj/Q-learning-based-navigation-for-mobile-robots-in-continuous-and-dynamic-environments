# Make sure to have the server side running in CoppeliaSim: 
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!
import numpy as np
try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time

print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')

    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    res,objs=sim.simxGetObjects(clientID,sim.sim_handle_all,sim.simx_opmode_blocking)
    if res==sim.simx_return_ok:
        print ('Number of objects in the scene: ',len(objs))
    else:
        print ('Remote API function call returned with error code: ',res)

    time.sleep(2)

   # ============= <define hands to various objects in our scene>=======================
      #====================================================================================
    err_code,left_motor=sim.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',sim.simx_opmode_blocking)       
    if err_code != sim.simx_return_ok:
       raise RuntimeError("Could not get handle to the Pioneer_p3dx_leftMotor")

    err_code,right_motor=sim.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',sim.simx_opmode_blocking)       
    if err_code != sim.simx_return_ok:
       raise RuntimeError("Could not get handle to the Pioneer_p3dx_rightMotor")
       
        #robot objet handl to get its position         
    err_code, robot        = sim.simxGetObjectHandle(clientID,'Pioneer_p3dx',sim.simx_opmode_blocking)
    err_code, rob_pos_vrep = sim.simxGetObjectPosition(clientID,robot,-1,sim.simx_opmode_streaming)
    err_code, rob_ori      = sim.simxGetObjectOrientation(clientID,robot,-1,sim.simx_opmode_streaming)
        
        
    #collision hendle
    #collision object
       
    res, robot_collisionID = sim.simxGetCollisionHandle(clientID, "Pioneer_p3dx", sim.simx_opmode_blocking) 
    res, status = sim.simxReadCollision(clientID, robot_collisionID, sim.simx_opmode_streaming)


    #Targets handles, if you want manu targets you can use many 
    err_code, target_handle   = sim.simxGetObjectHandle(clientID,'Target',sim.simx_opmode_blocking)
    err_code, target_pos_vrep = sim.simxGetObjectPosition(clientID,target_handle,-1,sim.simx_opmode_streaming)
 
    #Init Handler for all US sensor of the robot 
    #retrieve sensor arrays and initiate sensors
    sensor_h=[]                  #list for handles
    sensor_data_val=np.array([]) #array for sensor measurements
      
    for x in range(1,16+1):
        errorCode,sensor_handle=sim.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),sim.simx_opmode_oneshot_wait)
        sensor_h.append(sensor_handle) #keep list of handles        
        errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=sim.simxReadProximitySensor(clientID,sensor_handle,sim.simx_opmode_streaming)                
        sensor_data_val=np.append(sensor_data_val,np.linalg.norm(detectedPoint)) #get list of values
 

    # Now retrieve streaming data (i.e. in a non-blocking fashion):
    startTime=time.time()
    sim.simxGetIntegerParameter(clientID,sim.sim_intparam_mouse_x,sim.simx_opmode_streaming) # Initialize streaming
    while time.time()-startTime < 50:
          for x in range(1,16+1):
            errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=sim.simxReadProximitySensor(clientID,sensor_h[x-1],sim.simx_opmode_buffer)                
            sensor_data_val=np.append( sensor_data_val,np.linalg.norm(detectedPoint)) #get list of values
        
          print ('US data x: ',sensor_data_val) 
       
          time.sleep(0.05)

    # Now send some data to CoppeliaSim in a non-blocking fashion:
    sim.simxAddStatusbarMessage(clientID,'Hello CoppeliaSim!',sim.simx_opmode_oneshot)

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
