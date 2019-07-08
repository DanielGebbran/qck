#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:52:56 2019

@author: daniel
"""
import socket
import time
import datetime
import RT2_HouseModel as HouseADMM
import textwrap
import os
import sys

class Housing():
    def __init__(self,consumerfile,pricefile):
        self.Starting(consumerfile,pricefile)
        
    def Starting(self,consumerfile,pricefile):
        try:
            self.HouseX = HouseADMM.HouseholdModel(consumerfile,pricefile)
        except:
            print('Problem setting sub-problem. Please verify.')
            self.trigger_fail = 1
        else:
            self.trigger_fail = 0
            # Pass up information about results 
        #finally:
            # Pass up information trigger_fail 
            #print('Finished')
            
    def Solver(self):
        self.HouseX.solve()        
    
    def Init_Solver(self):
        self.HouseX.model.cost.deactivate()
        self.HouseX.model.cost_initial.activate()
        self.HouseX.solve()    
        self.HouseX.model.cost_initial.deactivate()
        self.HouseX.model.cost.activate()
        

if __name__ == '__main__':
    Time_start = time.time()
    #MyHouse = Housing('B1_H1_11_02_2011.csv','PriceSignal.csv')
    #MyHouse = Housing('/media/daniel/HDDfiles/Projects/CommProject/RaspberryPi/Test_2/RPi_4/B4_H4_11_02_2011.csv','/media/daniel/HDDfiles/Projects/CommProject/RaspberryPi/Test_2/RPi_4/PriceSignal.csv')
    MyHouse = Housing('/Py/MyHouse/B3_H3_11_02_2011.csv','/Py/MyHouse/PriceSignal.csv')

    print('Now establishing UDP client...')
    port = 5000
    host = '172.16.13.73'
    server = (host,port)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        s.bind(('0.0.0.0',5004))
        print ("Binding at 0.0.0.0 successful! Server IP:", host)
    except:
        print ("Could not bind port!")
        
    
    ### Starting connection. Required initial handshake for server to acquire IP address and ports from users. # Maybe create a method and use a function using self.s.sendto....
    s.sendto(str.encode("Connected to server!"), server) 
    s.settimeout(5.0)
    Conn_Attempts = 0
    while Conn_Attempts < 10:
        try:
            # Listen to Server
            data, addr = s.recvfrom(1024)
            Conn_Attempts = 11
            if data.decode('utf-8') == "F":
                s.close()
                #os.system('sudo poweroff')
                sys.exit('Input told me to turn OFF!')
            else:
                print ("<- Received confirmation:", data.decode('utf-8'))
        except socket.timeout:
            print("No data received within 5s. Sending data again!")
            s.sendto(str.encode("Connected to server!"), server) 
            Conn_Attempts += 1
        finally:
            if Conn_Attempts == 10:
                print ("No data received within 10 attempts. Sleeping for 10 s...")
                time.sleep(10.0)
                Conn_Attempts == 0
            else:
                None
                
    ## ADMM (outer) Loop ##
    #while 
    
# define a function for this? #############################################################################################################################
    th = 0
    while True:
        th += 1
        ## Stand-by mode # Awaiting order from Aggregator ##
        print ("Stand-by mode --- Awaiting order from Aggregator...")
        s.settimeout(60.0)
        confirm_s = 0
        while confirm_s == 0:
            try:
                # Listen to Server
                data, addr = s.recvfrom(1024)
                confirm_s = 1
            except socket.timeout:
                print("No data received within one minute. Still awaiting...") 
        
        # Verifies if input is 'F'inished
        if data.decode('utf-8') == "F":
            print ("Closing program!")
            s.close()
            #os.system('sudo poweroff')
            sys.exit('Input told me to turn OFF!')
        
        time_start = time.time()
        print ("Routine started!")    
        # Start of routine #
        iteration_k = 1
        finished = 0
        
        # When dealing with different time horizons, change settings from the model!
        MyHouse.Init_Solver()
        Comm_report = " "
        
        while finished == 0:
            MySchedule = " "
            
            ## If data is over 4096 bytes, split as previous files (RPi Test 1) or allocate higher buffer size on server
            for i in MyHouse.HouseX.T_set:
                MySchedule = MySchedule + str(i) + ',' + str(HouseADMM.value(MyHouse.HouseX.model.xHouse[i])) + ';'     
        
            print ("Size of data MUST BE LESS THAN 4050: ", len(str.encode(MySchedule)))
            
            # Sending data function #
            time_send = time.time()
            s.sendto(str.encode(MySchedule), server) 
            ### Wait for confirmation . def confirmation()? # Don't know how to use socket within this part on outside. Maybe create a method and use a function using self.s.sendto....
            s.settimeout(1.0)
            Conn_Attempts = 0
            while Conn_Attempts < 10:
                try:
                    # Listen to Server
                    data, addr = s.recvfrom(1024)
                    Conn_Attempts = 11
                    print ("<- Received confirmation: ", data.decode('utf-8'))
                except socket.timeout:
                    print("No data received within 1s. Sending data again!")
                    s.sendto(str.encode(MySchedule), server) 
                    Conn_Attempts += 1
                finally:
                    if Conn_Attempts == 10:
                        print ("No data received within 10 attempts. Sleeping for 10 s...")
                        time.sleep(10.0)
                        Conn_Attempts == 0
                    else:
                        None
                           
            time_latency = time.time() - time_send 
            Comm_report = Comm_report + str(iteration_k) + ',' + str(time_latency) + ';'
        
            # Listen to server's response (after aggregator solves grid subproblem), higher timeout
            s.settimeout(60.0)
            #s.settimeout(15.0)
            confirm = 0
            while confirm == 0:
                try:
                    # Listen to Server
                    #data, addr = s.recvfrom(4096)
                    data, addr = s.recvfrom(8192)
                    confirm = 1
                    print ("<- Received data!")
                except socket.timeout:
                    print("No data received within timeout. Waiting again!") 
                    
            #...    
            
            ## Preparation to re-solve problem
            
            # Separating input data
            MyInput = data.decode('utf-8').split(sep='L')
            # Verifies if input is 'F'inished
            if MyInput[0] == "F":
                finished = 1
                # Sending data function #
                s.sendto(str.encode(Comm_report), server)
                s.settimeout(1.0)
                Conn_Attempts = 0
                while Conn_Attempts < 10:
                    try:
                        # Listen to Server
                        data, addr = s.recvfrom(1024)
                        Conn_Attempts = 11
                        print ("<- Received confirmation: ", data.decode('utf-8'))
                    except socket.timeout:
                        print("No data received within 1s. Sending data again!")
                        s.sendto(str.encode(Comm_report), server) 
                        Conn_Attempts += 1
                    finally:
                        if Conn_Attempts == 10:
                            print ("No data received within 10 attempts. Sleeping for 10 s...")
                            time.sleep(10.0)
                            Conn_Attempts == 0
                        else:
                            None
                            
                # Verifies if input tells RPi to turn o'F'f 
                # On computer simulation, break outer loop to exit code... ? Outside function?!
                if MyInput[1] == "F":
                    s.close()
                    #os.system('sudo poweroff')
                    sys.exit('Input told me to turn OFF!')
            # If input is not finished, ADMM iteration follows
            else:
                Sub_xH_k = MyInput[0].split(sep=';')
                Sub_Lambda = MyInput[1].split(sep=';')
                # Reassigning household demand and lambda
                for i in range(len(Sub_xH_k)):
                    Aux_Array_H = Sub_xH_k[i].split(sep=',')    
                    Aux_Array_L = Sub_Lambda[i].split(sep=',')    
                    try:
                        MyHouse.HouseX.model.xHouse_k[float(Aux_Array_H[0])] = float(Aux_Array_H[1])
                        MyHouse.HouseX.model.Lambda_xHouse[float(Aux_Array_L[0])] = float(Aux_Array_L[1])
                    except:
                        if Aux_Array_H == ['']:
                            None#print ("This is ok!")
                        else:
                            print ("Error here! Please verify!")
            
                # Re-solving problem: Z kth iteration
                time_z = time.time()
                MyHouse.Solver()
                print("Solution of house subproblem obtained in", time.time()-time_z,"s, iteration number:", iteration_k)
                iteration_k += 1    
                
        # End while finished == 0
        print('Run time: ' , time.time() - time_start, ' s.')        
        print("Finished problem for time horizon number",th,'\n-----------------------------------------\n')
    
