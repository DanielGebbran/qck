#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:19:34 2019

@author: Daniel Gebbran
"""
import socket
import datetime
import time
import RT2_ADMM_5_bus as Aggregator
import sys
import numpy as np
from pprint import pprint
#import pyomo
#import pyomo.opt
from pyomo.environ import *

def trim_keys(Original, Headers):
    return {k:v for k,v in Original.items() if k not in Headers}
def recover_keys(Original, Headers):
    return {v for k,v in Original.items() if k in Headers}
def sortSecond(val): 
    return val[1]  

def Main():
    host = '172.16.13.73'
    port = 5000
    
    server = (host, port)
    
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(server)
    
    Headers = {'k','Cost'}
    CostSet = {'Cost'}
    IterationSet = {'k'}
    xHouse_Schedule = {}
    print('UDP Server Started on host', host, ', port', port)
    #print(s.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF))
    
    House_addr = [] # List to be populated with lists of addresses from houses: [[addr[0] is the IP ,addr[1] is the port], ..., ..., ...]
    AllHomes = 0
    while AllHomes != 4:
        recvdata, addr = s.recvfrom(1024)
        print ("Connection successful with house: ", addr[1]%5000, "using address:", str(addr))
        if addr in House_addr:
            print ("This address is already in file! Closing household through message.")
            Init_message = "F"
        else:
            House_addr.append(addr)
            Init_message = "Conntection established!"
            AllHomes += 1
        s.sendto(Init_message.encode('utf-8'),addr)
    print ("Acquired all addresses from users!", *House_addr, sep = ", ")
    House_addr.sort(key = sortSecond)  
    print("Reordering houses according to address:", House_addr) 
    
    th = 0    
    while True: ## Beginning of whole routine! ## 
        th += 1    
        ## Send Activation to activate Houses Stand-by mode ##
        number = input ("Enter 1 (or anything else) to start routine. Enter 2 to shut off households and exit this program.\n")
        if number == "2":
            Message = "F"
        else:
            Message = "Start!"

        time_start = time.time()
        print ("Passing message out of Stand-by for remote houses 1 to 4: ", end = "")
        for addr in House_addr:
            print (addr[1]%5000, end = " done, ")
            s.sendto(Message.encode('utf-8'),addr)
            
        if number == "2":
            s.close()
            sys.exit('Goodbye!')
        
        print ("Routine started!")    
        ## Start of routine ##
        HousesData, AllHomes = {}, []
        # Listening to xHouse values from 4 homes
        while sum(AllHomes) != 10:
            recvdata, addr = s.recvfrom(4096)
            print ("House", addr[1]%5000, "sent its consumption data.")
            s.sendto(str("Received!").encode('utf-8'),addr) #(confirm receipt of each value)
            if addr[1]%5000 in AllHomes:
                print ("This house already sent data!")
            else:
                HousesData[addr[1]%5000] = recvdata.decode('utf-8')
                AllHomes.append(addr[1]%5000) # Continues until values from all homes have been acquired
        print ("Acquired consumption data from all houses!")
        
        # Create network model from Aggregator
        AggregatorProblem = Aggregator.SubProblem('/media/daniel/HDDfiles/Projects/CommProject/RaspberryPi/Test_2/RPi_5_Aggr/1_4_200_298-11_02_2011_B.csv','/media/daniel/HDDfiles/Projects/CommProject/RaspberryPi/Test_2/RPi_5_Aggr/PriceSignal.csv')
        
        #Assign xHouse_k[i,k] according to inputs!
        for k in AggregatorProblem.H_set:
            Sub_xH_k = HousesData[k].split(sep=';')
            #pprint(Sub_xH_k)
            for i in range(len(Sub_xH_k)):
                #print("OKsf", end = " ")
                Aux_Array_H = Sub_xH_k[i].split(sep=',')      
                try:
                    #print (float(Aux_Array_H[0]), k, float(Aux_Array_H[1]), end = "  |  ")
                    AggregatorProblem.X.xHouse_k[float(Aux_Array_H[0]),k] = float(Aux_Array_H[1])
                except:
                    if Aux_Array_H == ['']:
                        None#print ("This is ok!")
                    else:
                        print ("Error here! Please verify!")
                    
        ### Insert ADMM routine HERE ###
        iteration_k = 1
        g_k_ro = 10
        g_register, X_V_E, X_V_F = [], [], []
        while g_k_ro > 0.0001:
            print ("Iteration:", iteration_k)
            g_step = []
            time_x = time.time()
            AggregatorProblem.solve() # Solves only X subproblem now!
            print("Solution of aggregator subproblem obtained in", time.time()-time_x,"s.")
            
            # Send new Lambda (Initial Lambda = 0) and xHouse_i information for all houses
            for k in AggregatorProblem.H_set:
                LambdaUpdate, HouseUpdate = "", ""
                for i in AggregatorProblem.T_set:
                    HouseUpdate = HouseUpdate + (str(i) + ',' + str(value(AggregatorProblem.X.xHouse_i[i,k])) + ';')
                    LambdaUpdate = LambdaUpdate + (str(i) + ',' + str(value(AggregatorProblem.X.Lambda_xHouse[i,k])) + ';')
                
                Message = HouseUpdate + 'L' + LambdaUpdate
                s.sendto(str(Message).encode('utf-8'),House_addr[k-1])
                #print("Sending data to house:", k, "on address:", House_addr[k-1],"\n\n", HouseUpdate)
            
            # Listening to xHouse values from 4 homes
            AllHomes = []
            while sum(AllHomes) != 10:
                recvdata, addr = s.recvfrom(4096)
                print ("House", addr[1]%5000, "sent its consumption data.", end = " ")
                s.sendto("New iteration data received!".encode('utf-8'),addr) #(confirm receipt of each value)
                #s.sendto(str("Received!").encode('utf-8'),addr) 
                if addr[1]%5000 in AllHomes:
                    print ("This house already sent data!")
                else:
                    HousesData[addr[1]%5000] = recvdata.decode('utf-8')
                    AllHomes.append(addr[1]%5000) # Continues until values from all homes have been acquired
                
            print ("Acquired consumption data from all houses!")
            
            # Update Lambda and X (local) values, calculates g_k_ro
            for k in AggregatorProblem.H_set:
                Sub_xH_k = HousesData[k].split(sep=';')
                #LambdaUpdate, HouseUpdate = [], []
                for i in range(len(Sub_xH_k)):
                    Aux_Array_H = Sub_xH_k[i].split(sep=',')      
                    try:
                        AggregatorProblem.X.xHouse_k[float(Aux_Array_H[0]),k] = float(Aux_Array_H[1])
                        residual = value(AggregatorProblem.X.xHouse_i[float(Aux_Array_H[0]),k]) - float(Aux_Array_H[1])
                        g_step.append(residual)
                        AggregatorProblem.X.Lambda_xHouse[float(Aux_Array_H[0]),k] += AggregatorProblem.Ro*residual    
                    except:
                        if Aux_Array_H == ['']:
                            None#print ("This is ok!")
                        else:
                            print ("Error here! Please verify!")
            
            
            g_k_ro = np.linalg.norm(np.asarray(g_step))
            print("Residual norm:", g_k_ro)
            # Registers values
            X_V_E.append(round(value(AggregatorProblem.X.xV_E_ij[23.5,4]),2))
            X_V_F.append(round(value(AggregatorProblem.X.xV_F_ij[23.5,4]),2))
            g_register.append(g_k_ro)
            iteration_k = iteration_k + 1
            
        print('---------------------------------\nSolutions achieved in ' ,time.time() - time_start, ' s. \n',X_V_E, '\n',X_V_F, '\n', ' \n')#Saving results of decreasing g...')
        #np.savetxt("g_all_iterations", np.array(g_register))
        print('Total number of iterations:', iteration_k-1, ', g_k_ro:', g_k_ro, '\nFinal cost:', value(AggregatorProblem.X.Xsub))
        print('End of ADMM optimization.') 
        
        # Send information: finished this time horizon!
        Message = "FL"
        # Message = "FLF" # turns off houses # Don't forger s.close() if implementing a server shutdown not related to user input defined above on 'while True'.
        for k in AggregatorProblem.H_set:
            s.sendto(str(Message).encode('utf-8'),House_addr[k-1])
            
        # Receive report about communication delays
        # Listening to xHouse values from 4 homes
        AllHomes = []
        while sum(AllHomes) != 10:
            recvdata, addr = s.recvfrom(4096)
            print ("House", addr[1]%5000, "sent its communication data.", end = " ")
            s.sendto("Communication data received!".encode('utf-8'),addr) #(confirm receipt of each value)
            #s.sendto(str("Received!").encode('utf-8'),addr) 
            if addr[1]%5000 in AllHomes:
                print ("This house already sent data!")
            else:
                HousesData[addr[1]%5000] = recvdata.decode('utf-8')
                AllHomes.append(addr[1]%5000) # Continues until values from all homes have been acquired
            
        print ("Acquired consumption data from all houses!")
        #(confirm receipt of each value until values from all homes have been acquired)
            
        
        print('Run time: ' , time.time() - time_start, ' s.')        
        print("Communications report:")
        pprint(HousesData)
            
        print('\n-----------------------------------------\n')
        print("Finished problem for time horizon number",th,'\n*******************************************************\n')
        

        
if __name__ == "__main__":
    Main()
    
    
    
           
#        s.settimeout(60.0)
#        confirm_s = 0
#        while confirm_s == 0:
#            try:
#                # Listen to Server
#                data, addr = s.recvfrom(1024)
#                confirm_s = 1
#            except socket.timeout:
#                print("No data received within one minute. Still awaiting...") 
#        
#        
#        # Receiving data from open ports
#        #recvdata, addr = s.recvfrom(1024)
#        recvdata, addr = s.recvfrom(4096)
#        
#        print (datetime.datetime.now().time(), "<- Received from: ", str(addr))
#        #print ("<- Received from: ", str(addr))
#        #print ("Data: ", recvdata.decode('utf-8'))
#        
#        if addr[1] > 1000:
#            Schedule_1 = {}
#            #PriceUpdate = []
#            PriceUpdate = ""
#            Sub_Schedule_1 = recvdata.decode('utf-8').split(sep=';')
#            #print(Sub_Schedule_1)
#            
#            for i in range(len(Sub_Schedule_1)):
#                Sub_xHouse_1 = Sub_Schedule_1[i].split(sep=',')    
#                try:
#                    Schedule_1[Sub_xHouse_1[0]] = Sub_xHouse_1[1]
#                except:
#                    print ("Failure 0")
#            #print('Converting to dictionary:', Schedule_1)
#            
#            
#            xHouse_Schedule[addr[1]%1000] = trim_keys(Schedule_1, Headers)
#            
#            try:
#                for thisset in recover_keys(Schedule_1, IterationSet):
#                    Iteration_k = thisset 
#                for thisset in recover_keys(Schedule_1, CostSet):
#                    ThisCost = thisset    
#                print("Household:",addr[1]%1000,"Iteration", Iteration_k,"Cost",ThisCost)
#            except:
#                None
#                        
#            for i in xHouse_Schedule[addr[1]%1000]:
#                #PriceUpdate = PriceUpdate + (str(i) + ',' + str(float(i)*0.12*float(Iteration_k)) + ';')
#                PriceUpdate = PriceUpdate + (str(i) + ',' + str(uniform(0,1)*float(Iteration_k)) + ';')
#                #PriceUpdate.append(str(i) + ',' + str(0.12) + ';')            
#            #print('New price update:', PriceUpdate)
#            #print('\n\nxHouse schedule:', xHouse_Schedule)
#
#            #time.sleep(uniform(0,5))            
#            try:
#                s.sendto(str(PriceUpdate).encode('utf-8'),addr)
#                #print('Sending data with string method encoding.')
#            except:
#                print('Failure enconding and/or sending message.')
#
#        else:
#            try:
#                senddata = float(recvdata.decode('utf-8'))/450
#                time.sleep(0.005)
#            except:
#                senddata = "Invalid data received."
#    
#            print (datetime.datetime.now().time(), "-> Sending messages: ", th, str(senddata))
#            try:
#                s.sendto(str(senddata).encode('utf-8'),addr)
#                print('Sending data with string method encoding.')
#            except:
#                #s.sendto(str.encode(senddata),addr)
#                #print('Second: new string encoding for numbers.')
#                print('Failure enconding and/or sending message.')
#
#        if recvdata.decode('utf-8') == "KILL":
#            print ("Requested Server Shutdown")
#            break