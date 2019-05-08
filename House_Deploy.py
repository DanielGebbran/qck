#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:55:18 2019

@author: daniel
"""
import socket
import time
import datetime
import HouseBiLevel as BiLvl
import textwrap

class Housing():
    def __init__(self,consumerfile,pricefile):
        self.Starting(consumerfile,pricefile)
        
    def Starting(self,consumerfile,pricefile):
        try:
            self.HouseX = BiLvl.HouseholdModel(consumerfile,pricefile)
        except:
            print('Problem setting sub-problem. Please verify.')
            self.trigger_fail = 1
            # Pass info to trigger self-solving into OPF: trigger_fail == 1
        else:
            self.HouseX.solve()    
            self.trigger_fail = 0
            # Pass up information about results 
        #finally:
            # Pass up information trigger_fail 
            #print('Finished')
            
    def Solver(self):
        self.HouseX.solve()        

if __name__ == '__main__':
    Time_start = time.time()
    MyHouse = Housing('/Py/MyHouse/B2_H2_11_02_2011.csv','/Py/MyHouse/PriceSignal.csv')
    #MyHouse.HouseX.model.xHouse.pprint()
#    for i in MyHouse.HouseX.T_set:
#        if i == 0 or i == 0.5:
#            print (type(i))
#        MyHouse.HouseX.model.c1[i] = 10
    
    #MyHouse.HouseX.model.c1.pprint()
    
    print('Now establishing UDP client...')

    #MyHouse.Solver()
    
    port = 5000; 
    host = '172.16.13.73'; 
    server = (host, port)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        s.bind(('0.0.0.0', 5002))
        print("Binding at 0.0.0.0 successfull!.")
    except:
        print("Could not bind port!")
    
    print ("UDP Client. Using server: ", host, port)
    #message = "Hello"
    #s.settimeout(2.0)
    s.settimeout(10.0)
    
    #print(s.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF))

    iteration_k = 0
    #while True:
    while iteration_k < 15:
    # Send Client to Server
    #print (datetime.datetime.now().time(), "Sending data to server: ", str(message))  
        iteration_k += 1
        #print("Current iteration:", iteration_k)
        MySchedule = " "
        SendSchedule = [" "," "," "," "," "," "," "," "," "," "]    
        message_divs = 0
        
        ## If data is over 4096 bytes, split as following or allocate higher buffer size on server
        for i in MyHouse.HouseX.T_set:
            MySchedule = MySchedule + str(i) + ',' + str(BiLvl.value(MyHouse.HouseX.model.xHouse[i])) + ';'     
            if len(MySchedule)//4050 == 0:
                SendSchedule[message_divs] = MySchedule 
            else:
                MySchedule = str(i) + ',' + str(BiLvl.value(MyHouse.HouseX.model.xHouse[i])) + ';'
                message_divs = message_divs + 1
                SendSchedule[message_divs] = MySchedule
        ## Assuming each of the above has at most 30 characters
    
        #for i in range(len(SendSchedule)):
        for i in range(message_divs+1):
            SendSchedule[i] = SendSchedule[i] + 'k,' + str(iteration_k) + ';Cost,' + str(MyHouse.HouseX.model.cost())
            #print ("Split number:",i,". Sending data to server... ")#, str(SendSchedule[i]))
            s.sendto(str.encode(SendSchedule[i]), server) 
            #print ("Size of data: ", len(str.encode(SendSchedule[i])))
    
        try:
        # Listen to Server
            data, addr = s.recvfrom(4096)
            #print ("<- Received! ")#, data.decode('utf-8'))
            Sub_Prices_1 = data.decode('utf-8').split(sep=';')
            #print('Separating data...')#, Sub_Prices_1)
            
            for i in range(len(Sub_Prices_1)):
                    Aux_Array = Sub_Prices_1[i].split(sep=',')    
                    try:
                        #Price[Sub_xHouse_1[0]] = Sub_xHouse_1[1]
                        MyHouse.HouseX.model.c1[float(Aux_Array[0])] = float(Aux_Array[1])/1000
                    except:
                        None
                        #print (type(Aux_Array[0]), type(Aux_Array[1]))
                        #print ("Problem at", i, "with auxiliary array:", Aux_Array)
                    #else:
                        #print('Including new results to prices was sucessful.')
            #print('Done!')
            #MyHouse.HouseX.model.c1.pprint()
            Time_it = time.time()
            MyHouse.Solver()     
            print("Solved iteration:", iteration_k, ", in", time.time() - Time_it, "seconds.")  
    #        try:
    #
    #        except:
    #            
    #        message = input("-> Type message: ")
        except socket.timeout:
            print ("No data received within 10s. Sleeping for 5s now...")
            #break
            time.sleep( 5 )
        except:
            print("Could not finish re-solving household sub-problem!")
            
    s.close()
    print('Total run time: ' , time.time() - Time_start, ' s.')
