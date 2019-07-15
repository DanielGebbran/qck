#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:52:56 2019

@author: daniel
"""

import HouseBiLevel as BiLvl

#House1 = BiLvl.HouseholdModel('/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/HouseholdsData/B1_H1_11_02_2011.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
#House1.solve()
#House2 = BiLvl.HouseholdModel('/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/HouseholdsData/B2_H2_11_02_2011.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
#House2.solve()
#House3 = BiLvl.HouseholdModel('/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/HouseholdsData/B3_H3_11_02_2011.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
#House3.solve()
#House4 = BiLvl.HouseholdModel('/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/HouseholdsData/B4_H4_11_02_2011.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
#House4.solve()

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


#HouseClass1 = Housing('/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/HouseholdsData/B1_H1_11_02_2011.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
#HouseClassX = {}
#for i in range (4):
#    HouseClassX[i] = Housing('/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/HouseholdsData/B1_H1_11_02_2011.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
#    
#HouseClassX[3].Solver()


#if HouseClass1.trigger_fail == 0:
#    HouseClass1.Solver()
#HouseClass1.Solver()
#for i in range(10):
#    HouseClass1.HouseX.model.c1[i] = 5000
#
#HouseClass1.Solver()
#HouseClass1.Solver()

'''HouseClass1 = ... Improve "Housing" to better reflect a real usage in a proper algorithm! '''

#House1.solve()
#House1.solve()
#
#House1.model.c1[23.5] = 1500
#House1.solve()
#House1.solve()
#House1.model.c1[23] = 1000
#House1.model.c1[23.5] = 1000
#House1.solve()
#House1.solve()
#House1.solve()
#House1.solve()