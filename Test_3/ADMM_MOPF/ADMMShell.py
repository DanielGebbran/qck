#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:52:56 2019

@author: daniel
"""

import ADMMHouseModel as HouseADMM
import time

class Housing():
    def __init__(self,consumerfile,pricefile,Ro):
        self.Starting(consumerfile,pricefile,Ro)
        
    def Starting(self,consumerfile,pricefile,Ro):
        time_start = time.time()
        self.HouseX = HouseADMM.HouseholdModel(consumerfile,pricefile,Ro)
        self.HouseX.model.cost.deactivate()
        self.HouseX.model.cost_initial.activate()
        self.HouseX.solve()    
        self.HouseX.model.cost_initial.deactivate()
        self.HouseX.model.cost.activate()
        #print('Cost: ', self.HouseX.model.cost_initial(), ', Solver time = ', time.time() - time_start, ' s.')
        #try:
            #self.HouseX = HouseADMM.HouseholdModel(consumerfile,pricefile)
        #except:
            #print('Problem setting sub-problem. Please verify.')
            #self.trigger_fail = 1
            # Pass info to trigger self-solving into OPF: trigger_fail == 1
        #else:
            #self.HouseX.model.cost.deactivate()
            #self.HouseX.model.cost_initial.activate()
            #self.HouseX.solve()    
            #self.HouseX.model.cost_initial.deactivate()
            #self.HouseX.model.cost.activate()
            #self.trigger_fail = 0
            
            # Pass up information about results 
        #finally:
            # Pass up information trigger_fail 
            #print('Finished')
            
    def Solver(self):
        self.HouseX.solve()        
