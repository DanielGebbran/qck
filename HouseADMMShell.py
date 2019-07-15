#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:52:56 2019

@author: daniel
"""

import HouseModel_ADMM as HouseADMM

class Housing():
    def __init__(self,consumerfile,pricefile):
        self.Starting(consumerfile,pricefile)
        
    def Starting(self,consumerfile,pricefile):
        try:
            self.HouseX = HouseADMM.HouseholdModel(consumerfile,pricefile)
        except:
            print('Problem setting sub-problem. Please verify.')
            self.trigger_fail = 1
            # Pass info to trigger self-solving into OPF: trigger_fail == 1
        else:
            #self.HouseX.solve()    
            self.trigger_fail = 0
            # Pass up information about results 
        #finally:
            # Pass up information trigger_fail 
            #print('Finished')
            
    def Solver(self):
        self.HouseX.solve()        
