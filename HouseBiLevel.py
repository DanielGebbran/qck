#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:27:54 2019

@author: daniel
"""
import pyomo
import pyomo.opt
import time
from pyomo.environ import *
import pandas as pd

class HouseholdModel():
    def __init__(self,consumerfile,pricefile):
        self.consumer_data = pd.read_csv(consumerfile)                      # Reading CSV files with Pandas
        self.price_data = pd.read_csv(pricefile)
        self.consumer_data.columns = self.consumer_data.columns.str.strip() # Removes spaces from header
        self.price_data.columns = self.price_data.columns.str.strip()
        # Defining index for sets:
        self.price_data.set_index('T',inplace=True)
        self.consumer_data.set_index('T',inplace=True)
        self.T_set = self.consumer_data.index.unique() #Equivalent to line below if only one index level
            
        # Calls function to create the whole model (concrete, not abstract model)
        self.createModel()
        
    def createModel(self):          # https://cfwebprod.sandia.gov/cfdocs/CompResearch/docs/pyomo_bilevel_sandreport.pdf
        time_start = time.time()
        self.model = ConcreteModel()
        
        # Either create one model and populate it with respective data, everytime it's used; or create as many models as households (but name of models would be tricky!);
        # >>> or create manually as many as required: more realistic; each house is either represented as a whole model or as xHouse from standard data.
        #    >>> If positive comms with House, replace local data with remote data, if not then run optimization with reference house data.
    
        ### Sets ###
        # Creates set T = time periods
        self.model.T = Set(initialize = self.T_set)
        self.model.T2 = Set(initialize = self.model.T - [0]) # Motif: used with time = k+1, previous updates, not battery initial SoC
        
        ### Parameters ###
        # Temporal parameters
        """ Modify: read and calculate from CSV data! """
        self.model.t0     = Param(default=0.0)
        self.model.tend   = Param(initialize=23.5)
        self.model.delta  = Param(default=0.5)
    
        kW = 1000 # Converts some units given in kW to W
    
        # Cost parameters
        def init_c0(model,j):
            return self.price_data.loc[j, 'c0'] / kW
        def init_c1(model,i):
            return self.price_data.ix[i, 'c1'] / kW
        def init_c2(model,i):
            return self.price_data.ix[i, 'c2'] / kW
        self.model.c2     = Param(self.model.T, initialize = init_c2)
        self.model.c1     = Param(self.model.T, initialize = init_c1, mutable = True)
        self.model.c0     = Param(self.model.T, initialize = init_c0)
        
        # Local demand and generation parameters, input from csv
        def init_demP(model,i):
            return kW * self.consumer_data.loc[i, 'demP'] # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
        def init_genP(model,i):
            return kW * self.consumer_data.loc[i, 'genP'] # Advanced indexing with hierarchical index
        self.model.genP   = Param(self.model.T, initialize = init_genP)
        self.model.demP   = Param(self.model.T, initialize = init_demP)
        # If loads have reactive power: self.model.demQ = ...
        
        # Battery parameters
        """ Possibly include in a separate 'control' file instead of in-line initializations """
        self.model.bmin   = Param(initialize =-1.75 * kW)
        self.model.bmax   = Param(initialize = 1.75 * kW)
        self.model.cmin   = Param(initialize = 0)
        self.model.cmax   = Param(initialize = 7.5 * kW)
        self.model.soc0   = Param(initialize = 3 * kW)
        self.model.etab   = Param(initialize = 0.92)
        # Household circuit breaker (assume bidirectional)
        self.model.hmin   = Param(initialize =-10 * kW)
        self.model.hmax   = Param(initialize = 10 * kW)
            
        ### Decision variables ###
        self.model.xHouse    = Var(self.model.T, bounds=(self.model.hmin,self.model.hmax))
        self.model.xPlus     = Var(self.model.T, within = NonNegativeReals)
        self.model.xMinus    = Var(self.model.T, within = NonNegativeReals)
        self.model.bSoC      = Var(self.model.T, bounds=(self.model.cmin,self.model.cmax))
        self.model.xBat      = Var(self.model.T, bounds=(self.model.bmin,self.model.bmax))
        # Create xHouse (demP + demFlexible) for flexible appliances demand response.
        
         ########################### Objective and constraints ###########################
        ### Objective function: Minimize total generation cost ###
        def cost_rule(model):
            return sum((((self.model.xPlus[i])**2)*(self.model.c2[i]) + self.model.c1[i]*self.model.xPlus[i] + self.model.c0[i]) for i in self.model.T)
            #return sum((((self.model.xHouse[i])**2)*(self.model.c2[i]) + self.model.c1[i]*self.model.xHouse[i] + self.model.c0[i]) for i in self.model.T)
            #return sum((self.model.c1[i]*sum((self.model.xHouse[i,k]) for k in self.model.H)) for i in self.model.T)
        self.model.cost = Objective(rule=cost_rule)
    
        # Grid power rule: Purchased power is expensive, sold power (xMinus) is too cheap and not accounted for!
        def local_power_rule (model, i):
            return self.model.xPlus[i] - self.model.xMinus[i] == self.model.xHouse[i]
        self.model.local_power = Constraint(self.model.T, rule=local_power_rule)
        
        # If an aggregator is used to concentrate more than one house in an area or a bus:
        #if trigger_Aggr == 1:
        #    def global_balance_rule (model, i):         # Total energy balance constraint
        #        return self.model.xAggr[i] == sum((self.model.xHouse[i,k]) for k in self.model.H)
        #    self.model.global_balance = Constraint(self.model.T, rule=global_balance_rule)
            
        def local_balance_rule (model, i):           # Local energy balance constraint
            return (self.model.xHouse[i] == (self.model.xBat[i] + self.model.demP[i] - self.model.genP[i]))
        self.model.local_balance = Constraint(self.model.T, rule=local_balance_rule)
        
        
            #return self.model.xHouse[i,k] == (self.model.xBat[i,k] + self.model.demP[i,k] - self.model.genP[i,k])
        #self.model.local_balance = Constraint(self.model.T, self.model.H, rule=local_balance_rule)
        
        def SoC_rule (model, i):                     # Local battery charging constraints
            return self.model.bSoC[i] == (self.model.bSoC[i-value(self.model.delta)] + self.model.etab*self.model.xBat[i])
        self.model.battery_SoC = Constraint(self.model.T2, rule=SoC_rule)
        
        self.startend=[0, 23.5]
        def StartEndSoC_rule (model, i):
            return (self.model.bSoC[i] == self.model.soc0)
        self.model.StartEndSoC = Constraint(self.startend, rule=StartEndSoC_rule)
    
        #print('Model built in ' ,time.time() - time_start, ' s. \n------------')
        
    def solve(self):
        
        time_start = time.time()
        solver = pyomo.opt.SolverFactory('ipopt')
        results = solver.solve(self.model)
            
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            print('Check solver not ok?')
            #logging.warning('Check solver not ok?')
        #else:
        #    print('Cost: ', self.model.cost(), ', Solver time = ', time.time() - time_start, ' s.')
            #print('Optimization successfully performed. \nCost: ', self.model.cost(), ', Solver time = ', time.time() - time_start, ' s.')
        
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            print('Check solver optimality?') 
            #logging.warning('Check solver optimality?')             
        
#if __name__ == '__main__':
#def startup:
#    Time_start = time.time()
#    mode = 1
#    solveprob1 = HouseholdModel('/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/HouseholdsData/B1_H1_11_02_2011.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
#    solveprob1.solve()
#    solveprob2 = HouseholdModel('/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/HouseholdsData/B2_H2_11_02_2011.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
#    solveprob2.solve()
#    solveprob3 = HouseholdModel('/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/HouseholdsData/B3_H3_11_02_2011.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
#    solveprob3.solve()
#    solveprob4 = HouseholdModel('/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/HouseholdsData/B4_H4_11_02_2011.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
#    solveprob4.solve()
#    
#    print('\n------------\nSolutions obtained in: ' ,time.time() - Time_start, ' s. \nTotal cost:', solveprob1.model.cost()+solveprob2.model.cost()+solveprob3.model.cost()+solveprob4.model.cost(), '\n------------')
#    print('Do you wish to see demand for individual households? Y/N')
#    x = input()
#    if x == 'Y':
#        print('House 1:')
#        solveprob1.model.xHouse.pprint()
#        print('House 2:')
#        solveprob2.model.xHouse.pprint()
#        print('House 3:')
#        solveprob3.model.xHouse.pprint()
#        print('House 4:')
#        solveprob4.model.xHouse.pprint()
        
