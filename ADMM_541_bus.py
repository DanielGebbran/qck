#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:28:34 2019

@author: daniel
"""
import pyomo
import pyomo.opt
import time
import ADMMShell as Shell
from pyomo.environ import *
# Possible to import pyomo.environ as pe and use pe.() for Param, Set, Var, ... Pyomo classes.
import pandas as pd
import numpy as np
import sys
import csv 
import scipy.io

# Panda has been able to deal with data, not necessary to use Numpy so far.

class SubProblem:
    def __init__(self,consumerfile,pricefile):
        self.consumer_data = pd.read_csv(consumerfile)                      # Reading CSV files with Pandas
        self.price_data = pd.read_csv(pricefile)
        self.consumer_data.columns = self.consumer_data.columns.str.strip() # Removes spaces from header
        self.price_data.columns = self.price_data.columns.str.strip()
        # Defining index for sets:
        self.consumer_data.set_index(['T','H','B'],inplace=True)        # Performs indexing of column 'T', 'H' & 'B' in place, i.e. on the run
        #self.T_set = self.= price_data.index.unique() #Equivalent to line below if only one index level
        self.T_set = self.consumer_data.index.levels[0] # Unique indexes are preserved, but not reordered. Selecting 'T' from multi-index, for time periods.
        self.H_set = self.consumer_data.index.levels[1] # Unique indexes are preserved, but not reordered. Selecting 'H' from multi-index, for households.    
        self.B_set = self.consumer_data.index.levels[2] # Unique indexes are preserved, but not reordered. Selecting 'B' from multi-index, for buses: [1, 2...].    
        self.consumer_data.reset_index(level='B',inplace=True)          # Resets bus data back to within dataframe format (utilized in distribution_network())
        self.price_data.set_index('T',inplace=True)
            
        # Calls functions to create X and Z subproblems
        self.distribution_network()     # Setting Y matrix, dictionaries for neighbouring buses and house aggregators
        self.kW = 1000 # Converts some units given in kW to W
        #self.Ro_PQ = 300; self.Ro_EF = 30000 # 860 iterations
        #self.Ro_PQ = 1000; self.Ro_EF = 100000 # 860 iterations
        #self.Ro_PQ = 0.01; self.Ro_EF = 0.01 # x iterations
        self.Ro = 1000
        time_z = time.time()
        print("Creating models for Households with ADMM coupled problem...")
        self.HousesADMM = {}
        for i in range(100):
            self.HousesADMM[i+1] = Shell.Housing('/media/daniel/HDDfiles/Projects/CommProject/JayssonNetwork/Households/'+str(i+1)+'.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
            #self.HousesADMM[i+1] = Shell.Housing('/Py/Network3/Houses/'+str(i+1)+'.csv','/Py/Aggregator/PriceSignal.csv')
            
        print("Rho used:", self.HousesADMM[1].HouseX.Ro, "Rho within this algorithm:", self.Ro)
        print("Finished creating models for Households with ADMM coupled problem. \nAll households built in:", time.time() - time_z, " s. \n------------")
              
        try:
            print ("Total households cost when minimizing energy import: ", sum(value(self.HousesADMM[k].HouseX.model.cost_initial()) for k in self.H_set), ".")        
            self.createX()
        except:
            print ("Failed to create or solve household subproblems, or create models X and Z.")        
        
    def createX(self):         # Generator, branches, prosumers
        time_start = time.time()
        self.X = ConcreteModel()
        ### Sets ###
        self.X.T = Set(initialize = self.T_set); self.X.T2 = Set(initialize = self.X.T - [0])
        self.X.H = Set(initialize = self.H_set); self.X.B = Set(initialize = self.B_set); 

        home_buses = []
        IDfile = open('/media/daniel/HDDfiles/Projects/CommProject/JayssonNetwork/Network_1/ID_Loads.csv', "r")
        #IDfile = open('/Py/Network3/ID_Loads.csv', "r")
        readerID = csv.reader(IDfile)
        for row2 in readerID:
            home_buses.append(int(row2[3]))
        
        add_buses = []
        for i in range(541):
            if i not in home_buses:
                add_buses.append(i)
        
        IDfile.close()

        for mysetn in add_buses:
            self.X.B.add(mysetn)   
        #self.X.B.pprint()

        def Lines_init(X):
            lst = []
            for i, n in self.Neigh.items():
                for j in n: lst.append((i, j))
            return lst
        self.X.L = Set(initialize = Lines_init, dimen=2)
        ### Parameters ###
        # Cost parameters
        def init_c0(X,j):
            return self.price_data.loc[j, 'c0'] / self.kW
        def init_c1(X,i):
            return self.price_data.ix[i, 'c1'] / self.kW
        def init_c2(X,i):
            return self.price_data.ix[i, 'c2'] / self.kW
        self.X.c2 = Param(self.X.T, initialize = init_c2); self.X.c1 = Param(self.X.T, initialize = init_c1); self.X.c0 = Param(self.X.T, initialize = init_c0)
        # Local demand and generation parameters, input from csv
        def init_demP(X,i,k):
            return self.kW * self.consumer_data.loc[(i, k), 'demP'] # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
        def init_genP(X,i,k):
            return self.kW * self.consumer_data.loc[(i, k), 'genP'] # Advanced indexing with hierarchical index
        self.X.genP = Param(self.X.T, self.X.H, initialize = init_genP); self.X.demP = Param(self.X.T, self.X.H, initialize = init_demP)
        # Voltage limits
        self.X.v2min = Param(initialize=43681); self.X.v2max = Param(initialize=53361) # 95% of 220V = 209V, 209**2 = 43681;  # 105% of 220V = 231V, 231**2 = 53361
        # Household circuit breaker (assume bidirectional)
        self.X.hmin = Param(initialize =-10 * self.kW); self.X.hmax = Param(initialize = 10 * self.kW)

        ### Decision Variables ###
        ## Variables Without Pairs on Z ##
        self.X.xPlus     = Var(self.X.T, within=NonNegativeReals)
        self.X.xMinus    = Var(self.X.T, within=NonNegativeReals)
        # Generator variables #
        self.X.xGrid     = Var(self.X.T) # Generated active power flow
        self.X.xQ        = Var(self.X.T) # Generated reactive power flow
        # Branches variables #
        self.X.Pij   = Var(self.X.T, self.X.L)
        self.X.Qij   = Var(self.X.T, self.X.L)
        # Bus variables #
        self.X.xV_E_ij      = Var(self.X.T, self.X.B, bounds=(100,231)) # Real voltage, in Volts
        self.X.xV_F_ij      = Var(self.X.T, self.X.B, bounds=(-100,100)) # Imaginary voltage, in Volts
        
        ## Variables With Pairs on Z##
        # Copies of neighbour variables and pairs #
        self.X.xHouse_i    = Var(self.X.T, self.X.H, bounds=(self.X.hmin, self.X.hmax))        
        self.X.xHouse_k    = Param(self.X.T, self.X.H, mutable = True)        
        ## Associated Lambdas ##
        self.X.Lambda_xHouse  = Param(self.X.T, self.X.H, mutable = True)
        for i in self.X.T:
            for h in self.X.H:
                self.X.Lambda_xHouse[i,h] = 0
        # Setting parameters to zero for initial run on X (not necessary for Z sub-problem)
        
        # Importing initial xHouse_k from bi-level household problems
        for k in self.H_set:
            for i in self.T_set:
                self.X.xHouse_k[i,k] = (value(self.HousesADMM[k].HouseX.model.xHouse[i]))
                
        ########################### Objective and constraints ###########################
        ### Objective function: X subproblem ###
        def Xsub_rule(X):
            return sum(((self.X.xPlus[i]**2)*(self.X.c2[i]) + self.X.c1[i]*self.X.xPlus[i] + self.X.c0[i]) +
                    sum ((1)*self.X.Lambda_xHouse[i,h]*(self.X.xHouse_i[i,h] - self.X.xHouse_k[i,h]) +
                    0.5*self.Ro * ((self.X.xHouse_i[i,h] - self.X.xHouse_k[i,h])**2) for h in self.X.H) for i in self.X.T)
        self.X.Xsub = Objective(rule=Xsub_rule)
        
        # Grid power rule
        def local_power_rule (X, i):
            return self.X.xPlus[i] - self.X.xMinus[i] == self.X.xGrid[i]
        self.X.local_power = Constraint(self.X.T, rule=local_power_rule)
        
        def voltage_limits_rule (X, i, b):      # Absolute Voltage limits rule and Active voltage reference
            if b == 0:                              # Slack (generation) bus
                return (self.X.xV_E_ij[i,b] == 220)
            else:
                return  self.X.v2min <= (self.X.xV_E_ij[i,b]**2 + self.X.xV_F_ij[i,b]**2) <= self.X.v2max
        self.X.voltage_limits = Constraint(self.X.T, self.X.B, rule=voltage_limits_rule)

        def voltage_limits_rule_Q (X, i, b):    # Reactive voltage reference 
            if b == 0:
                return (self.X.xV_F_ij[i,b] == 0)
            else: 
                return Constraint.Skip
        self.X.voltage_limits_Q = Constraint(self.X.T, self.X.B, rule=voltage_limits_rule_Q)
        
        def XPij_balance_rule(X, i, b, j):
            return self.X.Pij[i,(b,j)] == (self.M_g[b][j]*(self.X.xV_E_ij[i,b]**2+self.X.xV_F_ij[i,b]**2-self.X.xV_E_ij[i,b]*self.X.xV_E_ij[i,j]-self.X.xV_F_ij[i,b]*self.X.xV_F_ij[i,j])+
                       self.M_b[b][j]*(self.X.xV_E_ij[i,b]*self.X.xV_F_ij[i,j]-self.X.xV_F_ij[i,b]*self.X.xV_E_ij[i,j]))
        self.X.Pij_balance = Constraint(self.X.T, self.X.L, rule=XPij_balance_rule)
        
        def XQij_balance_rule(X, i, b, j):
            return self.X.Qij[i,(b,j)] == ((-1)*self.M_b[b][j]*(self.X.xV_E_ij[i,b]**2+self.X.xV_F_ij[i,b]**2-self.X.xV_E_ij[i,b]*self.X.xV_E_ij[i,j]-self.X.xV_F_ij[i,b]*self.X.xV_F_ij[i,j])+
                       self.M_g[b][j]*(self.X.xV_E_ij[i,b]*self.X.xV_F_ij[i,j]-self.X.xV_F_ij[i,b]*self.X.xV_E_ij[i,j]))
        self.X.Qij_balance = Constraint(self.X.T, self.X.L, rule=XQij_balance_rule)
    
        #Replaced L by actual L (not LuL), created LuL, error for index ((1,0) not valid for Pij anymore!). Trying to change balance rules: goes nowhere? Instead, using LuL as L, hence obsolete Qji & Pji.

        def Xbus_active_balance_rule (X, i, b):  # Bus active power balance constraint 
            if b == 0:      # Generation bus;  # Formula of power sent from node 0 to node 1
                return self.X.xGrid[i] == sum((self.X.Pij[i,(b,k)]) for k in self.Neigh[b])
            else:           # All consuming buses; each accounting for households which are members (m) of local aggregator on bus (b)
                return  (-1) * sum((self.X.xHouse_i[i,m]) for m in self.Aggre[b]) == sum((self.X.Pij[i,(b,k)]) for k in self.Neigh[b])
        self.X.local_active_balance = Constraint(self.X.T, self.X.B, rule=Xbus_active_balance_rule)
        # Are these required?? Not formulated in paper!
        def Xbus_reactive_balance_rule (X, i, b): # Bus active power balance constraint 
            if b == 0:      # Generation bus; 
                return self.X.xQ[i] == sum(self.X.Qij[i,(b,k)] for k in self.Neigh[b])
            else:           # All consuming buses; each accounting for households which are members (m) of local aggregator on bus (b)
                return  0 == sum(self.X.Qij[i,(b,k)] for k in self.Neigh[b])
        self.X.local_reactive_balance = Constraint(self.X.T, self.X.B, rule=Xbus_reactive_balance_rule)
        
        # Feeder capacity
        def feeder_capacity_rule (X, i):
            return -20000 <= self.X.xGrid[i] <= 20000
        self.X.feeder_capacity = Constraint(self.X.T, rule=feeder_capacity_rule)

        print('Model X built in ' ,time.time() - time_start, ' s. \n------------')
    
        
    ''' Defines a distribution network over existing buses, with admittance matrix, neighbouring buses and house aggregator dictionaries'''
    def distribution_network(self):     
        time_matrix = time.time()
        ## Extract admittance [Y = G + j B] 541x541 x2 matrixes from matlab file, transform each into
        # 2 lists of 541 lists.
        ###### Multiplied elements by -1 and Ignored self (11, 22, 33... on constructing B, G matlab matrixes from Y)
        Dict_B = scipy.io.loadmat('/media/daniel/HDDfiles/Projects/CommProject/JayssonNetwork/B.mat')
        Dict_G = scipy.io.loadmat('/media/daniel/HDDfiles/Projects/CommProject/JayssonNetwork/G.mat')
        #Dict_B = scipy.io.loadmat('/Py/Network3/B.mat')
        #Dict_G = scipy.io.loadmat('/Py/Network3/G.mat')
        
        self.M_g = Dict_B['B']
        self.M_b = Dict_G['G']
                
        print("Time to build matrixes:", time.time()-time_matrix)
   
        time_dict = time.time()
        for i in range(len(self.M_g)):                  # Creating dictionaries to correlate buses to neighbouring buses; and houses to buses, and diminishes calculation time for model creation
            lst = []                        # comparing to: (multiplying zeros from impedance matrix on non-neighbouring elements) or
            lst2 = []                       # (comparing to searching for all houses to corresponding buses).
            for m in self.H_set:     
                if i == self.consumer_data.loc[(0, m), 'B']:    # Searches each house, and compares its bus (B) to current search (i) bus. 
                    lst2.append(m)                              # If true, appends to list, passed into dictionary further ahead.
            for j in range(len(self.M_g)):              # multiplying zeros from impedance matrix on non-neighbouring elements)
                if self.M_g[i][j] != 0:
                    lst.append(j)
            if i == 0:
                self.Neigh = {0 : lst} 
                self.Aggre = {}       # Empty dictionary declaration
            else:
                self.Neigh.update({i : lst})    # Using self.Neigh looks enough, instead of self.model.Neigh.
                self.Aggre.update({i : lst2})       
                
        #print ("Neighbouring buses list: \n", self.Neigh.items(), '\n')
        #print ("Regional aggregator list: \n", self.Aggre.items(), '\n')
        print("Time to create dictionaries:", time.time()-time_dict)
    
    def solve(self):
        
        time_start = time.time()
        solver = pyomo.opt.SolverFactory('ipopt')
        print('Beginning ADMM solution of OPF problem.')

        g_register, X_V_E, X_V_F = [], [], []
        g_k_ro = 10
        t = 1
        while g_k_ro > 0.0001:
        #while g_k_ro > 0.1:
            print('Iteration:', t)
            g_step = [] # Clears array for usage on residual calculation
            time_x = time.time()
            self.result_x = solver.solve(self.X) # Solves k-th iteration in X
            #print('Sucessful solution for X. Cost:', value(self.X.Xsub), "solution time:", time.time() - time_x)
            print("Sucessful solution for X. Solution time:", time.time() - time_x)
            
            # Updates k+1 for Z sub-problem and solves Z
            time_z = time.time()
            for k in self.H_set:
                for i in self.T_set:
                    self.HousesADMM[k].HouseX.model.xHouse_k[i] = value(self.X.xHouse_i[i,k]) # Updates k+1 for Z sub-problem
                self.HousesADMM[k].HouseX.solve() # Solves k-th iteration in Z
            print('Sucessful solution for houses. Solution time:', time.time() - time_z)
            
            # Then update lambda for each house, and renew parameter values for self.X
            time_L = time.time()            
            for k in self.H_set:
                for i in self.T_set:
                     self.X.xHouse_k[i,k] = value(self.HousesADMM[k].HouseX.model.xHouse[i])
                     residual = value(self.X.xHouse_i[i,k]) - value(self.HousesADMM[k].HouseX.model.xHouse[i])
                     #print("Residual:", k, i, residual)
                     g_step.append(residual)
                     self.X.Lambda_xHouse[i,k] = value(self.X.Lambda_xHouse[i,k]) + self.Ro*residual
                     self.HousesADMM[k].HouseX.model.Lambda_xHouse[i] = value(self.HousesADMM[k].HouseX.model.Lambda_xHouse[i]) + self.HousesADMM[k].HouseX.Ro*residual

            print('Updated Lambdas. Step time:', time.time() - time_L)
            
            time_N = time.time()
            g_k_ro = np.linalg.norm(np.asarray(g_step)) 
            print("Residual norm:", g_k_ro, ". Step time:", time.time() - time_N)
            
            # Registers values
            X_V_E.append(round(value(self.X.xV_E_ij[23.5,539]),2))
            X_V_F.append(round(value(self.X.xV_F_ij[23.5,539]),2))
            g_register.append(g_k_ro)
            t = t + 1
            
        print('---------------------------------\nSolutions achieved in ' ,time.time() - time_start, ' s. \n',X_V_E, '\n',X_V_F, '\n', ' \n')#Saving results of decreasing g...')
        #np.savetxt("g_all_iterations", np.array(g_register))
        print('Total number of iterations:', t-1, ', g_k_ro:', g_k_ro, '\nFinal cost:', value(self.X.Xsub))
        print('End of ADMM optimization.') 
        
#        for k in sorted (self.H_set):
#            print("Energy consumption for house number:", k)
#            for i in sorted (self.T_set):
#                print(value(self.X.xHouse_k[i,k]))
#                #print(value(self.X.xHouse_k[i,k]), end=" ")
        print("xGrid:")
        for i in sorted (self.T_set):
            print(value(self.X.xGrid[i]))
        print("Voltages:")
        for i in self.X.B:
            print(i, (round(value(self.X.xV_E_ij[23.5,i]),2)), '+ j', (round(value(self.X.xV_E_ij[23.5,i]),2)))
        print("Total households cost when obeying aggregator problem:", sum (value(self.HousesADMM[k].HouseX.model.cost)for k in self.H_set))
        
        # Retrieve original cost for comparison with centralized solution
        #print('Optimization successfully performed. \nCost: ', self.model.cost(), ', Solver time = ', time.time() - time_start, ' s.')
            
if __name__ == '__main__':
        Time_start = time.time()
        solveprob = SubProblem('/media/daniel/HDDfiles/Projects/CommProject/JayssonNetwork/Network_3.csv','/media/daniel/HDDfiles/Projects/CommProject/PythonImplementation/BatteryModel/PriceSignal.csv')
        #solveprob = SubProblem('/Py/Network3.csv','/Py/Aggregator/PriceSignal.csv')
        solveprob.solve()

        print('---------------------------------')
        print('Total run time: ' , time.time() - Time_start, ' s.')