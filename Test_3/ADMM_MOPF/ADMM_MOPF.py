#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:28:34 2019

@author: daniel
"""
import pyomo
import pyomo.opt
import time
import datetime
import ADMMShell as Shell
#import Test_ADMMShell as Shell
from pyomo.environ import *
# Possible to import pyomo.environ as pe and use pe.() for Param, Set, Var, ... Pyomo classes.
import pandas as pd
import numpy as np
import sys
import csv 
import scipy.io

# Panda has been able to deal with data, not necessary to use Numpy so far.

class SubProblem:
    def __init__(self, In_network, In_price, set_rho):
        network = str(In_network); price = str(In_price)
        self.consumer_data = pd.read_csv('./Networks/Network'+ network +'.csv')                      # Reading CSV files with Pandas
        self.price_data = pd.read_csv('./Prices/PriceSignal_'+ price +'.csv')
        self.Ordenic = pd.read_csv('./Ordenic.csv')
        self.consumer_data.columns = self.consumer_data.columns.str.strip() # Removes spaces from header
        self.price_data.columns = self.price_data.columns.str.strip()
        self.Ordenic.columns = self.Ordenic.columns.str.strip()
        # Defining index for sets:
        self.consumer_data.set_index(['T','H','B'],inplace=True)        # Performs indexing of column 'T', 'H' & 'B' in place, i.e. on the run
        #self.T_set = self.= price_data.index.unique() #Equivalent to line below if only one index level
        self.T_set = self.consumer_data.index.levels[0] # Unique indexes are preserved, but not reordered. Selecting 'T' from multi-index, for time periods.
        self.H_set = self.consumer_data.index.levels[1] # Unique indexes are preserved, but not reordered. Selecting 'H' from multi-index, for households.    
        self.B_set = self.consumer_data.index.levels[2] # Unique indexes are preserved, but not reordered. Selecting 'B' from multi-index, for buses: [1, 2...].    
        self.consumer_data.reset_index(level='B',inplace=True)          # Resets bus data back to within dataframe format (utilized in distribution_network())
        
        self.price_data.set_index('T',inplace=True)
        self.Ordenic.set_index('Network',inplace=True)
            
        # Calls functions to create X and Z subproblems
        self.distribution_network(network)     # Setting Y matrix, dictionaries for neighbouring buses and house aggregators
        
        # EITHER SHOULD BE 1000 NOT BOTH.
        self.kW = 1 # Converts some units given in kW to W if 1000 instead of 1
        self.Y_kW = 1000 # Converts units given in W to kW if 1000 
        
#        input_Ro = float(input("\nPlease, select an appropriate Rho for the problem. \nType 0 for a list of defaults on your network: [$/(kWh^2)]\n"))
#        while input_Ro == 0:
#            print("Network 3: 0.025")    
#            input_Ro = float(input("\nPlease, select an appropriate Rho for the problem. \n"))
#            
#        self.Ro = input_Ro #self.Ro = 0.025
        self.Ro = set_rho
        
        time_z = time.time()
        
        print("Creating models for Households with ADMM coupled problem...")
        self.HousesADMM = {}
        
        #for i in range(100):
        for i in range (self.Ordenic.loc[int(network), 'Houses']):
            self.HousesADMM[i+1] = Shell.Housing('./Households/Network'+ network +'/' +str(i+1)+'.csv',
                                   './Prices/PriceSignal_'+ price +'.csv', self.Ro)
            #self.HousesADMM[i+1] = Shell.Housing('/Py/Network3/Houses/'+str(i+1)+'.csv','/Py/Aggregator/PriceSignal.csv')
            
        print("Rho used:", self.HousesADMM[1].HouseX.Ro, "Rho within this algorithm:", self.Ro)
        print("Finished creating models for Households with ADMM coupled problem. \nAll households built in:", time.time() - time_z, " s. \n------------")
              
        #try:
        #print ("Total households cost when minimizing energy import: ", sum(value(self.HousesADMM[k].HouseX.model.cost_initial()) for k in self.H_set), ".")        
        self.createX(network)
        #except:
        #    print ("Failed to create or solve household subproblems, or create models X and Z.")        
        
    def createX(self, network):         # Generator, branches, prosumers
        time_start = time.time()
        self.X = ConcreteModel()
        ### Sets ###
        self.X.T = Set(initialize = self.T_set); self.X.T2 = Set(initialize = self.X.T - [0])
        self.X.H = Set(initialize = self.H_set)
        self.X.B = Set(initialize = self.B_set)
        
        if network == "1" or network == "41" or network == "42":
            self.X.B.add(0)
            
        elif network == "2":
            for mysetn in [0, 1, 2, 3, 4, 9]:
                self.X.B.add(mysetn)

        else:
            home_buses = []
            #IDfile = open('/media/daniel/HDDfiles/Projects/CommProject/ADMM_MOPF/Networks/' + self.Ordenic.loc[int(network), 'ID'] + '.csv', "r")
            IDfile = open('./Networks/' + self.Ordenic.loc[int(network), 'ID'] + '.csv', "r")
            #IDfile = open('/Py/Network3/ID_Loads.csv', "r")
            readerID = csv.reader(IDfile)
            for row2 in readerID:
                home_buses.append(int(row2[3]))
            
            add_buses = []
            for i in range(self.Ordenic.loc[int(network), 'Buses']):
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
        # Cost parameters [$/kWh] converted to [$/Wh]
        def init_c0(X,j):
            return self.price_data.loc[j, 'c0'] / self.kW
        def init_c1(X,i):
            return self.price_data.ix[i, 'c1'] / self.kW
        def init_c2(X,i):
            return self.price_data.ix[i, 'c2'] / self.kW
        def init_ToU(X,i):
            return self.price_data.ix[i, 'ToU'] / self.kW
        self.X.c2 = Param(self.X.T, initialize = init_c2); self.X.c1 = Param(self.X.T, initialize = init_c1); self.X.c0 = Param(self.X.T, initialize = init_c0)
        self.X.ToU = Param(self.X.T, initialize = init_ToU)
        
        # Local demand and generation parameters, input from csv
        def init_demP(X,i,k):
            return self.kW * self.consumer_data.loc[(i, k), 'demP'] # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
        def init_genP(X,i,k):
            return self.kW * self.consumer_data.loc[(i, k), 'genP'] # Advanced indexing with hierarchical index
        self.X.genP = Param(self.X.T, self.X.H, initialize = init_genP); self.X.demP = Param(self.X.T, self.X.H, initialize = init_demP)
        # Voltage limits
        self.X.v2min = Param(initialize=46656); self.X.v2max = Param(initialize=64009) # -6% of 230V = 216V, 216**2 = 46656;  # 110% of 230V = 253V, 253**2 = 64009
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
        self.X.hPlus     = Var(self.X.T, self.X.H, within = NonNegativeReals)
        self.X.hMinus    = Var(self.X.T, self.X.H, within = NonNegativeReals)
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
                #self.X.xHouse_k[i,k] = 0
                
        ########################### Objective and constraints ###########################
        ### Objective function: X subproblem ###
        def Xsub_rule(X):
            return sum(( self.X.c2[i]*(self.X.xPlus[i]**2) + self.X.c1[i]*self.X.xPlus[i] + 
                   sum(( self.X.ToU[i] * (self.X.hPlus[i,h]) + (-0.10) * (self.X.hMinus[i,h]) +
                        (1)*self.X.Lambda_xHouse[i,h]*(self.X.xHouse_i[i,h] - self.X.xHouse_k[i,h]) +
                        0.5*self.Ro * ((self.X.xHouse_i[i,h] - self.X.xHouse_k[i,h])**2)) for h in self.X.H)) for i in self.X.T)
        self.X.Xsub = Objective(rule=Xsub_rule)
        
        #        def cost_rule(model):
#            return sum((self.model.ToU[i]*self.model.xPlus[i] + self.model.xMinus[i]*(-0.00010) +
#                       (-1)*self.model.Lambda_xHouse[i]*(self.model.xHouse[i] - self.model.xHouse_k[i]) +
#                       0.5*self.Ro * ((self.model.xHouse[i] - self.model.xHouse_k[i])**2))  for i in self.model.T)
#        self.model.cost = Objective(rule=cost_rule)
#    
        # Prosumers power rule
        def local_power_rule (X, i, h):
            return self.X.hPlus[i,h] - self.X.hMinus[i,h] == self.X.xHouse_i[i,h]
        self.X.local_power = Constraint(self.X.T, self.X.H, rule=local_power_rule)
        
        # Grid power rule
        def power_rule (X, i):
            return self.X.xPlus[i] - self.X.xMinus[i] == self.X.xGrid[i]
        self.X.power = Constraint(self.X.T, rule=power_rule)
        
        def voltage_limits_rule (X, i, b):      # Absolute Voltage limits rule and Active voltage reference
            if b == 0:                              # Slack (generation) bus
                return (self.X.xV_E_ij[i,b] == 230)
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
                       self.M_b[b][j]*(self.X.xV_E_ij[i,b]*self.X.xV_F_ij[i,j]-self.X.xV_F_ij[i,b]*self.X.xV_E_ij[i,j]))/self.Y_kW
        self.X.Pij_balance = Constraint(self.X.T, self.X.L, rule=XPij_balance_rule)
        
        def XQij_balance_rule(X, i, b, j):
            return self.X.Qij[i,(b,j)] == ((-1)*self.M_b[b][j]*(self.X.xV_E_ij[i,b]**2+self.X.xV_F_ij[i,b]**2-self.X.xV_E_ij[i,b]*self.X.xV_E_ij[i,j]-self.X.xV_F_ij[i,b]*self.X.xV_F_ij[i,j])+
                       self.M_g[b][j]*(self.X.xV_E_ij[i,b]*self.X.xV_F_ij[i,j]-self.X.xV_F_ij[i,b]*self.X.xV_E_ij[i,j]))/self.Y_kW
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
            return -100 <= self.X.xGrid[i] <= 3.7
        #self.X.feeder_capacity = Constraint(self.X.T, rule=feeder_capacity_rule)

        print('Model X built in ' ,time.time() - time_start, ' s. \n------------')
    
        
    ''' Defines a distribution network over existing buses, with admittance matrix, neighbouring buses and house aggregator dictionaries'''
    def distribution_network(self, network):     
        time_matrix = time.time()
        ## Extract admittance [Y = G + j B] 541x541 x2 matrixes from matlab file, transform each into
        # 2 lists of 541 lists.
        ###### Multiplied elements by -1 and Ignored self (11, 22, 33... on constructing B, G matlab matrixes from Y)
        if network == '1':
            #g =  0.166667 # Conductance for all lines. Equivalent impedance: (1.5 + j 2.6 = 3 | 60 degrees) Ohms for each line.
            #b = -0.288675 # Susceptance for all lines 
            #g =  1.66667 # Conductance for all lines. Equivalent impedance: (0.15 + j 0.26 = 0.3 | 60 degrees) Ohms for each line.
            #b = -2.88675 # Susceptance for all lines 
            
            # Creating Impedance Matrix
            x = int(len(self.B_set)) + 1    # Adds one unit due to bus number zero
            
            g =  5 # Conductance for all lines. Equivalent impedance: (0.05 + j 0.09 = 0.1 | 60 degrees) Ohms for each line.
            b = -8.6589 # Susceptance for all lines 
            
            self.M_g = [[0 for x in range(x)] for y in range(x)] # Initializing a blank matrix BxB
            self.M_b = [[0 for x in range(x)] for y in range(x)] # width and height, respectively
            for i in range(x):
                for j in range(x):  
                    if abs(i - j) == 1:
                        self.M_g[i][j] = g      # Using self.M_g looks enough, instead of self.model.M_g.
                        self.M_b[i][j] = b
                    
        elif network == '2':
            #g =  1.66667 # Conductance for all lines. Equivalent impedance: (0.15 + j 0.26 = 0.3 | 60 degrees) Ohms for each line.
            #b = -2.88675 # Susceptance for all lines 
            #g =  5 # Conductance for all lines. Equivalent impedance: (0.05 + j 0.09 = 0.1 | 60 degrees) Ohms for each line.
            #b = -8 # Susceptance for all lines.  Bi Lvl ## NOT OK ##
            #g =  10 # Conductance for all lines. Equivalent impedance: (0.05 + j 0.09 = 0.1 | 60 degrees) Ohms for each line.
            #b = -16 # Susceptance for all lines.  Bi Lvl ## NOT OK ##
            g =  15 # Conductance for all lines. Equivalent impedance: (0.05 + j 0.09 = 0.1 | 60 degrees) Ohms for each line.
            b = -24 # Susceptance for all lines. Bi Lvl ## OK ##
            
            self.M_g = [
                    [0, g*10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [g*10, 0, g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, g, 0, g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, g, 0, g*2, 0, 0, 0, g, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, g*2, 0, g*2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, g*2, 0, g*4, g*4, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, g*4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, g*4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, g, 0, 0, 0, 0, 0, g*3, 0, g, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, g*3, 0, g*4, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, g*4, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, g, g, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g, 0, 0, g*4, g*4],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g*4, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g*4, 0, 0]]
        
            self.M_b = [
                    [0, b*10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [b*10, 0, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, b, 0, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, b, 0, b*2, 0, 0, 0, b, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, b*2, 0, b*2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, b*2, 0, b*4, b*4, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, b*4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, b*4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, b, 0, 0, 0, 0, 0, b*3, 0, b, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, b*3, 0, b*4, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, b*4, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, b, 0, 0, 0, b, b, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, 0, 0, b*4, b*4],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b*4, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b*4, 0, 0]]
        
        else:
            #print(self.Ordenic.loc[int(network), 'G'], self.Ordenic.loc[int(network), 'B'])
            Dict_G = scipy.io.loadmat('./Networks/' + self.Ordenic.loc[int(network), 'G'] + '.mat')
            Dict_B = scipy.io.loadmat('./Networks/' + self.Ordenic.loc[int(network), 'B'] + '.mat')
            #Dict_G = scipy.io.loadmat('/Py/Network3/G.mat')
            #Dict_B = scipy.io.loadmat('/Py/Network3/B.mat')
            if network == "44" or network == "42":
                #self.M_g = (+0.05) * Dict_G[self.Ordenic.loc[int(network), 'G']]
                #self.M_b = (+0.05) * Dict_B[self.Ordenic.loc[int(network), 'B']]
                #self.M_g = (+0.3) * Dict_G[self.Ordenic.loc[int(network), 'G']] # 57 iter
                #self.M_b = (+0.3) * Dict_B[self.Ordenic.loc[int(network), 'B']]
                self.M_g = (+0.15) * Dict_G[self.Ordenic.loc[int(network), 'G']]
                self.M_b = (+0.15) * Dict_B[self.Ordenic.loc[int(network), 'B']]
            elif network == "41" or network == "43":
                self.M_g = (+0.07) * Dict_G[self.Ordenic.loc[int(network), 'G']]
                self.M_b = (+0.07) * Dict_B[self.Ordenic.loc[int(network), 'B']]
            else:
                self.M_g = (+1) * Dict_G[self.Ordenic.loc[int(network), 'G']]
                self.M_b = (+1) * Dict_B[self.Ordenic.loc[int(network), 'B']]
                
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
    
    def solve(self,network,price, criteria_abs, criteria_rel):
        
        #Ex_solver = input("Please type in a linear solver for IPOPT (standard: MA27; basic: MUMPS):\n")
        #Ex_acc = float(input("Please type in an acceptable termination criteria (Epsilon) (e.g. 0.25):\n"))
        #Ex_fin = float(input("Please type in a finishing termination criteria (Epsilon) (e.g. 0.05):\n"))
        
        #e_abs = 0.0001; e_rel = 0.001
        e_abs = criteria_abs; e_rel = criteria_rel
        
        algo_time = 0
        time_start = time.time()
        solver = pyomo.opt.SolverFactory('ipopt')
        #solver.options['linear_solver'] = Ex_solver
        solver.options['linear_solver'] = 'ma27'
        print('----------------------------------------------------------------'+
              '\nBeginning ADMM solution of OPF problem. Selected solver:', solver.options['linear_solver'])

        Ofile= open('/Py/Test_3/ADMM_MOPF/Results/' + 
                    str(datetime.datetime.now()) + '.csv', "w")
        writer = csv.writer(Ofile,delimiter=',', quotechar='|', quoting = csv.QUOTE_MINIMAL)
        writer.writerow(['Network','Price','Rho','e_abs','e_rel'])
        writer.writerow([network,price,self.Ro,e_abs,e_rel])
        writer.writerow(['k','Cost','Time','AvgHouse','r_k','s_k','e_p','e_d'])

        g_register, X_V_E, X_V_F = [], [], []
        g_k_ro = 10; d_k_ro = 10
        e_prim = 1; e_dual = 1;
        t = 1; pp = 0
        print('  k  |    Cost    | Time | Avg/House |   r_k   |   s_k   |  e_p  |  e_d  |')
        #while g_k_ro > Ex_fin or d_k_ro > Ex_fin:
        while g_k_ro > e_prim or d_k_ro > e_dual:
        
        
            if t % 30 == 0: print('  k  |    Cost    | Time | Avg/House |   r_k   |   s_k   |  e_p  |  e_d  |')
        
#            print('-----------------\nIteration:', t)
            g_step = [] # Clears array for usage on residual calculation
            d_step = []; D_step = []; dual_residual = []; lambda_vec = []; x_vec = []; z_vec = []
            time_x = time.time()
            self.result_x = solver.solve(self.X) # Solves k-th iteration in X
            #print('Sucessful solution for X. Cost:', value(self.X.Xsub), "solution time:", time.time() - time_x)
            algo_time += time.time() - time_x
            time_X = time.time() - time_x
#            print("Sucessful solution for X. Solution time:", time.time() - time_x)
#            print("Cost:", value(self.X.Xsub))
            
            # Updates k+1 for Z sub-problem and solves Z
            time_z = time.time()
            for k in self.H_set:
                for i in self.T_set:
                    self.HousesADMM[k].HouseX.model.xHouse_k[i] = value(self.X.xHouse_i[i,k]) # Updates k+1 for Z sub-problem
                    x_vec.append(value(self.X.xHouse_i[i,k]))
                    try:
                        d_step.append(value(self.HousesADMM[k].HouseX.model.xHouse[i]))
                    except:
                        d_step.append(0)
                self.HousesADMM[k].HouseX.solve() # Solves k-th iteration in Z
            algo_time += (time.time() - time_z)/self.Ordenic.loc[int(network), 'Houses']
            time_H = (time.time() - time_z)/self.Ordenic.loc[int(network), 'Houses']
#            print('Sucessful solution for houses. Solution time:', time.time() - time_z, 'average/house:', (time.time() - time_z)/self.Ordenic.loc[int(network), 'Houses'])
            
            # Then update lambda for each house, and renew parameter values for self.X
            time_L = time.time()            
            for k in self.H_set:
                for i in self.T_set:
                     self.X.xHouse_k[i,k] = value(self.HousesADMM[k].HouseX.model.xHouse[i])
                     z_vec.append(value(self.HousesADMM[k].HouseX.model.xHouse[i]))
                     residual = value(self.X.xHouse_i[i,k]) - value(self.HousesADMM[k].HouseX.model.xHouse[i])
                     #print("Residual:", k, i, residual)
                     g_step.append(residual)
                     D_step.append(value(self.HousesADMM[k].HouseX.model.xHouse[i]))
                     self.X.Lambda_xHouse[i,k] = value(self.X.Lambda_xHouse[i,k]) + self.Ro*residual
                     self.HousesADMM[k].HouseX.model.Lambda_xHouse[i] = value(self.HousesADMM[k].HouseX.model.Lambda_xHouse[i]) + self.HousesADMM[k].HouseX.Ro*residual
                     lambda_vec.append(value(self.X.Lambda_xHouse[i,k]))

            algo_time += time.time() - time_L
#            print('Updated Lambdas. Step time:', time.time() - time_L)
            
            time_N = time.time()
            for i in range(len(D_step)):
                dual_residual.append(((-1)*self.Ro)*(D_step[i] - d_step[i]))
            d_k_ro = np.linalg.norm(np.asarray(dual_residual)) 
            g_k_ro = np.linalg.norm(np.asarray(g_step)) 
            
            e_prim = e_abs*self.Ordenic.loc[int(network), 'Houses'] + e_rel*max( np.linalg.norm(np.asarray(x_vec)), np.linalg.norm(np.asarray(z_vec)) )
            e_dual = e_abs*self.Ordenic.loc[int(network), 'Houses'] + e_rel*np.linalg.norm(np.asarray(lambda_vec)) 
            
            
            algo_time += time.time() - time_N
#            print("Primal residual:", g_k_ro, "and dual:", d_k_ro, ". Step time:", time.time() - time_N)
            
            print('  ', t, ' | ', round(value(self.X.Xsub),4), ' | ', round(time_X,2), ' | ', round(time_H,2), ' | ', round(g_k_ro,5), ' | ', round(d_k_ro,5), ' | ', round(e_prim,7), ' | ', round(e_dual,7))
            
            writer.writerow([t, value(self.X.Xsub), time_X, time_H, g_k_ro, d_k_ro, e_prim, e_dual])
            
            # Registers values
            X_V_E.append(round(value(self.X.xV_E_ij[23.5, (self.Ordenic.loc[int(network), 'Buses'] - 1)]),2))
            X_V_F.append(round(value(self.X.xV_F_ij[23.5, (self.Ordenic.loc[int(network), 'Buses'] - 1)]),2))
            g_register.append(g_k_ro)
            t = t + 1
            if t > 3500:
                print("Did not converge within 3500 iterations! Exiting this routine.")
                break
#            #if g_k_ro < Ex_acc and d_k_ro < Ex_acc and pp == 0:
#            if g_k_ro < Ex_acc and pp == 0:
#                print("xGrid:")
#                for i in sorted (self.T_set):
#                    print(value(self.X.xGrid[i]))
#                print('---------------------------------\n Acceptable solutions achieved. \n Runtime: ',
#                      time.time() - time_start, ' s. Algorithm time:', algo_time, '\n',
#                      X_V_E, '\n',X_V_F, '\n', ' \n')
#                print('Total number of iterations:', t-1, '\ng_k_ro:', g_k_ro,'| d_k_ro:', d_k_ro,'\nFinal cost:', value(self.X.Xsub))
#                #print("Total households cost when obeying aggregator problem:", sum (value(self.HousesADMM[k].HouseX.model.cost)for k in self.H_set))
#                print("Total network cost:", sum(( value(self.X.c2[i])*(value(self.X.xPlus[i])**2) ) for i in self.X.T))
#                print("Total houses costs:", sum((sum((value(self.X.ToU[i])*value(self.X.hPlus[i,k])) for k in self.X.H) ) for i in self.X.T))
#                print("Total houses PV:", sum((sum(((-0.10)*value(self.X.hMinus[i,k])) for k in self.X.H) ) for i in self.X.T))
#                pp = 1
#                print('  k  |    Cost    | Time | Avg/House |   r_k   |   s_k   |')
#                print('*********************************************************')
                
        print('*********************************************************\n Final solutions achieved. \n Runtime: ',
                      time.time() - time_start, ' s. Algorithm time:', algo_time, '\n')#,
                      #X_V_E, '\n',X_V_F, '\n', ' \n')#Saving results of decreasing g...')
        #np.savetxt("g_all_iterations", np.array(g_register))
        #print('Total number of iterations:', t-1, ', g_k_ro:', g_k_ro, '\nFinal cost:', value(self.X.Xsub))
        print('End of ADMM optimization.') 
        
        writer.writerow(['AlgoTime', 'NetworkCost', 'HousesCost', 'HousesPV', 'Total Cost'])
        writer.writerow([algo_time, sum(( value(self.X.c2[i])*(value(self.X.xPlus[i])**2) ) for i in self.X.T
                        ), sum((sum((value(self.X.ToU[i])*value(self.X.hPlus[i,k])) for k in self.X.H) ) for i in self.X.T
                          ), sum((sum(((-0.10)*value(self.X.hMinus[i,k])) for k in self.X.H) ) for i in self.X.T
                            ), sum(( value(self.X.c2[i])*(value(self.X.xPlus[i])**2) ) for i in self.X.T 
                        ) + sum((sum((value(self.X.ToU[i])*value(self.X.hPlus[i,k])) for k in self.X.H) ) for i in self.X.T
                          ) + sum((sum(((-0.10)*value(self.X.hMinus[i,k])) for k in self.X.H) ) for i in self.X.T)])

        writer.writerow(['hat{P}_House','P_House', 'P_Bat', 'SoC'])
        for k in sorted (self.H_set):
            for i in sorted (self.T_set):
                writer.writerow([value(self.X.xHouse_i[i,k]), value(self.HousesADMM[k].HouseX.model.xHouse[i]), value(self.HousesADMM[k].HouseX.model.xBat[i]), value(self.HousesADMM[k].HouseX.model.bSoC[i])])
        writer.writerow(['P_Grid', 'Q_Grid'])
        for i in sorted (self.T_set):
            writer.writerow([value(self.X.xGrid[i]), value(self.X.xQ[i])])
        writer.writerow(['E', 'F'])
        for b in sorted (self.B_set):
            for i in sorted (self.T_set):
                writer.writerow([value(self.X.xV_E_ij[i,b]), value(self.X.xV_F_ij[i,b])])
        writer.writerow(['P_flow', 'Q_flow'])
        for b in sorted (self.B_set):
            for k in sorted (self.Neigh[b]):
                for i in sorted (self.T_set):
                    writer.writerow([value(self.X.Pij[i,(b,k)]), value(self.X.Qij[i,(b,k)])])
        
#        for k in sorted (self.H_set):
#            print("Energy consumption for house number:", k)
#            for i in sorted (self.T_set):
#                print(value(self.X.xHouse_k[i,k]))
#                #print(value(self.X.xHouse_k[i,k]), end=" ")
#        print("xGrid:")
#        for i in sorted (self.T_set):
#            print(value(self.X.xGrid[i]))
#        print("Voltages:")
#        print("-------------------------------------\nHigh price:")
#        for i in self.X.B:
#            print(i, (round(value(self.X.xV_E_ij[15.5,i]),2)), '+ j', (round(value(self.X.xV_F_ij[15.5,i]),2)))
#        print("-------------------------------------\nLow price:")
#        for i in self.X.B:
#            print(i, (round(value(self.X.xV_E_ij[23.5,i]),2)), '+ j', (round(value(self.X.xV_F_ij[23.5,i]),2)))
        #print("Total households cost when obeying aggregator problem:", sum (value(self.HousesADMM[k].HouseX.model.cost)for k in self.H_set))
#        print("Total network cost:", sum(( value(self.X.c2[i])*(value(self.X.xPlus[i])**2) ) for i in self.X.T))
#        print("Total houses costs:", sum((sum((value(self.X.ToU[i])*value(self.X.hPlus[i,k])) for k in self.X.H) ) for i in self.X.T))
#        print("Total houses PV:", sum((sum(((-0.10)*value(self.X.hMinus[i,k])) for k in self.X.H) ) for i in self.X.T))
        
        # Retrieve original cost for comparison with centralized solution
        #print('Optimization successfully performed. \nCost: ', self.model.cost(), ', Solver time = ', time.time() - time_start, ' s.')
        
        Ofile.close()
        
if __name__ == '__main__':
    
        #Criteria_abs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]#, 0.0000005, 0.0000001]    
        #Criteria_rel = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]#, 0.000005, 0.000001]
        
        #Set_Rho_41 = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
        #Set_Rho_42 = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        
        #Set_Rho_41 = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        #Set_Rho_42 = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        
        Ex_network = [41, 42]
        Ex_price = [5, 3]
        
        Time_start = time.time()
#        Ex_network = input("Please enter which network will be used, or 0 for the list of networks:\n")
#        while Ex_network == "0":
#            print("\n Number | Network | Buses | Houses: \n---------------------\n 1  |  Network 1  |  5  |  4  \n" + 
#                  " 2  |  Network 2  |  15 |  21 \n 3  |  Network 3  | 541 | 100 \n 31 | Network 3_1 | 100 |  40 " + 
#                  "\n 32 | Network 3_2 | 200 |  40 \n 33 | Network 3_3 | 140 |  70 \n 34 | Network 3_4 | 350 |  70 " + 
#                  "\n---------------------\n")
#            Ex_network = input("Please enter which network will be used:\n")
#            
#        if Ex_network == "41" or Ex_network == "42" or Ex_network == "43" or Ex_network == "44":
#            print("Adjust: request user to also input demand/PV profile!")
#            
#        Ex_price = input("Please enter which prices will be used, or 0 for the list of prices:\n")
#        while Ex_price == "0":
#            print("\n All prices in $/kWh. \n Number | Network: Quadratic & Linear | Houses: TOU & FIT\n---------------------\n" + 
#                  " 1  |  0.0001   &  0  |  0.12, 0.22, 0.52 & 0.10  \n" + 
#                  " 2  |  0.00002  &  0  |  0.12, 0.22, 0.52 & 0.10  \n" + 
#                  " 8  |  0.004    &  0  |  0.12, 0.22, 0.52 & 0.10  \n" + 
#                  "\n---------------------\n")
#            Ex_price = input("Please enter which network will be used:\n")
#        
#        solveprob = SubProblem(Ex_network, Ex_price)    
#        solveprob.solve(Ex_network, Ex_price)
#       
        #for net_ix, net_it in enumerate(Ex_network):
#        for crit_ix, crit_it in enumerate(Criteria_abs):
#            for r_ix, r_it in enumerate(Set_Rho_41):
#                print("Network",Ex_network[net_ix],"now using Rho =", Set_Rho_41[r_ix],"and e_[primal,dual]= [", Criteria_abs[crit_ix], ",", Criteria_rel[crit_ix],"].")
#                solveprob = SubProblem(Ex_network[net_ix], Ex_price[net_ix], Set_Rho_41[r_ix])    
#                solveprob.solve(Ex_network[net_ix], Ex_price[net_ix], Criteria_abs[crit_ix], Criteria_rel[crit_ix])
#                

        #for crit_ix, crit_it in enumerate(Criteria_abs):
            #for r_ix, r_it in enumerate(Set_Rho_42):
                #print("Network",Ex_network[1],"now using Rho =", Set_Rho_42[r_ix],"and e_[primal,dual]= [", Criteria_abs[crit_ix], ",", Criteria_rel[crit_ix],"].")
                #solveprob = SubProblem(Ex_network[1], Ex_price[1], Set_Rho_42[r_ix])    
                #solveprob.solve(Ex_network[1], Ex_price[1], Criteria_abs[crit_ix], Criteria_rel[crit_ix])
        print("Network",Ex_network[1],"now using Rho =", 0.75,"and e_[primal,dual]= [", 0.0001, ",", 0.001,"].")
        solveprob = SubProblem(Ex_network[1], Ex_price[1], 0.75)    
        solveprob.solve(Ex_network[1], Ex_price[1], 0.0001, 0.001)
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")

        #for crit_ix, crit_it in enumerate(Criteria_abs):
        #    for r_ix, r_it in enumerate(Set_Rho_41):
        print("Network",Ex_network[0],"now using Rho =", 0.4,"and e_[primal,dual]= [", 0.0001, ",", 0.001,"].")
        solveprob = SubProblem(Ex_network[0], Ex_price[0], 0.4)    
        solveprob.solve(Ex_network[0], Ex_price[0], 0.0001, 0.001)
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")

        print('---------------------------------')
        print('Total run time: ' , time.time() - Time_start, ' s.')