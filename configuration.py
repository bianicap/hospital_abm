### TO DO:
## 1. FIX LOGISTIC FUNCTION - done
## 2. ADD NOISE TO CONTACT NETWORK (CAN DO THIS AFTER/DURING SENSITIVITY ANALYSIS) - done
## 3. SENSITIVITY ANALYSIS -- 
    ## A. RUNS ON ONE SUBGRAPH (JUST SUBSET AGENTJOBS.CSV INPUT FILE), RUN FOR 2 SIMULATION WEEKS - done
    ## B. METRIC OUTPUTS -- % AGENTS RECEIVED KNOWLEDGE, TIME LAST AGENT RECEIVED KNOWLEDGE (DON"T NEED TO OUTPUT OTHER CSVs)
    
import sys
sys.version

from pylens import pylens as pylens
import numpy as np
import cProfile
from enum import Enum
import pandas as pd
import random
import time
import psycopg2 as pg
import datetime

from Knowledge_Sharing_Model import *

def modelRuns(cognitivePhase, annType, networkPhase, numRandomKnowledgeAgents, knowledgeAgentID, removeAgentID, startKnowledge, numSeeds, probSuccess, a, networkNoise, sensitivityAnalysis, subgraph, beta1, beta2, beta3, con, step_number, runs):
    
    # get system start time
    systemTime = datetime.datetime.now()
    
    print("model function started")

    for j in range(runs):

        ### Run the Model ###
        t0total = time.time()
        model = Knowledge_Sharing_Model(cognitivePhase, annType, networkPhase, numRandomKnowledgeAgents, knowledgeAgentID, removeAgentID, startKnowledge, numSeeds, probSuccess, a, networkNoise, sensitivityAnalysis, subgraph, beta1, beta2, beta3, con, step_number, systemTime) # to run full population, replace 50 with 'size' and comment out line 16

        for i in range(step_number):
            model.step()
        
        print("model complete and about to write results to file")
        # add system time to file names
        current_time = model.systemTime.strftime("%Y-%m-%d_%H:%M:%S")

        ### Export Knowledge Network ###
        results_KN = model.knowledge_table

        #results_KN = model.knowledge_table  
        KN_name = str(current_time)+'_Run_'+str(j)+'_Knowledge_Network.csv'
        results_KN.to_csv(KN_name)
        
        results_SN = model.social_table

        if not results_SN.empty:

            # Clean Contact Network Data
            results_SN.index.name = 'ID'
            results_SN.reset_index(inplace=True)

            # change object names to agent IDs
            IDs = nx.get_node_attributes(model.socialNetwork,'agentID')
            IDs_data = pd.DataFrame(list(IDs.items()),columns=['Object','agentID'])

            sn=pd.merge(results_SN, IDs_data, left_on='ID', right_on='Object',how='left')

            sn.drop('Object', axis=1, inplace=True)
            sn.drop('ID', axis=1, inplace=True)

            sn = sn.reindex_axis(['agentID'] + list(sn.columns[:-1]), axis=1)

            # change object names as column names to agent IDs
            colnames=list(sn)
            colnames=colnames[1:]
            colnames=colnames[:-4]

            newnames=[] # list to place updated column names

            for i in range(len(colnames)):
                name = colnames[i]
                # find agent ID that matches this object name
                name = IDs_data.loc[IDs_data['Object'] == name] 
                name = name.agentID.iloc[0]        
                newnames.append(name)

            newnames.insert(0, "agentID")
            newnames.insert(len(newnames)+1,"Step")
            newnames.insert(len(newnames)+1,"contactWeight")
            newnames.insert(len(newnames)+1,"attWeight")
            newnames.insert(len(newnames)+1,"roleWeight")
            
            sn.columns = newnames

            SN_name = str(current_time)+'_Run_'+str(j)+'_Social_Network.csv'
            sn.to_csv(SN_name)

        if model.sensitivityAnalysis == False:

            ### Export Attitude Results ###
            results_AT = model.attitude_table

            #results_KN = model.knowledge_table  
            AT_name = str(current_time)+'_Run_'+str(j)+'_Attitudes.csv'
            results_AT.to_csv(AT_name)

        t1total = time.time()
        total=t1total-t0total
        print("total time:", total)

# connect to postgres database
con = pg.connect(dbname='abm',
                 user='',
                 host='',
                 port='',
                password='')
cur = con.cursor()

sensitivityAnalysis = False
subgraph = True

step_number = 44000
runs = 1

### Select the phase of the model to run

# Phase 1: Cascasde model of knowledge diffusion
# Phase 2: Threshold model that incorporates attitude diffusion
# Phase 3: Cognitive model using the Reasoned Action Approach
cognitivePhase = 3

# if cognitive phase is 3, set if feedforward (1) or recurrent (2)
annType = 2

# Phase 1: The social network is the contact network
# Phase 2: Add exogenous effects (e.g., homophily)
networkPhase = 2

## simulation time to start knowledge spread
startKnowledge = 10080

### Other User specified variables
# probability that an agent that has knowledge will pass the knowledge to an interaction agents that does not have knowledge
# used in cognitive phase 1 and 2
probSuccess=.5
    
### Select how to give initial agents the knowledge: random agent(s) or specific agent(s)
# Number of agents to randomly give knowledge to
# If this is zero, knowledge will be assigned to specific agent(s)
numRandomKnowledgeAgents=10

# If knowledge will be given to a specific agent(s), assign those agents here
# and make sure numRandomKnowledgeAgents is set to 0
knowledgeAgentID=[]

# If agents are to be removed, this is the list of agents to remove
removeAgentID=[14727,174756,14810,174729,34735,14780,174754,174717,34728,34708]

## if running cognitive phase 3, this is the rate of curve used in logistic function that sets probability 
# of spreading knowledge to an interacting agent
a=5
# if running cognitive phase 3, Number of agents to seed with positive attitude
numSeeds=10

### ADD noise to contact network creation
# with a given probability agent will interact with a random agent after an activity
networkNoise = .001

# set social network weights
beta1 = .95
beta2 = .025
beta3 = .025

## if running cognitive phase 1 or 2, this is the simple success probability of spreading the knowledge 
# to an interacting agent
# if performing sensitivity analysis, want to loop through all combinations of parameters
if (sensitivityAnalysis == True):
    # list all possible values of parameters  
    #listProbSuccess=[1]
    #listNumKnowledgeAgents=[3]
    listBeta2=[0,.01,.02,.03]
  
    for i in range(len(listBeta2)):
        beta2=listBeta2[i]  
        beta3=listBeta2[i]
        beta1=1-(beta2 + beta3)
        modelRuns(cognitivePhase, annType, networkPhase, numRandomKnowledgeAgents, knowledgeAgentID, removeAgentID, startKnowledge, numSeeds, probSuccess, a, networkNoise, sensitivityAnalysis, subgraph, beta1, beta2, beta3, con, step_number, runs)

else:
    modelRuns(cognitivePhase, annType, networkPhase, numRandomKnowledgeAgents, knowledgeAgentID, removeAgentID, startKnowledge, numSeeds, probSuccess, a, networkNoise, sensitivityAnalysis, subgraph, beta1, beta2, beta3, con, step_number, runs)
    


