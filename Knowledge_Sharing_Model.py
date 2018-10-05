from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation
import pandas as pd
import random
import networkx as nx
from astropy.table import Table, Column
from enum import Enum
import numpy as np
import cProfile
import psycopg2 as pg
import time
from pylens import pylens

# Keep this with Agent; Load data with roles to assign to agents; Remove Non-worker agents
Role_File = pd.read_csv('AgentJobs.csv') 
Role_File = Role_File[Role_File.role != "Uknown/Remove"]
Role_File = Role_File[Role_File.role != "patient"]
Role_File = Role_File.reset_index(drop=True) # reindex so that row numbers are 0 to 2127

Role_File_Sub = pd.read_csv('AgentJobs_Sub.csv') 

class Knowledge_Sharing_Agent(Agent):
    def __init__(self, agent_number, model, hasKnowledge, valences):
        super().__init__(agent_number, model)
        
        self.agent_number = agent_number       
        getRole = Role_File[(Role_File.id==self.agent_number)] # Pulls from role file the agents role
        self.role = str(getRole.iloc[0][1])  
        self.currentStep=0   
        self.model = model
        self.valences = valences
        
        # if agent number is equal to 'start' input, then agent has knowledge
        # time step that knowledge was received
        self.timeKnowledge = np.nan
        self.hasKnowledge = hasKnowledge
        # track time that attitudes and belief valences changed
        self.timeNewAttitude = np.nan
        self.timeNewValences = np.nan
      
        # agent id from which knowledge was received
        self.receivedKnowledge = np.nan
        
        ### If running cognitive Phase 2, assign a positive attitude to the agent that has the knowledge
        # assign random thresholds, normal distribution with mean of 1
        self.attitude = 0
        if self.model.cognitivePhase == 2:
            numbers = [-1,1]
            r = random.choice(numbers)
            self.attitude = r
            if self.hasKnowledge == True:
                self.attitude = 1
                self.timeKnowledge = 0
        
        if self.hasKnowledge == True:
            self.timeKnowledge = 0
             
        # assign random threshold from a normal distribution
        mu=1
        sigma=10
        th = min(100, max(0, random.gauss(mu, sigma)))
        self.threshold = round(th)
        
        sql='select "min" from endtimes where "id.x"=' + str(self.agent_number)
        model.cur.execute(sql)
        rows = model.cur.fetchall()
        self.activityEnd=rows[0][0]
        
        # get list of all my activity end times
        sql='select "endTime" from tmp_allendtimesM3RB1 where "id"=' + str(self.agent_number) + ' and "endTime"<=' + str(model.totalSteps)
        self.myEndTimes=pd.read_sql_query(sql,model.con)
        
        # get dataframe with my interactions
        sql='select "id.y", "startTime.x","endTime.x","interactionTime" from tmp_cnaM3RB1 where "id.x"=' + str(self.agent_number) + 'and "endTime.x"<=' + str(model.totalSteps)
        self.mySchedule=pd.read_sql_query(sql,model.con)
        
    def step(self):
   
        # check if its the end of the agent's time at its current activity location
        # if so, update contact network to reflect any interactions that occurred   
        if (self.activityEnd == self.currentStep):
            # check what other agents are at the same location, at the same time as me
            activities = self.mySchedule[(self.mySchedule['endTime.x']==self.currentStep)]  
            # get agents interactions
    
            # check for any interactions that occured between agents
            if not activities.empty:
                   
                agentInteractions = self.interact(activities)
                
                # create or update contact tie between the agents that interacted
                self.updateContactNetwork(agentInteractions)
            
                # update social network based on updated contact network
                # the social network will also be updated later based on attitude homophily)
                self.updateSocialNetwork(agentInteractions)
         
                # run cognitive model (i.e., lens) to update attitudes based on current 
                # state and updated social network
                # return updated beliefs
                if self.model.cognitivePhase == 3:
                    self.runCognitiveModel(agentInteractions)
                
                # calculate new attitude
                if self.model.cognitivePhase != 1:
                    #self.attitude = self.calculateAttitude()
                    self.calculateAttitude()
                
                # if agent has knowledge, determine if agent should share knowledge with agent that currently 
                # interacting with 
                # (if this is multiple agents, then select agent with highest social tie)
                # for simplicity, randomly pick one agent to interact with (if there are multiple agents)
                if (self.hasKnowledge == True) & (self.currentStep > self.model.startKnowledge):
                    self.shareKnowledge(agentInteractions)
                    
            self.activityEnd = self.updateActivityEnd()
            
            ## update contact and social network at end of each week to reflect weakening of any ties
            if (self.currentStep % 10080 == 0):
                # get my ego network (nodes I'm directly linked to)
                ego=nx.neighbors(self.model.socialNetwork,self)
                # sum the ties of my connections that have an oppostive attitude as me
                tieSum=0
                
                beta1 = self.model.beta1 # weight given to contacts
                beta2 = self.model.beta2 # weight given to homophily
                beta3 = self.model.beta3
            
                # attitude effect is zero for first model
                if self.model.cognitivePhase == 1:
                    beta2 = 0
                    beta3 = self.model.beta2 + self.model.beta3
            
                for i in range(len(list(ego))):
                    node=ego[i]
                    # get interaction times
                    self.model.contactNetwork[self][node]['weight'] = (self.model.interactionTimes[self][node]['weight'] / float(self.currentStep))
                    contactEdgeWeight=contactEdgeWeight=self.model.contactNetwork[self][node]['weight']  
                    self.model.socialNetwork[self][node]['weight'] = beta1 * contactEdgeWeight + beta2 * np.exp(-np.abs(self.attitude - node.attitude)) + beta3 * (self.role == node.role)
                  
                    
        # Advance Step Counter
        self.currentStep += 1
   
    #### AGENT FUNCTIONS ###
    
    # creates object of activity end time to check if simulation is at the end of the agent's activity 
    def updateActivityEnd(self):
     
        ## get agents next activity
        #endTime = min(self.myEndTimes[(self.myEndTimes['endTime']>self.currentStep)])     
        endTime = self.myEndTimes[(self.myEndTimes['endTime']>self.currentStep)]

        if not endTime.empty:
            endTime = min(endTime['endTime'])
            activityEnd = int(endTime)
        else:    
            activityEnd = self.model.totalSteps
 
        return activityEnd
            
    # input is the agent's activity schedule
    def interact(self,otherAgentsActivities):
    
        # store list of potential agents to interact with and their interaction times
        # create a 2D list to store this information
        interactAgents = []
        interactAgents.append([])
        interactAgents.append([])
        
        ## calculate interaction time with other agents
        ## interate through each agent
        # if not otherAgentsActivities.empty:
        for i in range(0, len(otherAgentsActivities)):
            # get inforamtion about other agent
            otherAgentId=int(otherAgentsActivities.iloc[i]['id.y'])
            interactionTime=int(otherAgentsActivities.iloc[i]['interactionTime'])
            
            # get other agent
            otherAgent = [t for t in self.model.all_agents if t.agent_number == otherAgentId]
            #otherAgent = self.model.all_agents[otherAgentId]
            otherAgent = otherAgent[0]
            
            # add agent and interaction time to list
            interactAgents[0].append(otherAgent)
            interactAgents[1].append(interactionTime)
            
        # add noise to potential interactions
        # with a certain probability, select a random agent to interact with (could be an agent in the simulation)
        # randomly select an agent out of those I interacted with
        p=self.model.networkNoise
        rn=(random.randint(0,100))/100
        if rn<p:
            # select a random agent
            numbers = list(range(0,len(self.model.all_agents)))
            r = random.choice(numbers)
            otherAgent = self.model.all_agents[r]
            interactionTime = 10
            
            # add agent and interaction time to list
            interactAgents[0].append(otherAgent)
            interactAgents[1].append(interactionTime)
           
        return interactAgents
    
      
    def updateContactNetwork(self,agentInteractions):
        # loop through all agents and their interation times
        
        for i in range(len(agentInteractions[0])):
            # get agent and interaction time
            otherAgent=agentInteractions[0][i]
            interactionTime=agentInteractions[1][i]
            
            # check edge weight
            # if there is no edge, the output will be 0            
            edgeWeight=self.model.contactNetwork.get_edge_data(self,otherAgent,default=0)
             
            # if edge does not exist, create an edge
            # weight is equal to the interaction time
            if edgeWeight == 0:
                self.model.interactionTimes.add_edge(self,otherAgent,weight=interactionTime/2)
                self.model.contactNetwork.add_edge(self,otherAgent,weight=(self.model.interactionTimes[self][otherAgent]['weight'] / float(self.currentStep)))
            # if the edge does exist, update the edge weight by the interaction time
            else:
                self.model.interactionTimes[self][otherAgent]['weight'] += interactionTime/2
                self.model.contactNetwork[self][otherAgent]['weight'] = (self.model.interactionTimes[self][otherAgent]['weight'] / float(self.currentStep))
        
    
    # input is the agent (self) and another agent's contact tie, previous social tie (if it existed), 
    # and attitudes of both agents 
    def updateSocialNetwork(self,agentInteractions):
        # as a starting point, the social network will be the same as the contact network (and will not reflect 
        # attitude homophily)
        if self.model.networkPhase == 1:
            self.model.socialNetwork = self.model.contactNetwork
        
        elif self.model.networkPhase == 2:
            beta1 = self.model.beta1 # weight given to contacts
            beta2 = self.model.beta2 # weight given to homophily
            beta3 = self.model.beta3
            
            # attitude effect is zero for first model
            if self.model.cognitivePhase == 1:
                beta2 = 0
                beta3 = self.model.beta2+self.model.beta3
            
            # update social network ties
            for i in range(len(agentInteractions[0])):
                otherAgent=agentInteractions[0][i]
                socialEdgeWeight=self.model.socialNetwork.get_edge_data(self,otherAgent,default=0)
                contactEdgeWeight=self.model.contactNetwork[self][otherAgent]['weight']   
                
                # if edge does not exist, create an edge                
                if socialEdgeWeight == 0:
                    self.model.socialNetwork.add_edge(self,otherAgent,weight=beta1 * contactEdgeWeight + beta2 * np.exp(-np.abs(self.attitude - otherAgent.attitude)) + beta2 * (self.role == otherAgent.role))
                
                # if the edge does exist, update the edge weight
                else:
                    self.model.socialNetwork[self][otherAgent]['weight'] = beta1 * contactEdgeWeight + beta2 * np.exp(-np.abs(self.attitude - otherAgent.attitude)) + beta3 * (self.role == otherAgent.role)
                    
        else:
            self.model.socialNetwork = self.model.contactNetwork
    
    def runCognitiveModel(self, agentInteractions):
        # this is where we will link to LENS and it will return updated beliefs
      
        # randomly select an agent out of those I interacted with
        otherList = agentInteractions[0]
        numbers = list(range(0,len(otherList)))
        r = random.choice(numbers)
        otherAgent = otherList[r]
        
        ## Copy and pasted code from LENS
        # loop through valences
        for i in range(len(self.valences)):
            # get agent and interaction time
            otherValence=float(otherAgent.valences[i])
            myValence=float(self.valences[i])
            # create list to place interaction valences
            interactValence=[0]*len(self.valences)
            
            # get the weight of social tie
            sWeight=self.model.socialNetwork[self][otherAgent]['weight']
          
            # compute the interaction valences (which will be input into lens)
            interactValence[i]=(sWeight * (otherValence - myValence))/2 + myValence
        
        ## values are positive and negative valences of other agent
        # need to replace 1s and 0s with valences of other agent (as a list)
        pylens.write_ex_file('./lens/lens_output/input_'+str(self.agent_number)+'.ex',
                     [interactValence],
                     'B',
                     'agent_'+str(self.agent_number)+'_input')
        
        # with the examples you can now have lens return an output
        # given the wgt3.wt file
        # Note you call lens with a different in file
        if self.model.annType == 1:
            pylens.call_lens('./lens/02-global_cascades_update.in', {'agent_id': str(self.agent_number)})
        if self.model.annType == 2:
            pylens.call_lens('./lens/02-recurrent_update.in', {'agent_id': str(self.agent_number)})
        
        # the new results are saved to an out file
        # we now need to extract the new values from the outfile
        # pass in the outfile
        # num_lines refer to how many neural net nodes you have for the output
        # the split_index refers to which column of values in the outfile is the new state values
        # 0 for the first column
        # 1 for the second column, etc
        
        # returns positive and negative valences of 10 beliefs
        if self.model.annType == 1:
            new_state = pylens.get_new_state_from_outfile('./lens/lens_output/input_'+str(self.agent_number)+'.out',
                                                      num_lines=20,
                                                      split_index=0)
        if self.model.annType == 2:
            new_state = pylens.get_new_state_from_outfile('./lens/lens_output/AgentState.out',
                                                          num_lines=20,
                                                          split_index=0,
                                                          lens_type='recurrent')

        
        # making sure the length of the new state is correct
        # now you have the new state values you can use to assign back to your agent
        #assert len(new_state == 20)
        self.valences = new_state
        self.timeNewValences = self.currentStep
        
    def calculateAttitude(self):
        # calculate agents attitude score
        # return attitude score
        if self.model.cognitivePhase == 3:
            ## calculate attitude based on belief values
            # first half of valences are positive, second half are negative
            x_diff=0.0
            x=0.0
            for i in range(int(len(self.valences)/2)):    
                x_pos=float(self.valences[i])
                x_neg=float(self.valences[i+len(self.valences)/2])
                x_diff=x_pos-x_neg
                x=x+x_diff

            self.attitude=x/(len(self.valences)/2)
            self.timeNewAttitude = self.currentStep
            
        elif self.model.cognitivePhase == 2:
            # get agents in my social network, sum ties of agents with opposite attitude
            # if the sum of the ties exceeds my threshold, update my attitude
            
            # get my ego network (nodes I'm directly linked to)
            ego=nx.neighbors(self.model.socialNetwork,self)
            # sum the ties of my connections that have an oppostive attitude as me
            tieSum=0
            
            for i in range(len(ego)):
                node=ego[i]
                # sum tie strength of all my connections that have the oppostive attitude as me
                otherAttitude=node.attitude
                if otherAttitude != self.attitude:
                    # weight strenght of tie by step in the simulation
                    x=self.model.socialNetwork[self][node]['weight']
                    x=x/self.currentStep
                    tieSum=x+tieSum
               
            # check if sum of ties is greater than my threshold
            # if so, then flip my attitude
            if tieSum > self.threshold:   
                self.attitude = self.attitude * -1
                self.timeNewAttitude = self.currentStep
        else:
            pass
                      
    def shareKnowledge(self,agentInteractions):
        
        # select an agent to potentially share knowledge with
        otherList = agentInteractions[0]
        numbers = list(range(0,len(otherList)))
        r = random.choice(numbers)
        otherAgent = otherList[r]
          
        if self.model.cognitivePhase == 3:    
            # probability of spreading knowledge follows a logistic function
            #p=exp(self.attitude * self.model.a) / (1 + exp(self.attitude * self.model.a))
            p=1/(1+np.exp(-self.model.a*self.attitude))
            
            # if other agent has knowledge, then do nothing
            if otherAgent.hasKnowledge == False:
            # if knowledge was shared, update knowledge value of other agent
                rn=(random.randint(0,100))/100
                if rn<p:
                    otherAgent.hasKnowledge = True
                    self.receivedKnowledge = otherAgent.agent_number
                    self.timeKnowledge = self.currentStep
        
        # If running cognitive phase 1 or 2, then make knowledge spread a simple cascade model
        # randomly select an agent and spread the knowledge with a given probabilty of sucess
        if self.model.cognitivePhase == 2:
            if self.attitude == 1:
                # return whether agent shared knowledge with other agent using logistic probability function
                p=self.model.probSuccess # Success probability
                # if other agent has knowledge, then do nothing
                if otherAgent.hasKnowledge == False:
                # if knowledge was shared, update knowledge value of other agent
                    rn=(random.randint(0,100))/100
                    if rn<p:
                        otherAgent.hasKnowledge = True
                        self.receivedKnowledge = otherAgent.agent_number
                        self.timeKnowledge = self.currentStep
        
        if self.model.cognitivePhase == 1:
            # return whether agent shared knowledge with other agent using logistic probability function
            p=self.model.probSuccess # Success probability
            # if other agent has knowledge, then do nothing
            if (otherAgent.hasKnowledge == False):
            # if knowledge was shared, update knowledge value of other agent
                rn=(random.randint(0,100))/100
                if rn<p:
                    otherAgent.hasKnowledge = True
                    self.receivedKnowledge = otherAgent.agent_number
                    self.timeKnowledge = self.currentStep
                    
###############
#### MODEL ###
##############

class Knowledge_Sharing_Model(Model):
    def __init__(self, cognitivePhase, annType, networkPhase, numRandomKnowledgeAgents, knowledgeAgentID, removeAgentID, startKnowledge, numSeeds, probSuccess, a, networkNoise, sensitivityAnalysis, subgraph, beta1, beta2, beta3, con, step_number, systemTime):
        print("in model")
        self.cognitivePhase = cognitivePhase
        self.networkPhase = networkPhase
        self.probSuccess = probSuccess
        self.con = con
        self.cur = self.con.cursor()
        self.a = a
        self.networkNoise = networkNoise
        self.sensitivityAnalysis = sensitivityAnalysis
        self.subgraph = subgraph
        self.startKnowledge = startKnowledge
        self.numSeeds = numSeeds
        self.systemTime = systemTime
        self.numRandomKnowledgeAgents = numRandomKnowledgeAgents
        self.annType = annType
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        
        self.totalSteps=step_number
 
        self.cur.execute('DROP TABLE IF EXISTS tmp_cnaM3RB1')
        self.con.commit()
        self.cur.execute('DROP TABLE IF EXISTS tmp_allendtimesM3RB1')
        self.con.commit()
        
        # get agents to remove
        remove=[]
        for i in range(len(removeAgentID)): 
            x = removeAgentID[i]
            remove.append(x)
        
        # do not import schedules of removed agents
        if len(remove) > 0:
            removeStr=str(remove)

            removeStr=removeStr.replace("[", "(")
            removeStr=removeStr.replace("]", ")")
            
            # subset contact network for time of simulation
            sql='SELECT * INTO tmp_cnaM3RB1 FROM cnfin WHERE "endTime.x"<=' + str(self.totalSteps) + ' and "id.x" NOT IN ' + removeStr + ' and "id.y" NOT IN ' + removeStr
            self.cur.execute(sql)
            self.con.commit()

            # subset end times for time of simulation
            sql='SELECT * INTO tmp_allendtimesM3RB1 FROM allendtimes WHERE "endTime"<=' + str(self.totalSteps) + ' and "id" NOT IN ' + removeStr 
            self.cur.execute(sql)
            self.con.commit()
            print("removed agents")
        
        else:
            # subset contact network for time of simulation
            sql='SELECT * INTO tmp_cnaM3RB1 FROM cnfin WHERE "endTime.x"<=' + str(self.totalSteps)
            self.cur.execute(sql)
            self.con.commit()

            # subset end times for time of simulation
            sql='SELECT * INTO tmp_allendtimesM3RB1 FROM allendtimes WHERE "endTime"<=' + str(self.totalSteps)
            self.cur.execute(sql)
            self.con.commit()
        print("finished db calls")
        
        # if runs are for sensitivity analysis, use agents in one subgraph
        if self.subgraph == True:
            roles = Role_File_Sub
        else:
            roles = Role_File
        print("imported roles")
        
        # remove agents
        if len(remove) > 0:
            roles=roles[roles['id'].isin(remove)== False]
            roles = roles.reset_index(drop=True) # reindex so that row numbers are 0 to 2127
        
        # this is the number of agents
        size = len(roles.index)
        
        # get agents that have knowledge at initialization
        start=[]
        if self.numRandomKnowledgeAgents>0:
            for i in range(self.numRandomKnowledgeAgents):
                number = random.randrange(0,size)
                x = roles.ix[number,0]
                start.append(x)
        else:
            for i in range(len(knowledgeAgentID)):  
                x = knowledgeAgentID[i]
                start.append(x)
        
        # get agents that are seeded for purposes of LENS
        seed=[]
        if numSeeds>0:
            for i in range(numSeeds):
                number = random.randrange(0,size)
                
                x = roles.ix[number,0]
                seed.append(x)
            
        self.num_agents = size # Number of agents
        self.schedule = RandomActivation(self) # Random activation scheduler
     
        # Create contact and social networks
        # All agents will be part of the contact and social networks but no edges exist at the beginning of the simulation
        self.interactionTimes=nx.DiGraph() # change to DiGraph? error?
        self.contactNetwork=nx.DiGraph()
        self.socialNetwork=nx.DiGraph()
        
        # Create Bags for Network Data
        self.interaction_table=pd.DataFrame()
        self.contact_table=pd.DataFrame() # Empty contact network table to add to
        self.social_table=pd.DataFrame() # Empty social network table to add to
        
        self.knowledge_table=pd.DataFrame() # Empty knowledge table to add to
        self.attitude_table=pd.DataFrame() # Empty attitude table to add to
        self.valences_table=pd.DataFrame() # Empty belief valence table to add to
      
        self.all_agents=[] # Creates empty bag to place agents in    
        self.step_number=0 # Set step number
        
        # Create agents
        for i in range(self.num_agents):
            agent_number = roles.iat[i,0]
   
            ## see if agent has knowledge
            hasKnowledge = False
            if (agent_number in start):
                hasKnowledge = True
            
            ## create initial valence values, agents can have positive (postivie valences are 1, negative valences are 0), negative (positive valences are 0, negative valences 1). Positive valences are first 10, negative valences are last 10
            ## or neutral (all 0s) attitudes
            numbers = [0, 1]
            r = random.choice(numbers)
            n=10
            if r==1: # agent has a positive attitude towards knowledge sharing (canonical attitude)
                prototype = [1] * n + [0] * n
            else: # agent has a negative attitude towards knowledge sharing (canonical attitude)
                prototype = [0] * n + [1] * n
                
            # seed a certain number of agents, the rest will have valences of all 0
            valences = [0] * 20
            if (agent_number in seed):
                valences = [1] * n + [0] * n
                
            a = Knowledge_Sharing_Agent(agent_number, self, hasKnowledge, valences)
    
            self.all_agents.append(a)
            self.schedule.add(a)
            
            # add agent to contact and social networks
            self.interactionTimes.add_node(a,agentID=agent_number)
            self.contactNetwork.add_node(a,agentID=agent_number)
            self.socialNetwork.add_node(a,agentID=agent_number)
            
            if cognitivePhase == 3:
                # Copy and Pasted code from Lens
                # we need to create examples for the agent to learn
                # note that the examples_list is a 2d list
                # this example has 2 examples for the agent to train on
                fileName = './lens/lens_output/example_train_'+str(agent_number)+'.ex'
                
                # the mutate function will generate 50 instances of attitude valences and add variation to the canonical values
                pylens.write_ex_file(fileName,
                         [pylens.mutate(prototype, .5) for x in range(50)],
                         'B',
                         'agent_'+str(agent_number)+'_train')
                
                # train agent 3
                # you will get an output file called wgt3.wt for the agent 3 weight file
                if self.annType == 1:
                    pylens.call_lens('./lens/01-global_cascades_train.in', {'agent_id': str(agent_number)})
                
                if self.annType == 2:
                    pylens.call_lens('./lens/01-recurrent_train.in', {'agent_id': str(agent_number)})
                   
        print("simulation initialized")
           
    def step(self):
     
        # collect knowledge, attitude, and belief valence data
        # loop through each agent and only store agent info if necessary
        for i in range(len(self.all_agents)):
            a=self.all_agents[i]
           
            # for sensitivity analysis, only collect knowledge information
            # collect knowledge flows
            if ((a.timeKnowledge == a.currentStep-1) and (np.isnan(a.receivedKnowledge) == False)) or ((a.timeKnowledge == 0) and (a.currentStep == 0)):
                kn={'CurrentStep' : a.currentStep, 'MyAgentID' : a.agent_number, 'OtherAgentID' : a.receivedKnowledge,
                       'TimeKnowledgeReceived' : a.timeKnowledge, 'HasKnowledge' : a.hasKnowledge, 'Model' : self.cognitivePhase, 'NumInitialKnowledgeAgents' : self.numRandomKnowledgeAgents, 'NumberAgentsSeeded' : self.numSeeds, 'SuccessProbability' : self.probSuccess, 'RateOfLogisticCurve' : self.a, 'SensitivityAnalysis' : self.sensitivityAnalysis, 'Subgraph' : self.subgraph}
                kn=pd.DataFrame(data=kn, index=np.arange(0, 1))

                self.knowledge_table = self.knowledge_table.append(kn)  
            
            if self.sensitivityAnalysis == False:
                # collect attitude values if agent's attitude has changed
                if ((a.timeNewAttitude == a.currentStep-1) or (a.currentStep == 0)):
                    at={'CurrentStep' : a.currentStep, 'TimeNewAttitude' : a.timeNewAttitude, 'MyAgentID' : a.agent_number, 'Attitude' : a.attitude}
                    at=pd.DataFrame(data=at, index=np.arange(0, 1))
                    self.attitude_table = self.attitude_table.append(at)
        
        # collect contact network, social network, and interaction times
        if self.step_number % 1440 == 0:
            print("current step: ", self.step_number)

            if self.networkPhase != 1:
                # create table with social network
                #sn=nx.to_pandas_dataframe(self.socialNetwork)
                sn=nx.to_pandas_adjacency(self.socialNetwork)
                sn['Step']=self.step_number
                sn['contactWeight']=self.beta1
                sn['attWeight']=self.beta2
                sn['roleWeight']=self.beta3
                self.social_table = self.social_table.append(sn)
           
        self.step_number = self.step_number+1
        
        self.schedule.step()
        
        
        
        
        
