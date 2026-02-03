# Import required packages / files
import numpy as np
from sim_tools.distributions import Uniform, Beta
import networkx as nx
from agent import Agent

class AgentBasedModel:

	def __init__(self, parameters, seeds):

		# Parameters
		if sum(parameters['preference_weights'])=1:
			self.pref_weights = parameters['preference_weights']
		else:
			raise ValueError('Weights must sum to 1.')
			break

		# Store seeds
		self.seeds = seeds

	def generate_agents(self, num_agents):

		self.num_agents = num_agents	

		# Randomly draw agent attributes
		agent_probs = Uniform(low=0, high=1, random_seed=self.seeds[0])
		agent_distances = Beta(alpha=4, beta=2, random_seed=self.seeds[1])
		
		# Empty list for storing agents
		self.agent_list = []

		# Create agents
		for i in range(self.num_agents):
			attributes = {
				'id': i,
				'pref': 0.9*agent_probs[i] + 0.1*agent_distances[i],
				'distance': agent_distances[i]
			}
			self.agent_list.append(Agent(attributes))

		# Generate friendship network
		self.social_network = nx.newman_watts_strogatz_graph(self.num_agents, 2, 0.75)
		agent_map = dict(zip(range(self.num_agents), self.agent_list))
		nx.relabel_nodes(self.social_network, agent_map, copy=False)

		# Store friends as an agent attribute
		for i in self.agent_list:
			i.add_friends([j for j in social_network.neighbors(i)])

	def daily_step(self):

		for agent in agent_list:
			new_pref = self.pref_weights[0] * agent.pref[0] + \
			self.pref_weights[1] * agent.pref[2] + \ 
			self.pref_weights[2] * np.mean([x.member for x in agent.friends])
			agent.pref = np.append(agent.pref, new_pref)