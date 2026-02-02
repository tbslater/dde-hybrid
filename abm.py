# Import required packages / files
import numpy as np
from sim_tools.distributions import DistributionRegistry
import random
import networkx as nx
from agent import Agent

class AgentBasedModel:

	def __init__(self, parameters, seed):

		# Dictionary-based configuration (named distributions)
		config_dict = {
			'agent_probabilities': {
				'class_name': 'Uniform',
				'params': {'low': 0, 'high': 1}
			}, 
			'distance_from_gym': {
				'class_name': 'Gamma',
				'params': {'alpha': 2.5, 'beta': 3.5}
			}
		}
		
		# Create all distributions with a master seed
		distributions = DistributionRegistry.create_batch(config_dict, main_seed=seed)
		
		# Access distributions by name
		self.probs_dist = distributions['agent_probabilities']
		self.distance_dist = distributions['distance_from_gym']

	def generate_agents(self, max_agents):

		self.num_agents = max_agents	

		# Randomly draw agent attributes
		agent_probs = self.probs_dist.sample(self.num_agents)
		agent_distances = self.distance_dist.sample(self.num_agents)
		
		# Empty list for storing agents
		self.agent_list = []

		# Create agents
		for i in range(self.num_agents):
			attributes = {
				'id': i
				'prob': agent_probs[i],
				'distance': agent_distances[i]
			}
			self.agent_list.append(Agent(attributes))

		# Generate friendship network
		social_network = nx.newman_watts_strogartz_graph(self.num_agents, 2, 0.1)
		agent_map = dict(zip(range(self.num_agents), self.agent_list))
		nx.relabel_nodes(social_network, agent_map, copy=False)

		# Store friends as an agent attribute
		for i in self.agent_list:
			i.add_friends([j for j in social_network.neighbors(i)])