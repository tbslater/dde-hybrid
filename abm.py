# Import required packages / files
import numpy as np
from sim_tools.distributions import DistributionRegistry
import random
import networkx as nx
from agent import Agent

class AgentBasedModel:

	def __init__(self, parameters, seed):

		# Parameters
		if sum(parameters['preference_weights'])=1:
			self.pref_weights = parameters['preference_weights']
		else:
			raise ValueError('Weights must sum to 1.')
			break
		
		# Dictionary-based configuration (named distributions)
		config = {
			'agent_probabilities': {
				'class_name': 'Uniform',
				'params': {'low': 0, 'high': 1}
			}, 
			'relative_distance': {
				'class_name': 'Beta',
				'params': {'alpha': 4, 'beta': 2}
			}
		}
		
		# Create all distributions with a master seed
		distributions = DistributionRegistry.create_batch(config, main_seed=seed)
		
		# Access distributions by name
		self.probs_dist = distributions['agent_probabilities']
		self.distance_dist = distributions['relative_distance']

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
				'id': i,
				'pref': 0.9*agent_probs[i] + 0.1*agent_distances[i],
				'distance': agent_distances[i]
			}
			self.agent_list.append(Agent(attributes))

		# Generate friendship network
		social_network = nx.newman_watts_strogatz_graph(self.num_agents, 2, 0.75)
		agent_map = dict(zip(range(self.num_agents), self.agent_list))
		nx.relabel_nodes(social_network, agent_map, copy=False)
		nx.display(social_network, node_size=10, node_label=False, edge_label=False,
				   node_alpha=0.8,edge_alpha=0.8, edge_width=0.5)

		# Store friends as an agent attribute
		for i in self.agent_list:
			i.add_friends([j for j in social_network.neighbors(i)])

	def daily_step(self):

		for agent in agent_list:
			new_pref = self.pref_weights[0] * agent.pref[0] + \
			self.pref_weights[1] * agent.pref[2] + \ 
			self.pref_weights[2] * np.mean([x.member for x in agent.friends])
			agent.pref = np.append(agent.pref, new_pref)