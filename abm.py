# Import required packages/files
import numpy as np
from sim_tools.distributions import Beta, spawn_seeds
import networkx as nx
import math

class Agent:

	def __init__(self, threshold):

		# Threshold for becoming vaccinated
		self.threshold = threshold
		# Vaccination status
		self.vaccinated = 0

	def add_friends(self, friends):

		self.friends = friends
		self.num_friends = len(self.friends)

class AgentBasedModel:

	def __init__(self, parameters, main_seed):

		# Parameters
		self.max_daily_vax = parameters['max_daily_vax']
		self.influence_param = parameters['influence_param']
		self.beta_params = parameters['beta_params']
		if 0 <= parameters['weight'] <= 1:
			self.weight = parameters['weight']
		else:
			raise ValueError('Weight must be between 0 and 1.')

		# Store seeds
		self.seeds = spawn_seeds(2, main_seed)

		# Generator for sampling agents for vaccination
		self.generator = np.random.default_rng(self.seeds[0])

		# Store daily vaccinations
		self.daily_vax = np.array([0])

	def generate_agents(self, population):

		# Number of agents to generate
		self.population = population

		# Randomly draw thresholds from Unif(0,1)
		threshold_dist = Beta(alpha1=self.beta_params[0], 
							  alpha2=self.beta_params[1],
							  random_seed=self.seeds[1])
		thresholds = threshold_dist.sample(self.population)

		# Create agents
		self.agent_list = []
		for i in range(self.population):
			self.agent_list.append(Agent(thresholds[i]))

		# Generate friendship network
		self.social_network = nx.newman_watts_strogatz_graph(population, 4, 0.1)
		agent_map = dict(zip(range(self.population), self.agent_list))
		nx.relabel_nodes(self.social_network, agent_map, copy=False)

		# Store friends as an attribute
		for i in self.agent_list:
			i.add_friends([j for j in self.social_network.neighbors(i)])

	def daily_step(self, num_infections):

		infection_influence = 1 - \
		math.exp(-self.influence_param * (num_infections / self.population))

		sample_list = []
		unvaccinated = [x for x in self.agent_list if x.vaccinated==0]
		
		for agent in unvaccinated:

			social_influence = np.mean([x.vaccinated for x in agent.friends])
			total_influence = self.weight * infection_influence + \
			(1-self.weight) * social_influence

			if total_influence > agent.threshold:
				sample_list.append(agent)

		if len(sample_list) > self.max_daily_vax:
			vaccinated = self.generator.choice(sample_list, size=self.max_daily_vax)
		else:
			vaccinated = sample_list

		for agent in vaccinated:
			agent.vaccinated = 1

		self.daily_vax = np.append(self.daily_vax, len(vaccinated))