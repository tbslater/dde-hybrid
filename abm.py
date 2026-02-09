# Import required packages/files
import numpy as np
from sim_tools.distributions import Beta, spawn_seeds
import networkx as nx
import math

class Agent:
	'''
	Represents an individual in the population.

	Attributes
	----------
	threshold : float
		Individual threshold for getting vaccinated.
	vaccinated : int
		Vaccination status (1 is vaccinated).
	friends : array_like, shape (n,)
		List of objects representing the agent's friends.
	num_friends : int
		Number of friends.
	'''

	def __init__(self, threshold):
		'''
		Initialise a new agent.

		Parameters
		----------
		threshold : float
			Individual threshold for becoming vaccinated.
		vaccinated : int
			Vaccination status (1 is vaccinated).
		'''

		# Threshold for becoming vaccinated
		self.threshold = threshold
		# Vaccination status
		self.vaccinated = 0

	def add_friends(self, friends):
		'''
		Store the agent's friends.

		Parameters
		----------
		friends : array_like, shape (n,)
			List of objects representing the agent's friends.
		num_friends : int
			Number of friends.
		'''

		# List of friends
		self.friends = friends
		# Number of friends
		self.num_friends = len(self.friends)

class AgentBasedModel:
	'''
	Represents an agent-based model for vaccination behaviour.

	Attributes
	----------
	max_daily_vax : int
		Maximum vaccinations per day.
	influence_param : int or float
		Rate at which the number of infections affects vaccination 
		preference. 
	beta_params : array_like, shape (2,)
		Beta distribution parameters for generating thresholds.
	weight : float
		Weight for influence from the infection number. 
	seeds : array_like, shape (2,)
		Seeds for reproducibility.
	generator : Generator
		Random number generator for sampling agents.
	daily_vax : array_like, shape (n,)
		Number of vaccinations each day.
	population : int
		Total number of agents in the population.
	agent_list : array_like, shape (population,)
		List of agents.
	social_network : Newman-Watts-Strogartz graph object
		Graph representing the social network of the population.
	'''

	def __init__(self, parameters, main_seed):
		'''
		Initialise an agent-based model.

		Parameters
		----------
		parameters : dict
			Dictionary containing values for max_daily_vax, influence_param, 
			beta_params, weight.
		main_seed : int
			Seed for reproducibility.
		'''

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
		'''
		Generate a set of agents and create a social network. 

		Parameters
		----------
		population : int
			Number of agents in the population.

		Notes
		-----
		This section of code was inspired by Archbold et al., 2024 [1,2],
		in particular generation of the social network using a Newman-Watts-
		Strogatz graph [2].

		References
		----------
		.. [1] Archbold J, Clohessy S, Herath D, Griffiths N, Oyebode O
		(2024) An agent-based model of the spread of behavioural risk-factors
		for cardiovascular disease in city-scale populations. PLoS ONE 19(5):
		e0303051. https://doi.org/10.1371/journal.pone.0303051.
		.. [2] Newman M E J, Watts D J (1999) Renormalization group analysis
		of the small-world network model. Physics Letters A 263(4-6) pp.341-
		346. https://doi.org/10.1016/S0375-9601(99)00757-4.
		'''

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
		labels = dict(zip(range(self.population), self.agent_list))
		nx.relabel_nodes(self.social_network, labels, copy=False)

		# Store friends as an attribute
		for i in self.agent_list:
			i.add_friends([j for j in self.social_network.neighbors(i)])

	def daily_step(self, num_infections):
		'''
		Run the agent-based model for one day.

		For each agent calculate their daily influence and check if it meets
		the threshold for vaccination. Then sample agents for vaccination
		from those that meet their threshold.

		Parameters
		----------
		num_infections : float
			Number of infections that day.
		'''

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