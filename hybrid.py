# Import required packages / files
import numpy as np
from sim_tools.distributions import spawn_seeds
from sd import SystemDynamics
from abm import AgentBasedModel

class HybridSim(AgentBasedModel):

	def __init__(self, horizon, parameters, main_seed):

		# Generate multiple statistically independent random seeds
		seeds = spawn_seeds(5, main_seed)
		
		# Inherit attributes / functions from AgentBasedModel
		super().__init__(parameters, seeds[:1])

		# Generators for randomly choosing agents for state change
		self.member_rng = np.random.default_rng(seeds[2])
		self.ex_member_rng = np.random.default_rng(seeds[3])
		self.non_member_rng = np.random.default_rng(seeds[4])
		
		# Simulate until...
		self.horizon = horizon
		
		# Store SD model within the HS model class
		self.sd_model = SystemDynamics(parameters)

		# Change agent states in line with initial stock values
		if self.sd_model.M[-1] != 0:
			no_new_members = np.round(self.sd_model.M[-1])
			possible_agents = [x in self.agent_list if x.member==0 and x.ex_member==0]
			# Transform probs
			probs = [x.pref for x in possible_agents]
			transformed_probs = probs / sum(probs)
			# Randomly choose new members
			new_members = self.member_rng.choice(possible_agents, size=no_new_members,
												replace=False, p=transformed_probs)
			# Change states
			for agent in new_members:
				x.member = 1

		if self.sd_model.D[-1] != 0:
			no_new_dropouts = np.round(self.sd_model.D[-1])
			possible_agents = [x in self.agent_list if x.member]
			# Transform probs
			probs = [x.pref for x in possible_agents]
			transformed_probs = probs / sum(probs)
			# Randomly choose new members
			new_dropouts = self.member_rng.choice(possible_agents, size=no_new_dropouts,
												replace=False, p=transformed_probs)
			# Change states
			for agent in new_dropouts:
				x.dropout = 1

	def change_agent_states(self):
		## This needs tidying!! ##

		# Extract flows	
		no_new_members = round(self.sd_model.P[-2] - self.sd_model.P[-2] + \
		self.sd_model.N[-1])
		no_new_dropouts = round(self.sd_model.D[-1] - self.sd_model.D[-2] + \
		self.sd_model.N[-1])
		no_new_potentials = round(self.sd_model.N[-1])

		# New members
		if no_new_members != 0:
			possible_agents = [x in self.agent_list if x.member==0 and x.ex_member==0]
			# Transform probs
			probs = [x.pref for x in possible_agents]
			transformed_probs = probs / sum(probs)
			# Randomly choose new members
			new_members = self.member_rng.choice(possible_agents, 
												 size=no_new_members,
												 replace=False, 
												 p=transformed_probs)
			# Change states
			for agent in new_members:
				x.member = 1

		# New dropouts
		if no_new_dropouts != 0:
			possible_agents = [x in self.agent_list if x.member]
			# Transform probs
			probs = [x.pref for x in possible_agents]
			transformed_probs = probs / sum(probs)
			# Randomly choose new members
			new_dropouts = self.member_rng.choice(possible_agents, 
												  size=no_new_dropouts,
												  replace=False, 
												  p=transformed_probs)
			# Change states
			for agent in new_dropouts:
				x.member = 0
				x.dropout = 1

		# New potential members
		if no_new_potentials != 0:
			possible_agents = [x in self.agent_list if x.dropout]
			# Transform probs
			probs = [x.pref for x in possible_agents]
			transformed_probs = probs / sum(probs)
			# Randomly choose new members
			new_dropouts = self.member_rng.choice(possible_agents, 
												  size=no_new_potentials,
												  replace=False,
												  p=transformed_probs)
			
			# Change states
			for agent in new_dropouts:
				x.member = 0
				x.dropout = 0

	def update_sd_parameters(self):

		self.sd_model.joining_rate = # ...
		self.sd_model.dropout_rate = # ...

		return None

	def simulate(self):

		for t in range(1, self.horizon+1):

			self.daily_step()
			self.change_agent_states()
			self.update_sd_parameters()
			self.sd_model.solve(t)
			