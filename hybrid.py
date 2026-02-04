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
		self.converters = ['new_members', 'new_dropouts', 'new_potentials']
		self.generators = {
			self.converters[0]: np.random.default_rng(seeds[2]),
			self.converters[1]: np.random.default_rng(seeds[3]),
			self.converters[2]: np.random.default_rng(seeds[4])
		}
		
		# Simulate until...
		self.horizon = horizon
		
		# Store SD model within the HS model class
		self.sd_model = SystemDynamics(parameters)

		# Change agent states in line with initial stock values
		initial_vals = np.round([self.sd_model.M[-1], self.sd_model.D[-1]])
		for i in range(2):
			if initial_vals[i] != 0:
				poss = [x in self.agent_list if x.member==i and x.dropout==0]
				probs = [x.pref for x in possible_agents]
				transformed_probs = probs / sum(probs)
				to_change = \
				self.generators[self.converters[i]].choice(poss, size=initial_vals[i],
														  replace=False, p=transformed_probs)
				for agent in to_change:
					x.member == 1-i
					x.dropout == i

	def change_agent_states(self):

		# Extract flows	
		flow_vals = {
			self.converters[0]: \ 
			round(self.sd_model.P[-2] - self.sd_model.P[-2] + self.sd_model.N[-1]),
			self.converters[1]: \
			round(self.sd_model.D[-1] - self.sd_model.D[-2] + self.sd_model.N[-1]),
			self.converters[2]: \
			round(self.sd_model.N[-1])
		}

		for flow in enumerate(self.converters]):
			if flow_vals[flow] != 0:
				
				if i==0:
					possible = [x in self.agent_list if x.member==0 and x.dropout==0]
				if i==1: 
					possible = [x in self.agent_list if x.member]
				if i==2:
					possible = [x in self.agent_list if x.dropout]
					
				probs = [x.pref for x in possible_agents]
				transformed_probs = probs / sum(probs)
				
				to_change = self.generators[flow].choice(possible, size=flow_vals[flow], 
														 replace=False, p=transformed_probs)

				for agent in to_change:
					if i==0:
						x.member = 1
					if i==1:
						x.member = 0
						x.dropout = 1
					if i==2:
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