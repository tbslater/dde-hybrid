# Import required packages/files
import numpy as np
from sd import SystemDynamics
from abm import Agent, AgentBasedModel

class HybridSim(AgentBasedModel):

	def __init__(self, parameters):

		# Store additional params
		self.horizon = parameters['horizon']
		self.main_seed = parameters['main_seed']

		# Inherit attributes / functions from AgentBasedModel
		super().__init__(parameters, self.main_seed)

		# Store SD model within the HS model class
		self.sd_model = SystemDynamics(parameters)

		# Generate agents
		self.generate_agents(int(parameters['population']))

	def simulate(self):

		for t in range(1, self.horizon+1):
			
			# Solve SD equations
			self.sd_model.solve(t)

			# Run one step of the ABM
			self.daily_step(self.sd_model.I[-1])

			# Update SD parameter
			# self.sd_model.vaccination_fraction = self.daily_vax[-1] / \
			# self.population
			self.sd_model.vaccination_fraction = self.daily_vax[-1] / \
			self.sd_model.S[-1]

			# Print number of iterations completed
			if t % 10 == 0:
				print(f'Current timestep: {t}.')
				print(f'Simulation is {(t/self.horizon) * 100}% complete.')