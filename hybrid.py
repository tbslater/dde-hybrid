# Import required packages/files
import numpy as np
from sd import SystemDynamics
from abm import Agent, AgentBasedModel

class HybridSim(AgentBasedModel, SystemDynamics):
	'''
	Represents a hybrid simulation model for infectious disease
	modelling, combining a system dynamics approach with an agent-based
	model.

	Attributes
	----------
	horizon : int
		Number of days to run the simulation for.
	main_seed : int
		Seed for reproducibility.
	sd_model : SystemDynamics 
		Object representing the system dynamics model.

	Notes
	-----
	The style of this code was inspired by by the work of Palmer and
	Tian, 2021 [1]. In particular, we use inheritance for the agent-
	based model and store the system dynamics component as an attribute.

	References
	----------
	.. [1] Palmer G I, Tian Y (2023) Implementing hybrid simulations
	that integrate DES+SD in Python, Journal of Simulation, 17(3),
	pp. 240–256. doi: 10.1080/17477778.2021.1992312.
	'''

	def __init__(self, parameters):
		'''
		Initialise a hybrid simulation model. 

		Parameters
		----------
		parameters : dict
			Dictionary containing values for contact_rate, infectivity,
			symptom_delay, quarantine_length, vaccine_fraction, 
			quarantine_fraction, infectivity_length, population, max_daily_vax,
			influence_param, beta_params, weight, horizon, main_seed.
		'''
		
		# Store additional params
		self.horizon = parameters['general']['horizon']
		self.main_seed = parameters['general']['main_seed']

		# Inherit attributes / functions from sub-models
		SystemDynamics.__init__(self, parameters['system_dynamics'])
		AgentBasedModel.__init__(self, parameters['agent_based'], self.main_seed)

		# Generate agents
		self.generate_agents(int(self.population))

	def simulate(self, method='RK45', rtol=1e-9):
		'''
		Run the model until t=horizon.

		The order of logic each day is as follows:
			1) Solve the SD equations to obtain stock values.
			2) Run the ABM and change agent states.
			3) Calculate the proportion of vaccinations that day and update the
			SD parameter.

		Parameters
		----------
		method : str
			Method used by solve_ivp to solve stock equations. Default is 'RK45'.
		rtol : float
			Controls relative accuracy when using solve_ivp. Default is 1e-9.
		'''

		for t in range(1, self.horizon+1):
			
			# Solve SD equations
			self.solve(t, method)
			
			# Run one step of the ABM
			self.daily_step(self.I[-1])

			# Update SD parameter
			self.vaccination_fraction = self.daily_vax[-1]

			# Print number of iterations completed
			if t % 10 == 0:
				print(f'Current timestep: {t}.')
				print(f'Simulation is {(t/self.horizon) * 100}% complete.')