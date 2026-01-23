# Load require packages / files
import numpy as np
from scipy.integrate import solve_ivp, OdeSolution
from functions.SDFunctions import stock_delay, check_delay_time

class SDSetup:
	
	def __init__(self, params, initial_conditions):

		# Rate parameters
		self.contact_rate = params['contact_rate']
		self.infectivity = params['infectivity']
		self.vaccine_uptake = params['vaccine_uptake']

		# Delay parameters
		self.death_delay = params['death_delay']
		self.recovery_delay = params['recovery_delay']
		self.immunity_delay = params['immunity_delay']

		# Stocks
		self.susceptible = np.array([initial_conditions['susceptible']])
		self.infected = np.array([initial_conditions['infected']])
		self.immune = np.array([initial_conditions['immune']])
		self.dead = np.array([initial_conditions['dead']])

		# Store timepoints
		self.time = np.array([0])

	def infection_rate(self, t, y):

		# Stock values
		S, I, M, D = y

		# Calculate infection rate
		output = self.contact_rate * self.infectivity * S * I / (S + I + H)

		return output

	def vaccine_rate(self, t, y):

		# Stock values
		S, I, M, D = y

		# Calculate vaccine rate
		output = self.vaccine_uptake * S

		return output

class SDSolve(SDSetup):

	def __init__(self, params, initial_conditions):

		# Inherit methods/attributes from SDSetup
		super().__init__(params, initial_conditions)

		# Set up attribute for interpolation
		self.interpolator = 0

		# Calculate smallest delay time (required for triggering a solve)
		self.min_delay = min(self.death_delay, self.recovery_delay, self.immunity_delay)

	def stock_equations(self, t, y):

		# Infection rate
		infection_rate = self.infection_rate(t, y)

		# Vaccine rate
		vaccine_rate = self.vaccine_rate(t, y)

		# Death rate
		if check_delay_time(t, self.death_delay):
			delayed_y = stock_delay(t, self.death_delay, self.interpolator)
			death_rate = self.infection_rate(t, delayed_y)
		else:
			death_rate = 0

		# Recovery rate
		if check_delay_time(t, self.recovery_delay):
			delayed_y = stock_delay(t, self.recovery_delay, self.interpolator)
			recovery_rate = self.infection_rate(t, delayed_y)
		else:
			recovery_rate = 0

		# Susceptibility rate
		if check_delay_time(t, self.immunity_delay):
			delayed_y = stock_delay(t, self.immunity_delay, self.interpolator)
			susceptibility_rate = self.vaccine_rate(t, delayed_y) + \
			self.recovery_rate(t, delayed_y)
		else:
			susceptibility_rate = 0

		# Stock equations
		dSdt = susceptibility_rate - infection_rate - vaccine_rate
		dIdt = infection_rate - death_rate - recovery_rate
		dMdt = recovery_rate + vaccine_rate - susceptibility_rate
		dDdt = - death_rate

		return np.array([dSdt, dIdt, dMdt, dDdt])