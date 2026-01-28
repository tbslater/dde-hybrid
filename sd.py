# Load require packages / files
import numpy as np
from scipy.integrate import solve_ivp, OdeSolution
from functions import SDFunctions as sdf

class SDSetup:
	
	def __init__(self, params, initial_conditions):

		# Rate parameters
		self.contact_rate = params['contact_rate']
		self.infectivity = params['infectivity']
		self.vaccine_uptake = params['vaccine_uptake']
		self.death_prob = params['death_prob']

		# Delay parameters
		self.death_delay = params['death_delay']
		self.recovery_delay = params['recovery_delay']
		self.immunity_delay = params['immunity_delay']

		# Stocks
		self.S = np.array([initial_conditions['susceptible']])
		self.I = np.array([initial_conditions['infected']])
		self.M = np.array([initial_conditions['immune']])
		self.D = np.array([initial_conditions['dead']])

		# Store timepoints
		self.time = np.array([0])

	def infection_rate(self, t, y):

		# Stock values
		S, I, M, D = y

		# Calculate infection rate
		output = (self.contact_rate * self.infectivity * S * I) / (S + I + M)

		return output

	def vaccine_rate(self, t, y):

		# Stock values
		S, I, M, D = y
		
		# Calculate vaccine rate
		output = self.vaccine_uptake * S

		return output

	def recovery_rate(self, t, y):

		if sdf.check_delay_time(t, self.recovery_delay):

			# Delayed stock 
			y_delay = sdf.stock_delay(t, self.recovery_delay, self.interpolator)
	
			# Calculate recovery rate
			output = (1-self.death_prob) * self.infection_rate(t, y_delay)

			print([t, output, y, y_delay])

		else: 
			output = 0

		return output	

	def death_rate(self, t, y):

		if sdf.check_delay_time(t, self.death_delay):

			# Delayed stock 
			y_delay = sdf.stock_delay(t, self.death_delay, self.interpolator)
	
			# Calculate recovery rate
			output = self.death_prob * self.infection_rate(t, y_delay)

		else: 
			output = 0

		return output

	def susceptibility_rate(self, t, y):

		if sdf.check_delay_time(t, self.immunity_delay):

			# Delayed stock 
			y_delay = sdf.stock_delay(t, self.immunity_delay, self.interpolator)

			print(f'This is a SR calculation with delayed stock {y_delay}!')
	
			# Calculate recovery rate
			output = self.vaccine_rate(t, y_delay) + self.recovery_rate(t, y_delay)

			print(f'SR output is {output}.')

		else: 
			output = 0

		return output	

class SDSolve(SDSetup):

	def __init__(self, params, initial_conditions):

		# Inherit methods/attributes from SDSetup
		super().__init__(params, initial_conditions)

		# Set up attribute for interpolation
		self.interpolator = None

		# Calculate smallest delay time (required for triggering a solve)
		self.min_delay = min(self.death_delay, self.recovery_delay, self.immunity_delay)

	def stock_equations(self, t, y):

		# Calculate flows
		ir = self.infection_rate(t, y)
		vr = self.vaccine_rate(t, y)
		dr = self.death_rate(t, y)
		rr = self.recovery_rate(t, y)
		sr = self.susceptibility_rate(t, y)

		# Stock equations
		dSdt = sr - ir - vr
		dIdt = ir - dr - rr
		dMdt = rr + vr - sr
		dDdt = dr

		return np.array([dSdt, dIdt, dMdt, dDdt])

	def solve(self, t):

		while self.time[-1] < t:
			tmax = min(self.time[-1] + self.min_delay, t)
			
			# Initial conditions
			y0 = [self.S[-1], self.I[-1], self.M[-1], self.D[-1]]
		
			# Time domain
			time_domain = [self.time[-1], tmax]
		
			# Solve stock equations
			solutions = solve_ivp(self.stock_equations, time_domain, y0, 
								  dense_output=True, method='LSODA')
		
			S, I, M, D = solutions.y[:, -1]
			self.S = np.append(self.S, S)
			self.I = np.append(self.I, I)
			self.M = np.append(self.M, M)
			self.D = np.append(self.D, D)
		
			self.time = np.append(self.time, tmax)
		
			# Append interpolator
			if self.interpolator:
				ts = np.append(self.interpolator.ts, solutions.sol.ts[1:])
				interpolants = self.interpolator.interpolants + solutions.sol.interpolants
				self.interpolator = OdeSolution(ts, interpolants)
			else: 
				self.interpolator = solutions.sol