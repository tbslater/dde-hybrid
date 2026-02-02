# Import required packages 
import numpy as np
import math
from scipy.integrate import solve_ivp, OdeSolution

class SystemDynamics:

	def __init__(self, params, initial_conditions=None):

		# Parameters
		self.marketing = params['marketing']
		self.joining = params['joining']
		self.contact_rate = params['contact_rate']
		self.renewal = params['renewal']
		self.membership_length = params['membership_length']
		self.population = params['population']

		# Initial conditions
		if initial_conditions:
			self.N = np.array([initial_conditions['non_members']])
			self.M = np.array([initial_conditions['members']])
		else:
			self.N = np.array([self.population])
			self.M = np.array([0])

		# Store timepoints
		self.time = np.array([0])

		# Store interpolator
		self.interpolator = None

	def joining_rate(self, t, y):

		N, M = y

		output = (self.marketing * N) + \
		(self.contact_rate * self.joining * N * M) / self.population

		return output

	def renewal_rate(self, t):

		t_delay = t - self.membership_length
		if t_delay >= 0:
			y_delay = self.interpolator.__call__(t_delay)
			output = self.joining_rate(t_delay, y_delay)
		else: 
			output = 0

		return output

	def stock_equations(self, t, y):

		inflow = self.joining_rate(t, y) - self.renewal_rate(t)
		dNdt = - inflow
		dUdt = inflow

		return dNdt, dUdt

	def solve(self, t):

		while self.time[-1] < t:
			tmax = min(self.time[-1] + self.membership_length - 1, t)
			
			# Initial conditions
			y0 = [self.N[-1], self.M[-1]]
		
			# Time domain
			time_domain = [self.time[-1], tmax]
		
			# Solve stock equations
			solutions = solve_ivp(self.stock_equations, time_domain, y0, 
								  dense_output=True, method='LSODA')
		
			N, M = solutions.y[:, -1]
			self.N = np.append(self.N, N)
			self.M = np.append(self.M, M)
		
			self.time = np.append(self.time, tmax)
		
			# Append interpolator
			if self.interpolator:
				ts = np.append(self.interpolator.ts, solutions.sol.ts[1:])
				interpolants = self.interpolator.interpolants + solutions.sol.interpolants
				self.interpolator = OdeSolution(ts, interpolants)
			else: 
				self.interpolator = solutions.sol