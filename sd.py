# Import required packages 
import numpy as np
import math
from scipy.integrate import solve_ivp, OdeSolution

class SystemDynamics:

	def __init__(self, params, initial_conditions=None):

		# Parameters
		self.marketing = params['marketing']
		self.joining_rate = params['joining_rate']
		self.contact_rate = params['contact_rate']
		self.dropout_rate = params['dropout_rate']
		self.membership_length = params['membership_length']
		self.dropout_delay = params['dropout_delay']
		self.population = params['population']

		# Initial conditions
		if initial_conditions:
			self.P = np.array([initial_conditions['potential_members']])
			self.M = np.array([initial_conditions['members']])
			self.D = np.array([initial_conditions['dropouts']])
			# We'll add an extra stock for extracting flows for the hybrid model
			self.N = np.array([initial_conditions['new_potential_members']])
		else:
			self.P = np.array([self.population])
			self.M = np.array([0])
			self.D = np.array([0])
			# We'll add an extra stock for extracting flows for the hybrid model
			self.N = np.array([0])

		# Store timepoints
		self.time = np.array([0])

		# Store interpolator
		self.interpolator = None

	def member_inflow(self, y):

		P, M, D, N = y

		output = (self.marketing * P) + \
		(self.contact_rate * self.joining_rate * P * M) / self.population

		return output

	def dropout_inflow(self, t):

		t_delay = t - self.membership_length
		if t_delay >= 0:
			y_delay = self.interpolator.__call__(t_delay)
			output = self.dropout_rate * self.member_inflow(t_delay, y_delay)
		else: 
			output = 0

		return output

	def potential_inflow(self, y):

		P, M, D, N = y
		output = D / self.dropout_delay

		return output

	def stock_equations(self, t, y):

		# Evaluate flows
		member_inflow = self.member_inflow(y)
		ex_inflow = self.ex_inflow(t)
		potential_inflow = self.potential_inflow(y)

		# Stock equations
		dPdt = potential_inflow - member_inflow
		dMdt = member_inflow - ex_inflow
		dDdt = ex_inflow - potential_inflow
		dNdt = potential_inflow

		return dPdt, dMdt, dDdt, dNdt

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
		
			P, M, D, N = solutions.y[:, -1]
			self.P = np.append(self.P, P)
			self.M = np.append(self.M, M)
			self.D = np.append(self.D, D)
			self.N = np.append(self.N, N)
		
			self.time = np.append(self.time, tmax)
		
			# Append interpolator
			if self.interpolator:
				ts = np.append(self.interpolator.ts, solutions.sol.ts[1:])
				interpolants = self.interpolator.interpolants + solutions.sol.interpolants
				self.interpolator = OdeSolution(ts, interpolants)
			else: 
				self.interpolator = solutions.sol