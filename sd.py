# Import required packages
import numpy as np
from scipy.integrate import solve_ivp, OdeSolution

class SystemDynamics:

	def __init__(self, parameters, initial_conditions=None):

		# Parameters
		self.contact_rate = parameters['contact_rate']
		self.infectivity = parameters['infectivity']
		self.symptom_delay = parameters['symptom_delay']
		self.quarantine_length = parameters['quarantine_length']
		self.vaccine_fraction = 0
		self.quarantine_fraction = parameters['quarantine_fraction']
		self.infectivity_length = parameters['infectivity_length']

		# Initial_conditions
		if initial_conditions:
			self.S = np.array([initial_conditions['susceptible']])
			self.I = np.array([initial_conditions['infected']])
			self.Q = np.array([initial_conditions['quarantined']])
			self.R = np.array([initial_conditions['recovered']])
		else:
			self.S = np.array([parameters['population'] - 1])
			self.I = np.array([1])
			self.Q = np.array([0])
			self.R = np.array([0])

		# Store timepoints
		self.time = np.array([0])

		# Store interpolator
		self.interpolator = None

	def flow_equations(self, t, y):

		S, I, Q, R = y
		
		def quarantine_rate(I):

			output = (self.quarantine_fraction * I) / self.symptom_delay
			
			return output
		
		IR = (self.contact_rate * self.infectivity * S * I) / (S + I + R)

		IRR = ((1-self.quarantine_fraction) * I) / \
		(self.infectivity_length - self.symptom_delay)
		
		QR = quarantine_rate(I)
		
		VR = self.vaccine_fraction * S
		
		t_delay = t - self.quarantine_length
		if t_delay >= 0:
			I_delay = self.interpolator.__call__(t_delay)[1]
			QRR = quarantine_rate(I_delay)
		else:
			QRR = 0

		return IR, IRR, QR, VR, QRR

	def stock_equations(self, t, y):

		IR, IRR, QR, VR, QRR = self.flow_equations(t, y)

		dSdt = - IR - VR
		dIdt = IR - IRR - QR
		dQdt = QR - QRR
		dRdt = VR + IRR + QRR

		return dSdt, dIdt, dQdt, dRdt

	def solve(self, t):

		while self.time[-1] < t:
			tmax = min(self.time[-1] + self.quarantine_length - 1, t)
			
			# Initial conditions
			y0 = [self.S[-1], self.I[-1], self.Q[-1], self.R[-1]]
		
			# Time domain
			time_domain = [self.time[-1], tmax]
		
			# Solve stock equations
			solutions = solve_ivp(self.stock_equations, time_domain, y0, 
								  dense_output=True, method='LSODA')
		
			S, I, Q, R = solutions.y[:, -1]
			self.S = np.append(self.S, S)
			self.I = np.append(self.I, I)
			self.Q = np.append(self.Q, Q)
			self.R = np.append(self.R, R)
		
			self.time = np.append(self.time, tmax)
		
			# Append interpolator
			if self.interpolator:
				ts = np.append(self.interpolator.ts, solutions.sol.ts[1:])
				interpolants = self.interpolator.interpolants + solutions.sol.interpolants
				self.interpolator = OdeSolution(ts, interpolants)
			else: 
				self.interpolator = solutions.sol