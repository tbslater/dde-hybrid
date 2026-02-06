# Import required packages
import numpy as np
from scipy.integrate import solve_ivp, OdeSolution

class SystemDynamics:
	'''
	Represents a system dynamics model for infectious disease modelling.

	Attributes
	----------
	contact_rate : int or float
		Average number of contacts per day.
	infectivity : float
		Probability of transmission given contact.
	symptom_delay : int or float
		Average number of days following infection before symptoms show.
	quarantine_length : int
		Number of days in quarantine following symptoms. 
	vaccine_fraction : float
		Proportion of the population being vaccinated. 
	quarantine_fraction : float
		Proportion of people who quarantine given they have symptoms.
	infectivity_length : int or float
		Average duration of infectivity.
	S : array_like, shape (n,)
		Number of susceptible individuals at each time point in time.
	I : array_like, shape (n,)
		Number of infected individuals at each time point in time.
	Q : array_like, shape (n,)
		Number of quarantined individuals at each time point in time.
	R : array_like, shape (n,)
		Number of recovered individuals at each time point in time.
	time : array_like, shape (n,)
		Time points at which the equations have been solved.
	interpolator : OdeSolution
		Used for interpolating between previously solved points.

	Notes
	-----
	The structure of this code was inspired by the work of Palmer and 
	Tian, 2021 [1].

	References
	----------
	.. [1] Palmer G I, Tian Y (2023) Implementing hybrid simulations
	that integrate DES+SD in Python, Journal of Simulation, 17(3),
	pp. 240–256. doi: 10.1080/17477778.2021.1992312.
	'''

	def __init__(self, parameters, initial_conditions=None):
		'''
		Initialise a system dynamics model.

		Parameters
		----------
		parameters : dict
			Dictionary containing values for contact_rate, infectivity,
			symptom_delay, quarantine_length, vaccine_fraction, 
			quarantine_fraction, infectivity_length, population.
		initial_conditions : dict, optional
			Dicitionary containing initial stock values for susceptible,
			infected, quarantined and recovered individuals.
		'''

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
		'''
		Calculates flow values at time t.

		Parameters
		----------
		t : float
			Current time point. 
		y : array_like, shape (4,)
			Stock values at time t.

		Returns
		-------
		array_like, shape (5,)
			Flow values at time t.
		'''

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
		'''
		Calculates rate of change in stock at time t.

		Parameters
		----------
		t : float
			Current time point. 
		y : array_like, shape (4,)
			Stock values at time t.

		Returns
		-------
		array_like, shape (4,)
			Differential equation values at time t.
		'''

		IR, IRR, QR, VR, QRR = self.flow_equations(t, y)

		dSdt = - IR - VR
		dIdt = IR - IRR - QR
		dQdt = QR - QRR
		dRdt = VR + IRR + QRR

		return dSdt, dIdt, dQdt, dRdt

	def solve(self, t):
		'''
		Solves the stock differential equations until time t.

		Parameters
		----------
		t : float
			Solve until this time.
		'''

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