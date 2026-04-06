# Import required packages
import numpy as np
from scipy.integrate import solve_ivp, OdeSolution

class SDModel:
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
	vaccine_uptake : float
		Number of people vaccinated per day. 
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

	def __init__(self, parameters, method, initial_conditions=None):
		'''
		Initialise a system dynamics model.

		Parameters
		----------
		parameters : dict
			Dictionary containing values for contact_rate, infectivity,
			symptom_delay, quarantine_length, vaccine_uptake, 
			quarantine_fraction, infectivity_length, population.
		initial_conditions : dict, optional
			Dicitionary containing initial stock values for susceptible,
			infected, quarantined and recovered individuals.
		'''

		# Inherit interpolator class
		super().__init__()

		# Parameters
		self.contact_rate = parameters['contact_rate']
		self.infectivity = parameters['infectivity']
		self.symptom_delay = parameters['symptom_delay']
		self.quarantine_length = parameters['quarantine_length']
		self.quarantine_fraction = parameters['quarantine_fraction']
		self.infectivity_length = parameters['infectivity_length']
		self.population = parameters['population']

		# Method for solving pipeline delay
		if method=='LCT' or method=='interp':
			self.method = method
		else: 
			raise ValueError('Method must either be chain or interp.')

		if self.method == 'LCT':
			self.delay_order = parameters['delay_order']
			self.a = self.delay_order / self.quarantine_length
			# self.Z = np.zeros(self.delay_order - 1)
			self.Z = np.zeros(self.delay_order)
			# self.Z = self.Z.reshape((1,self.delay_order - 1))
			self.Z = self.Z.reshape((1,self.delay_order))
			if isinstance(self.delay_order, int) == False:
				raise ValueError('Order must be an integer.')
			
		# Initial_conditions
		if initial_conditions:
			self.S = np.array([initial_conditions['susceptible']])
			self.I = np.array([initial_conditions['infected']])
			self.Q = np.array([initial_conditions['quarantined']])
			self.R = np.array([initial_conditions['recovered']])
		else:
			self.S = np.array([self.population - 1])
			self.I = np.array([1])
			self.Q = np.array([0])
			self.R = np.array([0])

		# Store timepoints
		self.time = np.array([0])

		# Interpolator
		self.interpolator = None

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

		# Main stocks
		S, I, Q, R = y[:4]

		# Standard flows
		IR = (self.contact_rate * self.infectivity * S * I) / (S + I + R)
		IRR = ((1-self.quarantine_fraction) * I) / self.infectivity_length
		QR = (self.quarantine_fraction * I) / self.symptom_delay

		if self.method=='LCT':
			Z = y[4:]
			dZdt = np.zeros(self.delay_order)
			outflow = QR
			for i in range(self.delay_order):
				inflow = outflow
				outflow = self.a * Z[i]
				dZdt[i] = inflow - outflow

		if self.method=='interp':			
			t_delay = t - self.quarantine_length
			if t_delay >= 0:
				I_delay = self.interpolator(t_delay)[1] 
				outflow = (self.quarantine_fraction * I_delay) / self.symptom_delay
			else:
				outflow = 0

		# Stock equations
		dSdt = - IR
		dIdt = IR - IRR - QR
		dQdt = QR - outflow
		dRdt = IRR + outflow

		output = np.array([dSdt, dIdt, dQdt, dRdt])
		if self.method=='LCT':
			output = np.concatenate((output, dZdt), axis=None)

		return output

	def solve(self, t):
		'''
		Solves the stock differential equations until time t.

		Parameters
		----------
		t : float
			Solve until this time.
		method : str
			Method for solving. 
		rtol : float
			Controls relative accuracy when using solve_ivp.
		'''

		while self.time[-1] < t:
			# Solve until...
			tmax = min(self.time[-1] + self.quarantine_length - 1, t)
		
			# Initial conditions
			y0 = [self.S[-1], self.I[-1], self.Q[-1], self.R[-1]]
			if self.method=='LCT':
				y0 = np.concatenate((y0, self.Z[-1]), axis=None)
		
			# Time domain
			time_domain = [self.time[-1], tmax]
		
			# Solve stock equations
			solutions = solve_ivp(self.stock_equations, time_domain, y0, 
								  dense_output=True, method='LSODA', rtol=1e-6)
	
			# Update interpolator
			if self.interpolator: 
				# Extract and append time points
				ts = np.append(self.interpolator.ts, solutions.sol.ts[1:])
				# Extract and append list of interpolant objects
				interpolants = self.interpolator.interpolants + solutions.sol.interpolants
				# Create new OdeSolution object
				self.interpolator = OdeSolution(ts, interpolants)
			else: 
				# If the first time, we can just store the solve_ivp sol output
				self.interpolator = solutions.sol
			
			# Update stock values
			self.S = np.append(self.S, solutions.y[0,-1])
			self.I = np.append(self.I, solutions.y[1,-1])
			self.Q = np.append(self.Q, solutions.y[2,-1])
			self.R = np.append(self.R, solutions.y[3,-1])
			if self.method=='LCT':
				self.Z = np.vstack((self.Z, solutions.y[4:,-1]))
			self.time = np.append(self.time, tmax)
		