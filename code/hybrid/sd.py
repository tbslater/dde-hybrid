# Import required packages
import numpy as np
from scipy.integrate import solve_ivp, OdeSolution

class Interpolator:
	'''
	Object for interpolation of stock values.

	Attributes
	----------
	interpolator : OdeSolution
		Object containing functions for interpolation. 
	'''

	def __init__(self):
		'''
		Initialise an interpolator for the model. 
		'''

		# Set up interpolator object. We'll overwrite this later.
		self.interpolator = None

	def update_interp(self, solutions):
		'''
		Create a new OdeSolution object using by merging old and new.

		Parameters
		----------
		solutions : bunch object
			Output from solve_ivp. 
		'''
		
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

	def interp(self, t, adjust=True):
		'''
		Return interpolated value(s) at a singular or array of time points.
		Adjust the value in line with bounds (to account for error).

		Parameters
		----------
		t : float or array_like, shape (n,)
			Singular or array of time points to solve at.
		adjust : bool
			Adjusts stock values so bounds are not violated if True. 

		Returns
		-------
		array_like, shape (n,)
			Interpolated values. 
		'''

		# Use __call__ to return interpolated stock values
		vals = self.interpolator(t)
		# Adjust in line with boundaries if specified
		if adjust:
			vals[(vals < 0)] = 0
			vals[(vals > self.population)] = self.population

		return vals

class SystemDynamics(Interpolator):
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

	def __init__(self, parameters, initial_conditions=None):
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
		self.vaccine_uptake = 0
		self.quarantine_fraction = parameters['quarantine_fraction']
		self.infectivity_length = parameters['infectivity_length']
		self.population = parameters['population']

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

		def quarantine_rate(infections):

			output = (self.quarantine_fraction * infections) / self.symptom_delay
			
			return output
		
		IR = (self.contact_rate * self.infectivity * S * I) / (S + I + R)

		IRR = ((1-self.quarantine_fraction) * I) / self.infectivity_length
		
		QR = quarantine_rate(I)
		
		VR = self.vaccine_uptake * S
		
		t_delay = t - self.quarantine_length
		if t_delay >= 0:
			I_delay = self.interp(t_delay)[1] 
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

		# Flows
		IR, IRR, QR, VR, QRR = self.flow_equations(t, y)

		# Stock equations
		dSdt = - IR - VR
		dIdt = IR - IRR - QR
		dQdt = QR - QRR 
		dRdt = VR + IRR + QRR

		return dSdt, dIdt, dQdt, dRdt

	def solve(self, t, method, rtol):
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
		
			# Time domain
			time_domain = [self.time[-1], tmax]
		
			# Solve stock equations
			solutions = solve_ivp(self.stock_equations, time_domain, y0, 
								  dense_output=True, method=method, rtol=rtol)
		
			# Append interpolator
			self.update_interp(solutions)

			# Return last values
			S, I, Q, R = self.interp(tmax)
			
			# Update stock values
			self.S = np.append(self.S, S)
			self.I = np.append(self.I, I)
			self.Q = np.append(self.Q, Q)
			self.R = np.append(self.R, R)
			self.time = np.append(self.time, tmax)	