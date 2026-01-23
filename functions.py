# Load required packages
import numpy as np

class SDFunctions:
	
	def check_delay_time(t, delay):
		'''
		Check if the time delay is greater than the current time.

		Parameters
		----------
		t : float
			Current time point. 
		time_delay : float
			Length of the time delay.

		Returns
		-------
		bool
			True if the delay time is less than the current t. 
		'''

		if delayed_t < t:
			output = True
		else: 
			output = False

		return output
	
	def stock_delay(t, time_delay, interpolator):
		'''
		Return stock values at a delayed time.

		Parameters
		----------
		t : float
			Current time point. 
		time_delay : float
			Length of the time delay. 
		interpolator: OdeSolution
			Class for interpolating the delayed value. 
			
		Returns
		-------
		numpy.array
			Array of delayed stock values.
		'''

		# Calculate delayed time
		delayed_t = t - time_delay

		# Obtain delayed values
		output = interpolator.__call__(delayed_t)

		return output