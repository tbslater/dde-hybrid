# Load required packages
import numpy as np

class SDFunctions:
	
	def stock_delay(t, time_delay, interpolator, initial_value=None):
		'''
		Function to return stock values at a delayed time.

		Parameters
		----------
		t : float
			Current time point. 
		time_delay : float
			Length of the time delay. 
		interpolator: OdeSolution
			Class for interpolating the delayed value. 
		initial_value: list or numpy.array, optional
			Value(s) to return if t < time_delay
			
		Returns
		-------
		numpy.array
			Array of delayed stock values.
		'''

		# Calculate delayed time
		delayed_t = t - time_delay

		if delayed_t > 0:

			# Obtain delayed values
			output = interpolator.__call__(delayed_t)

		else: 

			# Return initial value
			if initial_value:
				output = initial_value
			else:
				output = interpolator.__call__(0)

		return output