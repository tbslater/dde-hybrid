# Load required packages
import numpy as np

class Agent:

	def __init__(self, attributes):

		self.member = 0
		self.ex_member = 0
		self.id = attributes['id']
		self.pref = np.array([attributes['pref']])
		self.distance = attributes['distance']

	def add_friends(self, friends):

		self.friends = friends
		self.num_friends = len(self.friends)