import numpy as np
import random

class Infection(object):
	
	def __init__(self, probability, distance):
		self.p = probability
		self.d = distance
		
	def prob_of_infection(self, distance):
		'''
		given distance between nodes
		return probability of contradiction at this distance
		according to inverse square relationship
		'''
		if distance == 0:
			distance = 0.01	
		c = self.p*(self.d)**2
		return c/(dis**2)
	
	def infect(self, distance):
		p = self.prob_of_infection(distance)
		return 1 if random.random() < p else 0
		
	def pair_wise_infection(self, infected_node, node):
		'''
		given two nodes, calculate probability of infection
		based on distance and return 1 if infected successfully, 0 otherwise
		'''
		distance = np.linalg.norm(infected_node - node)
		return self.infect(distance)