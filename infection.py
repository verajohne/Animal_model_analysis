import numpy as np
import random



class Infection(object):
	
	def __init__(self, p, d):
		self.p = p
		self.d = d
		
	
	def prob_of_infection(self, dis):
		'''
		given distance between nodes
		return probability of contradiction
		using inverse square relationship
		'''
		if dis == 0:
			dis = 0.01	
		c = self.p*(self.d)**2
		return c/(dis**2)
	
	def infect(self, dis):
		p = self.prob_of_infection(dis)
		return 1 if random.random() < p else 0
		
	def pair_wise_infection(self, infected_node, node):
		'''
		given two nodes, calculate probability of infection
		based on distance and return 1 if infected successfully, 0 otherwise
		'''
		dist = np.linalg.norm(infected_node - node)
		return self.infect(dist)