from __future__ import division
import numpy as np
import random
import scipy.io

import frnnr
import infection


#### CONSTANTS
INFECTION_RADIUS = 10

class Field(object):
	
	def __init__(self, flocks, infection):
		'''
		flocks are just 3D matrices of flock trajectory per time
		right now assumes flocks are equal size
		'''
		self.flocks = flocks
		
		n = flocks[0].shape[2]
		points = flocks[0]
		
		for i in range(1,len(self.flocks)):
			points = np.dstack((points, flocks[i]))
			n = n + flocks[i].shape[2]
		self.nodes = n
		self.points = points
		self.infection = infection
		self.time_samples = flocks[0].shape[1]
	
		
	def insert_infection(self, infection_map, n):
		inf = random.sample(range(0,self.nodes), n)
		for i in inf:
			infection_map[i] = 1
		return infection_map
	
	def run(self):
		infection_map = np.zeros(self.nodes, dtype= 'int64')
		infection_map = self.insert_infection(infection_map, 1)
		
		number_infected = []
		
		for ts in range(self.time_samples):
			infection_map = self.infect(ts, infection_map)
			
			#count nr infected in each herd at each time step for statistics
			offset = 0
			for f in range(len(self.flocks)):
				l = infection_map[offset:(offset+self.nodes/len(self.flocks))]
				try:
					ni = np.bincount(l)[1]
				except IndexError:
					ni = 0
				number_infected.append(ni)
				
				if ni >= 90:
					break
				
				offset = self.nodes/len(self.flocks)
					
		#matrix_file = scipy.io.savemat('flock1.mat', mdict={'epi': np.array(time90[0])}, format = '5' )
		#matrix_file = scipy.io.savemat('flock2.mat', mdict={'epi': np.array(time90[1])}, format = '5' )
		#return [number_infected_time_1, number_infected_time_2]
		#return [time90[0], time90[1]]
		print number_infected
		return number_infected
		
	def infect(self, ts, infection_map):
		'''
		given an infection_map over a field at a time instance
		return an updated infection map
		'''
		not_infected_indexes = []
		infected_indexes = []
		infected_points = []
		
		for i in range(len(infection_map)):
			if infection_map[i] == 1:
				infected_indexes.append(i)
			else:
				not_infected_indexes.append(i)

		#all coordinate points at ts
		x = self.points[0][ts]
		y = self.points[1][ts]
		
		#get list of coordinates of infected nodes
		for i in infected_indexes:
			p = np.array([self.points[0][ts][i], self.points[1][ts][i]])
			infected_points.append(p)
		
		#Get points within INFECTION RADIUS through fixed neighbour reporting
		f = frnnr.frnnr(INFECTION_RADIUS, infected_points)
		for i in not_infected_indexes:
			#get an uninfected node
			p = np.array([self.points[0][ts][i], self.points[1][ts][i]])
			#get list of distances to nearby infect-able nodes from uninfected node
			distances = f.get_distances(p)
			for d in distances:
				inf = self.infection.infect(d)
				if inf == 1:
					infection_map[i] = 1
					break		
		return infection_map



			
				