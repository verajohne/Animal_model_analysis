from __future__ import division
import numpy as np
import random

import frnnr
import infection


HERD_SIZE = 100


def update_dic(dictionary, nr_of_herds, infection_map):
	'''
	Assumes flocks are of equal size.
	Dictionary maps herd id to number of infected per time update_dic is called.
	Each herd_id, i, maps to a list of number of infected per time step.
	'''
	for i in range(nr_of_herds):
		index0 = i*100
		imap = infection_map[index0:index0 + HERD_SIZE]
		number_infected = np.bincount(imap)
		if len(number_infected) == 1:
			number_infected = 0
		else:
			number_infected = number_infected[1]
		dictionary[i+1].append(number_infected)
		
	return dictionary

class Field(object):
	
	def __init__(self, flocks, infection):
		'''
		flocks are the standard trajectory format
		flocks is a list of trajectory matrices
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
		'''
		infection_radius defines maximum distance between to nodes
		for an infection to be possible
		'''
		self.infection_radius = np.sqrt(infection.p*(infection.d**2)/0.001)
	
	def get_infection_dic(self):
		'''
		produces empty infection dic
		'''
		nr_of_herds = len(self.flocks)
		dictionary = {}
		for i in range(1, nr_of_herds + 1):
			dictionary[i] = []
		return dictionary
		
	def insert_infection(self, infection_map, n):
		inf = random.sample(range(0,self.nodes), n)
		for i in inf:
			infection_map[i] = 1
		return infection_map
	
	def run(self, dic = False):
		'''
		run infection simulation in field of sheep
		returns:
		1. dictionary of a mapping of herd to list of #infected/time (dic=True)
		2. time till 90% of total field infected
		returns -1 if 90% of flock failed to be infected.
		'''

		infection_map = np.zeros(self.nodes, dtype= 'int64')
		infection_map = self.insert_infection(infection_map, 1)
		
		time_to90 = -1
		t90 = False
		
		dictionary = {}
		if dic == True:
			dictionary = self.get_infection_dic()
		
		for ts in range(self.time_samples):
			infection_map = self.infect(ts, infection_map)
			if dic == True:
				dictionary = update_dic(dictionary, len(self.flocks), infection_map)
			
			number_infected = np.bincount(infection_map)[1]
			percentage_infected = number_infected /float( len(self.flocks)*HERD_SIZE )
			
			if percentage_infected >= 0.60 and t90 == False:
				time_to90 = ts
				t90 = True

		if dic == True:
			return (time_to90, dictionary)
		else:
			return time_to90
		
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
		'''
		Get points within the infectious radius through fixed neighbour reporting
		Only check node compared to infected nodes
		'''
		f = frnnr.frnnr(self.infection_radius, infected_points)
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



			
				