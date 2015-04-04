from __future__ import division
import numpy as np
import random
import scipy.io

import frnnr
import infection

#### CONSTANTS
'''
infection_radius defines maximum distance between to nodes
for an infection to be possible
'''
INFECTION_RADIUS = 10
HERD_SIZE = 100


def update_dic(dictionary, nr_of_herds, infection_map):
	'''Assumes flocks are of equal size.. 100 for now '''
	
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
		Assumes flocks are of equal size
		flocks is a list of trajectories
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
	
	def get_infection_dic(self):
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
		returns
		1. dictionary of a mapping of herd to list of #infected/time
		2. time till 90% of total field
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
			if percentage_infected >= 0.90 and t90 == False:
				time_to90 = ts
				t90 = True

		if dic == True:
			for key in dictionary.keys():
				file = 'stats_herd' + str(key) + '.mat'
				data = np.array([dictionary[key]])
				#matrix_file = scipy.io.savemat(filename, mdict={'stats': data}, format = '5' )
		
			return (time_to90, dictionary)
		else:
			#matrix_file = scipy.io.savemat('time_to_90.mat', mdict={'time_to_90': np.array([time_to90])}, format = '5' )
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



			
				