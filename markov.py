import dataprocessor as dp
import numpy as np
import random
import copy
import scipy.io as sio

'''
a hexagonal markov space
  __
 /  \
 \__/
  
'''

def get_markov_probabilities(trajectory):
	total_number_of_moves = (trajectory.shape[1]-1)*(trajectory.shape[2])

	markov_space_count = {}
	for i in range(1,7):
		markov_space_count[i] = 0
	
	for ts in range(trajectory.shape[1] - 1):
		#get positions at time ts and ts+1
		positions0 = dp.returnTimeMap(ts, trajectory)
		positions1 = dp.returnTimeMap(ts+1, trajectory)
		#get com at time ts
		com0 = dp.com(positions0)
	
		#center position matrixes around current com
		positions0 = dp.position_matrix_around_new_center(positions0, com0)
		positions1 = dp.position_matrix_around_new_center(positions1, com0)
		
		#get change in postions
		delta_pos = positions1 - positions0
		delta_pos = delta_pos.swapaxes(0,1)
		#get angle of points
		for p in delta_pos:	
			angle = dp.vector_to_angle(p)
			if angle <= np.pi/float(3):
				markov_space_count[1] += 1
			elif angle <= np.pi*(2/float(3)):
				markov_space_count[2] += 1
			elif angle <= np.pi:
				markov_space_count[3] += 1
			elif angle <= np.pi*(4/float(3)):
				markov_space_count[4] += 1
			elif angle <= np.pi*(5/float(3)):
				markov_space_count[5] += 1
			else:
				markov_space_count[6] += 1
	
	#get probabilities	
	for i in range(1,7):
		markov_space_count[i] = markov_space_count[i]/float(total_number_of_moves)
			
	return markov_space_count
	
def get_transformation(mp):
	#mp = dictionary of markov probabilities over hexagonal space
	
	p = random.random()
	a = random.uniform(0,np.pi/3)
	if p < mp[1]:
		angle = a
	elif p < mp[1] + mp[2]:
		#pi/3 - 2pi/3
		angle = np.pi/float(3) + a
		
	elif p < mp[1] + mp[2] + mp[3]:
		#2pi/3 - pi
		angle = 2*np.pi/float(3) + a
	elif p < mp[1] + mp[2] + mp[3] + mp[4]:
		#pi - 2
		angle = np.pi + a
	elif p < mp[1] + mp[2] + mp[3] + mp[4] + mp[5]:
		angle = 4*np.pi/float(3) + a
	else:
		angle = 5*np.pi/float(3) + a
	vector = dp.angle_to_unit_vector(angle)
	return vector
		
def create_markov_trajectory(initial_position, time, mp = None):
	'''
	given a 2D matrix create a trajectory for time
	using markov probabilities
	'''
	if mp == None:
		#1 minute
		trajectory0 = sio.loadmat('trajectory0.mat')['trajectory']
		mp = get_markov_probabilities(trajectory0)
	
	trajectory = initial_position.swapaxes(0,1)
	p0 = copy.deepcopy(trajectory)
	for ts in range(1,time):
		#print ts
		com = dp.com(p0.swapaxes(0,1))
		
		for i in range(p0.shape[0]):
			
			#the transformation if 	relative to COM
			v = get_transformation(mp)
			#p0[i] = np.add(p0[i], v)
			pt = p0[i] - com
			pt = np.add(pt, v)
			pt = pt + com
			p0[i] = pt
	
		trajectory = np.dstack((trajectory,p0.copy()))
	
	trajectory = trajectory.swapaxes(0,1)
	trajectory = trajectory.swapaxes(1,2)
	return trajectory


	
	
	
	
	
	
	
	
	
	

			
			
		