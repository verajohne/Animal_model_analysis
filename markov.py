import dataprocessor as dp
import numpy as np

import random

import scipy.io as sio


'''
Similar to markov 1
2 additional spaces
'''

STATIONARY_RADIUS = 0.1

RANDOM_RADIUS = 1

UPPER_RADIUS = 6


def get_markov_probabilities(trajectory):
	total_number_of_moves = (trajectory.shape[1]-1)*(trajectory.shape[2])
	
	markov_space_count = {}
	'''
	-1 = move anywhere within current hexagon
	0 = stationary
	1-6 = move to hexagon [1-6]
	'''
	for i in range(-1,7):
		markov_space_count[i] = 0
	
	distances = []
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
		
		for p in delta_pos:	
			distance = np.linalg.norm(p - np.zeros(2))
			distances.append(distance)
			angle = dp.vector_to_angle(p)
			
			if distance < STATIONARY_RADIUS:
				markov_space_count[0] += 1
			
			elif distance < RANDOM_RADIUS:
				markov_space_count[-1] +=1
			
			#which hexagon to move to
			elif angle <= np.pi/float(3):
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
	for i in range(-1,7):
		markov_space_count[i] = markov_space_count[i]/float(total_number_of_moves)
			
	return markov_space_count

def get_transformation(mp, step_size = None):
	'''
	return transition vector
	mp = markov_probabilities
	returns transition vector
	'''
	p = random.random()
	#a = random angle between 0-60 degrees
	a = random.uniform(0,np.pi/3)
	angle = 0
	d = 0
	if p < mp[-1]:
		#move random within current hexagon
		angle = random.uniform(0,2*np.pi)
		d = random.uniform(STATIONARY_RADIUS,RANDOM_RADIUS)
		#d = np.random.choice(step_size)
	elif p < mp[-1] + mp[0]:
		#stay in position
		return np.zeros(2)
	
	elif p <mp[-1] + mp[0] + mp[1]:
		angle = a
	elif p < mp[-1] + mp[0] + mp[1] + mp[2]:
		#pi/3 - 2pi/3
		angle = np.pi/float(3) + a
		
	elif p < mp[-1] + mp[0] + mp[1] + mp[2] + mp[3]:
		#2pi/3 - pi
		angle = 2*np.pi/float(3) + a
	elif p < mp[-1] + mp[0] + mp[1] + mp[2] + mp[3] + mp[4]:
		#pi - 2
		angle = np.pi + a
	elif p < mp[-1] + mp[0] + mp[1] + mp[2] + mp[3] + mp[4] + mp[5]:
		angle = 4*np.pi/float(3) + a
	else:
		angle = 5*np.pi/float(3) + a
		
	if d == 0:
		#d = random.uniform(RANDOM_RADIUS,UPPER_RADIUS)
		#d = np.random.choice(step_size)
		#param = 3.156
		#param = 3.86
		#param = 0.5827087
		#param = 0.890949
		#d = np.random.weibull(param) + np.random.uniform(0,0.5)
		#d = np.random.weibull(param)
		#add noise
		param = (3.5621113910571758, -0.20632901184649047, 1.3602469219787041)
		d = np.random.wald(param[0],param[2])
		
	vector = d*dp.angle_to_unit_vector(angle)

	return vector

def create_markov_trajectory(initial_position, time, mp = None, step_size = None):
	'''
	given a 2D matrix create a trajectory for time
	using markov probabilities
	'''
	
	'''
	if step_size == None:
		step_size = sio.loadmat('metric_stuff/step_size_traj0.mat')['stats'][0]
		v = np.percentile(step_size, 90)
		step_size = [x for x in step_size if x < v]
		step_size = [x for x in step_size if x > R]
	'''
	if mp == None:
		#1 minute
		trajectory0 = sio.loadmat('../basematrixes/trajectory0.mat')['trajectory']
		mp = get_markov_probabilities(trajectory0)
	
	trajectory = initial_position.swapaxes(0,1)
	p0 = initial_position.swapaxes(0,1)
	for ts in range(1,time):
		#com = dp.com(p0.swapaxes(0,1))
		com = dp.com(p0.transpose())
		#print ts
		for i in range(p0.shape[0]):
			#the transformation is	relative to COM
			v = get_transformation(mp, step_size)
			#p0[i] = np.add(p0[i], v)
			pt = p0[i] - com
			pt = np.add(pt, v)
			pt = pt + com
			p0[i] = pt
	
		trajectory = np.dstack((trajectory,p0.copy()))
		
	
	trajectory = trajectory.swapaxes(0,1)
	trajectory = trajectory.swapaxes(1,2)
	return trajectory
		
		
		
		
		
		
		
		
		