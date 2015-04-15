import numpy as np
import scipy.io
from scipy.spatial import ConvexHull
import random

'''
Standard trajectory format is [space_dimensions,time_steps, nodes]
Mostly a set of helper functions used in scripts and other modules
matrix implies 2D in this file. For example a position matrix for a time instance.
'''

def slice_time(trajectory, t0, t1):
	'''
	return subset of time interval: [t0,t1]
	'''
	return trajectory[0:2, t0:t1, 0:100]

def merge_matrixes(list_of_trajectories):
	'''
	trajectories in list must have same time dimension
	returns one big trajectory matrix combined of all trajectories in list_of_trajectories
	usually for simulate multiple trajectory matrixes in Simulation class
	'''
	result = list_of_trajectories[0]
	for i in range(1,len(list_of_trajectories)):
		result = np.dstack((result, list_of_trajectories[i]))
	return result

def random_walk(steps):
	temp = np.array([1000.0,1000.5])
	result = [temp]
	i = 0
	for step in range(steps -1):
		#if step % 120 == 0:
		#	i = i + np.pi/2
		#	i = i % (3*np.pi/2)
		#direction = np.random.uniform(0+i,np.pi/2 +i)
		'''
		direction = np.random.uniform(0,2*np.pi)
		dx = np.cos(direction)
		dy = np.sin(direction)
		d = np.array([dx,dy])
		temp = np.add(temp, d)
		result.append(temp)
		'''
		result.append(temp)
	return result	

def random_array_list(listlength): #???
	result = []
	step_size = 0.5
	for i in range(listlength):
		a = np.array([np.random.randint(0,3000),np.random.randint(0,3000)])
		result.append(a)
	return result

def partition_matrix(trajectory, partition_size):
	'''
	returns two trajectories
	'''
	x = trajectory[0][0]
	y = trajectory[1][0]
	
	set1_indexes = randomlist(trajectory.shape[2], partition_size)
	set2_indexes = [x for x in range(trajectory.shape[2]) if x not in set1_indexes]
	trajectory1 = getSubset(set1_indexes, trajectory)
	trajectory2 = getSubset(set2_indexes, trajectory)
	
	return [trajectory1, trajectory2]

def returnTimeMap(ts, trajectory):
	'''
	returns 2D matrix image of points at particular timestep
	'''
	#return trajectory.swapaxes(0,1)[ts]
	return np.array([trajectory[0][ts], trajectory[1][ts]])
	
def importData():
	'''
	helper function for quick import of matlab matrices during testing
	'''
	trajectory_mat = scipy.io.loadmat('../matrixes/trajectory.mat')
	trajectory = trajectory_mat['trajectory']
	return trajectory
	
def importMatrix(path, id):
	trajectory_mat = scipy.io.loadmat(path)
	trajectory = trajectory_mat[id]
	return trajectory
	
def randomlist(r, sample_size):
	'''
	r > sample_size
	return list of random numbers
	'''
	return random.sample(range(r), sample_size)
		
def getTimeSeriesForNode(i, trajectory):
	'''
	returns trajectory for a particular node over time
	'''
	matrix = np.transpose(trajectory)
	return np.transpose(trajectory[i])

def getSubset(indexes, trajectory):
	'''
	returns matrix for a subset of the nodes in matrix
	as specified in indexes
	'''
	subset = getTimeSeriesForNode(indexes[0], trajectory)
	for i in range(1,len(indexes)):
		temp = getTimeSeriesForNode(indexes[i], trajectory)
		subset = np.dstack((subset, temp))
	return subset

def getListPoints(x,y):
	'''
	takes list of x and y and returns list of np.array points
	'''
	points = []
	for i in range(len(x)):
		p = np.array([x[i],y[i]])
		points.append(p)
	return points

def angle_between_vectors(v,u):
	'''
	angle between v,u
	return difference of anticlockwise rotation v needs to rotate to equal u
	'''
	angle_v = vector_to_angle(v)
	angle_u = vector_to_angle(u)
	diff = angle_u - angle_v
	return diff

def rotate(vector, angle):
	#rotates anti clockwise about origin
	transition_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
	new_vector = np.dot(vector, transition_matrix)
	return new_vector

def rotate_about_point(vector, rotation_vector, angle):
	'''
	rotate about rotation_vector
	'''
	v = np.subtract(vector, rotation_vector)
	v = rotate(v, angle)
	v = np.add(v, rotation_vector)
	return v

def com(matrix):
	'''
	2D time instance matrix
	com = centre of mass
	'''
	return np.array([np.mean(matrix[0]), np.mean(matrix[1])])

def get_com(trajectory):
	'''
	returns a list of com
	'''
	com = []
	for ts in range(trajectory.shape[1]):
		xs = np.mean(trajectory[0, ts])
		ys = np.mean(trajectory[1, ts])
		comts = np.array([xs,ys])
		com.append(comts)	
	return np.array(com)

def get_delta_com(trajectory):
	'''
	returns a list of transition vectors between the center of mass
	at each time step
	'''
	com = get_com(trajectory)
	dcom = []
	for i in range(len(com) - 1):
		dc = np.subtract(com[i+1], com[i])
		dcom.append(dc)
	return dcom

def polygon_area(vertices):
	'''
	shoelace algorithm
	'''
	n = len(vertices)
	area = 0.0
	for i in range(n):
		j = (i + 1) % n
		area += vertices[i][0] * vertices[j][1]
		area -= vertices[j][0] * vertices[i][1]
	area = abs(area) / 2.0
	return area

def ch_area(matrix):
	'''
	given matrix, returns list of convex hull areas per time
	'''
	ch_area = []
	for ts in range(matrix.shape[1]):
		points = getListPoints(matrix[0][ts], matrix[1][ts])
		hull = ConvexHull(points)
		vertex_indexes = hull.vertices
		p = []
		for i in vertex_indexes:
			p.append(points[i])
		area = polygon_area(p)
		ch_area.append(area)
	return ch_area
	
def vector_to_angle(vector):
	'''returns positive angles '''
	x = vector[0]
	y = vector[1]
	phi = 0
	if x == 0:
		if y  > 0:
			phi = np.pi/2
		else:
			phi = np.pi*3/2
	else:
		phi = np.arctan(y/x)	# x > 0, y >0 || x > 0, y==0
		if x < 0:
			phi = phi + np.pi
		else:
			if y < 0: # x > 0, y < 0
				phi = phi + 2*np.pi
	return phi

def position_matrix_around_new_center(matrix, center):
	'''
	matrix should be a 2 X N matrix
	center should be a (x,y) array
	'''
	x = matrix[0] - center[0]
	y = matrix[1] - center[1]
	return np.vstack([x,y])

def angle_to_unit_vector(angle):
	return np.array([np.cos(angle), np.sin(angle)])

def get_list_of_rates(dic):
	'''
	given dictionary of infection rates
	return lists of matrixes of infection for stats plotting
	'''
	result = []
	for i in range(1,len(dic)+1):
		y = np.array(dic[i])
		x = np.arange(y.size)
		xy = np.array([x,y])
		result.append(xy)
	return result
	
		





