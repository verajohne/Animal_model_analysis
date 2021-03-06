import numpy as np
import scipy.io as sio
import scipy as sp
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
import random

import dataprocessor as dp
import infection
import field
import frnnr

'''
This module includes functions to aid
parameter tests for hypotheses 1 and 2
'''

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
	
def getListPoints(x,y):
	'''
	takes list of x and y and returns list of np.array points
	'''
	points = []
	for i in range(x.size):
		p = np.array([x[i],y[i]])
		points.append(p)
	return points

def avg_knn(trajectory, K):
	'''
	return list of distances to kth nearest neighbor for each point
	at each time step
	'''
	distances_to_KNN = np.empty(0)
	for ts in range(trajectory.shape[1]):
		xs = trajectory[0, ts]
		ys = trajectory[1, ts]
		points = np.array(getListPoints(xs,ys))
		#K+1 as algorithm includes itself
		nbrs = NearestNeighbors(n_neighbors = K+1, algorithm = 'ball_tree').fit(points)
		distances = nbrs.kneighbors(points)[0]
		distances = distances.swapaxes(0,1)
		distances_to_kth = distances[K]
		distances_to_KNN = np.concatenate((distances_to_KNN, distances_to_kth))h
	return distances_to_KNN

def diff_com(trajectory):
	'''
	return difference in com between 6 random nodes
	and the com of the trajectory at each time step
	'''
	six_loggers = random.sample(range(100), 6)
	diff_com = np.empty(0)
	for ts in range(trajectory.shape[1]):
		com6 = np.zeros(2)
		for log in six_loggers:
			com6 += np.array([trajectory[0,ts,log], trajectory[1,ts,log]])
		com6 = com6/float(6)
		xs = np.mean(trajectory[0, ts])
		ys = np.mean(trajectory[1, ts])
		comF = np.array([xs,ys])
		distance = np.linalg.norm(comF-com6)
		diff_com = np.append(diff_com, distance)
	
	return diff_com
	
def diff_com_vectors(trajectory):
	'''
	return list of vectors
	vectors is the vector between com of trajectory
	and com of 6 random sheep at each time step
	'''
	six_loggers = random.sample(range(100), 6)
	diff_com = []
	for ts in range(trajectory.shape[1]):
	#for ts in range(10):
		com6 = np.zeros(2)
		for log in six_loggers:
			com6 += np.array([trajectory[0,ts,log], trajectory[1,ts,log]])
		com6 = com6/float(6)
		xs = np.mean(trajectory[0, ts])
		ys = np.mean(trajectory[1, ts])
		comF = np.array([xs,ys])
		dv = np.subtract(comF,com6)
		diff_com.append(dv)
		
	return diff_com
		
def convexhull(trajectory):
	'''
	convex hull at each timestep
	'''
	ch = np.empty(0)
	
	for ts in range(trajectory.shape[1]):
		xs = trajectory[0,ts]
		ys = trajectory[1,ts]
		pointsF = zip(xs,ys)
		hull = ConvexHull(pointsF)
		vertex_indexes = hull.vertices
		p = []
		for i in vertex_indexes:
			p.append(pointsF[i])
		areaF = polygon_area(p)
		ch = np.append(ch, areaF)
	
	return ch

def number_of_neighbords_within_radius(trajectory, N):
	neighbors = []
	for ts in range(trajectory.shape[1]):
		matrix = dp.returnTimeMap(ts, trajectory)
		points = dp.getListPoints(matrix[0], matrix[1])
		f = frnnr.frnnr(N, points)
		for i in range(100):
			p = np.array([matrix[0][i],matrix[1][i]])
			distances = f.get_distances(p)
			neighbors.append(len(distances))
	return np.array(neighbors)
			
def distance_travelled(trajectory):
	'''
	distance travelled by a flock
	'''
	t = trajectory.shape[1] - 1
	matrix_0 = dp.returnTimeMap(0,trajectory)
	matrix_t = dp.returnTimeMap(t,trajectory)
	com0 = dp.com(matrix_0)
	comt = dp.com(matrix_t)
	return np.linalg.norm(com0- comt)

def infection_analysis(trajectory_list, p,d, runs):
	result = []
	i = infection.Infection(p, d)
	f = field.Field(trajectory_list, i)
	for i in range(runs):
		time = f.run()
		result.append(time)
	return np.array(result)

def leave_one_out_analysis(list_of_herd_trajectories, p,d):
	'''
	difference in spread of infection, varying which of
	the 14 flocks are present in field.
	'''
	herds = len(list_of_herd_trajectories)
	for i in range(herds):
		list = herds[:i] + herds[i+1 :]
		result = infection_analysis(list, p,d,100)
		fn = 'leave_out/LO_herd' + str(i+1) + '.mat'
		matrix_file = sio.savemat(fn, mdict={'stats': result}, format = '5' )

	
		
		
		
		
		
		
		
		
		
		
		
		
		
		