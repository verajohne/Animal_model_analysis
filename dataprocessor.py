import numpy as np
import scipy.io
from scipy.spatial import ConvexHull
import random

'''
Standard matrix format is [space_dimensions,time_steps, nodes]
Mostly a set of helper functions used in scripts and other modules
'''

def returnTimeMap(ts, matrix):
	'''
	returns 2D matrix image of points at particular timestep
	'''
	x = matrix[0][ts]
	y = matrix[1][ts]
	return np.vstack((x,y))
	
def importData():
	'''
	helper function for quick import of matlab matrices during testing
	'''
	trajectory_mat = scipy.io.loadmat('../matrixes/trajectory.mat')
	trajectory = trajectory_mat['trajectory']
	return trajectory
	
def importMatrix(matlabName, id):
	trajectory_mat = scipy.io.loadmat(matlabName)
	trajectory = trajectory_mat[id]
	return trajectory
	
def randomlist(r, sample_size):
	 return random.sample(range(r), sample_size)
	
	
def getTimeSeriesForNode(i, matrix):
	'''
	returns trajectory for a particular node over time
	'''
	matrix = np.transpose(matrix)
	return np.transpose(matrix[i])

def getSubset(indexes, matrix):
	'''
	returns matrix for a subset of the nodes in matrix
	as specified in indexes
	'''
	subset = getTimeSeriesForNode(indexes[0], matrix)
	for i in range(1,len(indexes)):
		temp = getTimeSeriesForNode(indexes[i], matrix)
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
	angle_v = vector_to_angle(v)
	angle_u = vector_to_angle(u)
	
	cosine_of_angle = np.dot(v,u)/np.linalg.norm(v)/np.linalg.norm(u) #check if this will be decimal division
	angle = np.arccos(cosine_of_angle)
	if np.isnan(angle):
		if (v == u).all():
			angle = 0.0
		else:
			angle = np.pi
	if angle_v > angle_u:
		angle = -angle
	
	return angle

def rotate(vector, angle):
	transition_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
	new_vector = np.dot(vector, transition_matrix)
	return new_vector
	
def com(matrix):
	#2D time instance matrix
	return np.array([np.mean(matrix[0]), np.mean(matrix[1])])

def get_com(matrix):
	com = []
	for ts in range(matrix.shape[1]):
		xs = np.mean(matrix[0, ts])
		ys = np.mean(matrix[1, ts])
		comts = np.array([xs,ys])
		com.append(comts)	
	return np.array(com)

def get_delta_com(matrix):
	com = get_com(matrix)
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





