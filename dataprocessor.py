import numpy as np
import scipy.io
from scipy.spatial import ConvexHull
import random

'''
Standard matrix format is [space_dimensions,time_steps, nodes] 

'''

def returnTimeMap(ts, matrix):
	x = matrix[0][ts]
	y = matrix[1][ts]
	return np.vstack((x,y))
	
def importData():
	trajectory_mat = scipy.io.loadmat('newTrajectory.mat')
	trajectory = trajectory_mat['trajectory']
	return trajectory
	
def randomlist(r, sample_size):
	 return random.sample(range(r), sample_size)
	
	
def getTimeSeriesForNode(i, matrix):
	matrix = matrix.reshape(matrix.shape[0],matrix.shape[2], matrix.shape[1])
	x = matrix[0][i]
	y = matrix[1][i]
	return np.vstack((x,y))

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
		
	






