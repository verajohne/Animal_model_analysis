import numpy as np
import scipy.io

import dataprocessor as dp

'''
1. start with 8am on day 1 topology
2. given data for 6 random

'''

def get_ch_flock(ch):
	return 13*ch

def get_com_flock(com):
	return c

def partition_matrix(matrix, partition_size):
	x = matrix[0][0]
	y = matrix[1][0]
	
	#nodes_indexes = dp.randomlist(matrix.shape[2], partition_size)
	nodes_indexes  = range(6)
	nodes_indexes.sort()
	nodes_indexes.reverse()
	
	for i in nodes_indexes:
		x = np.delete(x,i)
		y = np.delete(y,i)
		
	p1 = np.vstack((x,y))
	p2 = dp.getSubset(nodes_indexes, matrix)
	
	return (p2, p1)

def direction_predictor(matrix, subset = 6):
	'''
	1. calculate com6
	2. calculate com94
	3. get transition vector
	'''
	loggers = matrix.shape[2]
	predictions = loggers - subset
	data = partition_matrix(matrix, subset)
	#initially a 2D matrix at time0,postionXloggers
	#to be built into 3D trajectory
	predicting_matrix = data[1]
	#full 3D trajectory
	nodes_matrix = data[0]
	print nodes_matrix.shape
	
	'''
	For every TS, calculate com6, com94
	get transition vector
	add this to all points in 94
	'''
	com6 = dp.get_com(nodes_matrix)
	
	#same as iterating thought time
	temp = predicting_matrix
	
	for ts in range(matrix.shape[1] - 1):
		ts_position = np.zeros(2)
		com94 = dp.com(predicting_matrix)
		angle_between_com = dp.angle_between_vectors(com6[ts],com94)
		
		#create new position 2d matrix for ts
		for node_id in range(predictions):
			#this nodes position at previous ts
			position0 = np.swapaxes(temp,0,1)[node_id]
			position = dp.rotate(position0, angle_between_com)
			
			#position = np.transpose(temp)[node_id]
			#position = np.subtract(position0, com94)
			#position = np.add(position, com6[ts])
			#tran = np.subtract(com6[ts], com94)
			#position = np.add(position0, tran)
			if node_id == 0:
				ts_position = position
			else:
				ts_position = np.vstack((ts_position, position))
		ts_position = np.transpose(ts_position)
		temp = ts_position
		predicting_matrix = np.dstack((predicting_matrix, ts_position))
	
	print predicting_matrix.shape	
	
	predicting_matrix = np.swapaxes(predicting_matrix,1,2)
	print predicting_matrix.shape
	print nodes_matrix.shape
	result = np.dstack((predicting_matrix, nodes_matrix))
	print result.shape
	
	#matrix_file = scipy.io.savemat('newTrajectory2.mat', mdict={'trajectory': result}, format = '5' )
	
	return result
	

	
	
	
	
	
	