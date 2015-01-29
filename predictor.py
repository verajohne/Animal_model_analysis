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

def move(tran, current_position_matrix):
	x = np.add(tran[0], current_position_matrix[0])
	y = np.add(tran[1], current_position_matrix[1])
	return np.vstack((x,y))
	

def com_predictor(matrix):
	x = matrix[0][0]
	y = matrix[1][0]
	
	nodes_indexes = dp.randomlist(matrix.shape[2], 6)
	nodes_indexes.sort()
	nodes_indexes.reverse()
	
	for i in nodes_indexes:
		x = np.delete(x,i)
		y = np.delete(y,i)
		
	predicting_matrix = np.vstack((x,y))
	
	#predicting_indexes =[x for x in range(matric.shape[2]) if x not in nodes_indexes]
	nodes_matrix = dp.getSubset(nodes_indexes, matrix)
	dcom = dp.get_delta_com(matrix)
	i = 0
	temp = predicting_matrix
	for transition in dcom:
		print i
		i = i +1
		temp = move(transition, temp)
		predicting_matrix = np.dstack((predicting_matrix, temp))
	print "Predicting: ", predicting_matrix.shape
	print "Nodes: ", nodes_matrix.shape
	
	predicting_matrix = predicting_matrix.reshape(matrix.shape[0], matrix.shape[1], matrix.shape[2]-6)
	print "Predicting: ", predicting_matrix.shape
	result = np.dstack((predicting_matrix, nodes_matrix))
	print "Result: ", result.shape
	matrix_file = scipy.io.savemat('newTrajectory.mat', mdict={'trajectory': result}, format = '5' )
	return result
	#return predicting_matrix
	
	
	
	
	
	