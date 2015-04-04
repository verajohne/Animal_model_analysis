import numpy as np
import scipy.io as sio
import scipy as sp
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
import random

import dataprocessor as dp
import infection
import field

'''
Metrics for KS tests to evaluate models

1. Mean distance to nearest neighbor per time
2. com100-com6 per time
3. convehhull per time
4. time to 90% infection
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
	return list of avg distance to Kth neighbor
	'''
	mean_distances_to_KNN = []
	samples = trajectory.shape[2]
	for ts in range(trajectory.shape[1]):
	#for ts in range(10):
		xs = trajectory[0, ts]
		ys = trajectory[1, ts]
		points = np.array(getListPoints(xs,ys))
		#K+1 as algorithm includes itself
		nbrs = NearestNeighbors(n_neighbors = K+1, algorithm = 'ball_tree').fit(points)
		distances = nbrs.kneighbors(points)[0]
		distances = distances.swapaxes(0,1)
		distances_to_kth = distances[K]
		mean_d = sum(distances_to_kth)/float(samples)
		mean_distances_to_KNN.append(mean_d)
	
	return mean_distances_to_KNN

def diff_com(trajectory):
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
		distance = np.linalg.norm(comF-com6)
		diff_com.append(distance)
	
	return diff_com
		
def convexhull(trajectory):
	'''
	convex hull at each timestep
	'''
	ch = []
	
	for ts in range(trajectory.shape[1]):
	#for ts in range(10):
		xs = trajectory[0,ts]
		ys = trajectory[1,ts]
		pointsF = zip(xs,ys)
		hull = ConvexHull(pointsF)
		vertex_indexes = hull.vertices
		p = []
		for i in vertex_indexes:
			p.append(pointsF[i])
		areaF = polygon_area(p)
		ch.append(areaF)
	
	return ch

def infection_analysis(trajectory_list, p,d, runs):
	result = []
	i = infection.Infection(p, d)
	f = field.Field(trajectory_list, i)
	for i in range(runs):
		print i
		time = f.run()
		result.append(time)
	return np.array(result)

def leave_one_out_analysis(list_of_herd_trajectories, p,d):
	herds = len(list_of_herd_trajectories)
	for i in range(herds):
		list = herds[:i] + herds[i+1 :]
		result = infection_analysis(list, p,d,100)
		fn = 'LO_herd' + str(i+1) + '.mat'
		matrix_file = sio.savemat(fn, mdict={'stats': result}, format = '5' )


def main():

	trajectory0 = sio.loadmat('../basematrixes/trajectory0.mat')['trajectory']
	trajectory10s = sio.loadmat('../basematrixes/trajectory10s.mat')['trajectory']
	markov1 = sio.loadmat('../markov/markov1.mat')['trajectory']
	
	result = infection_analysis(markov1,0.02,1,100)
	matrix_file = sio.savemat('../metric_stuff/infection_markov1.mat', mdict={'stats': result}, format = '5' )
	
	'''
	result = []
	for i in range(100):
		print i
		ch = diff_com(markov1)
		result = result + ch
	result = np.array(result)
	n = '../metric_stuff/dcom_markov1.mat' 
	matrix_file = sio.savemat(n, mdict={'stats': result}, format = '5' )
	

	
	
	
	
	result = []
	for i in range(100):
		print i
		ch = convexhull(markov1)
		result = result + ch
	result = np.array(result)
	n = '../metric_stuff/ch_markov1.mat' 
	matrix_file = sio.savemat(n, mdict={'stats': result}, format = '5' )

	result = []
	for i in range(100):
		print i
		ch = convexhull(trajectory10s)
		result = result + ch
	result = np.array(result)
	n = '../metric_stuff/ch_trajectory10s.mat' 
	matrix_file = sio.savemat(n, mdict={'stats': result}, format = '5' )
	
	

	trajectory = sio.loadmat('../pred_matrix/trajectory10.mat')['trajectory']
	result = infection_analysis(trajectory, 0.2,1,100)
	n = '../metric_stuff/infection_d3.mat' 
	matrix_file = sio.savemat(n, mdict={'stats': result}, format = '5' )
	
	filename = '../pred_matrix/herd' + str(5) + '_100.mat'
	trajectory = sio.loadmat(filename)['herd']
	result = infection_analysis(trajectory, 0.2,1,100)
	n = '../metric_stuff/infection_herd' +str(5) + '.mat' 
	matrix_file = sio.savemat(n, mdict={'stats': result}, format = '5' )
	
	
	for j in range(1,15):
		filename = '../pred_matrix/herd' + str(j) + '_100.mat'
		trajectory = sio.loadmat(filename)['herd']
	
		result = []
		for i in range(100):
			l = avg_knn(trajectory,1)
			result = result + l
		result = np.array(result)
		n = '../metric_stuff/knn_herd' +str(j) + '.mat' 
		matrix_file = sio.savemat(n, mdict={'stats': result}, format = '5' )
	
	
		result = []
		for i in range(100):
			print i
			l = diff_com(trajectory)
			result = result + l
		result = np.array(result)
		n = '../metric_stuff/diffcom_herd' +str(j) + '.mat' 
		matrix_file = sio.savemat(n, mdict={'stats': result}, format = '5' )
	
		result = []
		for i in range(100):
			l = convexhull(trajectory)
			result = result + l
		result = np.array(result)
		n = '../metric_stuff/convexhullherd' +str(j) + '.mat' 
		matrix_file = sio.savemat(n, mdict={'stats': result}, format = '5' )
		
		#infection
		result = infection_analysis(trajectory, 0.2,1,100)
		n = '../metric_stuff/infection_herd' +str(j) + '.mat' 
		matrix_file = sio.savemat(n, mdict={'stats': result}, format = '5' )
		'''
	

if __name__ == '__main__':
	main()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		