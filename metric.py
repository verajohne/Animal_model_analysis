import numpy as np
import scipy.io as sio
import scipy as sp
from sklearn.neighbors import NearestNeighbors
import dataprocessor as dp

'''
Metrics for KS tests

1. Mean distance to nearest neighbor per time
2. com100-com6 per time
3. convehhull per time

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
		print ts
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
	
	return np.array(mean_distances_to_KNN)

def diff_com(trajectory):
	six_loggers = random.sample(range(self.nr_of_loggers), 6)
	diff_com = []
	for ts in range(trajectory.shape[1]):
		com6 = np.zeros(2)
		for log in six_loggers:
			com6 += np.array([self.logger_data[0,ts,log], self.logger_data[1,ts,log]])
		com6 = com6/float(6)
		xs = np.mean(trajectory[0, ts])
		ys = np.mean(trajectory[1, ts])
		comF = np.array([xs,ys])
		diff_com.append(abs(comF-com6))
	
	return diff_com
		
def convexhull(trajectory):
	ch = []
	
	for ts in range(trajectory.shape[1]):
		xs = self.logger_data[0,ts]
		ys = self.logger_data[1,ts]
		pointsF = zip(xs,ys)
		hull = ConvexHull(pointsF)
		vertex_indexes = hull.vertices
		p = []
		for i in vertex_indexes:
			p.append(pointsF[i])
		areaF = polygon_area(p)
		ch.append(areaF)
	
	return ch

def main():
	trajectory = sio.loadmat('../matrixes/trajectory.mat')['trajectory']
	result = []
	for i in range(100):
		l = avg_knn(trajectory,1)
		result = result + l
	result = np.array(result)
	matrix_file = scipy.io.savemat('knnD1.mat', mdict={'stats': result}, format = '5' )
	
	
	result = []
	for i in range(100):
		l = diff_com(trajectory)
		result = result + l
	result = np.array(result)
	matrix_file = scipy.io.savemat('diffcomD1.mat', mdict={'stats': result}, format = '5' )
	
	result = []
	for i in range(100):
		l = convexhull(trajectory)
		result = result + l
	result = np.array(result)
	matrix_file = scipy.io.savemat('convexhullD1.mat', mdict={'stats': result}, format = '5' )
	
	

if __name__ == '__main__':
	main()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		