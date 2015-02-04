import numpy as np

'''
Fixes-Radius Near Neighbour Reporting using bucket like algorithm
Given a cartesian space of points, 'p', report neighbors
within a given radius 'r' of this node.

The dictionary is a mapping from a bucket (a,b) to a list of points
contained in the bucket.
'''

class frnnr(object):
	
	def __init__(self,r, p):
		self.radius = r
		self.points = p
		self.dict = {}
		self.prepare_dict()
		
	def getBucket(self,point):
		point = point.astype(int)
		point = (point[0]/self.radius, point[1]/self.radius)
		return point
	
	def insert(self, point):
		bucket = self.getBucket(point)
		if bucket in self.dict:
			self.dict[bucket].append(point)
		else:
			self.dict[bucket] = []
			self.dict[bucket].append(point)
	
	def find(self, bucket):
		if bucket in self.dict:
			return self.dict[bucket]
		else:
			return []
	
	def prepare_dict(self):
		#set up dictionary
		
		for p in self.points:
			self.insert(p)
			
	
	def return_nodes_inrange_to_check(self, point):
		'''
		returns points in bucket and all neighboring buckets
		 __  __	 __	
		|__||__||__| 
		|__||__||__|
		|__||__||__|
		'''
		nodes = []
		bucket = self.getBucket(point)
		nodes = nodes + self.find(bucket)
		
		for x in range(-1,2):
			for y in range (-1,2):
				b = (bucket[0] + x, bucket[1] + y) 		
				nodes = nodes + self.find(b)
		
		return nodes
		
		
	def get_distances(self, point):
	
		nodes = self.return_nodes_inrange_to_check(point)
		
		distance = []
		for node in nodes:
			dist = np.linalg.norm(point - node)
			if dist < self.radius and dist != 0:
				distance.append(dist)
		return distance
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
			
		
	