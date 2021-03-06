import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.mlab as mlab

import dataset_time as dst

class Simulation(object):

	def __init__(self, trajectory_matrix):
		'''
		trajectory_matrix is a 3D matrix of movement per time
		'''
		self.logger_data = trajectory_matrix
		self.nr_of_loggers = trajectory_matrix.shape[2]
		self.nr_of_samples = trajectory_matrix.shape[1]
		self.dimensions = trajectory_matrix.shape[0]
	
	def update_plot(self, time_step, data, line):
		'''
		function used by animate_flock
		'''
		
		time = dst.get_time(time_step, only_time = True)
		#print time[0], ":", time[1]
		print time_step
		xs = self.logger_data[0, time_step]
		ys = self.logger_data[1, time_step]
		nodes = np.concatenate((xs,ys)).reshape(self.dimensions, self.nr_of_loggers)
		line.set_data(nodes)
		return line,

	def animate_flock(self):
		fig1 = plt.figure()
		xs = self.logger_data[0, 0]
		ys = self.logger_data[1, 0]
		
		data = np.concatenate((xs,ys)).reshape(2,self.nr_of_loggers)
		l, = plt.plot([], [], 'ro')
		plt.xlim(0,4000)
		plt.ylim(0,4000)
		#plt.axis([min(xs) - 10, max(xs)+10, min(ys)-10, max(ys)+10])
		plt.xlabel('x-coordinate')
		plt.xlabel('y-coordinate')
		plt.title('Sheep Simulation')
		line_ani = animation.FuncAnimation(fig1, self.update_plot, self.nr_of_samples, 
			fargs=(data, l), interval=1, blit=False, repeat = False)
		
		plt.show()