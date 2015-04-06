from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm  
import scipy.io
import dataset_time as dst

'''
For plotting of data
Some of these functions are particular to the structure of the dataset
'''

def plot_cdf(lists):
	#c = ['-g','-b']
	c= ['-r','-b', '-g', '-y', '-k', '-m', '-c', '#fe677f', '#FF00FF','#9900CC', '#663333', '#00FFCC','#FF9900', '#FFCC33']
	for list,col in zip(lists,c):
		sorted=np.sort(list)
		
		yvals=np.arange(sorted.shape[0])/float(sorted.shape[0])
		plt.plot( sorted, yvals, col )
	
	plt.xlim(-10,50)
	#plt.xlim(0,0.5*10**3)
	plt.ylabel('Proportion of flock infected')
	plt.xlabel('Time')
	plt.title('Field of 14 flocks starting with one infected sheet (p = 0.2, d = 1')
	#plt.text(1000, 0.1, 'Original Trajectory 1min (trajectory0)', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=10)
	#plt.text(1000, 0.2, 'Markov model',verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10)
	plt.show()

def import_data(file, ds):
	data = scipy.io.loadmat(file)
	data = data[ds]
	return data[0]

def import_area():
	data = scipy.io.loadmat('area100.mat')
	data = data['convexhull']
	data = data.reshape(2,1536100)
	data = data[1]/data[0]
	return data

def import_areaTod():
	set_graph()
	data = scipy.io.loadmat('areaTOD.mat')
	data = data['convexhull']
	mean = []
	for ts in range(1440):
		l = data[1][ts]/data[0][ts]
		l = trim_data(l,0,90)
		mean.append(np.mean(l))
		p = [np.max(l), np.min(l)]
		plt.plot([ts,ts], p, color = 'r')
	plt.plot(range(1440), mean)
	
	x_ticks = ['00:00', '04:00','08:00','12:00', '16:00', '20:00', '24:00']
	plt.xticks([0,4*60,8*60,12*60,16*60,20*60, 24*60], x_ticks)
	plt.show()

def plot_cdfs(sets):
	#set_graph()
	c= ['-r','-b', '-g', '-y', '-k', '-m', '-c', '#eeefff', '#FF00FF','#9900CC', '#663333', '#00FFCC','#FF9900', '#FFCC33']
	c = c + c
	for set,col in zip(sets,c):
		y = set[0]
		#if len(y) == 1:
		#	y = y[0]
		x = range(len(set[0]))
		plt.plot(x, y, col)
	
	plt.ylabel('% infected in each flock')
	plt.xlabel('Time (sec)')
	plt.xlim(0,5000)
	plt.ylim(0,100)
	#x_ticks = [dst.toString(dst.get_time(0)), dst.toString(dst.get_time(1000)),dst.toString(dst.get_time(2000)),dst.toString(dst.get_time(3000)),dst.toString(dst.get_time(4000)), dst.toString(dst.get_time(5000)), dst.toString(dst.get_time(6000)), dst.toString(dst.get_time(7000))]
	#plt.xticks([0, 1000, 2000, 3000, 4000, 5000,6000,7000], x_ticks)
	#locs, labels = plt.xticks()
	#plt.setp(labels, rotation=45)
	plt.show()

def plot(matrix):
	#plt.scatter(matrix[0],matrix[1], c = 'r')
	plt.xlabel('Time')
	plt.ylabel('Number Infected')
	plt.title('Number infected per time for herd 6 (10 min samples)')
	plt.plot( matrix[0], matrix[1], '-g' )
	plt.show()
	
def set_graph():
	sns.set_palette("deep", desat = .6)
	sns.set_context(rc = {"figure.figsize":(8,4)})
	
def trim_data(data, lower, upper):
	lwr = np.percentile(data, lower)
	upr = np.percentile(data, upper)
	new_data = [i for i in data if i >= lwr and i <= upr]
	return np.array(new_data)
	
def histogram(x, bins = 10, rp = False, weights = True, normed = False):
	set_graph()
	if rp == True:
		sns.rugplot(x);
	#want the sum of the y-values for bars to equal 1
	if weights:
		weights = np.ones_like(x)/len(x)
		plt.hist(x, bins = bins, weights = weights)
	else:
		plt.hist(x, bins = bins, normed = normed)
	plt.ylabel('Normalized probability')
	plt.xlabel('Difference in convex hull area')
	plt.xlim(0,3500000)
	plt.ylim(0,0.35)
	plt.axvline(np.mean(x), color='r', linestyle='dashed', linewidth=2)
	plt.show()

def kde(data):
	sns.kdeplot(data, shade=True)
	plt.axvline(np.mean(data), color='r', linestyle='dashed', linewidth=2)
	plt.ylabel('Normalized probability')
	plt.xlabel('Time to 90% of flock infected (100 runs)')
	#plt.title('100 samples of Trajectory0 (1 min samples')
	plt.show()

def shist(x, bins):
	mean = np.mean(x)
	sigma = np.std(x)

	x -= mean 

	#x_plot = np.linspace(min(x), max(x), 1000)                                                               

	fig = plt.figure()                                                               
	ax = fig.add_subplot(1,1,1)                                                      
	
	ax.hist(x, bins=bins, normed=True, label="data")
	#ax.plot(x_plot, norm.pdf(x_plot, mean, sigma), 'r-', label="pdf")                                                          

	ax.legend(loc='best')

	x_ticks = np.arange(-3*sigma, 3.1*sigma, sigma)                                  
	x_labels = [r"${} \sigma$".format(i) for i in range(-3,3)]                       

	ax.set_xticks(x_ticks)                                                           
	ax.set_xticklabels(x_labels)                                                     

	plt.show() 
	
