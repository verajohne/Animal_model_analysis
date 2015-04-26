from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm  
import scipy.io
import dataset_time as dst
import scipy.stats

'''
For plotting of data
Some of these functions are particular to the structure of the dataset
'''

def plot_cdf(lists):
	#c = ['-g','-b']
	c= ['-r','-b', '-g', '-y', '-k', '-m', '-c', '#fe677f', '#FF00FF','#9900CC', '#663333', '#00FFCC','#FF9900', '#FFCC33']
	#label = ['Dataset 3','setting 1','setting 2','setting 3','setting 4','setting 5']
	for list,col in zip(lists,c):
		sorted=np.sort(list)
		
		yvals=np.arange(sorted.shape[0])/float(sorted.shape[0])
		plt.plot( sorted, yvals, col, alpha = 0.7)
	
	#plt.legend(loc='lower right')
	plt.xlim(200000,0.9*10**6)
	#plt.xlim(0,150)
	#plt.xlim(0,6)
	#plt.ylim(0,1)
	plt.ylabel('probability')
	#plt.xlabel('Number of neighbours within a raidus of 20m')
	#plt.xlabel('Distance to nearest neighbor')
	plt.xlabel('Area of convex hull')
	#plt.title('Field of 14 flocks starting with one infected sheet (p = 0.2, d = 1')
	plt.show()

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

def plot(matrixes):
	'''
	takes list of 2D matrices and plots
	'''
	plt.xlabel('Time')
	plt.ylabel('Number Infected')
	
	c= ['-r','-b', '-g', '-y', '-k', '-m', '-c', '#fe677f', '#FF00FF','#9900CC', '#663333', '#00FFCC','#FF9900', '#FFCC33']
	labels = ['Dataset 3', 'Setting 1', 'Setting2','Setting 3', 'Setting4','Setting 5']
	for matrix,col,l in zip(matrixes,c,labels):
		plt.plot( matrix[0], matrix[1], col, label = l )
	plt.legend(loc='lower right')
	plt.show()
	
def set_graph():
	sns.set_palette("deep", desat = .6)
	sns.set_context(rc = {"figure.figsize":(8,4)})
	
def trim_data(data, lower, upper):
	lwr = np.percentile(data, lower)
	upr = np.percentile(data, upper)
	new_data = [i for i in data if i >= lwr and i <= upr]
	return np.array(new_data)
	
def distribution_fit(array):
	distributions = ['beta', 'weibull_min', 'wald','invgauss']
	h = plt.hist(array, bins=range(48), color='w')
	x = scipy.arange(array.size)
	for d in distributions:
		distribution = getattr(scipy.stats, d)
		param = distribution.fit(array)
		print d,' : ', param
		pdf_fitted = distribution.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * array.size
		plt.plot(pdf_fitted, label=d)
	plt.legend(loc='upper right')
	plt.xlim(0,20)
	plt.ylim(0,400000)
	plt.xlabel('Step size')
	plt.ylabel('Number of occurrences')
	plt.show()
	
def heat_map(x,y):
	'''
	x,y numpy array
	'''
	x = x.ravel()
	y = y.ravel()
	plt.hexbin(XX,YY,gridsize = 1000,cmap=plt.cm.jet, bins=None)
	cb = plt.colorbar()
	plt.show()
	
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
	plt.xlabel('Step Size')
	plt.xlim(0,100)
	plt.ylim(0,1)
	plt.axvline(np.mean(x), color='r', linestyle='dashed', linewidth=2)
	plt.show()

def kde(data):
	sns.kdeplot(data, shade=True)
	plt.axvline(np.mean(data), color='r', linestyle='dashed', linewidth=2)
	plt.ylabel('Normalized probability')
	plt.xlabel('time till 90% infected')
	plt.xlim(0,60)
	plt.show()

def multiple_kde(datasets):
	'''
	kernel density function
	datasets if list of np.arrays
	'''
	labels = ['Dataset 3', 'Setting 1']
	color = ['b','g']
	for data,l, c in zip(datasets, labels, color):
		sns.kdeplot(data)
		plt.axvline(np.mean(data), color=c, linestyle='dashed', linewidth=2, label = l)
	plt.legend(loc='upper right')
	plt.ylabel('Normalized probability')
	plt.xlabel('Distance to nearest neighbor')
	plt.show()

def histograms(data, bins = 10):
	label = ['Dataset 3', 'Setting 3','Setting 2', 'Setting 3']
	color = ['r','b','g','y','k','c']
	for d,l,c in zip(data, label, color):
		weights = np.ones_like(d)/len(d)
		plt.hist(d, bins = bins,color = c, weights = weights, label = l, alpha = 0.5)
		plt.axvline(np.mean(d), color=c, linestyle='dashed', linewidth=2)
	plt.ylabel('probability')
	plt.xlabel('Number of neighbors within 20 meters')
	plt.legend(loc='upper right')
	plt.xlim(0,30)
	#plt.ylim(0,1)
	plt.show()

	
	
	
	
	
	
	
	
	
