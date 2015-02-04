from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm  
import scipy.io
import dataset_time as dst

'''
For plotting of data (pdf)
Some of these functions are particular to my dataset
'''


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
	c= ['-r','-b', '-g']
	for set,col in zip(sets,c):
		y = set
		x = range(len(set))
		plt.plot(x, y, col)
	
	plt.ylabel('% infected in each flock')
	plt.xlabel('Time (sec)')
	#x_ticks = [dst.toString(dst.get_time(0)), dst.toString(dst.get_time(1000)),dst.toString(dst.get_time(2000)),dst.toString(dst.get_time(3000)),dst.toString(dst.get_time(4000)), dst.toString(dst.get_time(5000)), dst.toString(dst.get_time(6000)), dst.toString(dst.get_time(7000))]
	#plt.xticks([0, 1000, 2000, 3000, 4000, 5000,6000,7000], x_ticks)
	#locs, labels = plt.xticks()
	#plt.setp(labels, rotation=45)
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
		
	plt.axvline(np.mean(x), color='r', linestyle='dashed', linewidth=2)
	plt.show()

def kde(data):
	sns.kdeplot(data, shade=True)
	plt.axvline(np.mean(data), color='r', linestyle='dashed', linewidth=2)
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