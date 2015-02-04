import numpy as np

'''
This is for the specific dataset of the Trajectory matrix
Starting at 8am
'''	
	
def get_time_step(time, day_ts = False):
	'''
	takes a time array of this format:
	time = np.array([year, month, day, hr, minutes])
	returns corresponding timestep.
	Hard coded for this particular data.
	If day_ts == True. Time stamp returns is in mod 1440
	'''
	year = time[0]
	month = time[1]
	day = time[2]
	hr = time[3]
	minutes = time[4]
	time_step = 0
	if day == 3:
		hr = hr - 16
		time_step = hr*60 + minutes

	else:
		time_step = (day - 4)*(24*60) + hr*60 + minutes + 460

	if day_ts == True:
		return hr*60 + minutes
	else:
		return time_step


def get_time(time_step, only_time = False):
	'''
	given a timestep, return np array of time
	hard coded for this particular data.
	only_time == True, ignores day, month and year.
	'''
	if(time_step < 60):
		if only_time  == True:
			return np.array([16, time_step])
		return np.array([2012,2,3,16,time_step])
	if (time_step >= 60) & (time_step < 480):
		hrs = time_step/60 +16
		minutes = time_step % 60
		if only_time  == True:
			return np.array([hrs, minutes])
		return np.array([2012,2,3,hrs,minutes])
	if (time_step >= 480):
		time_step = time_step - 480
		day = 4 + time_step/1440
		hrs = time_step/60 % 24
		minutes = time_step % 60
		if only_time  == True:
			return np.array([hrs, minutes])
		return np.array([2012,2,day,hrs,minutes])
		
		
def toString(t):
	time = t[4], ":",t[3],":", t[2], "/", t[1]
	return time


