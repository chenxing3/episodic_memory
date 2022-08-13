import math
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 12})
from scipy.stats import mannwhitneyu
plt.rcParams["font.family"] = "Arial"


def exponential_fit(x, a, b, c):
    return a*np.log(b*x) + c

def checking_exp(samples, ref):
	high_exp_group = []
	low_exp_group = []

	for _, (bat, day, exp) in enumerate(samples):
		for _, (day_ref, exp_average) in enumerate(ref):
			
			if day == day_ref:
				# print('day_ref, exp_average: ', exp, exp_average)
				if exp >= exp_average:#-0.005*exp_average: #for one
				# if exp >= exp_average + 0.09*exp_average: #for four
					high_exp_group.append(bat)
				else:
					low_exp_group.append(bat)
				break
	print(high_exp_group, low_exp_group)
	return high_exp_group, low_exp_group


def plotting(array, one_trial, four_trial):

	# error bar line

	means = []
	ups = []
	downs = []
	Xs = []

	for _, (i, distances, times, tree_number, target_numbers) in enumerate(array):
		experiences = []
		for distance, time, tree_number, target_number in zip(distances, times,tree_number, target_numbers):
			tmp = distance*time
			experiences.append(tmp)

		alpha = 0.95
		p = ((1.0-alpha)/2.0) * 100
		up_tmp =  np.percentile(experiences, p)
		p = (alpha+((1.0-alpha)/2.0)) * 100
		down_tmp =  np.percentile(experiences, p)

		means.append(np.median(experiences))
		ups.append(up_tmp)
		downs.append(down_tmp)
		Xs.append(i)

	means = np.array(means)
	ups = np.array(ups)
	downs = np.array(downs)
	Xs = np.array(Xs)

	Xs_adjusted = [1]*len(Xs)

	y_means = np.log(means/Xs_adjusted)
	y_ups = np.log(ups/Xs_adjusted)
	y_downs = np.log(downs/Xs_adjusted)

	means_fitting_parameters, _ = curve_fit(exponential_fit, Xs, y_means)
	ups_fitting_parameters, _ = curve_fit(exponential_fit, Xs, y_ups)
	downs_fitting_parameters, _ = curve_fit(exponential_fit, Xs, y_downs)

	x_fit = np.linspace(min(Xs), max(Xs), 100)
	plt.scatter(Xs, y_means, marker='.', label='Average value', c= 'black')
	plt.plot(x_fit, exponential_fit(x_fit, *means_fitting_parameters), c= 'black',label='Fit curve')
	plt.fill_between(x_fit, exponential_fit(x_fit, *ups_fitting_parameters), exponential_fit(x_fit, *downs_fitting_parameters), alpha=0.5, facecolor='gray',linestyle='--')
	plt.ylabel('Estimated experience', fontweight='bold')
	plt.xlabel('Time [night]', fontweight='bold')


	a, b, c = means_fitting_parameters
	residuals = np.array(means) - y_means
	ss_res = np.sum(residuals**2)
	ss_tot = np.sum((means-np.mean(means))**2)
	r_squared = ss_res / ss_tot
	print('r_squared: ', r_squared, a, b, c)


	## a new method
	results = {}
	coeffs = np.polyfit(Xs, y_means, 1)

	 # Polynomial Coefficients
	results['polynomial'] = coeffs.tolist()

	# r-squared
	p = np.poly1d(coeffs)
	# fit values, and mean
	yhat = p(Xs)                         # or [p(z) for z in x]
	ybar = np.sum(y_means)/len(y_means)          # or sum(y)/len(y)
	ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
	sstot = np.sum((y_means - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
	results['determination'] = ssreg / sstot
	print(results)

	# sys.exit(1)
	exp_one = np.array([float(i)+1 for i in one_trial[:,2]])
	day_one = np.array([int(i) for i in one_trial[:,1]])
	y_one = np.log(exp_one/1)

	plt.scatter(day_one, y_one, marker='^', c= 'red', label='One-night trial')

	exp_four = np.array([float(i)+1 for i in four_trial[:,2]])
	day_four = np.array([int(i) for i in four_trial[:,1]])
	y_four = np.log(exp_four/1)

	plt.scatter(day_four, y_four, marker='x', c='blue', label='4/7-night trial')
	plt.plot([0,72], [15, 15], c='orange',  linestyle='--', label='threshold')
	plt.legend()

	plt.show()
	print()


def extracting_exp_information(trials, df):
	result = []
	for trial in trials:
		print(trial)
		MyBat = trial[0]
		Time = trial[1]
		flag = 0
		for i in range(len(df)):

			for columnname in df.columns:
				if MyBat in columnname:
					if int(df['Date_'+MyBat][i].replace('-', '')) >= int(Time):
						flag = 1
						break
			if flag == 1:
				distance = df['Distance_'+MyBat][i]
				Time = df['Time_'+MyBat][i]
				Tree_num = df['Tree_num_'+MyBat][i]
				Traget_num = df['Target_num_'+MyBat][i]

				exp = distance*Time
				# exp = Time*Time
				result.append([MyBat, int(i+1), float(exp)])
				flag = 0
				break

	return np.array(result)



def run():
	file = './Step3_summary_50_e.csv'
	df = pd.read_csv(file)

	experiments_one = [['Oskar_Tal', '20200329', 'one_day'],['Bane', '20200221', 'one_day'], 
						['Rasmi', '20200330', 'one_day'],['Matcha', '20200407', 'one_day'],
						['Haim_Shelanu', '20200426', 'one_day'],['Yumi', '20200301', 'one_day'], 
						['Pizza', '20200222', 'one_day'],['Avigur','20200209', 'one_day'],
						['Tzuzik','20210224', 'one_day'],['Michi', '20210402', 'one_day'],
						['Adva', '20210315', 'one_day'], ['Miles', '20210315', 'one_day'],
						['Dilla', '20210502', 'one_day']
						]

	experiments_four = [['Oskar_Tal', '20200211', 'four_day'], ['Bane', '20200323', 'four_day'],
						['Rasmi', '20200221', 'four_day'], ['Matcha', '20200222', 'four_day'],
						['Yumi', '20200429', 'four_day'], ['Pizza', '20200401', 'four_day'],
						['Yamit', '20200501', 'four_day'],['Michi', '20210310', 'four_day'], 
						['Prince_Edward', '20210216', 'four_day'],['Adva', '20210331', 'four_day'],
						['Miles', '20210215', 'four_day'],['Holy', '20210330','four_day'], 
						['Raja', '20210211', 'four_day'],
						['Dilla', '20210518', 'four_day'],
						['Aria', '20210506', 'four_day'],
						['Jafar', '20210506', 'four_day']

						]

	Experience_ref = []
	for i in range(len(df)):
		distances = []
		times = []
		tree_numbers = []
		target_numbers = []

		for columnname in df.columns:
			if 'Distance' in columnname:
				if not np.isnan(df[columnname][i]):
					distances.append(df[columnname][i])
			elif 'Time' in columnname:
				if not np.isnan(df[columnname][i]):
					times.append(df[columnname][i])
			elif 'Tree_num' in columnname:
				if not np.isnan(df[columnname][i]):
					tree_numbers.append(df[columnname][i])
			elif 'Target_num' in columnname:
				if not np.isnan(df[columnname][i]):
					target_numbers.append(df[columnname][i])

		exepriment_tmp = np.mean(distances)*np.mean(times)#*np.mean(target_numbers)
		if not np.isnan(exepriment_tmp) and i <= 81:
			Experience_ref.append([i+1, distances, times, tree_numbers, target_numbers])


	one_trial = extracting_exp_information(experiments_one, df)
	four_trial = extracting_exp_information(experiments_four, df)

	plotting(Experience_ref, one_trial, four_trial)
	plt.show()


	




if __name__ == "__main__":
	run()

