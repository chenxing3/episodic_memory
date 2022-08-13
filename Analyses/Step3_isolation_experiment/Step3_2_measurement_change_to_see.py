
import math
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exponential_fit(x, a, b, c):
    return a*np.log(b*x) + c

def checking_exp(samples, ref):
	high_exp_group = []
	low_exp_group = []

	for _, (bat, day, exp) in enumerate(samples):
		for _, (day_ref, exp_average) in enumerate(ref):
			
			if day == day_ref:
				# print('day_ref, exp_average: ', exp, exp_average)
				if exp >= exp_average-0.005*exp_average: #for one
				# if exp >= exp_average + 0.09*exp_average: #for four
					high_exp_group.append(bat)
				else:
					low_exp_group.append(bat)
				break
	print(high_exp_group, low_exp_group)
	return high_exp_group, low_exp_group

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

				exp = Time*Traget_num
				# exp = Time*Time
				result.append([float(exp),MyBat, int(i+1)])
				flag = 0
				break

	return result



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
			# my_meansurement = np.log(exepriment_tmp/(i+1))
			Experience_ref.append([i+1, distances, times, tree_numbers, target_numbers])

	one_trial = extracting_exp_information(experiments_one, df)
	four_trial = extracting_exp_information(experiments_four, df)

	print(sorted(one_trial))
	print(sorted(four_trial))







if __name__ == "__main__":
	run()

