import sys
import pandas as pd 
import numpy as np 
from scipy.stats import ttest_rel
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Arial"

from scipy.stats import mannwhitneyu
import seaborn as sns

def sta(records):
	'''
	get revisiting rates
	'''
	Num = 1
	Rate = np.nanmean(records['count_y']/records['count_x'])
	Nums = []
	Rates = []
	for i in records.index:
		Rates.append(records['count_y'][i]/records['count_x'][i])

	return Num, Rate, Nums, Rates

def get_parameters(array):
	'''
	get statistic parameters
	'''
	A = plt.boxplot(array)
	tmps = [item.get_ydata() for item in A['boxes']]
	tmps2 = [item.get_ydata() for item in A['medians']]
	q1 = tmps[0][0]
	q3 = tmps[0][2]
	median = tmps2[0][0]
	plt.close()
	return q1, q3, median

def run():
	file = "./Step3_sta_2——2_grouped_be_in_low.csv"
	df = pd.read_csv(file)
	groups = df.groupby('experimental_type_y')

	for exp_type, records in groups:
		if exp_type == 'fruit-four_day-after-shared':
			Num_fruit_4, Rate_fruit_4, Num_fruit_4s, Rate_fruit_4s = sta(records)

		elif exp_type == 'fruit-one_day-after-shared':
			Num_fruit_1, Rate_fruit_1, Num_fruit_1s, Rate_fruit_1s = sta(records)

		elif exp_type == 'nectar-four_day-after-shared':
			Num_nectar_4, Rate_nectar_4, Num_nectar_4s, Rate_nectar_4s = sta(records)

		elif exp_type == 'nectar-one_day-after-shared':
			Num_nectar_1, Rate_nectar_1, Num_nectar_1s, Rate_nectar_1s = sta(records)

	# get q1, q3 andd medians
	N1_q1, N1_q3, N1_median = get_parameters(Rate_nectar_1s)
	N4_q1, N4_q3, N4_median = get_parameters(Rate_nectar_4s)
	F1_q1, F1_q3, F1_median = get_parameters(Rate_fruit_1s)
	F4_q1, F4_q3, F4_median = get_parameters(Rate_fruit_4s)
	
	print('Rate_nectar_4s: ', N1_q1, N1_q3, N1_median, N4_median)
	print('Rate_fruit_4s: ', F4_q1, F4_q3, F1_median, F4_median)

	x = np.array([1, 4]) # xlim

	fig, ax1 = plt.subplots(1, 1)

	# nectar part: plot Q1 Q3 
	std_nectar_1 = (N1_q3 - N1_q1)/2+0.02
	std_nectar_4 = (N4_q3 - N4_q1)/2
	error_nectar = [std_nectar_1, std_nectar_4]
	nectar_rates = [(N1_q3+N1_q1)/2, (N4_q3+N4_q1)/2]
	(_, caps, _) = ax1.errorbar(x, nectar_rates, error_nectar, ecolor='blue', c='blue', linestyle='', capsize=3, elinewidth=1)
	for cap in caps:
		cap.set_markeredgewidth(1)

	# nectar part: plot line between medians
	nectar_rates = [N1_median, N4_median]
	(_, caps, _) = ax1.errorbar(x, nectar_rates, [0,0], ecolor='white', c='blue', linestyle='--')

	# fruit part: plot Q1 Q3 
	std_fruit_1 = (F1_q3 - F1_q1)/2
	std_fruit_4 = (F4_q3 - F4_q1)/2


	error_fruit = [std_fruit_1, std_fruit_4]
	print('std_fruit_4: ', std_fruit_1, std_fruit_4)

	fruit_rates = [(F1_q3+F1_q1)/2, (F4_q3+F4_q1)/2]
	print('std_fruit_4: ', Rate_fruit_1, Rate_fruit_4)

	(_, caps, _) = ax1.errorbar(x+0.01, fruit_rates, error_fruit, ecolor='red', c='red', linestyle='', capsize=3, elinewidth=1)
	for cap in caps:
		cap.set_markeredgewidth(1)

	# fruit part: plot line between medians
	fruit_rates = [F1_median, F4_median]
	(_, caps, _) = ax1.errorbar(x+0.01, fruit_rates, [0,0], ecolor='white', c='red', linestyle='--')

	ax1.set_xlim(0, 5)
	ax1.set_xticks([0,0,0,1,4])
	ax1.set_xticklabels(('', '', '', 'One-night trial', '4/7-nights trial'), fontweight='bold')
	ax1.set_ylim(0, 2)

	plt.show()

if __name__ == "__main__":
	# print("I am here!!")
	run()


