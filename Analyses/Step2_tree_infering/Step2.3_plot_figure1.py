import sys
import pandas as pd 
import numpy as np 
from scipy.stats import ttest_rel
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from scipy.stats import mannwhitneyu

def sta(records):
	Num = 1
	# Num = sum(records['count_time_y']/(sum(records['count_time_x'])+sum(records['count_time_y'])))
	Rate = sum(records['count_y'])/sum(records['count_x'])# + sum(records['count_y']))
	# Rate = sum(records['count_y'])/sum(records['count_x'])# + sum(records['count_y']))

	Nums = []
	Rates = []
	for i in records.index:
		# Nums.append(records['count_time_y'][i]/(records['count_time_x'][i]+records['count_time_y'][i]))
		Rates.append(records['count_y'][i]/records['count_x'][i])# + records['count_y'][i]))
		# Rates.append(records['count_y'][i]/records['count_x'][i])# + records['count_y'][i]))

	# print(Nums, Rates)
	# sys.exit(1)
	return Num, Rate, Nums, Rates



def run():
	file = "./sta_2——2_grouped_be_in_20_10_e.csv"
	df = pd.read_csv(file)
	groups = df.groupby('experimental_type_y')

	for exp_type, records in groups:
		# print(exp_type)
		# print("I am here!!")
		if exp_type == 'fruit-four_day-after-shared':
			# print("I am here!!")
			Num_fruit_4, Rate_fruit_4, Num_fruit_4s, Rate_fruit_4s = sta(records)

		elif exp_type == 'fruit-one_day-after-shared':
			# print("I am here!!")
			Num_fruit_1, Rate_fruit_1, Num_fruit_1s, Rate_fruit_1s = sta(records)

		elif exp_type == 'nectar-four_day-after-shared':
			# print("I am here!!")
			Num_nectar_4, Rate_nectar_4, Num_nectar_4s, Rate_nectar_4s = sta(records)

		
		elif exp_type == 'nectar-one_day-after-shared':
			# print("I am here!!")
			Num_nectar_1, Rate_nectar_1, Num_nectar_1s, Rate_nectar_1s = sta(records)


	print("Rate_nectar_1s,Rate_nectar_4s: ", Rate_nectar_1, Rate_nectar_4)
	P_value_nectar_rate = stats.ttest_ind(Rate_nectar_1s,Rate_nectar_4s, equal_var = False, alternative='greater')
	print("P_value_nectar_rate: ", P_value_nectar_rate)
	# P_value_nectar_num = stats.ttest_ind(Num_nectar_1s,Num_nectar_4s, equal_var = False, alternative='greater')

	P_value_fruit_rate = stats.ttest_ind(Rate_fruit_1s,Rate_fruit_4s, equal_var = False, alternative='greater')
	# P_value_fruit_num = stats.ttest_ind(Num_fruit_1s,Num_fruit_4s, equal_var = False, alternative='greater')


	
	# Rate_nectar_4s_c = [i for i in Rate_nectar_4s if i > 0]
	# Rate_nectar_1s_c = [i for i in Rate_nectar_1s if i > 0]
	df_test = pd.DataFrame()
	df_test_2 = pd.DataFrame()
	df_test['1s'] = Rate_nectar_1s
	df_test_2['4s'] = Rate_nectar_4s
	df_test.replace(0, np.nan, inplace=True)
	df_test_2.replace(0, np.nan, inplace=True)
	# print(Rate_nectar_4s, Rate_nectar_4s_c)
	U, p = mannwhitneyu(df_test['1s'].dropna(), df_test_2['4s'].dropna())
	print('mannwhitneyu: ', p)
	# nectar_nums = [Num_nectar_1, Num_nectar_4]
	# fruit_nums = [Num_fruit_1, Num_fruit_4]


	nectar_rates = [Rate_nectar_1, Rate_nectar_4]
	fruit_rates = [Rate_fruit_1, Rate_fruit_4]

	x = [1, 4]

	fig, ax1 = plt.subplots(1, 1)
	# ax1.plot(x, nectar_rates, c='orange', linewidth=2)
	# ax1.plot(x, fruit_rates, c='gray', linewidth=2)


	std_nectar_1 = np.std(Rate_nectar_1s)/np.sqrt(len(Rate_nectar_1s))
	std_nectar_4 = np.std(Rate_nectar_4s)/np.sqrt(len(Rate_nectar_4s))
	error_nectar = [std_nectar_1, std_nectar_4]

	std_fruit_1 = np.std(Rate_fruit_1s)/np.sqrt(len(Rate_fruit_1s))
	std_fruit_4 = np.std(Rate_fruit_4s)/np.sqrt(len(Rate_fruit_4s))

	error_fruit = [std_fruit_1, std_fruit_4]
	print('error_fruit: ', error_fruit)

	(_, caps, _) = ax1.errorbar(x, nectar_rates, error_nectar, c='blue', ecolor='blue', capsize=3, elinewidth=1, linewidth=2)
	(_, caps, _) = ax1.errorbar(x, fruit_rates, error_fruit, c='red', ecolor='red', capsize=3, elinewidth=1, linewidth=2)
	for cap in caps:
		cap.set_markeredgewidth(1)


	ax1.set_xlim(0, 5)
	ax1.set_xticks([0,0,0,1,4])
	ax1.set_xticklabels(('', '', '', 'One day', '4-7 days'), fontweight='bold')

	ax1.set_ylabel('Mean of Revisit Proportion', fontweight='bold')
	# ax1.set_ylim(0.25,0.55)

	# ax1.legend(['Nectar, P value: '+str(round(P_value_nectar_rate.pvalue,3)), 'Fruit, P value: '+str(round(P_value_fruit_rate.pvalue,3))])
	ax1.legend(['Nectar', 'Fruit'],facecolor='white', framealpha=1,edgecolor='white')


	# ax2.plot(x, nectar_nums, c='orange', linewidth=2)
	# ax2.plot(x, fruit_nums, c='gray', linewidth=2)
	# ax2.set_xlim(0, 5)
	# ax2.set_xticks([0,0,0,1,4])
	# ax2.set_xticklabels(('', '', '', 'One day', 'Four day'))

	# ax2.set_ylabel('Re-visit Number')
	# ax2.set_ylim(2, 5.5)
	# # ax2.legend(['Nectar: '+ P_value_nectar_num, 'Fruit: '+P_value_fruit_num])


	# ax2.legend(['Nectar, P value: '+str(round(P_value_nectar_num.pvalue,3)), 'Fruit, P value: '+str(round(P_value_fruit_num.pvalue,3))])




	plt.show()






if __name__ == "__main__":
	# print("I am here!!")
	run()


