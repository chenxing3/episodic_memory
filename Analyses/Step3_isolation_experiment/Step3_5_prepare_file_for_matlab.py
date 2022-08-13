

import numpy as np 
import pandas as pd 
import h3, sys, os

sys.path.append('../functions')

from My_hyper_parameters import * 
from My_functions import * 


def add_distance(df, point):
	distances = []
	for i in range(len(df)):
		tree_point = (df['lat'][i], df['lon'][i])

		dis = h3.point_dist(tree_point, point, unit="m")
		distances.append(round(dis,5))

	df['distance'] = distances
	return df

def add_distance2(df, fruit_dict):
	distances = []
	# print(fruit_dict)
	


	for i in range(len(df)):
		MyID = int(df['tree_id'][i])
		MyDistance = fruit_dict.loc[fruit_dict['id'] == MyID, ['distance']].values[0][0]
		distances.append(MyDistance)

	df['distance'] = distances

	return df

def co_responding(records, libs):
	days = []
	months = []
	years = []
	for i in records['bat'].values:
		for j in libs:
			if i == j[0]:
				days.append(j[1])
				months.append(j[2])
				years.append(j[3])
				break
	return days, months, years

def run():
	Imp = (32.111146,34.807771)

	# combine two file together
	file1 = 'Step3_sta_2——2_grouped_be_in_high.csv'
	file2 = 'Step3_sta_2——2_grouped_be_in_low.csv'

	df1 = pd.read_csv(file1)
	df2 = pd.read_csv(file2)

	df = df1.append(df2, ignore_index=True)

	# add distance
	fruit_dict = pd.read_excel(fruit_file)
	fruit_dict = add_distance(fruit_dict, Imp)
	df = add_distance2(df, fruit_dict)

	# start to analyze
	df_all = pd.DataFrame()

	one_day_experiments = [['Oskar_Tal',1, 'Mar', 2020], ['Bane', 1, 'Feb', 2020], ['Rasmi', 1, 'Mar', 2020],['Matcha', 1, 'Apr', 2020], ['Haim_Shelanu', 1, 'Apr', 2020],
							['Yumi', 1, 'Mar', 2020], ['Pizza', 1, 'Feb', 2020], ['Avigur',1, 'Feb', 2020],['Michi', 1, 'Apr', 2021], ['Adva', 1, 'Mar', 2021], ['Miles', 1, 'Mar', 2021], ['Tzuzik', 1, 'Feb', 2021],
							['Dilla', 1, 'May', 2021]
							]

	four_day_experiments = [['Oskar_Tal', 4, 'Feb', 2020],['Bane', 4, 'Mar', 2020],['Rasmi', 4, 'Feb', 2020], ['Matcha', 4, 'Feb', 2020],['Yumi', 4, 'Apr', 2020], ['Pizza', 4, 'Apr', 2020],['Yamit', 4, 'May', 2020],
							['Raja', 7, 'Feb', 2021], ['Michi', 7, 'Mar', 2021], ['Adva', 7, 'Apr', 2021], ['Miles', 7, 'Feb', 2021],['Holy', 7, 'Apr', 2021],['Prince_Edward', 7, 'Feb', 2021], 
							['Dilla', 7, 'May', 2021], ['Aria', 7, 'May', 2021],['Jafar', 7, 'May', 2021],
							]

	groups = df.groupby('experimental_type_x')

	for exp_type, records in groups:
		tmp = pd.DataFrame()

		tmp_b = records['count_x'].values
		tmp_a = records['count_y'].values
		# tmp['count_before'] = tmp_b
		# tmp['count_after'] = tmp_a
		tmp['count_x'] = tmp_b


		Revisitrates = []
		log_Revisitrates = []
		for before_tmp, after_tmp in zip(tmp_b, tmp_a):
			revisit_tmp = after_tmp/before_tmp
			Revisitrates.append(revisit_tmp)
			if after_tmp != 0:
				log_Revisitrates.append(np.log(revisit_tmp))
			else: 
				log_Revisitrates.append(-2)
		tmp['Revisitrates'] = Revisitrates
		tmp['log_Revisitrates'] = log_Revisitrates

		tmp['Batname'] = records['bat'].values
		tmp['Distance'] = records['distance'].values
		print(tmp['Distance'])

		rates = []
		for i,j in zip(tmp_b, tmp_a):
			rates.append(round(j/(i+j), 3))	
		tmp['rate'] = rates	

		if exp_type == 'fruit-four_day-before-shared':
			tmp['Fruit'] = ['F']*len(records)
			days, months, years = co_responding(records, four_day_experiments)

		elif exp_type == 'fruit-one_day-before-shared':
			tmp['Fruit'] = ['F']*len(records)
			days, months, years = co_responding(records, one_day_experiments)

		elif exp_type == 'nectar-four_day-before-shared':
			tmp['Fruit'] = ['N']*len(records)
			days, months, years = co_responding(records, four_day_experiments)

		elif exp_type == 'nectar-one_day-before-shared':
			tmp['Fruit'] = ['N']*len(records)
			days, months, years = co_responding(records, one_day_experiments)

		tmp['Nights'] = days
		tmp['Month'] = months
		tmp['Year'] = years

		df_all = df_all.append(tmp, ignore_index=True)

	file_res = file1.replace('.csv', '_all_matlab.csv')
	df_all.to_csv(file_res, index=None)

if __name__ == "__main__":
	run()

