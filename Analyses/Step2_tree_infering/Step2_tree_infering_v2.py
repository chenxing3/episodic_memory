

import pandas as pd 

import os
import sys
sys.path.append('../functions')

from My_hyper_parameters import * 
from My_functions import * 

def run():
	BatNames = [
				# 'Adva', 'Miles', 'Raja', 'Michi', 'Holy', 'Tzuzik', 'Prince_Edward', 
				# 'Aria', 'Dilla', 'Jafar', 'Malik','Peter_Pen','Avigur', 'Bane', 'Buffy', 
				# 'Haim_Shelanu', 'Luna', 'Malagasi','Matcha','Moana','Oskar_Tal', 'Pizza',]
				# 'Pliocene', 'Rasmi', 'Shlomzion', 'Yamit', 'Yumi','Ali','Balaz',]
				# 'Camila','Koral', 'MitMit','Ozvelian','Rosie','Shavit',]
				# 'Anka', 'Eli', 'Eva', 'Fima', 'K', 'Mazi',]
				# 'Nadav','Nature', 'Gana',]
				# 'Nazir', 'Odelia','Shem_Tov',]
				'Tishray', 'Tzedi', 'V']

	BatNames = ['Gana']
	sunset_file = '../ops/Sunset_Tel_Aviv.csv'
	sunset_df = pd.read_csv(sunset_file)

	for batname in BatNames:
		tmp_dir = '../Step0_preprocess/dataset/'
		file = tmp_dir + batname + '/' + batname + '_final_10_e.csv'
		df = pd.read_csv(file)
		# print(len(df))
		tree_pnts = fruit_trees(fruit_file)

		work_dir = './dataset/'
		ChkDir(work_dir)

		bat_work_dir = work_dir + batname + '/'
		ChkDir(bat_work_dir)

		# extract data for each night	
		Dates = []
		final_df = pd.DataFrame()
		for i in df['All_time']:
			date, mytimes = i.split(' ')
			Dates.append(date)

		df['Date'] = Dates
		night_list = all_night_list(min(Dates), max(Dates))
		# print(night_list)

		for night in night_list:

			try:
				sunset_time = sunset_df[sunset_df['night']==str(night)]['sunset'].item()
				# print(mysunset)
			except:
				print('checking the reference sunset file:', night)
				sys.exit(1)

			start = str(night + ' ' + sunset_time)
			start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
			end = start+timedelta(hours=day_delta)
			print('Night start (UTC time): ', batname, start, " to ", end)
			df_sub = extract_night_df(df, start, end)

			check_gps_points = pd.DataFrame()
			tmp1, tmp2, tmp3 = converting_array_tree_inferring(df_sub, hyper_acc)
			check_gps_points['Latitude'] = tmp1
			check_gps_points['Longitude'] = tmp2
			check_gps_points['Timestamp'] = tmp3

			# print("!!!!!", check_gps_points)
			# sys.exit()

			bat_tree_distances = pd.DataFrame()
			tmp1, tmp2, tmp3 = tree_infer(check_gps_points, tree_pnts, distance_bat_tree)
			bat_tree_distances['Timestamp'] = tmp1
			bat_tree_distances['Tree_id'] = tmp2
			bat_tree_distances['Tree_distance'] = tmp3


			final_df = final_df.append(bat_tree_distances, ignore_index = True)
			# print(bat_tree_distances)

		# print(final_df)
		# sys.exit()

		final_file = bat_work_dir + batname + '_tree_infer_' + str(distance_bat_tree) + '_10_e.csv'
		df_merged= pd.merge(df, final_df, on='Timestamp', how='left')
		df_merged.to_csv(final_file, index=None)


if __name__ == "__main__":
	run()


