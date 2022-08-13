
import numpy as np 
import pandas as pd 
from mypythonlib.common_functions import ChkFile, ChkDir
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import h3

import os,sys
sys.path.append('../functions')


from My_hyper_parameters import * 
from My_functions import * 

def run():
	BatNames = ['Adva', 'Miles', 'Raja', 'Michi', 'Holy', 'Tzuzik', 'Prince_Edward', 
				'Aria', 'Dilla', 'Jafar', 'Malik','Peter_Pen','Avigur', 'Bane', 'Buffy', 
				'Haim_Shelanu', 'Luna', 'Malagasi','Matcha','Moana','Oskar_Tal', 'Pizza', 
				'Pliocene', 'Rasmi', 'Shlomzion', 'Yamit', 'Yumi',
				'Ali','Balaz','Camila','Koral', 'MitMit','Ozvelian','Rosie','Shavit',
				'Anka', 'Eli', 'Eva', 'Fima', 'K', 'Mazi','Nadav','Nature','Gana',
				'Nazir', 'Odelia', 'Shem_Tov', 'Tishray', 'Tzedi', 'V']

	batnames = []
	dates = []
	tree_lists = []
	timestamp_lists = []

	for batname in tqdm(BatNames):
		bat_dir = '../Step2_tree_infering/dataset/' + batname +'/'
		fruit_bat_file = bat_dir + batname + '_data_s'+str(eps_in_meters)+'_grouped_threshold-' + str(threshold_acc)+ '_e.csv'
		ChkFile(fruit_bat_file)

		# print(fruit_bat_file)
		df = pd.read_csv(fruit_bat_file)
		Dates = []
		for i in df['All_time']:
			date, mytimes = i.split(' ')
			Dates.append(date)

		df['Date'] = Dates
		night_list = all_night_list(min(Dates), max(Dates))

		for night in night_list:
			start = str(night + ' ' + sunset_time)
			start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
			end = start+timedelta(hours=day_delta)

			df_sub = extract_night_df(df, start, end)

			batnames.append(batname)
			dates.append(night)

			if len(df_sub) > 3:
				tmp = df_sub[['Grouped_tree_id','Timestamp']]
				tmp = tmp.dropna()

				tmp_ids = []
				tmp_timestamps = []
				for tmp_id, tmp_timestamp in zip(tmp['Grouped_tree_id'], tmp['Timestamp']):
					tmp_ids.append(str(int(tmp_id)))
					tmp_timestamps.append(str(int(tmp_timestamp)))

				if len(tmp_ids) == 0:
					# bat go out in despite of no tree revisited
					tree_lists.append(-1)
					timestamp_lists.append(-1)
				else:
					tree_lists.append(','.join(tmp_ids))
					timestamp_lists.append(','.join(tmp_timestamps))
			else:
				# bat didn't go out, no GPS point
				tree_lists.append(None)
				timestamp_lists.append(None)


	res_file = './Step4_1_bat_visit_trees.csv'
	df_new = pd.DataFrame()
	df_new['batname'] = batnames
	df_new['date'] = dates
	df_new['tree_ids'] = tree_lists
	df_new['Timestamps'] = timestamp_lists

	df_new.to_csv(res_file, index=None)

		
if __name__ == "__main__":
	run()


