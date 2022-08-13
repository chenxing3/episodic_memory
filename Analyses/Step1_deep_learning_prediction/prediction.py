
import os
import sys
sys.path.append('../functions')

from My_hyper_parameters import * # import all the parameters
from My_functions import * # import all the functions

import pandas as pd
from datetime import datetime, timedelta
from predictor import Predictor

import warnings
warnings.filterwarnings('ignore')


def extracting_distance(pred, eign_file):
	'''
	extract specific information for each segments 
	'''

	count = 0
	for record in open(eign_file, 'r'):
		record = record.strip()

		if count == 0:
			pass
			count += 1
		elif count == 1:
			pred_plus = [i for i in record.split(',')] # label
			count += 1
		elif count == 2:
			Timestamp = [i for i in record.split(',')] # timestamp information
			count += 1
		elif count == 3:
			lng_tmp= [i for i in record.split(',')] # longitude
			count += 1
		elif count == 4:
			lat_tmp = [i for i in record.split(',')] # latitude
			count += 1

	result = []
	for tmp1, tmp2, tmp3, tmp4 in zip(pred_plus, Timestamp, lng_tmp, lat_tmp):
		result.append([float(tmp1)*pred, int(float(tmp2)), float(tmp3), float(tmp4)])
	return result



def check_points(preds, files):
	'''
	format the results
	'''
	if len(preds) == len(files):
		pools = []
		for i, (pred, eign_file) in enumerate(zip(preds, files)):
			pools.extend(extracting_distance(pred, eign_file)) # to get all the information in each segment

		pools = np.array(pools)
		IDs = list(set(pools[:,1]))

		My_IDs = []
		My_preds = []
		My_lngs = []
		My_lats = []
		for ID in sorted(IDs):
			tmp = []
			for record in pools:
				if ID == record[1]:
					tmp.append(record[0])
					tmp_lng = record[2]
					tmp_lat = record[3]
			My_IDs.append(int(ID))
			My_preds.append(np.mean(tmp))
			My_lngs.append(tmp_lng)
			My_lats.append(tmp_lat)

		new_df = pd.DataFrame()
		new_df['Timestamp'] = My_IDs
		new_df['Preds'] = My_preds
		new_df['Ad_lng'] = My_lngs
		new_df['Ad_lat'] = My_lats
		return new_df


def classify_timestamp(df, work_dir, batname):
	Dates = []
	for i in df['All_time']:
		date, mytimes = i.split(' ')
		Dates.append(date)

	df['Date'] = Dates
	night_list = all_night_list(min(Dates), max(Dates)) # to get all the nights 
	# print(night_list)
	# sys.exit()
	results = []

	for night in night_list:
		start = str(night + ' ' + sunset_time)  # to get start time for each night about 14:00 (winter) or 15:00 (summer) everyday
		# print('start: ', batname, start)
		start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
		end = start+timedelta(hours=day_delta) # to get end time (5:00 or 6:00AM)
		
		df_sub = extract_night_df(df, start, end) # to get data for each night 
		# sys.exit()

		if len(df_sub) > 10: # the row number of each file should > 10, otherwise it cannot be used to predict
			# add rad
			df_sub = create_radian_columns(df_sub) # convert coordinate to radian for distance calculation

			df_filled = filling_blank(df_sub) # filling missing data
			results.append(df_filled)
			# print(df_filled)
			# sys.exit()

	# eignlization
	work_tf_tmp = work_dir+'/tmp' 
	ChkDir(work_tf_tmp) # create work folder 
	eign_files = Converting(results, work_tf_tmp) # eignlization of the file 

	tf_file = work_dir + '/' + 'test.tf'
	making_df(eign_files, tf_file) # make tf for deep learning, only tf format file can be used here

	# loading model 
	predictor = Predictor(tf_file) #!! prediciton
	result = predictor.main()

	# to get result of dataframe
	df_preds = check_points(np.array(result), eign_files) 

	output_file = work_dir + '/' + batname + '_final_10_e.csv'
	df_final = pd.merge(df, df_preds, on='Timestamp', how='left')
	df_final.to_csv(output_file, index=None)
	print("Predict done!!\n")
	# sys.exit()

def run():

	BatNames = ['Adva', 'Miles', 'Raja', 'Michi', 'Holy', 'Tzuzik', 'Prince_Edward', 
				'Aria', 'Dilla', 'Jafar', 'Malik','Peter_Pen','Avigur', 'Bane', 'Buffy', 
				'Haim_Shelanu', 'Luna', 'Malagasi','Matcha','Moana','Oskar_Tal', 'Pizza', 
				'Pliocene', 'Rasmi', 'Shlomzion', 'Yamit', 'Yumi',
				'Ali','Balaz','Camila','Koral', 'MitMit','Ozvelian','Rosie','Shavit',
				'Anka', 'Eli', 'Eva', 'Fima', 'K', 'Mazi','Nadav','Nature', 'Gana',
				'Nazir', 'Odelia', 'Shem_Tov', 'Tishray', 'Tzedi', 'V']

				
	for batname in BatNames:
		print("-----> ", batname, " starting to predict...")
		# read files
		tmp_dir = '../Step0_preprocess/dataset/'
		data_file = tmp_dir + batname + '/' + batname + '_timestamp.csv'
		df = pd.read_csv(data_file)

		# find out all the missing data and prediction by input each segemnt
		work_tf_tmp = tmp_dir + batname
		df = classify_timestamp(df, work_tf_tmp, batname)
	
if __name__ == "__main__":
	run()




