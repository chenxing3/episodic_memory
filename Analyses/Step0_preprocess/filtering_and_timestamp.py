

import os
import sys
sys.path.append('../functions')

from My_hyper_parameters import *
from My_functions import * 



def cleanup_dataframe(df):
	acc = df.fillna("", inplace=False)
	# print(df)
	acc = acc.loc[acc.Latitude <= 90.0]
	acc = acc.loc[acc.Latitude >= -90.0]

	acc = acc.loc[acc.Longitude <= 180.0]
	acc = acc.loc[acc.Longitude >= -180.0]
	return acc


def run():

	BatNames = ['Adva', 'Miles', 'Raja', 'Michi', 'Holy', 'Tzuzik', 'Prince_Edward', 
				'Aria', 'Dilla', 'Jafar', 'Malik','Peter_Pen','Avigur', 'Bane', 'Buffy', 
				'Haim_Shelanu', 'Luna', 'Malagasi','Matcha','Moana','Oskar_Tal', 'Pizza', 
				'Pliocene', 'Rasmi', 'Shlomzion', 'Yamit', 'Yumi',
				'Ali','Balaz','Camila','Koral', 'MitMit','Ozvelian','Rosie','Shavit',
				'Anka', 'Eli', 'Eva', 'Fima', 'K', 'Mazi','Nadav','Nature', 'Gana',
				'Nazir', 'Odelia', 'Shem_Tov', 'Tishray', 'Tzedi', 'V']

	# 
	database_dir = '../../Dataset/'

	for batname in BatNames:
		dataset_file = database_dir + batname +'.csv'

		# read file and set timestamp for each GPS points
		if os.path.isfile(dataset_file):
			df = pd.read_csv(dataset_file)
			print(dataset_file)

			tmp_dir = './dataset/'
			bat_dir = tmp_dir + batname

			ChkDir(tmp_dir) # create dataset folder
			ChkDir(bat_dir) # create bat work folder

			time_interval = search_time_interval(df) # to get time sampling interval of 15s or 30s
			# print(time_interval)
			if not isNaN(time_interval):
				clean_data = cleanup_dataframe(df) # clean the df
				df_filtered = cleanup_GPS(clean_data, time_interval, sunset_time, dusk_time) # clean the files

				data_file = bat_dir + '/' + batname + '_timestamp.csv'
				df_filtered.to_csv(data_file, index=None)

	
if __name__ == "__main__":
	run()




