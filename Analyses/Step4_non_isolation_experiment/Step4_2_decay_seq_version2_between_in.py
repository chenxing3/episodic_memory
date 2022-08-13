
import math
import sys
import numpy as np 
import pandas as pd 
from mypythonlib.common_functions import ChkFile, ChkDir
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import h3
import matplotlib.pyplot as plt
from tqdm import tqdm 

from collections import Counter
import operator


def extracting(species_file, id_file, type):
	species_data = pd.read_excel(species_file)
	id_data = pd.read_excel(id_file)

	speciess = species_data.loc[species_data["foodType"]==type, ['species']].values

	speceiss = [i[0] for i in speciess]
	# print('speciess: ', speciess)

	result = []
	for i in range(len(id_data)):
		if id_data['species'][i] in speciess:
			my_output = int(id_data['id'][i])
			result.append(my_output)
	return result


def be_in(seq):
	output = [seq[0]]
	ref = seq[0]
	for i in range(1, len(seq)):
		if ref != seq[i]:
			output.append(seq[i])
			ref = seq[i]

	# print(output)
	return output

def isNaN(string):
	return string != string


def converting(id, records, threshold):

	results = []
	interval_count = 0
	tree_seq = []
	tree_nan = []
	flag = 0
	for index in records.index:
		trees = str(records['tree_ids'][index])
		# print("!!!!: ", trees)
		# sys.exit(1)
		if not isNaN(trees):
			trees2 = trees.split(',')
			trees2 = be_in(trees2)
			# print(id, trees2)
			# sys.exit(1)
			if str(int(id)) in trees2: # check if traget id in the list of a specific day
					number_tmp = trees2.count(str(int(id))) # the number of the visit
				# if number_tmp >= 2:
					tree_seq.append(int(id))
					results.append([int(id), interval_count, number_tmp, records['date'][index]])
		interval_count += 1

	return results

def spliting(seqs, segment_day):

	alls = []
	acc = 0
	tmps = [[seqs[0][1], seqs[0][2]]]

	for i in range(1, len(seqs)):
		# print(seqs)
		my_tree_id = seqs[i][0]
		my_interval = seqs[i][1] - seqs[i-1][1]
		my_visit_times = seqs[i][2]

		if my_interval > segment_day:
			alls.append(tmps)
			tmps  = [[seqs[i][1], my_visit_times]]
			# tmps.append()
		else:
			tmps.append([seqs[i][1], my_visit_times])

	results = []
	for records in alls:
		if len(records) >= 2:
			results.append(records)
	
	return results

def converting_to_rates(seqs, night_bins, segment_day):
	
	full_arrays = []
	for seq in seqs:
		tmps = []
		suppose_night_interval = 1
		for j in range(1, len(seq)):

			current_night_interval = seq[j][0] - seq[0][0]
			if current_night_interval == suppose_night_interval:
				tmps.append(seq[j][1]/seq[j-1][1])
				suppose_night_interval += 1
			else:
				add_length = [0]*(current_night_interval-suppose_night_interval)
				add_length.append(seq[j][1]/seq[j-1][1])
				suppose_night_interval += 1 + current_night_interval-suppose_night_interval
				tmps.extend(add_length)

		## align
		if len(tmps) < segment_day:
			add_length = [0]*(segment_day-len(tmps))
			tmps.extend(add_length)
		else:
			tmps = tmps[:segment_day]

		full_arrays.append(tmps)

	# print(full_arrays)
	### convert to rate 
	rates_arrays = []
	if len(full_arrays) > 0:
		for records in full_arrays:
			# print('records: ', len(records), records)
			# for each bins 
			my_array = []
			for bins in night_bins:
				my_tmp =[]
				for bin_num in bins:
					if records[bin_num] > 0:
						my_tmp.append(records[bin_num])
				if len(my_tmp) > 0:
					my_array.append(np.mean(my_tmp))
				else:
					my_array.append(0)
			if sum(my_array) > 0:
				rates_arrays.append(my_array)
				# print("rates_arrays: ", rates_arrays)
	return rates_arrays


def run():
	fruit = 2
	nectar = 4
	Imp = (32.111146,34.807771)
	threshold = 30

	tree_file = "../ops/Trees_with_food_alldata.xlsx"
	id_file = '../ops/TreesAllYears_final_slite.xlsx'

	fruit_tree_ids = extracting(tree_file, id_file, fruit)
	nectar_tree_ids = extracting(tree_file, id_file, nectar)

	print(fruit_tree_ids)

	data_file = './Step4_1_bat_visit_trees.csv'
	df = pd.read_csv(data_file)
	groups = df.groupby('batname')

	segment_day = 31
	night_bins = [[0,1],[2,3,4],[5,6,7],[8,9,10,11], [12,13,14,15],[16,17,18,19],[20,21,22,23,24],[25,26,27,28,29]]
	night_bins = [[0,1],[2,3,4],[5,6,7],[8,9],[10],[11],[12], [13],[14], [15],[16],[17],[18],[19,20],[21,22],[23,24,25],[26,27,28],[29,30]]

	df_fruit = pd.DataFrame()
	
	res_f_file = './Step4_2_fruit2.csv'

	for MyID in fruit_tree_ids:
		print(MyID)
		flag = 0
		total_seq_n = 0
		interval_days = []
		rates = []
		
		
		rates = []
		for batname, records in sorted(groups):
			# print(MyID, batname)
			# MyID = 2000654
			all_seqs = converting(MyID, records, threshold)

			if len(all_seqs)  > 2:
				segment_seqs = spliting(all_seqs, segment_day)
				if len(segment_seqs) > 0:
					# print("segment_seqs: ", segment_seqs)

					eign_rates = converting_to_rates(segment_seqs, night_bins, segment_day)

					if len(eign_rates) > 0:
						rates.extend(eign_rates)
		
		
		if len(rates) > 0:
			rates = np.array(rates)

			df_tmp2 = pd.DataFrame()
			df_tmp2['tree_id'] = [str(MyID)]

			for num in range(len(rates[0])):
				aaa = np.mean(rates[:,num])
				df_tmp2['night_bin_%s' % num] = [aaa]

			df_fruit = df_fruit.append(df_tmp2, ignore_index=True)
			print('df_fruit: ', df_fruit)
			# print(df_tmp)
		# print(df_tmp)
	
	print(df_fruit)
	df_fruit.to_csv(res_f_file, index=None)

	sys.exit(1)



if __name__ == "__main__":
	run()





