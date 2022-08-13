
import numpy as np 
import pandas as pd 
from datetime import datetime, time, timedelta
import re, os, sys
import math
from mypythonlib.common_functions import ChkFile, ChkDir
import operator
import h3


import os,sys
sys.path.append('../functions')

from My_hyper_parameters import * 
from My_functions import * 



def isNaN(string):
	return string != string
	
def extracting(species_file, id_file, type):
	species_data = pd.read_excel(species_file)
	id_data = pd.read_excel(id_file)

	speciess = species_data.loc[species_data["foodType"]==type, ['species']].values

	speceiss = [i[0] for i in speciess]
	# print('speciess: ', speciess)

	result = []
	for i in range(len(id_data)):
		if id_data['species'][i] in speciess:
			my_output = [int(id_data['id'][i]), id_data['species'][i]]
			result.append(my_output)
	return result


def checking(df, start, end):
	df['All_time'] = pd.to_datetime(df['All_time'])
	df2 = df[(df['All_time'] > start) & (df['All_time'] <= end)]

	return df2



def to_select_tree(df,):
	selected_pool = []
	for i in df['Grouped_tree_id'].index:
		if not isNaN(df['Grouped_tree_id'][i]):
			selected_pool.append(int(df['Grouped_tree_id'][i]))
		else:
			selected_pool.append(math.nan)

	return selected_pool


def extract_data(df, keep_start, keep_end, experimental_type, day_delta_isolation=2):
	# print("day_delta: ", day_delta_isolation)
	# 第一步，指定可能性最大的三棵树
	selected_trees = to_select_tree(df)
	df['selected_trees'] = selected_trees

	keep_start = datetime.strptime(keep_start, '%Y%m%d %H:%M:%S')
	keep_end = datetime.strptime(keep_end, '%Y%m%d %H:%M:%S')

	# 根据时间，保留有效区
	before_df = checking(df, keep_start-timedelta(hours=24*2), keep_start)
	# print("before_trees: ", experimental_type, before_df)
	after_df = checking(df, keep_end, keep_end+timedelta(hours=24*day_delta_isolation))
	# print("after_trees: ", experimental_type, after_df)
	return before_df, after_df


def repeat_exclude(df):
	df = df.dropna()
	output = list(set(df))
	return [str(int(i)) for i in output]


def model(array, threshold=3):
	value = 1

	all_ids = []
	for index, (id_tmp, distance) in enumerate(array):
		if id_tmp not in all_ids:
			all_ids.append(id_tmp)

	# print("all_ids: ", all_ids)
	tmp_pool = []
	for id in all_ids:
		value = 1
		count = 0
		for index, (id_tmp, distance) in enumerate(array):
			if id == id_tmp:
				if distance == 0:
					distance = 0.1
				value = value*math.exp(1/distance)
				count += 1
		tmp_pool.append([id, value*count])

	tmp_pool = sorted(tmp_pool, key=operator.itemgetter(1), reverse=True)
	# print("tmp_pool: ", tmp_pool)
	output = []
	for i in tmp_pool[:threshold]:
		output.append(i[0])
	# print("best trees:", output)
	return output

def be_in(seq):

	output = [seq[0]]
	ref = seq[0]
	print(seq)

	for i in range(1, len(seq)-1):
		if ref != seq[i]:
			# if seq[i] == seq[i+1] or seq[i] == seq[max(0, i-1)]:
			# if len(output) > 0:
			if seq[i] != output[-1]:
				output.append(seq[i])
			# else:
			# 	output.append(seq[i])
			ref = seq[i]

	# print(output)
	# sys.exit(1)
	return output

def counting(df, tree_ids, type):
	# print("df:", df)
	# sys.exit(1)

	tree_df = df['Grouped_tree_id']
	tree_df = tree_df.dropna()
	tree_pool = be_in([int(i) for i in tree_df])
	print(tree_pool)
	

	ids = []
	speciess = []
	counts = []
	types = []


	for index, (id, species) in enumerate(tree_ids):

		if id in tree_pool:
			counts.append(tree_pool.count(id))
			types.append(type)
			ids.append(id)
			speciess.append(species)

		# 首先把他进行分组


	# 			tree_df = records['Tree_id']
	# 			tree_df = tree_df.dropna()
	# 			tree_pool = [int(i) for i in list(tree_df)]

	# 			print("tree_df: ", tree_df)
	# 			sys.exit(1)

	# 			for tree_tmp in tree_pool:
	# 				# print("!!!!checking: ", tree_tmp, id)
	# 				if id == tree_tmp:
	# 					if df.loc[df['Tree_id']==id, ['Tree_distance']].values[0] <= 20:
	# 						# print("I am here!!")
	# 						count += 1
	# 						be_in_seqs
	# 						# break

	# 		else:
	# 			update_pool = []

	# 			for tmp_id, tmp_distance in zip(records['Tree_id'].dropna(), records['Tree_distance'].dropna()):
	# 				update_pool.append([int(tmp_id), float(tmp_distance)])
	# 			possible_trees = model(update_pool, threshold=1)
				
	# 			if id in possible_trees:
	# 				# print("I am here!!!")
	# 				count += 1

	# 			# tree_df = records['Tree_id']
	# 			# tree_df = tree_df.dropna()
	# 			# tree_pool = [int(i) for i in list(tree_df)]
	# 			# for tree_tmp in tree_pool:
	# 			# 	# print("!!!!checking: ", tree_tmp, id)
	# 			# 	if id == tree_tmp:
	# 			# 		print("I am here!!")
	# 			# 		count += 1
	# 			# 		# break

	# 	# print(id, species, count, type)
	# 	if count > 0:
	# 		ids.append(id)
	# 		speciess.append(species)
	# 		counts.append(count)
	# 		types.append(type)

	res_df = pd.DataFrame()
	res_df['species'] = speciess
	res_df['tree_id'] = ids
	res_df['count'] = counts
	res_df['experimental_type'] = types
	print(res_df)
	# sys.exit(1)

	return res_df


def check_tree_sets(df_before, df_after, trees, tree_type):
	before_tree_ids = repeat_exclude(df_before['selected_trees'])
	after_tree_ids = repeat_exclude(df_after['selected_trees'])

	# print(before_trees)

	ref_tree_ids = [str(tree_id) for index, (tree_id, tree_species) in enumerate(trees)]

	shared_trees_tmp = (set(before_tree_ids) & set(after_tree_ids)) & set(ref_tree_ids)

	before_trees = []
	after_trees = []
	shared_trees = []
	for index, (tree_id, tree_species) in enumerate(trees):
		if str(tree_id) in shared_trees_tmp:
			shared_trees.append([tree_id, tree_species])

		if str(tree_id) in before_tree_ids:
			before_trees.append([tree_id, tree_species])

		if str(tree_id) in after_tree_ids:
			after_trees.append([tree_id, tree_species])


	# print("shared_trees: ", shared_trees)

	if len(shared_trees) > 0:
		df_before_shared = counting(df_before, before_trees, tree_type + "-before-shared")
		df_after_shared = counting(df_after, after_trees, tree_type + "-after-shared")
		df_merged= pd.merge(df_before_shared, df_after_shared, on='tree_id', how='left')
	else:
		df_merged = None


	return df_merged

def filter(Imp, nectar_tree_ids, file, threshold):
	df = pd.read_excel(file)
	# print(df)
	res = []
	for i in nectar_tree_ids:
		# print(df['id'], i[0])
		gpss = df.loc[df['id']==i[0], ['lat', 'lon']]

		# print(gpss)

		# for i in aa:
		# 	print(i)
		for gps in gpss.values:
			# print(gps)
			gps = set(gps)
			dis = h3.point_dist(Imp, gps, unit="m")
			# print(i, dis)
			if dis >= threshold:
				if i not in res:
					res.append(i)

	return res

def run():

	fruit_tree_ids = extracting(tree_file, id_file, fruit)
	nectar_tree_ids = extracting(tree_file, id_file, nectar)

	# eps_in_meters = 25

	final_df = pd.DataFrame()
	result_file = './sta_2——2_grouped_be_in_'+str(eps_in_meters) + '_' + str(threshold_acc)+'_e.csv'


	for record in experiments:
		print("record: ", record)
		batname = record[0]
		keep_start = record[1]
		keep_end = record[2]
		experimental_type = record[3]

		fruit_bat_file = './dataset/' + batname + '/' + batname + '_data_s'+str(eps_in_meters)+'_grouped_threshold-' + str(threshold_acc)+'_e.csv'

		ChkFile(fruit_bat_file)

		print(fruit_bat_file)
		# 目的是看蝙蝠去果树和蜜树的次数，对于不同的类型的植物，然后范围是参数，
		
		fruit_bats = pd.read_csv(fruit_bat_file)
		before_df, after_df = extract_data(fruit_bats, keep_start, keep_end, experimental_type, day_delta_isolation)
		# print(before_df)
		# sys.exit(1)
		df_fruits = check_tree_sets(before_df, after_df, fruit_tree_ids, 'fruit-'+experimental_type)
		df_nectar = check_tree_sets(before_df, after_df, nectar_tree_ids, 'nectar-'+experimental_type)

		final_df_tmp = pd.DataFrame()
		final_df_tmp = final_df_tmp.append(df_fruits, ignore_index = True)
		final_df_tmp = final_df_tmp.append(df_nectar, ignore_index = True)

		# print("len(final_df_tmp): ", len(final_df_tmp))

		
		my_bats = []
		for i in range(len(final_df_tmp)):
			my_bats.append(batname)

		final_df_tmp['bat'] = my_bats




		final_df = final_df.append(final_df_tmp, ignore_index = True)

	final_df.to_csv(result_file)


if __name__ == "__main__":
	run()
