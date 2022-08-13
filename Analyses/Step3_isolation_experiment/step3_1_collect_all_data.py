

import os
import sys
sys.path.append('../functions')

from My_hyper_parameters import *  
from My_functions import *  

import math
import sys
import numpy as np
import pandas as pd 

# 地理信息包
import h3
import pyproj
import geopandas as gpd
from pyproj import Transformer
from datetime import datetime, time, timedelta
from shapely.geometry import Point
from shapely.ops import transform
from shapely.ops import cascaded_union

from shapely.geometry.polygon import Polygon

from mypythonlib.common_functions import ChkFile, ChkDir
from tqdm import tqdm

import folium



def buffer_in_meters(lng, lat, radius):
	proj_meters = pyproj.CRS('EPSG:6991')
	proj_latlng = pyproj.CRS('EPSG:4326')

	project_to_meters = Transformer.from_crs(proj_latlng, proj_meters, always_xy=True).transform
	# project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform

	project_to_latlng = Transformer.from_crs(proj_meters, proj_latlng, always_xy=True).transform

	pt_latlng = Point(lng, lat)
	pt_meters = transform(project_to_meters, pt_latlng)

	buffer_meters = pt_meters.buffer(radius)
	buffer_latlng = transform(project_to_latlng, buffer_meters)
	return buffer_latlng

def creat_polygons(df, range):
	buffer_radius = range*0.6

	# groups = df.groupby('cluster')

	clusters = list()
	blobs = list()
	# count = 0

	buffers = [buffer_in_meters(lon, lat, buffer_radius) for lon, lat in zip(df['lon'], df['lat'])]
	blob = unary_union(buffers)

	cluster_gdf = gpd.GeoDataFrame({'value':[str(1)]}, geometry=[blob])
	cluster_gdf = cluster_gdf.set_crs("EPSG:4326")

	return cluster_gdf


def plot(polygons):
		m = folium.Map(location=[32.11,34.80], zoom_start=14, control_scale=True,tiles='CartoDB positron')

		for index, geom in zip(polygons['value'], polygons['geometry']):
			geo_j = gpd.GeoSeries(geom).to_json()
			geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'fillColor': 'orange'})
			folium.Popup('this is range: ' + str(index)).add_to(geo_j)
			geo_j.add_to(m)

		m.save('Step3_Target_range_index.html')


def check_visit_home_range(df, polygons):

	res = []
	for i in range(len(df)):
		if df['Timestamp_forage'][i] != '-1':
			point = Point(df["Longitude"][i], df["Latitude"][i])
			# gpd_tree_points = gpd.GeoDataFrame({'id':['1']}, geometry=[point])
			# sys.exit()
			# print(point.within(polygons['geometry'][0]))
			if point.within(polygons['geometry'][0]):
				res.append(1)
			else:
				res.append(0)
		else:
			res.append(0)
	# sys.exit()
	df['Home_range'] = res
	return df

def day_number(start, end):
	start_format = datetime.strptime(start, '%Y-%m-%d')
	end_format = datetime.strptime(end, '%Y-%m-%d')
	# print(start_format, end_format)
	tmp = end_format - start_format

	output = []
	for day in range(tmp.days+1):
		a_date = start_format + timedelta(days = day)
		a_format = str(a_date).split(' ')[0]
		output.append(a_format)
	return output

def date_diff_in_min(dt2, dt1):
	timedelta = dt2 - dt1
	return timedelta.days * 24 * 3600 + timedelta.seconds
	
# def Distance(point1, point2):



def be_in(seq):

	output = [seq[0]]
	ref = seq[0]
	# print(seq)

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

def extracting_df(date, df, traget_tree_id):

	start = date + ' 12:00:00'
	start_format = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
	end_format = start_format + timedelta(days = 0.7)

	# print(start_format, end_format)

	my_tree_pool = []
	count = 1
	my_delta_time = 0 
	my_distance = 0 

	for i in range(len(df)):
		if df['Home_range'][i] == 1:
			my_time = df['Date'][i] + ' ' + df['UTC'][i]
			my_location = (df['Ad_lat'][i]*180/math.pi, df['Ad_lng'][i]*180/math.pi)
			try:
				my_time = datetime.strptime(my_time, '%Y-%m-%d %H:%M')
			except:
				my_time = datetime.strptime(my_time, '%Y-%m-%d %H:%M:%S')

			if my_time > start_format and my_time < end_format:
				
				if count == 1:
					ref_time = my_time
					ref_location = (df['Ad_lat'][i]*180/math.pi, df['Ad_lng'][i]*180/math.pi)
					if not np.isnan(df['Grouped_tree_id'][i]):
						my_tree_pool.append(df['Grouped_tree_id'][i])
					count += 1
				else:
					# print("I am here!!", my_location ,ref_location)
					my_delta_time_tmp = date_diff_in_min(my_time ,ref_time)/60
					my_distance_tmp = h3.point_dist(my_location, ref_location, unit="m")

					if not np.isnan(df['Grouped_tree_id'][i]):
						my_tree_pool.append(df['Grouped_tree_id'][i])
					# print('my_distance_tmp: ', my_distance_tmp)
					# print('my_delta_time_tmp: ', my_delta_time_tmp)
					if my_delta_time_tmp < 20:
						my_delta_time += my_delta_time_tmp
						my_distance += my_distance_tmp
						
						# count = 1
					
					ref_time = my_time
					ref_location = my_location
					

			elif my_time > end_format:
				# print('my_distance: ', my_distance)
				# print('my_delta_time: ', my_delta_time/60)
				
				break
	if len(my_tree_pool) > 1:
		my_tree_pool = be_in(my_tree_pool)

	tree_count = 0
	for i in my_tree_pool:
		if i in traget_tree_id:
			tree_count += 1
	# print(my_delta_time/60, len(my_tree_pool), tree_count)
	# sys.exit()
	return my_distance, my_delta_time/60, len(my_tree_pool), tree_count

def time_machine(df, batname, traget_tree_id):

	Dates = []
	for i in df['All_time']:
		date, mytimes = i.split(' ')
		Dates.append(date)

	df['Date'] = Dates
	date_list = day_number(min(Dates), max(Dates))

	pool = []
	count = 1
	dis , mytime, tree_nums, target_nums = 0, 0 ,0, 0

	for date in date_list: 
		dis_tmp, mytime_tmp, tree_nums_tmp, target_num_tmp = extracting_df(date, df, traget_tree_id)
		dis += dis_tmp
		mytime += mytime_tmp
		tree_nums += tree_nums_tmp
		target_nums += target_num_tmp

		if mytime_tmp > 0:
			pool.append([str(count), date, dis, mytime, tree_nums, target_nums])
			count += 1
			# print(count, date, dis, mytime, tree_nums)
	# # print(pool)
	# sys.exit()
	pool = np.array(pool)
	df_new = pd.DataFrame()
	df_new['Count'] = pool[:,0]
	df_new['Date_' + batname] = pool[:,1]
	df_new['Distance_' + batname] = pool[:,2]
	df_new['Time_' + batname] = pool[:,3]
	df_new['Tree_num_' + batname] = pool[:,4]
	df_new['Target_num_' + batname] = pool[:,5]
	print('batname: ', len(df_new))
	return df_new


def run():
	captivity_file = '../Step2_tree_infering/sta_2——2_grouped_be_in_20_10_e.csv'
	tree_file = fruit_file

	traget_tree_df = pd.read_csv(captivity_file)
	all_tree_id_df = pd.read_excel(tree_file)

	traget_tree_id = list(set(traget_tree_df['tree_id']))
	df_target = pd.DataFrame()
	for i in traget_tree_id:
		record = all_tree_id_df.loc[all_tree_id_df["id"]==i, ['lat', 'lon', 'id']]
		df_target = df_target.append(record, ignore_index=True)
	
	polygons_file = './Step3_polygons_target_range.json'

	eps_in_meters_range = 50
	polygons = creat_polygons(df_target, eps_in_meters_range)
	polygons.to_file(polygons_file, driver="GeoJSON")
	
	# 作图
	polygons = gpd.read_file(polygons_file)
	plot(polygons) # 做图

	BatNames = ['Avigur', 'Bane', 'Buffy', 'Haim_Shelanu', 'Luna', 'Malagasi','Matcha',
				'Moana','Oskar_Tal', 'Pizza', 'Pliocene', 'Rasmi', 'Shlomzion', 'Yamit', 
				'Yumi','Adva','Aria', 'Dilla', 'Jafar', 'Malik','Peter_Pen','Miles', 
				'Raja', 'Michi', 'Holy','Tzuzik', 'Prince_Edward','Gana',
				'Ali','Balaz','Camila','Koral', 'MitMit','Ozvelian','Rosie','Shavit',
				'Anka', 'Eli', 'Eva', 'Fima', 'K', 'Mazi','Nadav','Nature', 
				'Nazir', 'Odelia', 'Shem_Tov', 'Tishray','Tzedi', 'V']

	BatNames = ['Adva', 'Miles', 'Raja', 'Michi', 'Holy', 'Tzuzik', 'Prince_Edward', 
				'Aria', 'Dilla', 'Jafar', 'Malik','Peter_Pen','Avigur', 'Bane', 'Buffy', 
				'Haim_Shelanu', 'Luna', 'Malagasi','Matcha','Moana','Oskar_Tal', 'Pizza', 
				'Pliocene', 'Rasmi', 'Shlomzion', 'Yamit', 'Yumi']

	eps_in_meters = 20
	threshold = threshold_acc
	for batname in tqdm(BatNames):
		print(batname)
		bat_dir = '../Step2_tree_infering/dataset/' + batname +'/'


		data_file = bat_dir + batname + '_data_s'+str(eps_in_meters)+'_grouped_threshold-' + str(threshold_acc)+ '_e.csv'
		ChkFile(data_file)
		df_bats = pd.read_csv(data_file)

		polygons = gpd.read_file(polygons_file)
		df_checked = check_visit_home_range(df_bats, polygons)

		checked_file = data_file.replace('.csv', '_checked_50.csv')
		df_checked.to_csv(checked_file)


	# 最后一个部分，统计数量和时间
	BatNames = ['Avigur', 'Bane', 'Buffy', 'Haim_Shelanu', 'Luna', 'Malagasi','Matcha',
				'Moana','Oskar_Tal', 'Pizza', 'Pliocene', 'Rasmi', 'Shlomzion', 'Yamit', 
				'Yumi','Adva','Aria', 'Dilla', 'Jafar', 'Malik','Peter_Pen','Miles', 
				'Raja', 'Michi', 'Holy','Tzuzik', 'Prince_Edward','Gana',
				'Ali','Balaz','Camila','Koral', 'MitMit','Ozvelian','Rosie','Shavit',
				'Anka', 'Eli', 'Eva', 'Fima', 'K', 'Mazi','Nadav','Nature', 
				'Nazir', 'Odelia', 'Shem_Tov', 'Tishray','Tzedi', 'V']

	BatNames = ['Adva', 'Miles', 'Raja', 'Michi', 'Holy', 'Tzuzik', 'Prince_Edward', 
				'Aria', 'Dilla', 'Jafar', 'Malik','Peter_Pen','Avigur', 'Bane', 'Buffy', 
				'Haim_Shelanu', 'Luna', 'Malagasi','Matcha','Moana','Oskar_Tal', 'Pizza', 
				'Pliocene', 'Rasmi', 'Shlomzion', 'Yamit', 'Yumi']

	df_summary = pd.DataFrame()
	df_summary['Count'] = [str(i) for i in range(1,200)]
	print("df_summary: ", df_summary['Count'])
	for batname in BatNames:
		print(batname)
		bat_dir = '../Step2_tree_infering/dataset/' + batname +'/'

		data_file = bat_dir + batname + '_data_s'+str(eps_in_meters)+'_grouped_threshold-' + str(threshold_acc)+ '_e.csv'
		checked_file = data_file.replace('.csv', '_checked_50.csv')
		ChkFile(checked_file)
		df_checked = pd.read_csv(checked_file)
		df_add = time_machine(df_checked, batname, traget_tree_id)
		df_summary= pd.merge(df_summary, df_add, on='Count', how='left')

		
	file_summary = './Step3_summary_' + str(eps_in_meters_range) + '_e.csv'
	df_summary.to_csv(file_summary)




if __name__ == "__main__":
	run()

