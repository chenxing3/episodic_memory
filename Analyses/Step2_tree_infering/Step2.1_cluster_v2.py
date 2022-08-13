

# normal packages
import os,sys
sys.path.append('/Users/xingchen/Documents/PhD_learning/Bat_navigation/All_database/Vesper/Recheck_ep_memory/functions')
from My_hyper_parameters import * 
from My_functions import * 

import re
import math
import numpy as np 
import pandas as pd 
import operator
from tqdm import tqdm
import traces

from numba import jit, njit
import numba
import warnings 
warnings.simplefilter(action='ignore')


# geographic packages
import h3
import pyproj

import geopandas as gpd

from pyproj import Transformer
from sklearn.cluster import DBSCAN
from datetime import datetime, time
from shapely.geometry import Point
from shapely.ops import transform
from shapely.ops import cascaded_union
import folium
from shapely import wkt




@jit(nopython=True)
def is_inside_sm(polygon, point):
	length = len(polygon)-1
	dy2 = point[1] - polygon[0][1]
	intersections = 0
	ii = 0
	jj = 1

	while ii<length:
		dy  = dy2
		dy2 = point[1] - polygon[jj][1]

		# consider only lines which are not completely above/bellow/right from the point
		if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

			# non-horizontal line
			if dy<0 or dy2<0:
				F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

				if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
					intersections += 1
				elif point[0] == F: # point on line
					return 2

			# point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
			elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
				return 2

		ii = jj
		jj += 1

	#print 'intersections =', intersections
	return intersections & 1  


@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
	ln = len(points)
	D = np.empty(ln, dtype=numba.boolean) 
	for i in numba.prange(ln):
		D[i] = is_inside_sm(polygon,points[i])
	return D  

def isNaN(string):
	return string != string

def get_close_trees(lon, lat, tree_points, threshold):
	res = []
	for _, (point, my_range, my_lon, my_lat) in enumerate(tree_points):
		_, _, dis = geodesic.inv(lon, lat, my_lon, my_lat)
		if dis < threshold:
			res.append([point, dis, my_range, [my_lon, my_lat]])
	return res

def get_trajactory(right_time, df):

	start = right_time - timedelta(seconds = 30)
	end = right_time + timedelta(seconds = 30)
	# print(start, end)

	df['All_time'] = pd.to_datetime(df["All_time"].astype(str), format='%Y/%m/%d %H:%M:%S')
	latitudes = []
	longtiudes = []
	for mytime, my_lng, my_lat in zip(df['All_time'], df['Ad_lng'], df['Ad_lat']):
		latitudes.append([mytime, my_lat])
		longtiudes.append([mytime, my_lng])

	latitudes = traces.TimeSeries(latitudes)
	longtiudes = traces.TimeSeries(longtiudes)

	tmp_lat = latitudes.sample(
		sampling_period=timedelta(seconds=0.1),
		start=start,
		end=end,
		interpolate='linear',
	)

	tmp_lon = longtiudes.sample(
		sampling_period=timedelta(seconds=0.1),
		start=start,
		end=end,
		interpolate='linear',
	)

	df_res = pd.DataFrame()
	df_res['All_time'] = np.array(tmp_lat)[:,0]
	df_res['lat'] = np.array(tmp_lat)[:,1]
	df_res['lon'] = np.array(tmp_lon)[:,1]
	df_res = df_res.replace('None', np.nan).dropna(how='any')
	df_res = df_res.reset_index()

	results = []
	for mylat, mylon in zip(df_res['lat'], df_res['lon']):
		results.append([mylon*180/np.pi, mylat*180/np.pi])
	return np.array(results)




def indexing(df, threshold=0):
	'''
	give id to all the possible foraging points
	'''
	count = 1
	count_list = []

	for index in df.index:
		if not np.isnan(df['Preds'][index]):
			if df['Preds'][index] > threshold:
				count += 1
				count_list.append("-1")
			else:
				tmp = str(int(df['Night_stamp'][index])) + '_' + str(count)
				count_list.append(tmp)
		else:
			count_list.append("-1")

	df['Timestamp_forage'] = count_list

	return df


def cluster_foraging(df, eps_in_meters=25, num_samples=5):
	'''
	using DBScan to filter data not in a cluster
	'''
	earth_perimeter = 40070000.0
	eps_in_radians = eps_in_meters/earth_perimeter* (2*math.pi)

	db_scan = DBSCAN(eps=eps_in_radians, min_samples=num_samples, metric='haversine')
	# print(df[['Ad_lat', 'Ad_lng']])
	return db_scan.fit_predict(df[['Ad_lat', 'Ad_lng']])


def spatial_cluster(df, eps_in_meters=25, min_sample_per_cluster=3):
	'''
	to get data for DBScan 
	'''
	groups = df.groupby('Timestamp_forage')
	range_cluster = []

	for cluster_id, points in groups:
		# print(cluster_id)
		# sys.exit(1)
		if len(points) > min_sample_per_cluster and cluster_id != '-1':
			temp = cluster_foraging(points, eps_in_meters, min_sample_per_cluster)
			# print('Timestamp_forage cluster: ', temp)
			# sys.exit()
			temp_pool = []
			for i in temp:
				if i >= 0:
					temp_id = str(cluster_id).zfill(3) + '-' + str(i+1)
				else:
					temp_id = "0" # 如果没有被聚类，都设为0
				temp_pool.append(temp_id)

			range_cluster.extend(temp_pool)
		else:
			tmp = [0]*len(points)
			tmp_id = [str(i) for i in tmp]
			range_cluster.extend(tmp_id)
	return range_cluster


def buffer_in_meters(lng, lat, radius):
	'''
	to get buffer range of a point
	'''
	proj_meters = pyproj.CRS('EPSG:6691') # Israel range
	proj_latlng = pyproj.CRS('EPSG:4326') # WGS84

	project_to_meters = Transformer.from_crs(proj_latlng, proj_meters, always_xy=True).transform
	# project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform

	project_to_latlng = Transformer.from_crs(proj_meters, proj_latlng, always_xy=True).transform

	pt_latlng = Point(lng, lat)
	pt_meters = transform(project_to_meters, pt_latlng)

	buffer_meters = pt_meters.buffer(radius)
	buffer_latlng = transform(project_to_latlng, buffer_meters)
	return buffer_latlng

def Fruit_trees(fruit_file, threshold):
	fruit_trees = pd.read_csv('../ops/TreesAllYears_final_slite_20_buffer_range.csv')

	fruit_trees['geometry'] = fruit_trees['geometry'].apply(wkt.loads)

	tree_points = []
	reps = []

	for i in range(len(fruit_trees)):
		if fruit_trees["id"][i] not in reps:

			x_poly, y_poly  = fruit_trees["geometry"][i].exterior.coords.xy
			x_poly = np.array(x_poly)
			y_poly = np.array(y_poly)
			MyPolygon = np.array([[x, y] for x, y in zip(x_poly, y_poly)])
			# print(MyPolygon)
			# sys.exit(1)
			# MyRange = buffer_in_meters(fruit_trees["lon"][i], fruit_trees["lat"][i], threshold)
			tree_points.append([fruit_trees["id"][i], MyPolygon, fruit_trees["lon"][i], fruit_trees["lat"][i]])
			reps.append(fruit_trees["id"][i])
	return tree_points


def cut_time(df, tree_points):
	max_time_trees = []
	max_times = []
	shorest_tree_times = []
	for i in tqdm(range(len(df))):

		if not isNaN(df['Tree_id'][i]):

			# to get trajactory within 1min for a specific point
			min_index = max(i-10, 0)
			max_index = min(i+10, len(df)-1)
			indexes = np.arange(min_index, max_index)
			selected_df = df.iloc[indexes]
			right_time = pd.to_datetime(str(df['All_time'][i]), format='%Y/%m/%d %H:%M:%S')
			trajactory_points = get_trajactory(right_time, selected_df)
			# sys.exit(1)
			# print(df['Tree_id'][i])

			# 看这个附近的tree
			selected_lat = df['Ad_lat'][i]*180/np.pi
			selected_lon = df['Ad_lng'][i]*180/np.pi

			# print(selected_lat, selected_lon)
			shortest_tree = df['Tree_id'][i]
			selected_trees = get_close_trees(selected_lon, selected_lat, tree_points, eps_in_meters)
			# print(selected_trees)

			if len(selected_trees) > 0:
				possible_trees = []
				sum_times = []
				my_shorest_time = None
				for _, (tree_id, mydis, my_range, (lon, lat)) in enumerate(selected_trees):
					# print("trajactory_points: ", trajactory_points)
					# print("my_range:", my_range)
					Mybool = is_inside_sm_parallel(trajactory_points, my_range)
					# print(len(trajactory_points))

					# sys.exit(1)
					if len(trajactory_points) == 301 or len(trajactory_points) ==300: # when only half of the data avaliable, unified them into 60s
						mytime_tmp  = sum(Mybool)*2-1
					else:
						mytime_tmp  = sum(Mybool)

					possible_trees.append(tree_id)
					sum_times.append(mytime_tmp/10)

					if tree_id == shortest_tree:
						my_shorest_time = mytime_tmp/10


				shorest_tree_times.append(my_shorest_time)
				if my_shorest_time >= 59.9:
					# right_trees += 1
					max_time_trees.append(shortest_tree)
					max_times.append(my_shorest_time)
				else:
					# print(possible_trees, sum_times)
					max_time = max(sum_times)
					tmp = sum_times.index(max_time)
					max_time_tree = possible_trees[tmp]

					# if shortest_tree - max_time_tree > -1:
					# 	right_trees += 1
					# 	# print(shortest_tree, max_time_tree, max_time)
					# else:
					# 	wrong_trees += 1

					max_time_trees.append(max_time_tree)
					max_times.append(max_time)
					# break
			else:
				max_time_trees.append(None)
				max_times.append(None)
				shorest_tree_times.append(None)
		else:

			max_time_trees.append(None)
			max_times.append(None)
			shorest_tree_times.append(None)

	df['max_time_tree'] = max_time_trees
	df['max_time'] = max_times
	df['shorest_tree_time'] = shorest_tree_times

	# file_res = file.replace('.csv', '_add_time.csv')
	# df.to_csv(file_res, index=None)
	return df


def grouping_trees(df, threshold, threshold_acc):
	# to filter preds

	# deep learning part
	for i in range(len(df)):
		if not isNaN(df['Preds'][i]):
			if df['Preds'][i] > threshold_acc:
				df['cluster'][i] = str(0)
			# else:
			# 	df['cluster'][i] = df['cluster'][i]

	print(df)
	df = df.sort_values(by='cluster')

	groups = df.groupby('cluster')

	res_list = []
	for group_id, records in groups:
		if group_id == "0":
			for index in records.index:
				# res_list.append(records['Tree_id'][index])
				res_list.append(None)
		else:

			for index in records.index:
				if records['Tree_distance'][index] <= threshold:
					if records['shorest_tree_time'][index] >= 30:
						res_list.append(records['Tree_id'][index])
					elif records['max_time'][index] >= 30:
						res_list.append(records['max_time_tree'][index])
					else:
						res_list.append(None)
				else:
					res_list.append(None)

	df['Grouped_tree_id'] = res_list
	df = df.sort_values(by='Timestamp')
	return df



def run():
	BatNames = [
				# 'Adva', 'Miles', 'Raja', 'Michi', 'Holy', 'Tzuzik', 'Prince_Edward', 
				# 'Aria', 'Dilla', 'Jafar', 'Malik','Peter_Pen','Avigur', 'Bane', 'Buffy', 
				# 'Haim_Shelanu', 'Luna', 'Malagasi','Matcha','Moana','Oskar_Tal', 'Pizza', 
				# 'Pliocene', 'Rasmi', 'Shlomzion', 'Yamit', 'Yumi',
				# 'Ali','Balaz','Camila','Koral', 
				# 'MitMit','Ozvelian','Rosie','Shavit',
				# 'Anka', 'Eli', 'Eva', 'Fima', 'K', 
				# 'Mazi','Nadav','Nature', 
				'Gana', 
				# 'Nazir', 'Odelia', 
				# 'Shem_Tov', 
				# 'Tishray', 
				# 'Tzedi', 'V'
				]

	# BatNames = ['Adva']

	eps_in_meters = 20
	threshold_acc = 10

	tree_points = Fruit_trees(fruit_file, eps_in_meters)


	for batname in tqdm(BatNames):
		print(batname)
		bat_dir = './dataset/' + batname +'/'
		data_file = bat_dir + batname + '_tree_infer_30_'+str(threshold_acc)+'_e.csv'
		ChkFile(data_file)

		df = pd.read_csv(data_file)


		# filter the trees when the time that a bat spend near by a tree is less then 30s
		df_filter = cut_time(df, tree_points)
		# file_res = data_file.replace('.csv', '_add_time.csv')
		# df_filter.to_csv(file_res, index=None)

		# deep learning filter
		# df_filter = pd.read_csv(file_res)

		df_filter = add_night_stamp_group(df_filter)
		df_filter = indexing(df_filter, threshold_acc)
		# DBscan part
		
		

		df_filter = df_filter.sort_values(by='Timestamp_forage')
		cluster_list = spatial_cluster(df_filter, eps_in_meters_dmso)
		df_filter['cluster'] = cluster_list # filter tree using DB Scan
		print(df_filter)
		# sys.exit()

		df_grouped = grouping_trees(df_filter, eps_in_meters, threshold_acc)
		res_file = bat_dir + batname + '_data_s'+str(eps_in_meters)+'_grouped_threshold-' + str(threshold_acc)+ '_e.csv'
		df_grouped.to_csv(res_file, index=None)





if __name__ == "__main__":
	run()

