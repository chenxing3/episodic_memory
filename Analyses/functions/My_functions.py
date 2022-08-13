import math
import os
import sys
import re
import numpy as np
import pandas as pd
import glob
from decimal import *
from datetime import datetime, time, timedelta
from zipfile import ZipFile


import pyproj
geodesic = pyproj.Geod(ellps='WGS84')

from lxml import html
from pykml import parser

import tfrecord
from collections import Counter
from astropy.convolution import convolve
import tfrecord

from mypythonlib.common_functions import ChkDir, ChkFile

from My_hyper_parameters import *

from shapely.geometry import Point
import geopandas as gpd
import h3




# step 1 collect data
###########################

def collect_dirs(dir):
	# 就是为了载入文件夹

	all_dirs = []
	files = os.listdir(dir)
	for fn in files:
		real_path = os.path.join(dir,fn)
		
		if os.path.isdir(real_path):
			temp = collect_dirs(real_path)
			all_dirs.append([fn, real_path])
	
	return all_dirs
			
def osPath(url):
	# 载入文件夹内的所有文件
	output = []
	files=os.listdir(url)#此时files是一个容器
	for f in files:#遍历容器
		real_path = os.path.join(url,f)#拼接路径
		if os.path.isfile(real_path):#判断是否是文件
			output.append(os.path.abspath(real_path))
			# print(os.path.abspath(real_path))#打印文件的绝对路径
		elif os.path.isdir(real_path):#判断是否是文件夹
			#此时是一个文件夹
			#需要使用递归继续进行查找
			temp = osPath(real_path)#继续调用函数完成递归
			output.extend(temp)
	return output

# 这个改版，就是为了kmz文件的搜索
def osPath_v2(url):
	# 载入文件夹内的所有文件
	output = []
	files=os.listdir(url)#此时files是一个容器
	for f in files:#遍历容器
		real_path = os.path.join(url,f)#拼接路径
		if os.path.isfile(real_path):#判断是否是文件
			output.append([f.replace('.kmz', ''), os.path.abspath(real_path)])
			# print(os.path.abspath(real_path))#打印文件的绝对路径
		elif os.path.isdir(real_path):#判断是否是文件夹
			#此时是一个文件夹
			#需要使用递归继续进行查找
			temp = osPath_v2(real_path)#继续调用函数完成递归
			output.extend(temp)
	return output



#就是为了查重geotag文件

def name_match(strg, search=re.compile(r'Track-geoTag.csv').search):
	'''
	search csv files
	'''
	return bool(search(strg))


# 为了查找 geotag里面的所有数据
def extracting_csv(dir):
	df2 = pd.DataFrame()

	all_file = osPath(dir)
	for file in all_file:
		if name_match(file):
			file_df = pd.read_csv(file)
			file_df['All_time'] = file_df[['Date', 'UTC']].agg(' '.join, axis=1)
			file_df['All_time'] = pd.to_datetime(file_df["All_time"].astype(str), format='%Y/%m/%d %H:%M:%S')

			df2 = df2.append(file_df, ignore_index=True)

	df2 = df2.sort_values(by='All_time')

	# 改一下lat和long的名字
	names = df2.columns.tolist()
	names[names.index('Latitude [DD]')] = 'Latitude'
	names[names.index(' Longitude [DD]')] = 'Longitude'
	df2.columns = names


	return df2



def time_convert(time_input):
	my_time =  time_input - 367.0000028935
	# my_time = 736644.734370833
	# print(my_time)
	my_time_new = my_time * timedelta(days=1) + datetime(1, 1, 1, 0, 0, 0)
	return my_time_new.strftime('%Y/%m/%d %H:%M:%S') # show Local version of date


# 为了查找txt里面的所有数据
def extracting_txt(dir):
	df2 = pd.DataFrame()
	all_file = osPath(dir)
	for file in glob.glob(dir + '/*.txt'):
		df = pd.read_csv(file)
		if len(df) > 1: 
			normal_time = []
			UTCs = []
			for i in range(len(df)):
				temp_time = time_convert(df['time'][i])
				normal_time.append(temp_time)
				UTC = temp_time.split(' ')[1]
				UTCs.append(UTC)

			df['UTC'] = UTCs
			df['All_time'] = normal_time
			df['All_time'] = pd.to_datetime(df["All_time"].astype(str), format='%Y/%m/%d %H:%M:%S')

			df2 = df2.append(df, ignore_index=True)

	df2 = df2.sort_values(by='All_time')



	# 改一下lat和long的名字
	names = df2.columns.tolist()
	names[names.index('lat')] = 'Latitude'
	names[names.index('lon')] = 'Longitude'
	df2.columns = names


	return df2


# kmz的解析文件


def extract_lat_lon(file):

	times = []
	my_lons = []
	my_lats = []
	utcs = []
	my_indexes = []
	count = 1

	# print("file", file)
	kmz = ZipFile(file, 'r')

	with kmz.open('Plot.kml', 'r') as f: # 有可能是这种解析
	# with kmz.open('plot.kml', 'r') as f:
		kml = parser.parse(f).getroot()
	# print(kml)
	# sys.exit()
	for pm_large in kml.Document.Folder:
		try:
			for pm in pm_large.Placemark:

				my_time = pm.name
				# print(my_time)
				my_coordinates = pm.LineString.coordinates
				# my_latitude = kml.Document.Folder.Placemark.description.latitude

				my_coord = str(my_coordinates).split(',')
				my_lon = my_coord[0]
				my_lat = my_coord[1]
				# print(my_lon, my_lat)

				all_time_tmp = str(my_time)
				utc= all_time_tmp.split(' ')[1]
				times.append(all_time_tmp)
				utcs.append(utc)
				my_lons.append(float(my_lon))
				my_lats.append(float(my_lat))
				my_indexes.append(count)
				count+= 1
		except:
			pass


	return times, utcs, my_lons, my_lats, my_indexes

# 为了查找kmz里面的所有数据
def extracting_kmz(file):

	df = pd.DataFrame()
	# for file in glob.glob(files):
	my_times, my_utcs, my_lons, my_lats, my_indexes = extract_lat_lon(file)
	# print(my_times)
	# sys.exit()
	df['All_time'] = my_times
	df['UTC'] = my_utcs
	df['Latitude'] = my_lats
	df['Longitude'] = my_lons
	df['index'] = my_indexes

	df['All_time'] = pd.to_datetime(df["All_time"].astype(str), format='%d-%b-%Y %H:%M:%S') #
	df = df.sort_values(by='All_time')

	return df




# step 2 timestamp
###########################

def delta_timestamp(start_time, end_time):
	start_time = datetime.strptime(start_time, '%H:%M:%S')
	end_time = datetime.strptime(end_time, '%H:%M:%S')
	timedelta = end_time - start_time

	return (timedelta.days * 24 * 3600 + timedelta.seconds) # every 30 seconds

def isNaN(string):
	return string != string

def search_time_interval(file_df):
	# update 2021.10
	output = []

	tmps = []
	if len(file_df) > 1: # 如果record只有一条，则没有必要去考虑了
		for i in range(1,len(file_df['UTC'])):
			start_index = file_df['UTC'].index[i-1]
			end_index = file_df['UTC'].index[i]
			tmp = delta_timestamp(file_df['UTC'][start_index], file_df['UTC'][end_index])
			if tmp < 60 and tmp > 0: # 如果这个时间长度大于2分钟，则不予考虑
				tmps.append(tmp-0.01)

		# 处理一下，分为几个部分：
		tmp_median = np.median(tmps)

		if tmp_median > 25: #如果median 大于25 应该就是30s
			tmp_median = Decimal(tmp_median).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
			# print("30: ", tmp_median)
		else: 
			if tmp_median > 13: #如果median 大于25 应该就是15s
				tmps2 = []
				for i in tmps:
					if i < 30:
						tmps2.append(i)
				tmp2_median = np.median(tmps2)
				tmp_median = Decimal(np.median(tmp2_median)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)

				if tmp_median < 15:
					print("Warning!!!!Please check here!!!!!!!")
					sys.exit()

		if not isNaN(tmp_median):
			return tmp_median
		else:
			return 30



def date_diff_in_min(dt2, dt1, time_interval):
	timedelta = dt2 - dt1
	return (timedelta.days * 24 * 3600 + timedelta.seconds)/time_interval # every 30 seconds


def convert_timestamp(my_time, start_time, time_interval):
	# start_time = parser.parse(start_time)
	# end_time = parser.parse(my_time)
	# print(start_time, end_time)
	# sys.exit(1)

	start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
	mytime = datetime.strptime(my_time, '%Y-%m-%d %H:%M:%S')
	ID = date_diff_in_min(mytime, start_time, time_interval)
	ID_c = Decimal(ID).quantize(Decimal('0'), rounding=ROUND_HALF_UP) #四舍五入
	return ID_c


def cleanup_GPS(df, time_interval, sunset_time, dusk_time):
	'''
	filter test time
	'''

	## choose the data only from 16:00 to 8:00 AM
	df2 = pd.DataFrame()
	# dusk_time = '5:00:00'
	# sunset_time = '15:00:00'
	df['UTC_new'] = pd.to_datetime(df['UTC'])

	idx1 =(df.set_index('UTC_new')
			.between_time('00:00:00', dusk_time)
			.reset_index()
			.reindex(columns=df.columns))

	# print(len(idx1), len(df))

	idx2 =(df.set_index('UTC_new')
			.between_time(sunset_time, '23:59:59')
			.reset_index()
			.reindex(columns=df.columns))
	
	# print(len(idx2), len(df))
	df2 = df2.append(idx2, ignore_index=True)
	df2 = df2.append(idx1, ignore_index=True)

	# df2['All_time'] = df2[['Date', 'UTC']].agg(' '.join, axis=1)

	# df2.to_csv('test.csv')

	df2.drop('UTC_new', axis=1, inplace=True) 
	df2.reset_index(inplace=True, drop=True)
	# df2 = def2(ignore_index=True)
	
	# add time stamp 
	Timestamp = [convert_timestamp(i, df2['All_time'][0], time_interval) for i in df2['All_time']]
	# print(Timestamp, time_interval)
	df2['Timestamp'] = Timestamp
	# df2['Timestamp_group'] = classify_timestamp(Timestamp)
	df2 = df2.sort_values(by=['Timestamp'])
	return df2




def all_night_list(start, end):
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


def extract_night_df(df, start, end): 
	# start_format = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
	# end_format = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')

	# print(start, end)

	df['All_time'] = pd.to_datetime(df['All_time'])
	df2 = df[(df['All_time'] > start) & (df['All_time'] <= end)]

	return df2


def create_radian_columns(df):
	df['rad_lng'] = np.radians(df['Longitude'].values)*100000
	df['rad_lat'] = np.radians(df['Latitude'].values)*100000
	return df

def filling_blank(df):
	df.reset_index(inplace=True, drop=True)
	index_start = df['Timestamp'][0]
	index_end = df['Timestamp'][len(df)-1]+1
	# print(index_start, index_end)

	output = []
	for i in range(int(index_start), int(index_end)):
		flag = 0
		for df_i in range(len(df)):
			if i == df['Timestamp'][df_i]:
				# print(df['lat'][df_i], df['lon'][df_i], df['Timestamp'][df_i])
				output.append([df['Longitude'][df_i], df['Latitude'][df_i], df['rad_lng'][df_i], df['rad_lat'][df_i], int(i)])
				flag = 1
				break
		
		if flag == 0:
			# print(0, 0 , i)
			output.append([math.nan, math.nan, math.nan, math.nan, int(i)])
	# print("output: ", output)
	return output



def Threshold(array, my_length, max_value):
	MyThreshold = 0
	# count = 1
	for index in range(0,len(array)-1):
		# index = index+1
		tmp = math.sqrt((array[index+1][0] - array[index][0])**2 + (array[index+1][1] - array[index][1])**2)
		if tmp <= max_value:
			MyThreshold += tmp
			# print(MyThreshold, count)
			# count += 1

	return MyThreshold/my_length


def converting_base(array, threshold):

	bases_seq = []
	values_seq = []
	# 首先求差值序列
	for index in range(0, len(array)-1):
		tmp_value = math.sqrt((array[index+1][0] - array[index][0])**2 + (array[index+1][1] - array[index][1])**2)
		tmp_length = array[index+1][2] - array[index][2]
		if tmp_length != 0:
			my_value = tmp_value/tmp_length
		else:
			my_value = tmp_value
			print('time stamp is not correct')

		if my_value >= threshold:
			bases_seq.append(1)
		else:
			bases_seq.append(0)

		values_seq.append(my_value/threshold)
	return bases_seq, values_seq


def segmentation_indexes(total_length, seg_length, step):
	ranges = []
	fragment_number = int((total_length - seg_length)/step) +1
	for i in range(fragment_number):
		start = i*step
		end = i*step+seg_length
		if end <= total_length:
			ranges.append([start, end])
	# print(ranges, total_length)
	return ranges

def Conv(array, kernel=kernel):
	array2 = [array[0]]
	array2.extend(array)
	array2.append(array[-1])

	conv_array_1 = convolve(array2, kernel, nan_treatment='interpolate',boundary='extend')
	conv_array_1 = conv_array_1[1:-1]

	# array = array[1:-1]
	array_mask = np.isnan(array)
	conv_array_signal = np.where(array_mask, np.nan, conv_array_1)
	# print(array)
	# print(conv_array_signal)
	# sys.exit()

	conv_array_noise = array - conv_array_signal

	return conv_array_signal, conv_array_noise



def eign_count(array):
	counts = Counter(array)
	return counts[0], counts[1]

def convert_eign(array, eign_length):

	counts = Counter(array)

	res = []
	for i in range(eign_length):
		res.append(counts[i])
	# print(array, res)
	# sys.exit(1)
	return res


def big_seg_array(array, indexes):
	min_index = min(indexes)
	max_index = max(indexes)

	return array[min_index:max_index]

def not_nan(x):
	return x[~np.isnan(x)]

def Converting2Eign(array_raw, array, array_values, indexes, seg_length, large_seg_number):

	results = []
	eign_values = []
	mean_array = []
	selected_indexes = []
	my_array = []
	min_indexes = []

	# eign_0s = []
	# eign_1s = []
	for i, index in enumerate(indexes):
		i = i+1
		sub_array = array[index[0]:index[1]]


		eign_0, eign_1 = eign_count(sub_array)
		eign_values.extend(convert_eign([eign_0], index[1]-index[0]))
		eign_values.extend(convert_eign([eign_1], index[1]-index[0]))

		# eign_values.extend(convert_eign(sub_array, index[1]-index[0]))
		# print("mean: ", eign_values, len(eign_values))
		# sys.exit()
		mean_array.append(np.mean(array_values[index[0]:index[1]]))
		selected_indexes.extend([index[0], index[1]])

		if i % large_seg_number == 0:

			# print("mean_array, eign_values: ", eign_values)
			# sys.exit()
			results.append([mean_array, eign_values])
			my_array.append(array_raw[min(selected_indexes):max(selected_indexes)])
			min_indexes.append(min(selected_indexes))

			mean_array = []
			eign_values = []
			selected_indexes = []
			# eign_0s = []
			# eign_1s = []
		elif i == len(indexes):
			add_length = seg_length*large_seg_number*2 - len(eign_values)
			add_eign = [0]*add_length
			add_values = [0]*(large_seg_number - len(mean_array))
			mean_array.extend(add_values)
			eign_values.extend(add_eign)

			# print('I am here!!: ', len(add_eign), mean_array, eign_values, len(eign_values))
			results.append([mean_array, eign_values])
			my_array.append(array_raw[min(selected_indexes):max(selected_indexes)])
			min_indexes.append(min(selected_indexes))
		# print('my_array: ', my_array)
		# sys.exit()
	# print(results, len(results))
	# sys.exit()
	return results, my_array, min_indexes


def Converting(sequences, workdir):
	count = 1
	my_files = []
	for seq in sequences:
		seq_filter_lng, seq_noise_lng = Conv(np.array(seq)[:,2])
		seq_filter_lat, seq_noise_lat = Conv(np.array(seq)[:,3])


		seq_filter_signal = []
		# seq_filter_noise = []
		for tmp1, tmp2, tmp1_noise, tmp2_noise, MyID in zip(seq_filter_lng, seq_filter_lat, seq_noise_lng, seq_noise_lat, np.array(seq)[:,4]):
			if not isNaN(tmp1):
				seq_filter_signal.append([tmp1, tmp2, MyID])
				# seq_filter_noise.append([tmp1_noise, tmp2_noise, MyID])
			# else:

		seq_filter_signal = np.array(seq_filter_signal)
		# seq_filter_noise = np.array(seq_filter_noise)


		# for a, b in zip(seq_filter_signal[:,2], not_nan(np.array(seq)[:,4])):
		# 	print(a, b)
		# # print('seq_filter_signal: ', len(seq_filter_signal[:,2]), "\n", np.array(seq)[:,4])
		# sys.exit()

		threshold_signal = Threshold(seq_filter_signal, len(seq_filter_lng), 7)
		# threshold_noise = Threshold(seq_filter_noise, len(not_nan(seq_noise_lng)), 100)
		s_bases_seq, s_values_seq = converting_base(seq_filter_signal, threshold_signal)


		# # 然后进行分段，每7个点一段，步长为1，没7组一大段
		# big_segment_length = 9
		# seg_length = 4
		# # seg_length = 3
		# step = 1

		large_seg_number = int((big_segment_length-seg_length)/step + 1)

		indexes = segmentation_indexes(len(s_bases_seq), seg_length, step)

		eigns, sub_array, min_indexes = Converting2Eign(seq_filter_signal, s_bases_seq, s_values_seq, indexes, seg_length, large_seg_number)


		for eign, my_array, min_index in zip(eigns, sub_array, min_indexes):
			my_values = eign[0]
			my_eigns = eign[1]
			my_time_stamps = np.array(my_array)[:,2]

			## extract adjusted rad information
			my_adjusted_rad_lng = []
			my_adjusted_rad_lat = []
			for i in my_time_stamps:
				for record in seq_filter_signal:
					if i == record[2]:
						my_adjusted_rad_lng.append(str(record[0]/100000))
						my_adjusted_rad_lat.append(str(record[1]/100000))
						break


			# print('my_time_stamp: ', my_time_stamp)

			# print('my_array: ', my_array)
			# print('my_eigns: ', my_eigns)
			# print('my_values: ', my_values)
			# print('min_index: ', min_index)
			# sys.exit()

			filename_plot = workdir + '/' + str(count).zfill(4) + '.png'
			filename_text = workdir + '/' + str(count).zfill(4) + '.txt'

			# if len(seq)/5 > seg_length:
			# 	# print(seq[min_index:min_index+int(len(seq)/5)])
			# 	# sys.exit(1)
			# 	plotting(my_array, seq_filter_signal[min_index:min_index+int(len(seq_filter_signal)/2)], filename_plot)
			# else:
			# 	plotting(my_array, seq_filter_signal[min_index:min(min_index+int(len(seq_filter_signal)/2),len(seq_filter_signal))], filename_plot)

			text_handle = open(filename_text, 'w')
			text_handle.write(','.join([str(i) for i in my_eigns])+'\n')
			text_handle.write(','.join([str(i) for i in my_values])+'\n')
			text_handle.write(','.join([str(i) for i in my_time_stamps])+'\n')
			text_handle.write(','.join([i for i in my_adjusted_rad_lng])+'\n')
			text_handle.write(','.join([i for i in my_adjusted_rad_lat])+'\n')
			text_handle.close()

			my_files.append(filename_text)
			count += 1
	return my_files


def extracting_eign(file):

	count = 0
	for i in open(file, 'r'):
		i = i.strip()
		i = i.split(',')
		# print(i)
		if count == 0:
			eigns = [int(tmp) for tmp in i]
			count += 1
		elif count == 1:
			values = [float(tmp) for tmp in i]
			count += 1
	return values, eigns


def Convert_tf(data, tf_file):
	writer = tfrecord.TFRecordWriter(tf_file) # write

	for index, (label, value, array) in enumerate(data):
		# print('len(arrays): ', len(array))

		# print("array1: ", len(array1))

		writer.write({
			"text": (array, "float"),
			"value": (value, "float"),
			"label": (int(label), "int"),
		})
	writer.close()


def making_df(files, tf_file):

	pool = []
	for file in files:
		label = 0
		value_tmp, eign_tmp = extracting_eign(file)
		pool.append([label, value_tmp, eign_tmp])

	# print("len(pool): ", len(pool))
	Convert_tf(pool, tf_file)
	return tf_file



# 第三步骤，开始学会找树了
#===================================

def fruit_trees(fruit_file):
	fruit_trees = pd.read_excel(fruit_file)

	tree_points = []
	tree_id = []

	for i in range(len(fruit_trees)):
		if fruit_trees["id"][i] not in tree_id:
			tree_points.append(Point(fruit_trees["lon"][i], fruit_trees["lat"][i]))
			tree_id.append(fruit_trees["id"][i])

	gpd_tree_points = gpd.GeoDataFrame({'id':tree_id}, geometry=tree_points)

	return gpd_tree_points




def converting_array_tree_inferring(df, threshold = 0):

	lats = []
	lons = []
	ids = []
	for lat_tmp, lon_tmp, id_tmp, forage_tmp in zip(df['Ad_lat'], df['Ad_lng'], df['Timestamp'], df['Preds']):
		# print(lat_tmp, lon_tmp, id_tmp, forage_tmp)
		if not np.isnan(forage_tmp):
			if forage_tmp <= threshold:
				lats.append(lat_tmp*180/math.pi)
				lons.append(lon_tmp*180/math.pi)
				ids.append(id_tmp)

	return lats, lons, ids




def tree_infer(df, tree_gps, distance_bat_tree):
	'''
	对每一个gps点进行tree点距离计算
	'''

	

	tree_distances = []
	timestamps = []
	tree_ids = []
	for i in range(len(df)):
		flag = 0
		distance_tmp = 9999
		tree_tmp = math.nan
		bat_gps_point = (df["Latitude"][i], df["Longitude"][i])

		# print(bat_gps_point)
		# sys.exit()
		timestamp = df['Timestamp'][i]

		distance_pool = []
		for tree_id, tree_point in zip(tree_gps['id'], tree_gps['geometry']):

			# dis = h3.point_dist((tree_point.y,tree_point.x), bat_gps_point, unit="m")
			_, _, dis = geodesic.inv(tree_point.x, tree_point.y, df["Longitude"][i], df["Latitude"][i])
			# print(dis, dis_2)
			# sys.exit(1)
			if dis < distance_bat_tree:
				if dis < distance_tmp:
					tree_tmp = tree_id
					distance_tmp = dis
					flag = 1
		timestamps.append(timestamp)
		if flag == 0:
			tree_distances.append(math.nan)
			tree_ids.append(math.nan)
		else:
			tree_distances.append(round(distance_tmp, 1))
			tree_ids.append(int(tree_tmp))

	# print(tree_distance)
	return timestamps, tree_ids, tree_distances


def add_night_stamp_group(df):

	Dates = []
	final_df = pd.DataFrame()
	for i in df['All_time']:
		date, mytimes = i.split(' ')
		Dates.append(date)

	df['Date'] = Dates
	night_list = all_night_list(min(Dates), max(Dates))

	count = 1
	for night in night_list:
		# 一天的结束，是另一天的开始
		start = str(night + ' ' + sunset_time)
		# print('night: ', start)
		start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
		end = start+timedelta(hours=day_delta)

		df_sub = extract_night_df(df, start, end)

		if len(df_sub) > 0:
			df_sub['Night_stamp'] = [str(count)]*len(df_sub)

		count += 1

		final_df = final_df.append(df_sub, ignore_index = True)


	final_df = final_df.sort_values(by='All_time')

	return final_df



