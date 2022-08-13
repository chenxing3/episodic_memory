import numpy as np 
from astropy.convolution import Gaussian1DKernel, convolve

import sys
sys.path.append('./')

# all the bat names 

# 2016-2017
BatNames_2017 = ['Ali','Balaz','Camila','Koral', 'MitMit','Ozvelian',
				 'Rosie','Shavit', 'Asha', 'BA', 'Eliko', 'Fin', 
				 'Threedots', 'Uri']

# # 2017-2018  'Noam' = 'K',  'V' = 'Victor' , 'Gana' = ?
BatNames_2018 = ['Anka', 'Eli', 'Eva', 'Fima', 'K', 'Mazi','Nadav',
				 'Nature', 'Gana', 'Nazir', 'Odelia', 'Shem_Tov', 
				 'Tishray', 'Tzedi', 'V']


# 2019-2020
BatNames_2020 = ['Avigur', 'Bane', 'Buffy', 'Haim_Shelanu', 'Luna', 
				 'Malagasi','Matcha','Moana', 'Oskar_Tal', 'Pizza', 
				 'Pliocene', 'Rasmi', 'Shlomzion', 'Yamit', 'Yumi']

# 2020-2021
BatNames_2021 = ['Adva', 'Miles', 'Raja', 'Michi', 'Holy', 'Tzuzik', 
				 'Prince_Edward', 'Aria', 'Dilla', 'Jafar', 'Malik',
				 'Peter_Pen']

sunset_time = '14:00:00' # In the evening, it is actually London time, this place can take 14, it should be 14+3 = 17 o'clock in summer in Israel, and 14 + 2 = 16 o'clock in winter
dusk_time = '4:00:00' # In the morning, London time, it should be 4+3 o'clock in summer and 4+2 o'clock in winter in Israel
day_delta = 14

# segment length
big_segment_length = 9
seg_length = 4
step = 1

# kernel = np.array([1, 1, 1])/3
kernel = Gaussian1DKernel(stddev=1) # gaussian kernel

# step 3
fruit_file = '../ops/TreesAllYears_final_slite.xlsx'
hyper_acc = 10000 # test parameter
distance_bat_tree = 30 # tree and trajactary distance


# step 3.1
threshold_acc = 10
eps_in_meters_dmso = 30 # point distance in DBSCAN
eps_in_meters = 20 # shorter tree and trajactary distance


# step 3.2
tree_file = '../ops/Trees_with_food_alldata.xlsx'
id_file = '../ops/TreesAllYears_final_slite.xlsx'
fruit = 2 # fruit ID
nectar = 4 # nectar ID
day_delta_isolation = 2 # max day after isolation

experiments = [['Oskar_Tal', '20200330 4:00:00', '20200331 12:00:00', 'one_day'], ['Oskar_Tal', '20200213 4:00:00', '20200217 12:00:00', 'four_day'], 
				['Bane', '20200223 4:00:00', '20200224 12:00:00', 'one_day'], ['Bane', '20200326 4:00:00', '20200330 12:00:00', 'four_day'],
				['Rasmi', '20200331 4:00:00', '20200401 12:00:00', 'one_day'], ['Rasmi', '20200223 4:00:00', '20200227 12:00:00', 'four_day'], 
				['Matcha', '20200406 4:00:00', '20200407 12:00:00', 'one_day'], ['Matcha', '20200223 4:00:00', '20200227 12:00:00', 'four_day'],
				['Haim_Shelanu', '20200427 4:00:00', '20200428 12:00:00', 'one_day'],#['Haim_Shelanu', '20200303 4:00:00', '20200311 12:00:00', 'four_day'], #这个之后其实没有数据
				['Yumi', '20200303 4:00:00', '20200304 12:00:00', 'one_day'], ['Yumi', '20200430 4:00:00', '20200504 12:00:00', 'four_day'], 
				['Pizza', '20200223 4:00:00', '20200224 12:00:00', 'one_day'], ['Pizza', '20200402 4:00:00', '20200406 12:00:00', 'four_day'],
				['Avigur','20200210 4:00:00','20200211 12:00:00', 'one_day'],['Yamit', '20200503 4:00:00', '20200507 12:00:00', 'four_day'],


				['Tzuzik','20210225 4:00:00','20210226 12:00:00', 'one_day'],
				['Michi', '20210404 4:00:00', '20210405 12:00:00', 'one_day'], ['Michi', '20210311 4:00:00', '20210320 12:00:00', 'four_day'], 
				['Prince_Edward', '20210218 4:00:00', '20210225 12:00:00', 'four_day'], #['Prince_Edward', '20210118 4:00:00', '2021019 12:00:00', 'one_day'], 
				['Adva', '20210316 4:00:00', '20210317 12:00:00', 'one_day'], ['Adva', '20210401 4:00:00', '20210408 12:00:00', 'four_day'],
				['Miles', '20210316 4:00:00', '20210317 12:00:00', 'one_day'], ['Miles', '20210218 4:00:00', '20210225 12:00:00', 'four_day'],
				['Holy', '20210401 4:00:00', '20210408 12:00:00', 'four_day'], ['Raja', '20210211 4:00:00', '20210218 12:00:00', 'four_day'], 
				['Dilla', '20210517 4:00:00', '20210526 12:00:00', 'four_day'], ['Dilla', '20210502 4:00:00', '20210503 12:00:00', 'one_day'],
				['Jafar', '20210506 4:00:00', '20210513 12:00:00', 'four_day'],
				['Aria', '20210506 4:00:00', '20210513 12:00:00', 'four_day'], # fake


				]



