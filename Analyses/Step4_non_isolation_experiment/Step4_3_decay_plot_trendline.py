import numpy as np 
import pandas as pd 
import random
import sys
import math 
import scipy.optimize as sci

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy import stats
from matplotlib import rcParams
import itertools
from scipy.stats import mannwhitneyu

plt.rcParams.update({'font.size': 14})

plt.rcParams["font.family"] = "Arial"

def generate_means(df, x, number):
	seqs = []
	quarters = []
	for i in range(0, number):
		means = [1]
		random_n = random.sample(list(df.index), int((len(df)/1.2)+1))
		df_new = df.iloc[random_n]
		for j in range(0,13):
			means.append(df_new['night_bin_%s' % j].mean())

		seqs.append(means)

		print(means)
		# sys.exit()
		try:
			popt, _ = sci.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  means)
			my_quarter = math.log(0.5/popt[0], math.e)/popt[1]
			# print(my_quarter)
			quarters.append(my_quarter)
		except:
			pass

	# print(quarters)
	# sys.exit(1)
	return seqs, quarters


def plotting(revisit_groups, nonvisit_groups, color='gray'):
	# data = [[i for i in revisit_groups if i>2.7], [i for i in nonvisit_groups if i<6]]
	data = [revisit_groups, nonvisit_groups]

	# m1 = [np.mean([i for i in revisit_groups if i>2.5]), np.mean([i for i in nonvisit_groups if i<6])]
	m1 = [np.mean(revisit_groups), np.mean(nonvisit_groups)]
	st1 = [np.std(revisit_groups), np.std(nonvisit_groups)]

	fig, ax = plt.subplots()
	red_circle = dict(markerfacecolor='red', marker='+', markeredgecolor='red')
	bp = ax.boxplot(data, medianprops={"linewidth": 1,"solid_capstyle": "butt", 'color': 'r'},flierprops=red_circle)


	# for item in ['medians']:
		

	for i, line in enumerate(bp['medians']):
		x, y = line.get_xydata()[1]
		text = ' μ={:.2f}\n σ={:.2f}'.format(m1[i], st1[i])
		ax.annotate(text, xy=(x, y))
	    # plt.setp(x, color='gray')

	ax.set_ylabel('Half-life-time [night]', fontweight='bold')
	ax.set_xticklabels(('Fruit trees', 'Nectar trees'), fontweight='bold')


	# plt.ylim(0,6)
	plt.show()

	P_value = stats.ttest_ind(revisit_groups, nonvisit_groups, equal_var = False, alternative='less')
	U, p = mannwhitneyu(revisit_groups, nonvisit_groups)
	print("P_value, p: ", P_value, p)

def func(x, a, b):
	return a*np.exp(b*x)


def generate_decay(seqs, x):
	seqs = np.array(seqs)

	means = []
	ups = []
	downs = []


	for i in range(14):
		mean_tmp = np.mean(seqs[:,i])
		alpha = 0.95
		p = ((1.0-alpha)/2.0) * 100
		up_tmp =  np.percentile(seqs[:,i], p)
		p = (alpha+((1.0-alpha)/2.0)) * 100
		down_tmp =  np.percentile(seqs[:,i], p)

		# std = np.std(seqs[:,i])/mean_tmp
		# print('std: ', std)
		means.append(mean_tmp)
		ups.append(up_tmp)
		downs.append(down_tmp)


	popt, _ = sci.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  means)
	popt_up, _ = sci.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  ups)
	popt_down, _ = sci.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  downs)
	print("a, b:", popt[0], popt[1])
	x_linspace = np.linspace(min(x), max(x), 100)
	exp = popt[0]*np.exp(popt[1]*x_linspace)
	exp_u = popt_up[0]*np.exp(popt_up[1]*x_linspace)
	exp_d = popt_down[0]*np.exp(popt_down[1]*x_linspace)

	print("I ma here!!!")
	x_linspace = np.linspace(min(x), max(x), 14)
	print('len(means): ', len(means))
	residuals = np.array(means) - popt[0]*np.exp(popt[1]*x_linspace)
	ss_res = np.sum(residuals**2)
	ss_tot = np.sum((means-np.mean(means))**2)
	r_squared = 1 - (ss_res / ss_tot)
	print("r_squared: ", r_squared)

	return exp, exp_u, exp_d


# length = [[1],[2,3],[4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17],[18,19,20]]
# length = [[1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17],[18,19,20],[21,22,23]]
# x = [0, 1.5, 4 ,7, 10, 13, 16, 19]
# x = [0, 1.5, 4, 7, 9, 12, 15, 18]
# x = [0, 1.5, 3.5, 5.5, 7.5,9.5, 11.5, 14]


# x = [0, 1, 2 ,4, 7, 10, 13, 16]
# x = [0,1,2,3,4.5,6.5,8.5,10.5]

# x = [1, 4 ,7, 10, 15.5, 17.5, 22]
x = [0, 1.5, 4 ,7, 9.5, 11, 12,13,14,15,16,17,18,19]




file_n = 'Step4_2_nectar2.csv'
df_n = pd.read_csv(file_n)
# df_n = df_n.replace(np.nan, 0)
print(df_n)

file_f = 'Step4_2_fruit2.csv'
df_f = pd.read_csv(file_f)
# df_f = df_f.replace(np.nan, 0)

means_n, quarters_n = generate_means(df_n, x, 100)
means_f, quarters_f = generate_means(df_f, x, 100)
# print("means_n, means_f: ", means_n, means_f)
# sys.exit(1)
plotting(quarters_f, quarters_n)


fruit_exp, fruit_exp_u, fruit_exp_d = generate_decay(means_f, x)
nectar_exp, nectar_exp_u, nectar_exp_d = generate_decay(means_n, x)
x_linspace = np.linspace(min(x), max(x), 100)

plt.plot(x_linspace, fruit_exp, label='Fruit', c='red')
plt.plot(x_linspace, nectar_exp, label='Nectar', c='blue')


plt.fill_between(x_linspace, nectar_exp_u, nectar_exp_d, alpha=0.5, facecolor='gray',linestyle='--')
plt.fill_between(x_linspace, fruit_exp_u, fruit_exp_d, alpha=0.5, facecolor='gray',linestyle='--')
# plt.grid(True)
# xaxis.grid(True)
plt.ylabel("Mean of Revisiting Rate", fontweight='bold')
plt.xlabel("Time [Night]", fontweight='bold')
# plt.ylim(0, 1)
# plt.yticks(np.arange(0, 1.1,0.25)) 
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(1e-2, 1)
# plt.xlim(0,10)
legend = plt.legend(loc='upper right',facecolor='white', framealpha=1,edgecolor='white')

plt.show()




