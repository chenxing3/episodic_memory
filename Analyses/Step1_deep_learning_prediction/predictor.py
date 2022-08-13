


import torch
import torch.nn as nn
from torch.optim import Adam
# from Imdb import get_dataloader
import torch.nn.functional as F 
# import lib
import os, sys

from tfrecord.torch.dataset import TFRecordDataset
from model import Model

from torch.autograd import Variable


import numpy as np




class Predictor:
	def __init__(self, tf_file):

		self.BATCH_SIZE = 300
		self.imdb_model = Model.MyModel()
		self.optimizer = Adam(self.imdb_model.parameters(), 0.001)


		if os.path.exists("./model/model.pkl"):
			self.imdb_model.load_state_dict(torch.load("./model/model.pkl"))
			self.optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))
			print('loading the model!!!')
		else:
			print('not loading the model!!!')


		index_path = None
		shuffle_queue_size = self.BATCH_SIZE
		description = {"text": "float", "value":"float", "label": "int"} ## float
		gc_mass_datasets = {}
		gc_mass_datasets['test'] = TFRecordDataset(tf_file, index_path, description)

		self.dataloaders = {}
		self.dataloaders['test'] = torch.utils.data.DataLoader(
			gc_mass_datasets['test'],
			batch_size = self.BATCH_SIZE,
			)

	def main(self):

		
		with torch.no_grad():
			self.imdb_model.eval()
			preds = []
			for idx, data in enumerate(self.dataloaders['test']):
				inputs, values, labels = data['text'], data['value'], data['label']
				labels = labels.squeeze_()

				labels = labels.squeeze_()
				# print(inputs.shape)
				inputs = inputs.view(-1, 6, 4*2)

				inputs = Variable(inputs)  # .cuda()
				labels = Variable(labels).long()

				weighted_probs = []
				for each_file_values in values:
					tmps = []
					for each_value in each_file_values:
						tmps.append([1/2, each_value/2])
					weighted_probs.append(tmps)
				# print('weighted_probs: ', weighted_probs)
				# print("inputs: ", inputs)
				# sys.exit()
				# optimizer.zero_grad()
				outputs = self.imdb_model(inputs, torch.tensor(weighted_probs))
				# print("output: ", outputs)
				# print("---------------------")
				pred_tmp = torch.max(outputs, dim=-1, keepdim=False)[-1]
				preds.extend(pred_tmp.numpy())
		# print(preds, len(preds))

		# sys.exit()
		return preds#, np.array(weighted_probs)[:,1]


