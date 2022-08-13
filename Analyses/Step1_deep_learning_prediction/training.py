import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F 

import os, sys
from tfrecord.torch.dataset import TFRecordDataset
from model import Model

from torch.autograd import Variable


BATCH_SIZE = 300


imdb_model = Model.MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(imdb_model.parameters(), 0.001)


if os.path.exists("./model/model.pkl"):
	imdb_model.load_state_dict(torch.load("./model/model.pkl"))
	optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))

workdir = ""
tfdir = workdir + "tf_dataset/"
train_tf = tfdir + 'train.tf'
valid_tf = tfdir + 'valid.tf'

index_path = None
shuffle_queue_size = BATCH_SIZE
description = {"text": "float", "value":"float", "label": "int"} ## float
gc_mass_datasets = {}
gc_mass_datasets['train'] = TFRecordDataset(train_tf, index_path, description, shuffle_queue_size)
gc_mass_datasets['valid'] = TFRecordDataset(valid_tf, index_path, description)

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(
	gc_mass_datasets['train'],
	batch_size = BATCH_SIZE,
	)
dataloaders['valid'] = torch.utils.data.DataLoader(
	gc_mass_datasets['valid'],
	batch_size = BATCH_SIZE,
	)

def train(epoch):

	for phase in ['train']:
		if phase == 'train':
			imdb_model.train(True)

		for idx, data in enumerate(dataloaders['train']):
			inputs, values, labels = data['text'], data['value'], data['label']
			# print(inputs.shape)
			# sys.exit(1)
			labels = labels.squeeze_()
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
			# sys.exit(1)

			optimizer.zero_grad()
			outputs = imdb_model(inputs, torch.tensor(weighted_probs))

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			print(epoch, idx, loss.item())

			# if idx % 100 == 0:
			# 	torch.save(imdb_model.state_dict(), "./model/model.pkl")
			# 	torch.save(optimizer.state_dict(), "./model/optimizer.pkl")


def test(best_test_acc, index):
	test_loss = 0
	correct = 0
	sample_num = 0
	test_batch_size = 256
	
	# mode = False
	imdb_model.eval()

	# test_dataloader = get_dataloader(train=False)
	print("testing.....\n")
	with torch.no_grad():
		for phase in ['valid']:
			if phase == 'valid':
				imdb_model.train(False)


			for idx, data in enumerate(dataloaders['valid']):
				inputs, values, labels = data['text'], data['value'], data['label']


				labels = labels.squeeze_()
				print(inputs.shape)
				inputs = inputs.view(-1, 6, 4*2)

				inputs = Variable(inputs)  # .cuda()
				labels = Variable(labels).long()

				weighted_probs = []
				for each_file_values in values:
					tmps = []
					for each_value in each_file_values:
						tmps.append([1/2, each_value/2])
					weighted_probs.append(tmps)

				# optimizer.zero_grad()
				outputs = imdb_model(inputs, torch.tensor(weighted_probs))
				print("=====================")
				print("output: ", outputs)
				print("---------------------")
				print("labels: ", labels)
				print("=====================")
				test_loss += criterion(outputs, labels)
				pred = torch.max(outputs, dim=-1, keepdim=False)[-1]
				correct += pred.eq(labels.data).sum()
				sample_num += len(labels)

				print('sub correction: ', correct/len(labels))
				# sys.exit(1)

			test_acc = correct.item()/sample_num
			print("correct: ", test_acc, flush=True)
			torch.save(imdb_model.state_dict(), "./model/model_" + str(index) +".pkl")
			torch.save(optimizer.state_dict(), "./model/optimizer_" + str(index) +".pkl")

			test_loss = test_loss.data/(idx+1)
			print("loss: ", test_loss.item(),'\n', flush=True)
			if test_acc >= best_test_acc:
				print('test_acc, best_test_acc: ', test_acc, best_test_acc)
				best_test_acc = test_acc
				torch.save(imdb_model.state_dict(), "./model/model.pkl")
				torch.save(optimizer.state_dict(), "./model/optimizer.pkl")

	return best_test_acc



if __name__ == '__main__':
	torch.set_num_threads(16)
	if os.path.exists("./model/model.pkl"):
		os.remove("./model/model.pkl")
		os.remove("./model/optimizer.pkl")
	best_test_acc = 0

	for i in range(80):
		train(i)
		best_test_acc = test(best_test_acc, i)




