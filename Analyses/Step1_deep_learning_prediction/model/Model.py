
import torch
import torch.nn as nn

import sys


hidden_size = 512
num_layers = 2
bidirectional = True
dropout = 0.6


class MyModel(nn.Module):
	def __init__(self, num_classes=2):
		super(MyModel, self).__init__()
		self.lstm = nn.LSTM(input_size=4*2, hidden_size=hidden_size, num_layers=num_layers,
							batch_first=True, bidirectional=bidirectional, dropout=dropout)

		self.w = nn.Parameter(torch.rand((1,), requires_grad=True))#.view(2,2))

		self.layers = nn.Sequential(
			nn.Linear(hidden_size*2, 200),
			nn.ReLU(True),
			nn.BatchNorm1d(200),
			nn.Dropout(dropout),

			nn.Linear(200, 200),
			nn.ReLU(True),
			nn.BatchNorm1d(200),
			nn.Dropout(dropout),

			nn.Linear(200, num_classes),
			nn.Sigmoid()
		)

		self.layers2 = nn.Sequential(
			nn.Linear(hidden_size*2, 200),
			nn.ReLU(True),

			nn.Linear(200, num_classes),
			nn.Sigmoid()
		)

	def forward(self, input1, query_frame_probs):
		'''
		input [batch_size, len]
		'''
		x, (h_n, c_n) = self.lstm(input1)
		frame_prob = self.layers2(x)

		# concat forward and backward results
		output_fw = h_n[-2, :, :] # forward
		output_bw = h_n[-1, :, :] # backward
		output = torch.cat([output_fw, output_bw], dim=-1) #[batch_size, hidden_size*2]
		global_prob = self.layers(output)

		attention_weights = nn.functional.softmax(-((query_frame_probs - frame_prob)*self.w)**2 / 2, dim=1)

		attention_weights = attention_weights.permute(0, 2, 1)
		global_prob = global_prob.unsqueeze(1).repeat(1, 6, 1)

		overall_prob = torch.bmm(attention_weights, global_prob)
		overall_prob_reduce = torch.diagonal(overall_prob, dim1=-2, dim2=-1)
		# print('overall_prob_reduce: ', overall_prob_reduce.shape)

		return overall_prob_reduce#, frame_prob

