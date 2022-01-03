import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from args import read_args
import numpy as np
import string
import re
import math
args = read_args()


class HetAgg(nn.Module):
	def __init__(self, args, feature_list, u_neigh_list_train, c_neigh_list_train):
		super(HetAgg, self).__init__()
		embed_d = args.embed_d
		in_f_d = args.in_f_d
		self.args = args 
		self.C_n = args.C_n
		self.U_n = args.U_n

		self.feature_list = feature_list
		self.u_neigh_list_train = u_neigh_list_train
		self.c_neigh_list_train = c_neigh_list_train

			
		self.u_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.c_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)


		self.u_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.c_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)


		self.u_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)
		self.c_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)


		self.softmax = nn.Softmax(dim = 1)
		self.act = nn.LeakyReLU()
		self.drop = nn.Dropout(p = 0.5)
		self.bn = nn.BatchNorm1d(embed_d)
		self.embed_d = embed_d


	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
				nn.init.xavier_normal_(m.weight.data)
				m.bias.data.fill_(0.1)


	def u_content_agg(self, id_batch): #heterogeneous user aggregation
		embed_d = self.embed_d
		u_net_embed_batch = self.feature_list[4][id_batch]
		u_i_embed_batch = self.feature_list[6][id_batch]
		u_text_embed_batch_1 = self.feature_list[5][id_batch, :embed_d][0]
		u_text_embed_batch_2 = self.feature_list[5][id_batch, embed_d : embed_d * 2][0]
		u_text_embed_batch_3 = self.feature_list[5][id_batch, embed_d * 2 : embed_d * 3][0]
		u_text_embed_batch_4 = self.feature_list[5][id_batch, embed_d * 3: embed_d * 4][0]
		u_text_embed_batch_5 = self.feature_list[5][id_batch, embed_d * 4: embed_d * 5][0]
		u_text_embed_batch_6 = self.feature_list[5][id_batch, embed_d * 5: embed_d * 6][0]
		u_text_embed_batch_7 = self.feature_list[5][id_batch, embed_d * 6: embed_d * 7][0]
		u_text_embed_batch_8 = self.feature_list[5][id_batch, embed_d * 7: embed_d * 8][0]
		u_text_embed_batch_9 = self.feature_list[5][id_batch, embed_d * 8: embed_d * 9][0]
		u_text_embed_batch_10 = self.feature_list[5][id_batch, embed_d * 9:][0]

		concate_embed = torch.cat((u_net_embed_batch, u_i_embed_batch, u_text_embed_batch_1, u_text_embed_batch_2,\
		 u_text_embed_batch_3,u_text_embed_batch_4,u_text_embed_batch_5,u_text_embed_batch_6,u_text_embed_batch_7, \
		 u_text_embed_batch_8,u_text_embed_batch_9,u_text_embed_batch_10), 1).view(len(id_batch[0]), 12, embed_d)

		concate_embed = torch.transpose(concate_embed, 0, 1)
		all_state, last_state = self.u_content_rnn(concate_embed)

		return torch.mean(all_state, 0)


	def c_content_agg(self, id_batch):
		embed_d = self.embed_d
		c_u_embed_batch = self.feature_list[2][id_batch]
		c_t_embed_batch = self.feature_list[0][id_batch]
		c_i_embed_batch = self.feature_list[1][id_batch]
		c_net_embed_batch = self.feature_list[3][id_batch]

		concate_embed = torch.cat((c_u_embed_batch, c_t_embed_batch, c_i_embed_batch, \
								   c_net_embed_batch), 1).view(len(id_batch[0]), 4, embed_d)

		concate_embed = torch.transpose(concate_embed, 0, 1)
		all_state, last_state = self.c_content_rnn(concate_embed)

		return torch.mean(all_state, 0)


	def node_neigh_agg(self, id_batch, node_type): #type based neighbor aggregation with rnn 
		embed_d = self.embed_d

		if node_type == 1:
			batch_s = int(len(id_batch[0]) / 5)
		else:
			batch_s = int(len(id_batch[0]) / 10)

		if node_type == 1:
			neigh_agg = self.u_content_agg(id_batch).view(batch_s, 5, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.u_neigh_rnn(neigh_agg)
		elif node_type == 2:
			neigh_agg = self.c_content_agg(id_batch).view(batch_s, 10, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.c_neigh_rnn(neigh_agg)
		neigh_agg = torch.mean(all_state, 0).view(batch_s, embed_d)
		
		return neigh_agg


	def node_het_agg(self, id_batch, node_type): #heterogeneous neighbor aggregation
		if node_type == 1:
			u_neigh_batch = [[0] * 5] * len(id_batch)
			c_neigh_batch = [[0] * 10] * len(id_batch)
		if node_type == 2:
			u_neigh_batch = [[0] * 5] * len(id_batch)
			c_neigh_batch = [[0] * 10] * len(id_batch)
		for i in range(len(id_batch)):
			if node_type == 1:
				u_neigh_batch[i] = self.u_neigh_list_train[0][id_batch[i]]
				if u_neigh_batch[i]==[]:
					u_neigh_batch[i] = [i]*5
				c_neigh_batch[i] = self.u_neigh_list_train[1][id_batch[i]]
				if c_neigh_batch[i]==[]:
					print("error!comment is empty!")

			elif node_type == 2:
				u_neigh_batch[i] = self.c_neigh_list_train[0][id_batch[i]]
				c_neigh_batch[i] = self.c_neigh_list_train[1][id_batch[i]]
				if c_neigh_batch[i]==[]:
					c_neigh_batch[i] = [i]*10
				if u_neigh_batch[i]==[]:
					print("error!user is empty!")


		u_neigh_batch = np.reshape(u_neigh_batch, (1, -1))
		u_agg_batch = self.node_neigh_agg(u_neigh_batch, 1)
		c_neigh_batch = np.reshape(c_neigh_batch, (1, -1))
		c_agg_batch = self.node_neigh_agg(c_neigh_batch, 2)


		#attention module
		id_batch = np.reshape(id_batch, (1, -1))
		if node_type == 1:
			cat_agg_batch = self.u_content_agg(id_batch)
		elif node_type == 2:
			cat_agg_batch = self.c_content_agg(id_batch)


		cat_agg_batch_2 = torch.cat((cat_agg_batch, cat_agg_batch), 1).view(len(cat_agg_batch), self.embed_d * 2)
		u_agg_batch_2 = torch.cat((cat_agg_batch, u_agg_batch), 1).view(len(cat_agg_batch), self.embed_d * 2)
		c_agg_batch_2 = torch.cat((cat_agg_batch, c_agg_batch), 1).view(len(cat_agg_batch), self.embed_d * 2)


		#compute weights
		concate_embed = torch.cat((cat_agg_batch_2, u_agg_batch_2, c_agg_batch_2), 1).view(len(cat_agg_batch), 3, self.embed_d * 2)
		if node_type == 1:
			atten_w = self.act(torch.bmm(concate_embed, self.u_neigh_att.unsqueeze(0).expand(len(cat_agg_batch),\
			 *self.u_neigh_att.size())))
		elif node_type == 2:
			atten_w = self.act(torch.bmm(concate_embed, self.c_neigh_att.unsqueeze(0).expand(len(cat_agg_batch),\
			 *self.u_neigh_att.size())))
		atten_w = self.softmax(atten_w).view(len(cat_agg_batch), 1, 3)

		#weighted combination
		concate_embed = torch.cat((u_agg_batch, u_agg_batch, c_agg_batch), 1).view(len(cat_agg_batch), 3, self.embed_d)
		weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(cat_agg_batch), self.embed_d)

		return weight_agg_batch


	def het_agg(self, triple_index, c_id_batch, pos_id_batch, neg_id_batch):
		embed_d = self.embed_d
		if triple_index == 0:
			c_agg = self.node_het_agg(c_id_batch, 1)
			p_agg = self.node_het_agg(pos_id_batch, 2)
			n_agg = self.node_het_agg(neg_id_batch, 2)
		elif triple_index == 1:
			c_agg = self.node_het_agg(c_id_batch, 2)
			p_agg = self.node_het_agg(pos_id_batch, 1)
			n_agg = self.node_het_agg(neg_id_batch, 1)
		elif triple_index == 2:
			c_agg = self.node_het_agg(c_id_batch, 2)
			p_agg = self.node_het_agg(pos_id_batch, 2)
			n_agg = self.node_het_agg(neg_id_batch, 2)


		return c_agg, p_agg, n_agg


	def aggregate_all(self, triple_list_batch, triple_index):
		c_id_batch = [x[0] for x in triple_list_batch]
		pos_id_batch = [x[1] for x in triple_list_batch]
		neg_id_batch = [x[2] for x in triple_list_batch]

		c_agg, pos_agg, neg_agg = self.het_agg(triple_index, c_id_batch, pos_id_batch, neg_id_batch)

		return c_agg, pos_agg, neg_agg


	def forward(self, triple_list_batch, triple_index):
		c_out, p_out, n_out = self.aggregate_all(triple_list_batch, triple_index)
		return c_out, p_out, n_out

