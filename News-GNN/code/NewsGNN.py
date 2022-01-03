import torch
import torch.optim as optim
import data_generator
import tools
from args import read_args
from torch.autograd import Variable
import numpy as np
import random
torch.set_num_threads(2)
import os
import math
os.environ['CUDA_VISIBLE_DEVICES']='0'


class model_class(object):
	def __init__(self, args):
		super(model_class, self).__init__()
		self.args = args
		self.gpu = args.cuda

		input_data = data_generator.input_data(args = self.args)
		self.input_data = input_data
		print("input finish!")

		feature_list = [input_data.c_text_embed, input_data.c_info_embed, input_data.c_u_net_embed,\
		input_data.c_net_embed, input_data.u_net_embed, input_data.u_text_embed,input_data.u_info_embed]

		for i in range(len(feature_list)):
			feature_list[i] = torch.from_numpy(np.array(feature_list[i])).float()

		if self.gpu:
			for i in range(len(feature_list)):
				feature_list[i] = feature_list[i].cuda()

		u_neigh_list_train = input_data.u_neigh_list_train
		c_neigh_list_train = input_data.c_neigh_list_train


		self.model = tools.HetAgg(args, feature_list, u_neigh_list_train, c_neigh_list_train)

		if self.gpu:
			self.model.cuda()

		self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay = 0)
		self.model.init_weights()

		users = []
		for i in range(1, self.args.U_n):
			users.append(i)
		comments = []
		for i in range(1, self.args.C_n):
			comments.append(i)

		batch_count = math.ceil(len(users) / 500)
		users_out = np.empty((0, 200), float)

		for i in range(batch_count):
			if (i + 1) * 500 <= len(users):
				temp_users = users[i * 500:(i + 1) * 500]
			else:
				temp_users = users[i * 500:]
			temp_out = self.model.node_het_agg(temp_users, 1).cpu().detach().numpy()
			users_out = np.append(users_out, temp_out, axis=0)

		f_users_vector = open(self.args.data_path + "politifact_users_vector.txt", "w")
		for j in range(len(users_out)):
			f_users_vector.write(str(j + 1) + "$:$:" + self.numpy_vector_str(users_out[j]) + "\n")
		print("users ok!")
		batch_count = math.ceil(len(comments) / 500)
		comments_out = np.empty((0, 200), float)
		for i in range(batch_count):
			if (i + 1) * 500 <= len(comments):
				temp_comments = comments[i * 500:(i + 1) * 500]
			else:
				temp_comments = comments[i * 500:]
			temp_out = self.model.node_het_agg(temp_comments, 2).cpu().detach().numpy()
			comments_out = np.append(comments_out, temp_out, axis=0)
		f_comments_vector = open(self.args.data_path + "politifact_comments_vector.txt", "w")

		for j in range(len(comments_out)):
			f_comments_vector.write(str(j + 1) + "$:$:" + self.numpy_vector_str(comments_out[j]) + "\n")
		
	def numpy_vector_str(slef,n):
		s = ""
		for i in range(len(n) - 1):
			s += str(n[i]) + " "
		s += str(n[len(n) - 1])
		return s


if __name__ == '__main__':
	args = read_args()
	print("------arguments-------")
	for k, v in vars(args).items():
		print(k + ': ' + str(v))

	#fix random seed
	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed_all(args.random_seed)

	#model 
	model_object = model_class(args)


