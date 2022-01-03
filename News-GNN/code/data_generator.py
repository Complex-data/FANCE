import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *


class input_data(object):
	def __init__(self, args):
		self.args = args

		u_c_list_train = [[] for k in range(self.args.U_n)]
		c_u_list_train = [[] for k in range(self.args.C_n)]
		c_c_list_train = [[] for k in range(self.args.C_n)]
		relation_f = ["politifact_comment_user.txt", "politifact_user_comment.txt","politifact_c_c.txt"]

		#store comments relational data
		for i in range(len(relation_f)):
			f_name = relation_f[i]
			neigh_f = open(self.args.data_path + f_name, "r")
			for line in neigh_f:
				line = line.strip()
				node_id = int(re.split(':', line)[0])
				neigh_list = re.split(':', line)[1]
				neigh_list_id = re.split(',', neigh_list)
				if f_name == 'politifact_user_comment.txt':
					for j in range(len(neigh_list_id)):
						u_c_list_train[node_id].append('c'+str(neigh_list_id[j]))
				elif f_name == 'politifact_comment_user.txt':
					for j in range(len(neigh_list_id)):
						c_u_list_train[node_id].append('u'+str(neigh_list_id[j]))
				elif f_name == 'politifact_c_c.txt':
					for j in range(len(neigh_list_id)):
						c_c_list_train[node_id].append('c'+str(neigh_list_id[j]))
			neigh_f.close()


		#user neighbor: user + comment
		c_neigh_list_train = [[] for k in range(self.args.C_n)]
		for i in range(self.args.C_n):
			c_neigh_list_train[i] += c_u_list_train[i]
			c_neigh_list_train[i] += c_c_list_train[i]


		self.c_u_list_train =  c_u_list_train
		self.u_c_list_train = u_c_list_train
		self.c_c_list_train = c_c_list_train
		self.c_neigh_list_train = c_neigh_list_train


		if self.args.train_test_label != 2:
			self.triple_sample_p = self.compute_sample_p()

			# store comment text pre-trained embedding
			c_text_embed = np.zeros((self.args.C_n, self.args.in_f_d))
			c_t_e_f = open(self.args.data_path + "politifact_comment_text.txt", "r")
			for line in islice(c_t_e_f, 0, None):
				values = line.split()
				index = int(values[0])
				embeds = np.asarray(values[1:], dtype='float32')
				c_text_embed[index] = embeds
			c_t_e_f.close()
			print("comment_text!!")
			c_info_embed = np.zeros((self.args.C_n, self.args.in_f_d))
			c_i_e_f = open(self.args.data_path + "politifact_c_info.txt", "r")
			for line in islice(c_i_e_f, 0, None):
				values = line.split()
				index = int(values[0])
				embeds = np.asarray(values[1:], dtype='float32')
				c_info_embed[index] = embeds
			c_i_e_f.close()
			print("c_info!!")
			u_info_embed = np.zeros((self.args.U_n, self.args.in_f_d))
			u_i_e_f = open(self.args.data_path + "politifact_u_info.txt", "r")
			for line in islice(u_i_e_f, 1, None):
				values = line.split()
				index = int(values[0])
				embeds = np.asarray(values[1:], dtype='float32')
				u_info_embed[index] = embeds
			u_i_e_f.close()
			print("u_info!!")

			self.c_text_embed = c_text_embed
			self.c_info_embed = c_info_embed
			self.u_info_embed = u_info_embed

			#store pre-trained network/content embedding
			u_net_embed = np.zeros((self.args.U_n, self.args.in_f_d))
			c_net_embed = np.zeros((self.args.C_n, self.args.in_f_d))
			net_e_f = open(self.args.data_path + "politifact_node_embedding.txt", "r")
			for line in islice(net_e_f, 1, None):
				line = line.strip()
				index = re.split(' ', line)[0]
				if len(index) and (index[0] == 'u' or index[0] == 'c'):
					embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
					if index[0] == 'u':
						u_net_embed[int(index[1:])] = embeds
					else:
						c_net_embed[int(index[1:])] = embeds
			net_e_f.close()
			print("node_embedding!!")

			c_u_net_embed = np.zeros((self.args.C_n, self.args.in_f_d))
			for i in range(self.args.C_n):
				if len(c_u_list_train[i]):
					for j in range(len(c_u_list_train[i])):
						u_id = int(c_u_list_train[i][j][1:])
						c_u_net_embed[i] = np.add(c_u_net_embed[i], u_net_embed[u_id])
					c_u_net_embed[i] = c_u_net_embed[i] / len(c_u_list_train[i])
			print("c_u_net!!")
		
			u_text_embed = np.zeros((self.args.U_n, self.args.in_f_d * 10))
			for i in range(self.args.U_n):
				if len(u_c_list_train[i]):
					feature_temp = []
					if len(u_c_list_train[i]) >= 10:
						for j in range(10):
							feature_temp.append(c_text_embed[int(u_c_list_train[i][j][1:])])
					else:
						for j in range(len(u_c_list_train[i])):
							feature_temp.append(c_text_embed[int(u_c_list_train[i][j][1:])])
						for k in range(len(u_c_list_train[i]), 10):
							feature_temp.append(c_text_embed[int(u_c_list_train[i][-1][1:])])

					feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
					u_text_embed[i] = feature_temp
			print("u_text!!")
			
			self.c_u_net_embed = c_u_net_embed
			self.c_net_embed = c_net_embed
			self.u_net_embed = u_net_embed
			self.u_text_embed = u_text_embed

			#store neighbor set from random walk sequence 
			u_neigh_list_train = [[[] for i in range(self.args.U_n)] for j in range(2)]
			c_neigh_list_train = [[[] for i in range(self.args.C_n)] for j in range(2)]

			het_neigh_train_f = open(self.args.data_path + "het_neigh_train.txt", "r")
			for line in het_neigh_train_f:
				line = line.strip()
				node_id = re.split(':', line)[0]
				neigh = re.split(':', line)[1]
				neigh_list = re.split(',', neigh)
				if node_id[0] == 'u' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'u':
							u_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'c':
							u_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif node_id[0] == 'c' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'u':
							c_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						if neigh_list[j][0] == 'c':
							c_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
			het_neigh_train_f.close()
			print("het_neigh!!")

			u_neigh_list_train_top = [[[] for i in range(self.args.U_n)] for j in range(2)]
			c_neigh_list_train_top = [[[] for i in range(self.args.C_n)] for j in range(2)]

			top_k = [5, 10] #fix each neighor type size
			for i in range(self.args.U_n):
				for j in range(2):
					u_neigh_list_train_temp = Counter(u_neigh_list_train[j][i])
					top_list = u_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 0:
						neigh_size = 5
					else:
						neigh_size = 10
					for k in range(len(top_list)):
						u_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(u_neigh_list_train_top[j][i]) and len(u_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(u_neigh_list_train_top[j][i]), neigh_size):
							u_neigh_list_train_top[j][i].append(random.choice(u_neigh_list_train_top[j][i]))
			top_k = [5, 10]
			for i in range(self.args.C_n):
				for j in range(2):
					c_neigh_list_train_temp = Counter(c_neigh_list_train[j][i])
					top_list = c_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 0:
						neigh_size = 5
					else:
						neigh_size = 10
					for k in range(len(top_list)):
						c_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(c_neigh_list_train_top[j][i]) and len(c_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(c_neigh_list_train_top[j][i]), neigh_size):
							c_neigh_list_train_top[j][i].append(random.choice(c_neigh_list_train_top[j][i]))

			u_neigh_list_train[:] = []
			c_neigh_list_train[:] = []


			self.u_neigh_list_train = u_neigh_list_train_top
			self.c_neigh_list_train = c_neigh_list_train_top



	def het_walk_restart(self):
		u_neigh_list_train = [[] for k in range(self.args.U_n)]
		c_neigh_list_train = [[] for k in range(self.args.C_n)]

		#generate neighbor set via random walk with restart
		node_n = [self.args.U_n, self.args.C_n]
		for i in range(2):
			for j in range(node_n[i]):
				print(j)
				if i == 0:
					neigh_temp = self.u_c_list_train[j]
					neigh_train = u_neigh_list_train[j]
					curNode = "u" + str(j)
				elif i == 1:
					neigh_temp = self.c_u_list_train[j]
					neigh_train = c_neigh_list_train[j]
					curNode = "c" + str(j)
				if len(neigh_temp):
					neigh_L = 0
					a_L = 0
					p_L = 0
					while neigh_L < 50: 
						rand_p = random.random() 
						if rand_p > 0.5:
							if curNode[0] == "c":
								curNode = random.choice(self.c_neigh_list_train[int(curNode[1:])])
								if curNode[0] == 'u' and a_L < 16: 
									neigh_train.append(curNode)
									neigh_L += 1
									a_L += 1
								elif curNode[0] == 'c' and p_L < 36:
									neigh_train.append(curNode)
									neigh_L += 1
									p_L += 1

							elif curNode[0] == "u":
								curNode = random.choice(self.u_c_list_train[int(curNode[1:])])
								if curNode[0] == 'c' and p_L < 36:
									neigh_train.append(curNode)
									neigh_L += 1
									p_L += 1
						else:
							if i == 0:
								curNode = ('u' + str(j))
							elif i == 1:
								curNode = ('c' + str(j))

		for i in range(2):
			for j in range(node_n[i]):
				if i == 0:
					u_neigh_list_train[i] = list(u_neigh_list_train[i])
				elif i == 1:
					c_neigh_list_train[j] = list(c_neigh_list_train[j])


		neigh_f = open(self.args.data_path + "het_neigh_train.txt", "w")
		for i in range(2):
			for j in range(node_n[i]):
				if i == 0:
					neigh_train = u_neigh_list_train[j]
					curNode = "u" + str(j)
				elif i == 1:
					neigh_train = c_neigh_list_train[j]
					curNode = "c" + str(j)
				if len(neigh_train):
					neigh_f.write(curNode + ":")
					for k in range(len(neigh_train) - 1):
						neigh_f.write(neigh_train[k] + ",")
					neigh_f.write(neigh_train[-1] + "\n")
		neigh_f.close()


	def compute_sample_p(self):
		print("computing sampling ratio for each kind of double ...")
		window = self.args.window
		walk_L = self.args.walk_L
		C_n = self.args.C_n
		U_n = self.args.U_n

		total_triple_n = [0.0] * 3 
		het_walk_f = open(self.args.data_path + "politifact_random_walk.txt", "r")
		centerNode = ''
		neighNode = ''

		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'u':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'c':
									total_triple_n[0] += 1
					elif centerNode[0]=='c':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'u':
									total_triple_n[1] += 1
								elif neighNode[0] == 'c':
									total_triple_n[2] += 1

		het_walk_f.close()

		for i in range(len(total_triple_n)):
			total_triple_n[i] = self.args.batch_s / total_triple_n[i]
		print("sampling ratio computing finish.")
		print("total_triple_n 0:", total_triple_n[0])
		print("total_triple_n 1:", total_triple_n[1])
		print("total_triple_n 2:", total_triple_n[2])
		return total_triple_n


	def sample_het_walk_triple(self):
		print ("sampling triple relations ...")
		triple_list = [[] for k in range(3)]
		window = self.args.window
		walk_L = self.args.walk_L
		U_n = self.args.U_n
		C_n = self.args.C_n

		triple_sample_p = self.triple_sample_p 

		het_walk_f = open(self.args.data_path + "politifact_random_walk.txt", "r")
		centerNode = ''
		neighNode = ''
		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'u':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'c' and random.random() < triple_sample_p[1]:
									negNode = random.randint(0, C_n - 1)
									while len(self.c_u_list_train[negNode]) == 0:
										negNode = random.randint(0, C_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[0].append(triple)
					elif centerNode[0]=='c':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'u' and random.random() < triple_sample_p[1]:
									negNode = random.randint(0, U_n - 1)
									while len(self.u_c_list_train[negNode]) == 0:
										negNode = random.randint(0, U_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[1].append(triple)
								elif neighNode[0] == 'c' and random.random() < triple_sample_p[2]:
									negNode = random.randint(0, C_n - 1)
									while len(self.c_u_list_train[negNode]) == 0:
										negNode = random.randint(0, C_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[2].append(triple)
		het_walk_f.close()

		return triple_list

