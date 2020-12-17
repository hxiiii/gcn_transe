# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
# from tqdm import tqdm

class Trainer(object):

	def __init__(self, 
				 model = None,
				 data_loader = None,
				 test_dataloader=None,
				 test_dataset=None,
				 p_norm=1,
				 train_times = 1000,
				 alpha = 0.5,
				 use_gpu = True,
				 opt_method = "adam",
				 save_steps = None,
				 checkpoint_dir = None):

		self.work_threads = 8
		self.p_norm=p_norm
		self.train_times = train_times

		self.opt_method = opt_method
		self.optimizer = None
		self.lr_decay = 0
		self.weight_decay = 0
# 		self.weight_decay = 1e-5
		self.alpha = alpha

		self.model = model
		self.data_loader = data_loader
		self.test_dataloader=test_dataloader
		self.test_dataset=test_dataset
		self.use_gpu = use_gpu
		self.save_steps = save_steps
		self.checkpoint_dir = checkpoint_dir
		self.early_stop=True

	def train_one_step(self, data):
		self.optimizer.zero_grad()
		loss = self.model(data)
		loss.backward()
# 		print(self.model.model.alpha.is_leaf,self.model.model.alpha.requires_grad,self.model.model.alpha.grad)
		self.optimizer.step()		 
		return loss.cpu().item()

	def run(self):
		if self.use_gpu:
			self.model.cuda()

		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		else:
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				momentum=0.9,
				weight_decay=self.weight_decay,
			)
		print("Finish initializing...")
		min_loss= float('inf')
		early_step=0
# 		training_range = tqdm(range(self.train_times))
		for epoch in range(self.train_times):
			self.model.train()
			res = 0.0
			i=0
			for data in self.data_loader:
				i+=1
				if self.use_gpu:
					loss=self.train_one_step(data)
				else:
					loss = self.train_one_step(data)
				res += loss
				if i%100==0:
					print('epoch:%d,step:%d loss:%f'%(epoch,i,res/i))
# 			training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
			if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
				print("Epoch %d has finished, saving..." % (epoch))
				self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))
			if (epoch+1)%100==0:
				self.evaluate(epoch+1,self.checkpoint_dir)


	def evaluate(self,epoch,cheak_dir): 
		self.model.eval()
		if self.use_gpu:
			self.model.cuda()
		'''Raw'''
		head_meanrank_raw = 0
		head_hits10_raw = 0
		tail_meanrank_raw = 0
		tail_hits10_raw = 0
		'''Filter'''
		head_meanrank_filter = 0
		head_hits10_filter = 0
		tail_meanrank_filter = 0
		tail_hits10_filter = 0
		with torch.no_grad():           
			start=time.time()
			des_embeddings = self.model.get_embeddings(self.test_dataset.entity_num)
			i = 0
			ent_embeddings = []
			r_embeddings=[]
			while i < self.test_dataset.relation_num:
				e, r = self.model.model.evaluate(torch.arange(i,min(i+3,self.test_dataset.relation_num)).cuda(), des_embeddings)
				ent_embeddings.append(e)
				r_embeddings.append(r)
				i+=3
				print(time.time()-start)
			ent_embeddings = torch.cat(ent_embeddings)
			r_embeddings = torch.cat(r_embeddings)

			for data in self.test_dataloader:

				row_h, filter_h = self.cal_head(ent_embeddings, r_embeddings, data)

				head_meanrank_raw+=np.sum(row_h)
				head_meanrank_filter+=np.sum(filter_h)
				row_h=np.where(row_h>10,0,1)
				head_hits10_raw+=np.sum(row_h)
				filter_h = np.where(filter_h>10,0,1)
				head_hits10_filter+=np.sum(filter_h)
        
				row_t, filter_t = self.cal_tail(ent_embeddings, r_embeddings, data)

				tail_meanrank_raw+=np.sum(row_t)
				tail_meanrank_filter+=np.sum(filter_t)
				row_t = np.where(row_t > 10, 0, 1)
				tail_hits10_raw += np.sum(row_t)
				filter_t = np.where(filter_t > 10, 0, 1)
				tail_hits10_filter += np.sum(filter_t)
			print('-----Raw-----')
			head_meanrank_raw /= len(self.test_dataset)
			head_hits10_raw /= len(self.test_dataset)
			tail_meanrank_raw /= len(self.test_dataset)
			tail_hits10_raw /= len(self.test_dataset)
			print('-----Head prediction-----')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
			print('-----Tail prediction-----')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
			print('------Average------')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
															 (head_hits10_raw + tail_hits10_raw) / 2))
			print('-----Filter-----')
			head_meanrank_filter /= len(self.test_dataset)
			head_hits10_filter /= len(self.test_dataset)
			tail_meanrank_filter /= len(self.test_dataset)
			tail_hits10_filter /= len(self.test_dataset)
			print('-----Head prediction-----')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
			print('-----Tail prediction-----')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
			print('-----Average-----')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
															 (head_hits10_filter + tail_hits10_filter) / 2))
			if not os.path.exists(os.path.join(cheak_dir+'-'+'result.txt')):
				f=open(os.path.join(cheak_dir+'-'+ 'result.txt'), 'w', encoding='utf-8')
				f.write('******')
				f.close()
			with open(os.path.join(cheak_dir+'-'+'result.txt'),'a',encoding='utf-8') as f:
				f.write(str(epoch)+'\n')
				f.write('-----Raw-----'+'\n')
				f.write('-----Head prediction-----'+'\n')
				f.write('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw)+'\n')
				f.write('-----Tail prediction-----'+'\n')
				f.write('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw)+'\n')
				f.write('------Average------'+'\n')
				f.write('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
															 (head_hits10_raw + tail_hits10_raw) / 2)+'\n')
				f.write('-----Filter-----'+'\n')
				f.write('-----Head prediction-----'+'\n')
				f.write('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter)+'\n')
				f.write('-----Tail prediction-----'+'\n')
				f.write('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter)+'\n')
				f.write('------Average------'+'\n')
				f.write('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
															 (head_hits10_filter + tail_hits10_filter) / 2)+'\n')

	def predict(self,datas):
# 		print(len(datas))
		self.model.eval()
		if self.use_gpu:
			self.model.cuda()
		'''Raw'''
		head_meanrank_raw = 0
		head_hits10_raw = 0
		tail_meanrank_raw = 0
		tail_hits10_raw = 0
		'''Filter'''
		head_meanrank_filter = 0
		head_hits10_filter = 0
		tail_meanrank_filter = 0
		tail_hits10_filter = 0
		with torch.no_grad():
# 			ent_embeddings = self.model.get_embeddings(self.test_dataset.entity_num)

			start=time.time()
			des_embeddings = self.model.get_embeddings(self.test_dataset.entity_num)
			i = 0
			ent_embeddings = []
			r_embeddings=[]
			while i < self.test_dataset.relation_num:
				e, r = self.model.model.evaluate(torch.arange(i,min(i+3,self.test_dataset.relation_num)).cuda(), des_embeddings)
				ent_embeddings.append(e)
				r_embeddings.append(r)
				i+=3
				print(time.time()-start)
			ent_embeddings = torch.cat(ent_embeddings)
			r_embeddings = torch.cat(r_embeddings)
			f=open('1.txt','a')
			for data in torch.split(datas, 3000):
				start = time.time()
# 				h, t, r, rel_embeddings = self.model.model.evaluate(data.cuda())
# 				row_h, filter_h = self.cal_head(t, r, ent_embeddings, data)

				row_h, filter_h = self.cal_head(ent_embeddings, r_embeddings, data)
                
                
				print(time.time() - start)
				head_meanrank_raw += np.sum(row_h)
				head_meanrank_filter += np.sum(filter_h)
				row_h = np.where(row_h > 10, 0, 1)
				head_hits10_raw += np.sum(row_h)
				filter_h = np.where(filter_h > 10, 0, 1)
				head_hits10_filter += np.sum(filter_h)
                
# 				row_t, filter_t = self.cal_tail(h, r, ent_embeddings, data)
				row_t, filter_t = self.cal_tail(ent_embeddings, r_embeddings, data)
 
				tail_meanrank_raw += np.sum(row_t)
				tail_meanrank_filter += np.sum(filter_t)
				row_t = np.where(row_t > 10, 0, 1)
				tail_hits10_raw += np.sum(row_t)
				filter_t = np.where(filter_t > 10, 0, 1)
				tail_hits10_filter += np.sum(filter_t)
				print(time.time() - start)
			print('-----Raw-----')
			head_meanrank_raw /= len(datas)
			head_hits10_raw /= len(datas)
			tail_meanrank_raw /= len(datas)
			tail_hits10_raw /= len(datas)
			print('-----Head prediction-----')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
			print('-----Tail prediction-----')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
			print('------Average------')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
															 (head_hits10_raw + tail_hits10_raw) / 2))
			print('-----Filter-----')
			head_meanrank_filter /=len(datas)
			head_hits10_filter /= len(datas)
			tail_meanrank_filter /= len(datas)
			tail_hits10_filter /= len(datas)
			print('-----Head prediction-----')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
			print('-----Tail prediction-----')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
			print('-----Average-----')
			print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
															 (head_hits10_filter + tail_hits10_filter) / 2))
			f.write('------Average------'+'\n')
			f.write('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
														 (head_hits10_filter + tail_hits10_filter) / 2)+'\n')
			f.close()

	def cal_head(self, ent_embeddings, r_embeddings, data):
		row_list = []
		filter_list = []

		batch_t = data[:, 1]
		batch_r = data[:, 2]
		dis = []
		for i, r in enumerate(r_embeddings[batch_r]):
			t = ent_embeddings[batch_r[i]][batch_t[i]].cuda()
			h_p = t - r.cuda()
			dis.append(torch.norm(h_p.unsqueeze(0) - ent_embeddings[batch_r[i]].cuda(), self.p_norm, -1).unsqueeze(0))
		dis = torch.cat(dis)
        # h_p = t - r
        # dis = torch.norm(h_p.unsqueeze(1) - ent_embeddings.unsqueeze(0), self.p_norm, -1)
		for i, t in enumerate(dis):
			row = 1
			fil = 1
    # dis = torch.norm(t - ent_embeddings, self.p_norm, -1)
    # print('head-',i,dis.shape)
			_, index = torch.topk(t, self.test_dataset.entity_num, largest=False)
			for e in index.cpu():
				if e == data[i][0]:
					break
				else:
					row += 1
				if (e.item(), data[i][1].item(), data[i][2].item()) not in self.test_dataset.all_triples:
					fil += 1
			row_list.append(row)
			filter_list.append(fil)
		return np.array(row_list), np.array(filter_list)

    
	def cal_tail(self, ent_embeddings, r_embeddings, data):
		row_list = []
		filter_list = []

		batch_h = data[:, 0]
		batch_r = data[:, 2]
		dis = []
		for i, r in enumerate(r_embeddings[batch_r]):
			h = ent_embeddings[batch_r[i]][batch_h[i]].cuda()
			t_p = h + r.cuda()
			dis.append(torch.norm(t_p.unsqueeze(0) - ent_embeddings[batch_r[i]].cuda(), self.p_norm, -1).unsqueeze(0))
		dis = torch.cat(dis)


		# row_list = []
		# filter_list = []
		# t_p = h + r
		# dis = torch.norm(t_p.unsqueeze(1) - ent_embeddings.unsqueeze(0), self.p_norm, -1)
		for i, t in enumerate(dis):
			row = 1
			fil = 1
			# dis=torch.norm(t-ent_embeddings,self.p_norm,-1)
			_, index = torch.topk(t, self.test_dataset.entity_num, largest=False)
			for e in index.cpu():
				if e == data[i][1]:
					break
				else:
					row += 1
				if (data[i][0].item(), e.item(), data[i][2].item()) not in self.test_dataset.all_triples:
					fil += 1
			row_list.append(row)
			filter_list.append(fil)
		return np.array(row_list), np.array(filter_list)
    
    
	def set_model(self, model):
		self.model = model

	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_lr_decay(self, lr_decay):
		self.lr_decay = lr_decay

	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay

	def set_opt_method(self, opt_method):
		self.opt_method = opt_method

	def set_train_times(self, train_times):
		self.train_times = train_times

	def set_save_steps(self, save_steps, checkpoint_dir = None):
		self.save_steps = save_steps
		if not self.checkpoint_dir:
			self.set_checkpoint_dir(checkpoint_dir)

	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir