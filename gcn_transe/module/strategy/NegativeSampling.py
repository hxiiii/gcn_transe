from .Strategy import Strategy
import torch
import os
import pickle
import random

class NegativeSampling(Strategy):

	def __init__(self, model = None, loss = None, entity_num=14905,all_triples=None, dataset=None,regul_rate = 0.0, l3_regul_rate = 0.0):
		super(NegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		self.entity_num = entity_num
		self.all_triples=all_triples
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate
		with open(os.path.join('/home/jsj201-1/mount1/hx/gcn_transe/OpenKE-OpenKE-PyTorch/openke/data/'+dataset,  'head_tail_proportion.pkl'), 'rb') as fr:
			self.tail_per_head = pickle.load(fr)
			self.head_per_tail = pickle.load(fr)

	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

	def forward(self, data):
		pos=data
# 		neg=self.neg_sample(pos)
		neg=self.neg_sample_bernoulli(pos)
# 		pos,neg=data
		# score = self.model(data)
		# p_score = self._get_positive_score(score)
		# n_score = self._get_negative_score(score)
		p_score=self.model(pos.cuda())
		n_score=self.model(neg.cuda())
		loss_res = self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res

	def neg_sample(self, pos):
		neg=pos.clone().detach()
		index=torch.rand(pos.shape[0])
		sample=torch.randint(self.entity_num,(pos.shape[0],))
		neg[:,0]=torch.where(index>0.5,pos[:,0],sample)
		neg[:,1]=torch.where(index<0.5,pos[:,1],sample)
		return neg
    
	def neg_sample_bernoulli(self,pos):
		neg=[]
		for triple in pos:
			rel=triple[2]
			split = self.tail_per_head[rel] / (self.tail_per_head[rel] + self.head_per_tail[rel])
			random_number = random.random()
			if random_number < split:
				neg.append(self.corrupt_head_filter(triple))
			else:
				neg.append(self.corrupt_tail_filter(triple))
		return torch.cat(neg).reshape(-1,3)

	def corrupt_head_filter(self,triple):
		newTriple = triple.clone().detach()
		while True:
			newHead = random.randrange(self.entity_num)
# 			break
			if (newHead, newTriple[1].item(), newTriple[2].item()) not in self.all_triples:
				break
		newTriple[0] = newHead
		return newTriple

	def corrupt_tail_filter(self,triple):
		newTriple = triple.clone().detach()
		while True:
			newTail = random.randrange(self.entity_num)
# 			break
			if (newTriple[0].item(), newTail, newTriple[2].item()) not in self.all_triples:
				break
		newTriple[1] = newTail
		return newTriple
    
	def get_embeddings(self,length):
		print(length)
		i=0
		ent_embeddings=[]
		while i<length:
			ent_embeddings.append(self.model.get_embeddings(torch.arange(i,min(i+2000,length)).cuda()))
			i += 2000
		return torch.cat(ent_embeddings)