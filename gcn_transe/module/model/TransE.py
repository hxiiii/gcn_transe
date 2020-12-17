import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import time
from openke.module.gcn.utils import read_data,read_embedding
class TransE(Model):

	def __init__(self, model=None,ent_tot=0, rel_tot=0, dim = 100, p_norm = 1, vec_dim=20, norm_flag = True,entity_em=None,rel_em=None, margin = None, epsilon = None):
		super(TransE, self).__init__(ent_tot, rel_tot)

		self.GCN = model
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.vec_dim=vec_dim
        
		self.entity_em=entity_em
		self.rel_em=rel_em

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.line = nn.Linear(768, 100)

		self.alpha = nn.Parameter(torch.FloatTensor(self.dim))
		nn.init.uniform_(self.alpha)

		self.w_matrix = nn.Linear(self.dim, self.vec_dim)
		self.u_matrix = nn.Linear(self.dim, self.vec_dim, bias=False)
		self.v_vector = nn.Linear(self.vec_dim, 1, bias=False)

		self.all_adj,self.all_features,self.mask=read_data()

# 		sentence_embedding
		self.sentence_embedding=read_embedding()
      
		if self.entity_em is not None and self.rel_em is not None:
			self.ent_embeddings.weight.data=self.entity_em
			self.rel_embeddings.weight.data=self.rel_em
# 			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		elif margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False


	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score


	def forward(self, data):
		mode='normal'
		batch_h=data[:,0].flatten()
		batch_t=data[:,1].flatten()
		adj=self.all_adj[batch_h]
		x=self.all_features[batch_h]
		h_d=self.GCN(x.cuda(),adj.cuda())
		batch_t=data[:,1].flatten()
		adj=self.all_adj[batch_t]
		x=self.all_features[batch_t]
		t_d=self.GCN(x.cuda(),adj.cuda())
		batch_r=data[:,2].flatten()
		r = self.rel_embeddings(batch_r)
        
# 		h_d = self.dot_attention(h_d, r,self.mask[batch_h].cuda())
# 		t_d = self.dot_attention(t_d, r,self.mask[batch_t].cuda())

		h_d = self.additive_attention(h_d, r,self.mask[batch_h].cuda())
		t_d = self.additive_attention(t_d, r,self.mask[batch_t].cuda())

		self.alpha.data=torch.sigmoid(self.alpha.data)   
		h = self.alpha.mul(self.ent_embeddings(batch_h)) + (1 -self.alpha).mul(h_d+self.line.cuda()(self.sentence_embedding[batch_h].cuda()))
		t = self.alpha.mul(self.ent_embeddings(batch_t)) + (1 - self.alpha).mul(t_d+self.line(self.sentence_embedding[batch_t].cuda()))

		score = self._calc(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def evaluate(self,indices, des_embeddings):
		self.w_matrix.cpu()
		self.u_matrix.cpu()
		self.v_vector.cpu()
		r = self.rel_embeddings(indices).cpu()
		des_embeddings = des_embeddings.unsqueeze(0).cpu()
		e_s = self.ent_embeddings.weight.data.cpu()
		intermediate = self.w_matrix(r).unsqueeze(1).unsqueeze(2) + self.u_matrix(des_embeddings)
		intermediate = torch.tanh(intermediate)
		score = self.v_vector(intermediate).squeeze(-1)
		score = score.masked_fill(self.mask.cpu() == 0, -1e9)
		score = torch.softmax(score, -1)
		e_d = torch.matmul(score.unsqueeze(2), des_embeddings).squeeze(-2)
              
		alpha = F.sigmoid(self.alpha).cpu()  

		e = alpha.mul(e_s).unsqueeze(0) + (1 - alpha).mul(e_d+self.line(self.sentence_embedding.data.cpu()))     
        
		if self.norm_flag:
			e = F.normalize(e, 2, -1)
			r = F.normalize(r.squeeze(), 2, -1)
		return e, r        


	def get_embeddings(self,indices):
		adj = self.all_adj[indices]
		x = self.all_features[indices]
		e = self.GCN(x.cuda(), adj.cuda())
		return e.cpu()        
        

	def dot_attention(self,e_d,r,mask):
		r_temp=r.unsqueeze(-1)
		score=torch.div(torch.matmul(e_d,r_temp).squeeze(),e_d.shape[-1]**0.5)
		score=score.masked_fill(mask==0,-1e9)
		score=torch.softmax(score,-1)
		result=torch.matmul(score.unsqueeze(1),e_d).squeeze()
		return result

	def additive_attention(self,e_d,r,mask):
		self.w_matrix.cuda()
		self.u_matrix.cuda()
		self.v_vector.cuda()
		intermediate = self.w_matrix(r).unsqueeze(1) + self.u_matrix(e_d)
		intermediate = torch.tanh(intermediate)
		score=self.v_vector(intermediate).squeeze(2)
		score = score.masked_fill(mask == 0, -1e9)
		score = torch.softmax(score, 1)
		result = torch.matmul(score.unsqueeze(1), e_d).squeeze(1)
		return result
