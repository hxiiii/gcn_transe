# import gcn_transe
import sys
import torch


from gcn_transe.config import Trainer
from gcn_transe.module.model import TransE
from gcn_transe.module.loss import MarginLoss,SigmoidLoss
from gcn_transe.module.strategy import NegativeSampling
from gcn_transe.data.TrainDataSet import *
from gcn_transe.data.TestDataSet import *
from torch.utils.data import DataLoader
from gcn_transe.module.gcn.GCN import *
import argparse


argparser=argparse.ArgumentParser()
# input file path
argparser.add_argument('-tp','--triples_path',help="train triples file",type=str,default='../gcn_transe/data/WN18/train.txt')
argparser.add_argument('-ep','--entity2id_path',help="entity file",type=str,default='../gcn_transe/data/WN18/entity2id.txt')
argparser.add_argument('-rp','--relation2id_path',help="relation  file",type=str,default='../gcn_transe/data/WN18/relation2id.txt')
argparser.add_argument('-vp','--valid_path',help="valid triples file",type=str,default='../gcn_transe/data/WN18/valid.txt')
argparser.add_argument('-ts','--test_path',help="test triples file",type=str,default='../gcn_transe/data/WN18/test.txt')
# Transe model
argparser.add_argument('-ed','--embeddingdim',help='embedding dimention',type=int,default=50)
argparser.add_argument('-ln','--L_norm',help='distance L1 or L2',type=int,default=1)
argparser.add_argument('-mg','--margin',help='margin ',type=float,default=1.0)
argparser.add_argument('-bs','--batch_size',help='batch size',type=int,default=512)
argparser.add_argument('-e','--epochs',help='epoch times',type=int,default=4900)
argparser.add_argument('-ug','--use_gpu',help='use gpu',action='store_true',default=True)
argparser.add_argument('-lr','--learning_rate',help='learning rate',type=float,default=0.001)
# GCN model
argparser.add_argument('-nf','--nfeat',help='GCN input dimention',type=int,default=100)
argparser.add_argument('-hd','--nhid',help='GCN hidden size',type=int,default=20)
argparser.add_argument('-nc','--nclass',help='GCN output dimention',type=int,default=50)
argparser.add_argument('-dp','--dropout',help='dropout',type=float,default='0.5')
argparser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
argparser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')

argparser.add_argument('-se','--save_epoch',type=int,default=100)
argparser.add_argument('-cd','--checkpoint_dir',type=str,default='./checkpoint_WN18/WN18_checkpoint_transe_enhenced_gcn_one_sigmoid_glove_additive_attention_bernoulli_residual_50_margin_1.0')
# ./checkpoint_transe_gcn_one_sigmoid
args=argparser.parse_args()
args.use_gpu=torch.cuda.is_available() and args.use_gpu
print(args.use_gpu,torch.cuda.is_available())

# dataloader for training

train_dataset=TrainDataSet(args.triples_path, args.entity2id_path, args.relation2id_path)
train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)

# dataloader for test
test_dataset=TestDataSet(args.triples_path, args.entity2id_path, args.relation2id_path,args.valid_path,args.test_path)
test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)

# load pretrained embedding
# path='./checkpoint_WN18/WN18_checkpoint_transe_50-999.ckpt'
# paras=torch.load(path)
# entity_embedding=paras['model.ent_embeddings.weight']
# relation_embedding=paras['model.rel_embeddings.weight']

GCN_model=GCN(args.nfeat, args.nhid, args.embeddingdim, args.dropout)
# GCN_model=GAT(args.nfeat, args.nhid, args.embeddingdim, args.dropout,nheads=args.nb_heads, alpha=args.alpha)


# define the model
transe = TransE(
	model=GCN_model,
	ent_tot = train_dataset.get_entity_num(),
	rel_tot = train_dataset.get_relation_num(),
	dim = args.embeddingdim,
	p_norm = args.L_norm,
	norm_flag = True,
	entity_em=entity_embedding,
	rel_em=relation_embedding
)


# define the loss function
model = NegativeSampling(
	model = transe, 
# 	loss = SigmoidLoss(),    
	loss = MarginLoss(margin = args.margin),
	entity_num=train_dataset.get_entity_num(),
	all_triples=test_dataset.all_triples,
	dataset='WN18'
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, test_dataloader=test_dataloader,test_dataset=test_dataset,p_norm = args.L_norm,train_times = args.epochs, alpha = args.learning_rate, use_gpu = args.use_gpu,save_steps =args.save_epoch ,checkpoint_dir= args.checkpoint_dir)
trainer.run()
