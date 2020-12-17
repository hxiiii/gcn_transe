import numpy as np
import scipy.sparse as sp
import torch
import time

# dataset='WN18'
dataset='FB15k'
# FB15k 14904 300
# WN18 40943 100
sentence_path='/home/jsj201-1/mount1/hx/gcn_transe/OpenKE-OpenKE-PyTorch/openke/data/'+dataset+'/sentence_embedding/embedding.npy'

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx=mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
#     mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def read_data(num=14904,max_len=300):
    features_list=[]
    adj_list=[]
    mask=np.zeros((num,max_len))
    for id in range(num):
        features_list.append(np.load('/home/jsj201-1/mount1/hx/gcn_transe/OpenKE-OpenKE-PyTorch/openke/data/'+dataset+'/data_embedding/glove_%d.npz'%id)['arr_0'].astype(np.float32))
        adj = sp.coo_matrix(np.load('/home/jsj201-1/mount1/hx/gcn_transe/OpenKE-OpenKE-PyTorch/openke/data/'+dataset+'/glove_data_embedding/enhenced_%d.npy' % id))
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_list.append(adj.toarray().astype(np.float32))
    for i,feature in enumerate(features_list):
        if feature.shape[0]<max_len:
            feature=np.vstack((feature,np.zeros((max_len-feature.shape[0],feature.shape[1]),dtype=np.float32)))
            mask[i,:feature.shape[0]]=1
        else:
            feature=feature[:max_len,:]
            mask[i]=1
        features_list[i] = feature
    for i,adj in enumerate(adj_list):
        if adj.shape[0]<max_len:
            adj=np.vstack((adj,np.zeros((max_len-adj.shape[0],adj.shape[1]),dtype=np.float32)))
            adj=np.hstack((adj,np.zeros((max_len,max_len-adj.shape[1]),dtype=np.float32)))
        else:
            adj=adj[:max_len,:max_len]
        adj_list[i]=adj
    adj=torch.from_numpy(np.stack(adj_list)).cuda()
    features=torch.from_numpy(np.stack(features_list)).cuda()
    mask=torch.from_numpy(mask).cuda()
    return adj,features,mask



def read_embedding(path=sentence_path):
    return torch.from_numpy(np.load(path)).cuda()
