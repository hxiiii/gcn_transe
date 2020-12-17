from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
import json
from collections import Counter


class TrainDataSet(Dataset):
    def __init__(self, triples_path, entity2id_path, relation2id_path):
        super(TrainDataSet,self).__init__()
        self.triples_path = triples_path
        self.entity2id_path = entity2id_path
        self.relation2id_path = relation2id_path
        self.entity_num = 0
        self.relation_num = 0
        self.entity_list = []
        self.relation_list = set()
        self.entity2id_dict = {}
        self.relation2id_dict = {}
        self.triples = []
        self.read_triples()

    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, item):
        return np.array(self.triples[item], dtype=np.int64)

#     def __getitem__(self, item):
#         return np.array(self.triples[item][0:3],dtype=np.int64), np.array(self.triples[item][3:6],dtype=np.int64)

    def read_entity(self):
        entity_df = pd.read_csv(self.entity2id_path, sep='\t', header=None)
        self.entity2id_dict = dict(zip(entity_df[0], entity_df[1]))
        self.entity_num = len(self.entity2id_dict)
        self.entity_list = list(self.entity2id_dict.values())

    def get_entity_num(self):
        return self.entity_num

    def read_relation(self):
        relation_df = pd.read_csv(self.relation2id_path, sep='\t', header=None)
        self.relation2id_dict = dict(zip(relation_df[0], relation_df[1]))
        self.relation_num = len(self.relation2id_dict)

    def get_relation_num(self):
        return self.relation_num

    def read_triples(self):
        self.read_entity()
        self.read_relation()
        triples_df = pd.read_csv(self.triples_path, sep='\t', header=None)
        for i in range(len(triples_df)):
            sample = (self.entity2id_dict[triples_df.iloc[i, 0]], self.entity2id_dict[triples_df.iloc[i, 1]],
                      self.relation2id_dict[triples_df.iloc[i, 2]])
#             sample = sample + self.neg_sample(sample)
            self.relation_list.add(triples_df.iloc[i, 2])
            self.triples.append(sample)
    
    
    def calculate_head_tail_proportion(self):
        # You should sort first before groupby!
        tripleList=sorted(self.triples,key=lambda x: x[-1])
        grouped = [(k, list(g)) for k, g in groupby(tripleList, key=lambda x:x[-1])]
        num_of_relations = len(grouped)
        head_per_tail_list = [0] * num_of_relations
        tail_per_head_list = [0] * num_of_relations
        for elem in grouped:
            headList = []
            tailList = []
            for triple in elem[1]:
                headList.append(triple[0])
                tailList.append(triple[1])
            headSet = set(headList)
            tailSet = set(tailList)
            head_per_tail = len(headList) / len(tailSet)
            tail_per_head = len(tailList) / len(headSet)
            head_per_tail_list[elem[0]] = head_per_tail
            tail_per_head_list[elem[0]] = tail_per_head
        with open(os.path.join('./FB15k',  'head_tail_proportion.pkl'), 'wb') as fw:
            pickle.dump(tail_per_head_list, fw)
            pickle.dump(head_per_tail_list, fw)
            
            
    def get_sparse_entity(self,n):
#         sparse_triples=[]
        sparse_entities={}
#         sparse_entities=[]
        l=[]
#         s=set()
        for t in self.triples:
            l.extend(t[:2])
#             s.add(t[0])
#             s.add(t[1])
        c=Counter(l)
        for key in c.keys():
            if c[key]==n:
                sparse_entities.setdefault(n,[]).append(key)
#                 sparse_entities.append(key)
#         for t in self.triples:
#             for e in sparse_entities[n]:
#                 if e==t[0] or e==t[1]:
#                     sparse_triples.append(t)
#         with open('sparse_entities.txt','w') as fp:
#             json.dump(sparse_entities,fp)
#         print('sparse entities num:{},triples num:{}'.format(len(sparse_entities[n]),len(sparse_triples)))
        return sparse_entities
#         return np.array(sparse_triples, dtype=np.int64)


    def neg_sample(self, pos):
        rd = random.uniform(0, 1)
        h = pos[0]
        t = pos[1]
        r = pos[2]
        if rd < 0.5:
            h_neg = random.choice(self.entity_list)
            return h_neg, t, r
        else:
            t_neg = random.choice(self.entity_list)
            return h, t_neg, r


# t = TrainDataSet('./FB15k/T.txt', './FB15k/entity2id.txt', './FB15k/R.txt')
# print(len(t))
# print(t[0], t[1])
