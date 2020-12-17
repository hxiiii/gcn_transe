import numpy as np
import pandas as pd
import random
import json
from .TrainDataSet import TrainDataSet

class TestDataSet(TrainDataSet):
    def __init__(self,triples_path, entity2id_path, relation2id_path,valid_path,test_path,valid=True):
        super(TestDataSet,self).__init__(triples_path, entity2id_path, relation2id_path)
        self.valid_path=valid_path
        self.test_path=test_path
        self.valid_triples=[]
        self.test_triples=[]
        self.valid=valid
        self.read_valid()
        self.read_test()
        self.all_triples = set(self.triples)|set(self.valid_triples)|set(self.test_triples)

    def __len__(self):
        if self.valid:
            return len(self.valid_triples)
        return len(self.test_triples)

    def __getitem__(self,item):
        if self.valid:
            return np.array(self.valid_triples[item], dtype=np.int64)
        return np.array(self.test_triples[item], dtype=np.int64)

    def read_valid(self):
        triples_df = pd.read_csv(self.valid_path, sep='\t', header=None)
        for i in range(len(triples_df)):
            if triples_df.iloc[i,0] not in self.entity2id_dict or triples_df.iloc[i,1] not in self.entity2id_dict or triples_df.iloc[i, 2] not in self.relation2id_dict:
                continue
            sample = (self.entity2id_dict[triples_df.iloc[i, 0]], self.entity2id_dict[triples_df.iloc[i, 1]],
                        self.relation2id_dict[triples_df.iloc[i, 2]])
            self.valid_triples.append(sample)

    def read_test(self):
        triples_df = pd.read_csv(self.test_path, sep='\t', header=None)
        for i in range(len(triples_df)):
            if triples_df.iloc[i, 0] not in self.entity2id_dict or triples_df.iloc[i, 1] not in self.entity2id_dict or triples_df.iloc[i, 2] not in self.relation2id_dict:
                continue
            sample = (self.entity2id_dict[triples_df.iloc[i, 0]], self.entity2id_dict[triples_df.iloc[i, 1]],
                        self.relation2id_dict[triples_df.iloc[i, 2]])
            self.test_triples.append(sample)
            
    def choose_sparse_entity(self, n):
        sparse_entities=self.get_sparse_entity(n)
#         with open('sparse_entities.txt', 'r') as fp:
#             d = json.load(fp)
        sparse_triples=[]
        if self.valid:
            triples = self.valid_triples
        else:
            triples = self.test_triples
        for t in triples:
            for e in sparse_entities[n]:
                if e == t[0] or e == t[1]:
                    sparse_triples.append(t)
        print('sparse entities num:{},triples num:{}'.format(len(sparse_entities[n]),len(sparse_triples)))
        return np.array(sparse_triples, dtype=np.int64)

# if __name__=='__main__':
#     print(TestDataSet().entity2id_dict)
