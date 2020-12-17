import numpy as np
import pandas as pd
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from bert_serving.client import BertClient
'''转换向量过程'''
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

# 已有的glove词向量
# glove_file = './FB15k/glove.6B.100d.txt'
# 指定转化为word2vec格式后文件的位置
tmp_file = "./FB15k/word2vec.txt"
# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec(glove_file, tmp_file)

# 加载转化后的文件
wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
# 使用gensim载入word2vec词向量

vocab_size=len(wvmodel.vocab)
vector_size=wvmodel.vector_size

# 随机生成weight
weight = np.zeros((vocab_size, vector_size))

words=wvmodel.wv.vocab

word_to_idx = {word: i for i, word in enumerate(words)}
# 定义了一个unknown的词.
# word_to_idx['<unk>'] = 0
idx_to_word = {i: word for i, word in enumerate(words)}
# idx_to_word[0] = '<unk>'
# print(idx_to_word[vocab_size-1])

for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmodel.index2word[i]]
    except:
        continue
    vector=wvmodel.wv.get_vector(idx_to_word[word_to_idx[wvmodel.index2word[i]]])
    weight[index, :] = vector

parser=CoreNLPParser(url='http://localhost:9000')
dpparser = CoreNLPDependencyParser(url='http://localhost:9000')
id=0
data=pd.read_csv('./FB15k/description.txt',delimiter='\t',header=None)
for line in data[1]:
    tokens=list(parser.tokenize(line))
    embed=[weight[word_to_idx.get(token,400000)] for token in tokens]
    np.savez('./FB15k/data_embedding/glove_%d.npz'%id,embed)
    id+=1

def get_adj_matrix(sentences):
    tokens = list(parser.tokenize(sentences))
    length = len(tokens)
    matrix = np.zeros((length, length), dtype=int)
    row = 0
    offset = 0
    nn_list = ['NN', 'NNS', 'NNP', 'NNPS']
    cross_sentence = np.zeros((length, 3), dtype=np.int32)

    for index, sentence in enumerate(dpparser.parse_text(sentences)):
        s = sentence.to_conll(4)
        for item in s.split('\n')[:-1]:
            item = item.split('\t')
            if item[1] in nn_list:
                cross_sentence[row] = [index, row - offset, 1]
            else:
                cross_sentence[row] = [index, row - offset, 0]
            if int(item[2]) != 0:
                matrix[row][offset + int(item[2]) - 1] = 1
                matrix[offset + int(item[2]) - 1][row] = 1
            row += 1
        offset = row
    return matrix, cross_sentence


def enhenced_edge(matrix, results, cross_sentence):
    index = np.argwhere(cross_sentence[:, 2] == 1)
    print(index)
    for i in range(len(index) - 1):
        for j in range(i + 1, len(index)):
            if cross_sentence[index[i], 0] == cross_sentence[index[j], 0]:
                continue
            else:
                sim = cos_sim(results[index[i]], results[index[j]])
                if sim > 0.7:
                    print(sim, index[i], index[j])
                    matrix[index[i], index[j]] = 1
                    matrix[index[j], index[i]] = 1
    return matrix


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos

id=0
data=pd.read_csv('./FB15k/description.txt',delimiter='\t',header=None)
for i,line in enumerate(data[1]):
    matrix,cross_sentence=get_adj_matrix(line)
    results=np.load('./FB15k/data_embedding/glove_%d.npz'%id)['arr_0']
    assert results.shape[0]==matrix.shape[0],'error!!!!!'
    matrix=enhenced_edge(matrix,results,cross_sentence)
    np.save('./FB15k/glove_data_embedding/enhenced_%d.npy'%id,matrix)
    id+=1

bc = BertClient()
id=0
data=pd.read_csv('./FB15k/description.txt',delimiter='\t',header=None)
results=[]
for line in data[1]:
    results.append(bc.encode([line]))
embedding=np.vstack(results)
np.save('./FB15k/sentence_embedding/embedding.npy',embedding)