# -*- coding:  UTF-8 -*-
from __future__ import division
import tensorflow as tf
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

import keras.preprocessing.sequence as S

plt.switch_backend('agg')
# import matplotlib; matplotlib.use('TkAgg')
import collections.abc
from collections import defaultdict, OrderedDict

##FakeMove on NYC
online = False
batch_size = 64  # 64
voc_poi = list()  # pois in dictionary including <PAD>, <GO> and <EOS>
table_X = {}
new_table_X = {}
# read embeddings offline way
if online == False:
    print("this is the offline learning")
    def getXs():  # 读取轨迹向量
        fpointvec = open('embeddings/Yelp_deepwalk_2022_64.dat', 'r')  # 获取check-in向量 已经用word2vec训练得到 tor traindata/
        #     table_X={}  #建立字典索引
        item = 0
        for line in fpointvec.readlines():
            lineArr = line.split()
            if (len(lineArr) < 3):  # delete fist row #
                print('yes---',len(lineArr))
                continue;
            item += 1  # 统计条目数
            X = list()
            for i in lineArr[1:]:
                X.append(float(i))  # 读取向量数据
            if lineArr[0] == '8888888':
                table_X['<PAD>'] = X  # dictionary is a string  it is not a int type
            if lineArr[0] == '9999999':
                table_X['<EOS>'] = X  # dictionary is a string  it is not a int type
            if lineArr[0] == '7777777':
                table_X['<GO>'] = X  # dictionary is a string  it is not a int type
                #table_X['<GO>'] = X  # dictionary is a string  it is not a int type
            else:
                table_X[lineArr[0]] = X
        print('finish read vector of POIs')
        return table_X
else:
    print('we will use a online learning for POI embeddings')
#print(table_X,'yyyyy')
# read trajectories
# Read data
def read_pois_r():
    Train_DATA = []
    Train_USER = []
    Test_DATA = []
    Test_USER = []
    T_DATA = {}
    fread_train = open('data/Yelp/Yelp_train.txt', 'r')
    for lines in fread_train.readlines():
        line = lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Train_DATA.append(data_line)
        Train_USER.append(line[0])
        T_DATA.setdefault(line[0], []).append(data_line)
    fread_train = open('data/Yelp/Yelp_test.txt', 'r')
    for lines in fread_train.readlines():
        line = lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Test_DATA.append(data_line)
        Test_USER.append(line[0])
    print('Train Size', len(Train_DATA))
    print('total trajectory', len(Test_DATA) + len(Train_DATA))
    return T_DATA, Train_DATA, Train_USER, Test_DATA, Test_USER

# common functions--------------------------------------------------------------------
# read fake data
# Read data
def read_pois_f():
    Train_DATA = []
    Train_USER = []
    Test_DATA = []
    Test_USER = []
    T_DATA = {}
    fread_train = open('data/Yelp/Yelp_train0.15.txt', 'r')#finaltrain.txt
    for lines in fread_train.readlines():
        line = lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Train_DATA.append(data_line)
        Train_USER.append(line[0])
        T_DATA.setdefault(line[0], []).append(data_line)
    fread_train = open('data/Yelp/Yelp_test0.15.txt', 'r')
    for lines in fread_train.readlines():
        line = lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Test_DATA.append(data_line)
        Test_USER.append(line[0])
    print('Train Size', len(Train_DATA))
    print('total trajectory', len(Test_DATA) + len(Train_DATA))
    return T_DATA, Train_DATA, Train_USER, Test_DATA, Test_USER

# @1 convert POI identity to a index
def extract_words_vocab():
    int_to_vocab = {idx: word for idx, word in enumerate(voc_poi)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

# @2 filling the Mask
def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])  # 取最大长度
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def eos_sentence_batch(sentence_batch, eos_in):
    return [sentence + [eos_in] for sentence in sentence_batch]


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.abc.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


# data preprocessing part---------------------------------------------------------------
# read pois
getXs()  # read vectors

RH_DATA, RTrain_DATA, RTrain_USER, RTest_DATA, RTest_USER = read_pois_r()
FH_DATA, FTrain_DATA, FTrain_USER, FTest_DATA, FTest_USER = read_pois_f()

T = RTrain_DATA + RTest_DATA
total_check = len(flatten(T))
total_user = set(flatten(RTrain_USER + RTest_USER))
user_number = len(set(total_user))

for i_ in range(len(T)):
    for j_ in range(len(T[i_])):
        #print(T[i_][j_])
        new_table_X[T[i_][j_]] = table_X[T[i_][j_]]  # new pois dictionary for dataset
# add additional characters
new_table_X['<GO>'] = table_X['<GO>']
new_table_X['<EOS>'] = table_X['<EOS>']
new_table_X['<PAD>'] = table_X['<PAD>']
for poi in new_table_X:  # 按照保存的顺序读的
    voc_poi.append(poi)
print('lens', len(voc_poi))
int_to_vocab, vocab_to_int = extract_words_vocab()
print('Dictionary Length', len(int_to_vocab), 'real POI number', len(int_to_vocab) - 3)
TOTAL_POI = len(int_to_vocab)
print('Total check-ins', total_check, TOTAL_POI)
print('Total Users', user_number)

# 在这里我给出了虚假历史轨迹的构建，以便模型扩展，不一定用到
in_A = []
out_A = []
u_History = {}  # 已经转成index模式了
u_unique_H = {}

for key in RH_DATA.keys():  # index char
    temp = RH_DATA[key]
    temp = flatten(temp)
    new_temp = []
    for i in temp:
        new_temp.append(vocab_to_int[i])
        # print vocab_to_int[i]
    u_History[key] = new_temp
    u_unique_H[key] = np.unique(new_temp)
Popularity_POI={}
for T in RTrain_DATA:  # index char
    #print (T)
    for i in T:
        #print('-----------',vocab_to_int[i])
        if i in Popularity_POI: #如果在字典里，则增加计数否则创建一个
            value=Popularity_POI[i]
            Popularity_POI[i]=value+1
        else:
            Popularity_POI[i]=1 #这里不能设置为很小的数否则会失衡。。

length = []
for key in u_History:
    unique_values = np.unique(u_History[key])
    length.append(len(unique_values))
print('avg_user_length', np.mean(length), np.max(length), np.min(length))

in_A = OrderedDict()
out_A = OrderedDict()

# 构建个人轨迹关系图
for key in RH_DATA.keys():
    temps = RH_DATA[key]
    temp_Auo = np.zeros([len(u_unique_H[key]), len(u_unique_H[key])])  # ,dtype=np.float
    #temp_Aui = np.zeros([len(u_unique_H[key]), len(u_unique_H[key])])  # ,dtype=np.float
    value = u_unique_H[key]
    for subtemp in temps:  # 统计每一个用户的
        for i in range(len(subtemp) - 1):
            # print(vocab_to_int[subtemp[i]])
            v1 = np.where(value == vocab_to_int[subtemp[i]])[0][0]
            v2 = np.where(value == vocab_to_int[subtemp[i + 1]])[0][0]

            count = temp_Auo[v1][v2]
            temp_Auo[v1][v2] = count+1#
            #
            # count2 = temp_Aui[v2][v1]
            # temp_Aui[v2][v1] =1 # count2 +
    # 频度处理
    u_sum_in = np.sum(temp_Auo, 0)
    u_sum_in[np.where(u_sum_in == 0)] = 1
    u_A_in = np.divide(temp_Auo, u_sum_in)

    u_sum_out = np.sum(temp_Auo, 1)
    u_sum_out[np.where(u_sum_out == 0)] = 1
    u_A_out = np.divide(temp_Auo.transpose(), u_sum_out)

    # freq_o=np.sum(temp_Auo,axis=1)
    # vo=np.divide(temp_Auo.T, freq_o, out=np.zeros_like(temp_Auo.T), where=freq_o != np.zeros_like(freq_o))
    out_A[key] = u_A_out  # vo.T
    # freq_i=np.sum(temp_Aui,axis=1)
    # vi=np.divide(temp_Aui.T, freq_i, out=np.zeros_like(temp_Aui.T), where=freq_i != np.zeros_like(freq_i))
    in_A[key] = u_A_in  # vi.T

# convert data 将字符型POI转化成index表示
new_trainT = list()
fnew_trainT = list()
for i in range(len(RTrain_DATA)):  # TRAIN
    temp = list()
    ftemp = list()
    temp.append(vocab_to_int['<GO>'])  # 添加一个终止符号
    ftemp.append(vocab_to_int['<GO>'])  # 添加一个终止符号
    for j in range(len(RTrain_DATA[i])):
        temp.append(vocab_to_int[RTrain_DATA[i][j]])
        ftemp.append(vocab_to_int[FTrain_DATA[i][j]])
    # temp.append(vocab_to_int['<EOS>']) #添加一个终止符号
    new_trainT.append(temp)
    fnew_trainT.append(ftemp)
# max_check=max(np.array(new_trainT).flatten())
# print (max_check)

new_testT = list()
fnew_testT = list()
for i in range(len(RTest_DATA)):
    temp = list()
    ftemp = list()
    temp.append(vocab_to_int['<GO>'])  # 添加一个终止符号
    ftemp.append(vocab_to_int['<GO>'])  # 添加一个终止符号
    for j in range(len(RTest_DATA[i])):
        temp.append(vocab_to_int[RTest_DATA[i][j]])
        ftemp.append(vocab_to_int[FTest_DATA[i][j]])
    # temp.append(vocab_to_int['<EOS>'])  # 添加一个终止符号
    new_testT.append(temp)
    fnew_testT.append(ftemp)

# 需要对数据进行简单处理，进行排序减少空白符的插入
# sort original data
index_T = {}
trainT = []
trainU = []

findex_T = {}
ftrainT = []
ftrainU = []
for i in range(len(new_trainT)):
    index_T[i] = len(new_trainT[i])
temp_size = sorted(index_T.items(), key=lambda item: item[1])
for i in range(len(temp_size)):
    id = temp_size[i][0]
    trainT.append(new_trainT[id])
    trainU.append(RTrain_USER[id])

    ftrainT.append(fnew_trainT[id])
    ftrainU.append(FTrain_USER[id])
print('jsut a train')
print(ftrainT[:2])
print(trainT[:2])
# sort for test dataset
index_T = {}
testT = []
testU = []
findex_T = {}
ftestT = []
ftestU = []
for i in range(len(new_testT)):
    index_T[i] = len(new_testT[i])
temp_size = sorted(index_T.items(), key=lambda item: item[1])
for i in range(len(temp_size)):
    id = temp_size[i][0]
    testT.append(new_testT[id])
    testU.append(RTest_USER[id])

    ftestT.append(fnew_testT[id])
    ftestU.append(FTest_USER[id])
print('jsut a test')
print(ftestT[:2])
print(testT[:2])


# Creat dictionary embeddings构建一个嵌入字典，其实就是一个list
Pop_Scores=list()
#print('yes',Popularity_POI.keys())

def dic_em():
    #counts = 0
    dic_embeddings = list()
    for key in new_table_X:
        dic_embeddings.append(new_table_X[key])
        if key in Popularity_POI.keys():
            Pop_Scores.append(Popularity_POI[key])
        else:
            Pop_Scores.append(1.)
            #counts=counts+1
    #print(counts)
    return dic_embeddings

dic_embeddings = tf.constant(dic_em())
#dic_embeddings=tf.eye(len(new_table_X))
#dic_embeddings=tf.random.normal([len(vocab_to_int),256])
#
# graph_A=np.loadtxt('context_A_NYC.dat')
# graph_embeddings=tf.constant(graph_A,dtype=tf.float32)
# 模型参数设置
# 模型参数设置

# epoch
train_size = len(trainT) % batch_size
test_size = len(testT) % batch_size
trainT = trainT + trainT[-(batch_size - train_size):]  # copy data and fill the last batch size
trainU = trainU + trainU[-(batch_size - train_size):]

testT = testT + testT[-(batch_size - test_size):]  # copy data and fill the last batch size
testU = testU + testU[-(batch_size - test_size):]

# fake
ftrainT = ftrainT + ftrainT[-(batch_size - train_size):]  # copy data and fill the last batch size
ftrainU = ftrainU + ftrainU[-(batch_size - train_size):]
ftestT = ftestT + ftestT[-(batch_size - test_size):]  # copy data and fill the last batch size
ftestU = ftestU + ftestU[-(batch_size - test_size):]

# 深度学习相关函数-----------------------------------------------------------

# 位置编码
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# 遮挡 注意我在这里使用vocab_to_int['<PAD>']作为填充符号的
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, vocab_to_int['<PAD>']), tf.float32)

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# 前瞻遮挡
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

# x = tf.constant([[7, 6, vocab_to_int['<PAD>'], vocab_to_int['<PAD>'], 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
# print (create_padding_mask(x))
# 注意力模块
# 注意力机制
def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
      q: 请求的形状 == (..., seq_len_q, depth)
      k: 主键的形状 == (..., seq_len_k, depth)
      v: 数值的形状 == (..., seq_len_v, depth_v)
      mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
      输出，注意力权重
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

# 多头注意力层
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# ffn
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)

    def call(self, query, values):
        matmul_qk = tf.matmul(self.W2(query), self.W1(values), transpose_b=True)
        # 缩放 matmul_qk
        # dk = tf.cast(tf.shape()[-1], tf.float32)
        # scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # 将 mask 加入到缩放的张量上。
        # print(tf.shape(matmul_qk))
        # print(tf.shape(mask))
        # if mask is not None:
        #   matmul_qk += (mask * -1e9)
        attention_weights = tf.nn.softmax(matmul_qk, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, values)
        # print(attention_weights)
        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.batt = BahdanauAttention(d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        # self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dense_out = tf.keras.layers.Dense(d_model)
        self.dense_out2 = tf.keras.layers.Dense(d_model)
        # self.dense_out3 = tf.keras.layers.Dense(d_model)
        self.dense_in = tf.keras.layers.Dense(d_model)
        self.dense_in2 = tf.keras.layers.Dense(d_model)

    def call(self, x, pad_Au_out, pad_Au_in, training, mask):  # encoder 核心代码

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model) #
        ffn_output = self.dropout2(ffn_output, training=training)  # 残差网络
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        # print('out',tf.shape(out2))
        # print('pad', tf.shape(pad_Au_out))
        pad_Au_out = tf.cast(pad_Au_out, dtype=tf.float32)
        pad_Au_in = tf.cast(pad_Au_in, dtype=tf.float32)
        gnn_out = tf.nn.relu(self.dense_out(pad_Au_out))#tf.matmul(pad_Au_out, hx, transpose_a=True)
        gnn_out2 = self.dense_out2(gnn_out)
        gnn_in =tf.nn.relu(self.dense_out(pad_Au_in))
        gnn_in2=self.dense_in2(gnn_in)
        gnn = tf.concat([gnn_in2,gnn_out2],axis=2)# * tf.nn.sigmoid(gnn_out2)  # tf.concat([gnn_out,gnn_in],axis=2)#*tf.nn.sigmoid(gnn_out2)#+gnn_in*tf.nn.sigmoid(gnn_in)
        # print(tf.shape(gnn))
        his_attn_output, attention_weights = self.batt(out2, gnn)  # (batch_size, input_seq_len, d_model)#k,v,q 我修改了一下子
        # his_attn_output=self.layernorm3(his_attn_output)
        # print('out--',tf.shape(his_attn_output))
        out2 = tf.concat([out2, his_attn_output], axis=2)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

        # self.att=BahdanauAttention(d_model)
        # self.gnn=tf.keras.layers.Dense(d_model)
        # self.gnn2 = tf.keras.layers.Dense(d_model)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # if training is False:
        #     print('dec',look_ahead_mask.shape)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        # if training is False:
        #     print('dec',out1.shape)
        # print(tf.shape(enc_output),tf.shape(out1),tf.shape(padding_mask))

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)


        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model) #这个是随机嵌入我们不用
        self.embedding = dic_embeddings
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.mlp = tf.keras.layers.Dense(d_model)

    def call(self, x, pad_Au_out, pad_Au_in, training, mask):
        #hx =tf.nn.embedding_lookup(self.embedding, historical_tra)
        # hseq_len = tf.shape(hx)[1]
        # hx *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # hx += self.pos_encoding[:, :hseq_len, :]
        # hx = self.dropout(hx, training=training)
        seq_len = tf.shape(x)[1]
        # 将嵌入和位置编码相加。
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x = tf.nn.embedding_lookup(self.embedding, x)#self.embedding
        x = self.mlp(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, pad_Au_out, pad_Au_in, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # = tf.keras.layers.Embedding(target_vocab_size, d_model)#不使用随机初始化
        self.embedding =dic_embeddings
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.gnn = tf.keras.layers.Dense(d_model)
        self.gnn2 = tf.keras.layers.Dense(d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.mlp = tf.keras.layers.Dense(d_model)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        # if training is False:
        #     print ('seq_len',seq_len)
        attention_weights = {}
        # emb = tf.math.tanh(self.gnn(self.embedding))
        # emb=self.gnn2(emb)
        # # x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x = tf.nn.embedding_lookup(self.embedding, x)
        x = self.mlp(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        # if training is False:
        #     print ('x',x.shape)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)


        self.final_layer = tf.keras.layers.Dense(tf.shape(dic_embeddings)[0])  # 这一步很关键他是一个 tf.shape(dic_embeddings)[1]
        #self.att = BahdanauAttention(d_model)
        #self.att=BahdanauAttention(d_model)
        # self.gnn=tf.keras.layers.Dense(d_model)
        # self.gnn2 = tf.keras.layers.Dense(d_model)

    def call(self, inp, tar,pad_Au_out, pad_Au_in, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, pad_Au_out, pad_Au_in, training, enc_padding_mask
                                )  # (batch_size, inp_seq_len, d_model)

        # 再写一个。。。
        # if training==False:
        #     print(enc_output.shape)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        #print(len(Pop_Scores))
        # Pop_Scores2=tf.expand_dims(tf.nn.softmax(Pop_Scores),axis=-1)
        # score_ems=tf.multiply(Pop_Scores2,dic_embeddings)
        # dec_output2=self.att(dec_output,score_ems)
        # out=tf.concat([dec_output,dec_output2],axis=2)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        #final_output=tf.nn.softmax(tf.matmul(final_output,dic_embeddings,transpose_b=True))# tf.keras.

        # graph_embedding=tf.nn.relu(self.gnn(graph_embeddings))
        # graph_embedding=self.gnn2(graph_embedding)
        # out_att,_=self.att(dec_output,graph_embedding)
        # #print(tf.shape(out_att))
        # dec_output=tf.concat([dec_output, out_att], axis=2)
        # final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        # final_output = tf.nn.softmax(tf.matmul(final_output, dic_embeddings, transpose_b=True))

        # Pop_Scores_rev = tf.math.reciprocal(Pop_Scores)
        # Pop_Scores_rev =tf.expand_dims(tf.nn.softmax(Pop_Scores_rev), axis=0)
        # Pop_Scores_rev=tf.tile(Pop_Scores_rev,[tf.shape(final_output)[1],1])
        # final_output=(final_output+Pop_Scores_rev*0.1) #tf.nn.softmax
        return final_output, attention_weights


# 参数列表
num_layers = 1
d_model_in = 64  # 表示位置嵌入大小？
d_model = 128  # 表示隐藏层大小
dff = 1024  # 2048
num_heads = 2
input_vocab_size = len(int_to_vocab)  # 8000
target_vocab_size = len(int_to_vocab)  # 8500
dropout_rate = 0.1

#
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


# adam优化器
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
# define the training learn rate
exps = tf.keras.optimizers.schedules.ExponentialDecay(
    0.1,
    decay_steps=1,
    decay_rate=0.9,
    staircase=False)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# 损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, vocab_to_int[
        '<PAD>']))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
train_accuracy_T = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy_T')


def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# checkpoint_path = "./checkpoints/train_Yelp"
#
# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)
#
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果检查点存在，则恢复最新的检查点。
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')

# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
# 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
# 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
# 更多的通用形状。

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar, pad_Au_out, pad_Au_in):  # inp 是包含虚假位置的轨迹 而tar是真实轨迹
    # 剔除GO-go,1,2,3->1,2,3,pad,pad,pad
    # inp=inp[:, :-1]
    # inp=tf.convert_to_tensor(inp)
    # print('inp', inp)
    inp = tf.keras.preprocessing.sequence.pad_sequences(inp, padding='post', value=vocab_to_int[
        '<PAD>'])  # pad_sentence_batch(inp, vocab_to_int['<PAD>'])#

    # print('inp', inp)
    inp = inp[:, 1:]  # 移除GO符号
    # print ('inps',inp)
    # print('inp', inp)
    # 选择teacherforce 训练模式，其实我认为这种方式不完美
    tar = eos_sentence_batch(tar, vocab_to_int[
        '<EOS>'])  # tf.keras.preprocessing.sequence.pad_sequences(tar,padding='post',maxlen=1,value=vocab_to_int['<EOS>'])#eos_sentence_batch(tar, vocab_to_int['<EOS>'])
    # print('tar', tar)
    # print (vocab_to_int['<EOS>'])
    tar = tf.keras.preprocessing.sequence.pad_sequences(tar, padding='post', value=vocab_to_int[
        '<PAD>'])  # pad_sentence_batch(tar_inp, vocab_to_int['<PAD>'])#
    tar_inp = tar[:, :-1]  # go,1,2,3->go,1,2,3,pad,pad

    tar_real = tar[:, 1:]  # go,1,2,3->1,2,3,eos,pad,pad

    # print('tari', tar_inp)
    # print('tarr',tar_real)
    # tar_real=eos_sentence_batch(tar_real,vocab_to_int['<EOS>'])
    # print('tar_real', tar_real)
    # tar_real=tf.keras.preprocessing.sequence.pad_sequences(tar_inp,padding='post',value=vocab_to_int['<PAD>'])#pad_sentence_batch(tar_real, vocab_to_int['<PAD>'])
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)


    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, pad_Au_out, pad_Au_in,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # train_loss(loss)
    # train_accuracy(tar_real, predictions)
    return loss, tar_real, predictions


def evaluate(inp, tar, pad_Au_out, pad_Au_in):  # inp 是包含虚假位置的轨迹 而tar是真实轨迹
    # recall=0
    # f1=0
    # print('real',inp[:10]) #[9815, 938, 1835]
    # print('fake',tar[0]) #[9815, 938, 1835]
    # reals=np.array(inp)[:, 1:]
    original = tar
    inp = tf.keras.preprocessing.sequence.pad_sequences(inp, padding='post', value=vocab_to_int[
        '<PAD>'])  # pad_sentence_batch(inp, vocab_to_int['<PAD>'])#
    inp = inp[:, 1:]  # 移除GO符号
    tar = eos_sentence_batch(tar, vocab_to_int[
        '<EOS>'])  # tf.keras.preprocessing.sequence.pad_sequences(tar,padding='post',maxlen=1,value=vocab_to_int['<EOS>'])#eos_sentence_batch(tar, vocab_to_int['<EOS>'])
    # print('tar', tar)
    # print (vocab_to_int['<EOS>'])
    # print('real---',inp[0]) #[9815, 938, 1835]
    # print('fake--',tar[0]) #[9815, 938, 1835, 9816] 最后一个符号是EOS
    tar = tf.keras.preprocessing.sequence.pad_sequences(tar, padding='post', value=vocab_to_int[
        '<PAD>'])  # pad_sentence_batch(tar_inp, vocab_to_int['<PAD>'])#
    tar_inp = tar[:, :-1]  # go,1,2,3->go,1,2,3,pad,pad

    tar_real = tar[:, 1:]  # go,1,2,3->1,2,3,eos,pad,pad
    # #print('real---',inp[0]) #[9815, 938, 1835]
    # print('fake--',tar_inp[0],vocab_to_int[
    #     '<EOS>'],vocab_to_int[
    #     '<PAD>']) #[9815, 938, 1835, 9816] 最后一个符号是EOS id是9816， PAD id——>9817 GO是9815
    decoder_input = tar_inp[:, 0]
    # print(tar_inp.shape,decoder_input.shape)
    decoder_input = tf.expand_dims(decoder_input, 1)
    # print(tar_inp.shape, decoder_input.shape)
    pred_result = []
    ACC = 0
    #his_mask = create_padding_mask(pad_historical_tra)
    for t in range(0, tar_inp.shape[1]):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, decoder_input)
        # print(enc_padding_mask[:4])
        # print(combined_mask[:4])
        # print(dec_padding_mask[:4])
        predictions, _ = transformer(inp, decoder_input,pad_Au_out, pad_Au_in,
                                     False,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # print(predicted_id)
        pred_result.append(predicted_id.numpy().tolist())
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)
    pred_result = np.array(pred_result).T[0]
    # print((np.shape(pred_result)))
    # print(pred_result[0])
    # print(tar_real[0])
    # print(reals[0])
    batch_recall = []
    batch_f1 = []
    batch_recall = []
    for i in range(len(inp)):
        temp = []
        length = len(original[i]) - 1  # 删除go符号
        y = tar_real[i][:length]  # [1:]#剔除GO符号
        py = pred_result[i][:length]
        if (y == py).all():
            ACC += 1
        for j in range(len(y)):
            if y[j] == py[j]:
                temp.append(1.)
                # print('true')
            else:
                temp.append(0.)
        # recall,f1=calc_F1(y,list(py))
        # batch_recall.append(recall)
        # batch_f1.append(f1)
        # recall=recall_score(y,py,average='macro',zero_division=0)
        f1 = f1_score(y, py, average='macro')
        batch_f1.append(f1)
        # batch_recall.append(recall)
        batch_recall.append(np.mean(temp))
    return np.mean(batch_recall), ACC, np.mean(batch_f1)


# 预备训练
EPOCHS = 100
# 需要对数据进行简单处理，进行排序减少空白符的插入

for epoch in range(EPOCHS):
    start = time.time()
    print('training epoch', epoch)

    train_loss.reset_states()
    train_accuracy.reset_states()
    # print('data',(trainT[20]))

    step = 0
    print(len(trainT))
    batch_acc = []
    T_pred = []
    while step < len(trainU) // batch_size:
        train_accuracy_T.reset_states()
        start_i = step * batch_size
        input_x = trainT[start_i:start_i + batch_size]
        finput_x = ftrainT[start_i:start_i + batch_size]
        input_u = trainU[start_i:start_i + batch_size]
        # print(input_u)
        historical_tra = []
        Au_out = []
        Au_in = []
        for u in range(len(input_u)):
            # print(np.shape(u_unique_H[u]))
            emb_fx=tf.nn.embedding_lookup(dic_embeddings, u_unique_H[input_u[u]])
            A1=tf.cast(out_A[input_u[u]],tf.float32)
            A2 = tf.cast(in_A[input_u[u]], tf.float32)
            context =tf.matmul(A1,emb_fx)
            context2 = tf.matmul(A2, emb_fx)
            #print(tf.shape(context))
            # print(tf.shape(emb_fx))
            Au_out.append(context)
            Au_in.append(context2)
            historical_tra.append(u_unique_H[input_u[u]])
        # print(historical_tra[10])
        pad_historical_tra = tf.keras.preprocessing.sequence.pad_sequences(historical_tra, padding='post',
                                                                           value=vocab_to_int['<PAD>'])
        pad_Au_out=[]
        pad_Au_in=[]
        max_length = tf.shape(pad_historical_tra)[1]
        for index in range(tf.shape(pad_historical_tra)[0]):
            pad=tf.ones([max_length-len(Au_out[index]),d_model_in])*1e-9
            # pad_s = tf.expand_dims(new_table_X['<PAD>'], axis=0)
            # pad = tf.tile(pad_s, [max_length - len(Au_out[index]), 1])
            A1=tf.concat([Au_out[index],pad],axis=0)
            pad_Au_out.append(A1)
            A2=tf.concat([Au_in[index],pad],axis=0)
            pad_Au_in.append(A2)

        loss, ttar_real, tpredictions = train_step(finput_x, input_x, pad_Au_out, pad_Au_in)

        # print(ttar_real[0])
        train_accuracy_T(ttar_real, tpredictions)
        batch_train_accuracy_T = train_accuracy_T.result() * batch_size
        batch_acc.append(batch_train_accuracy_T)
        step += 1
    train_acc=np.sum(batch_acc) / (step * batch_size)
    print('Epoch', epoch + 1, 'acc', np.sum(batch_acc) / (step * batch_size))

    # -----------------------------------------------
    step = 0
    T_recall = []
    T_f1 = []
    ACC = 0
    start2 = time.time()
    while step < len(testU) // batch_size:
        start_i = step * batch_size
        input_x = ftestT[start_i:start_i + batch_size]  # 这是假轨迹
        input_y = testT[start_i:start_i + batch_size]  # 这是真轨迹
        input_u = testU[start_i:start_i + batch_size]
        # print(input_u)
        historical_tra = []
        Au_out = []
        Au_in = []
        for u in range(len(input_u)):
            # print(np.shape(u_unique_H[u]))
            emb_fx = tf.nn.embedding_lookup(dic_embeddings, u_unique_H[input_u[u]])
            A1 = tf.cast(out_A[input_u[u]], tf.float32)
            A2 = tf.cast(in_A[input_u[u]], tf.float32)
            context = tf.matmul(A1, emb_fx)
            context2 = tf.matmul(A2, emb_fx)
            Au_out.append(context)
            Au_in.append(context2)
            historical_tra.append(u_unique_H[input_u[u]])
        # print(historical_tra[10])
        pad_historical_tra = tf.keras.preprocessing.sequence.pad_sequences(historical_tra, padding='post',
                                                                           value=vocab_to_int['<PAD>'])
        pad_Au_out = []
        pad_Au_in = []
        max_length = tf.shape(pad_historical_tra)[1]
        for index in range(tf.shape(pad_historical_tra)[0]):
            # pad_s=tf.expand_dims(new_table_X['<PAD>'],axis=0)
            # pad=tf.tile(pad_s,[max_length - len(Au_out[index]),1])
            #print(pad_s)
            pad = tf.ones([max_length - len(Au_out[index]), d_model_in])*1e-9
            A1 = tf.concat([Au_out[index], pad], axis=0)
            pad_Au_out.append(A1)
            A2 = tf.concat([Au_in[index], pad], axis=0)
            pad_Au_in.append(A2)
        b_r, acc, f1, = evaluate(input_x, input_y, pad_Au_out, pad_Au_in)
        T_recall.append(b_r)
        T_f1.append((f1))
        ACC += acc
        step += 1

    print('The Test results are >>>>>>>>>\n')
    print('Recall', np.mean(T_recall), 'ACC', ACC / ((step * batch_size)), 'f1', np.mean(T_f1))
    print('----------------------------------------------')
    end2 = time.time()
    print('time cost', (end2 - start2))
    #ckpt_save_path = ckpt_manager.save()
    # 记录结果的文件
    # file = open('./result/NYC_graph_EM1103_512_ICDE.txt', 'a+')
    # file.write('epoch\t' + str(epoch) + '\ttrain acc:\t' + str(train_acc) + '\ttest recall:\t' + str(
    #     np.mean(T_recall)) + '\ttest f1:\t' + str(np.mean(np.mean(T_f1))) + '\ttest acc:\t' + str(
    #     ACC / ((step * batch_size))) + '\n')
    end = time.time()
    print('time cost',(end-start))

print('finish files!')
