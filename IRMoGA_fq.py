# -*- coding:  UTF-8 -*-
from __future__ import division
import tensorflow as tf
import time
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import collections.abc
from collections import  OrderedDict

online = False
batch_size = 64  # 64
voc_poi = list()  # pois in dictionary including <PAD>, <GO> and <EOS>
table_X = {}
new_table_X = {}
# read embeddings offline way
if online == False:
    print("this is the offline learning")
    def getXs():
        fpointvec = open('embeddings/FQ_embeddings_graph.dat', 'r')  # obtain the embeddings
        item = 0
        for line in fpointvec.readlines():
            lineArr = line.split()
            if (len(lineArr) < 256):  # delete fist row #
                continue;
            item += 1
            X = list()
            for i in lineArr[1:]:
                X.append(float(i))
            if lineArr[0] == '</s>':
                table_X['<PAD>'] = X  # dictionary is a string  it is not a int type
            else:
                table_X[lineArr[0]] = X
        print('finish read vector of POIs')
        return table_X
else:
    print('we will use a online learning for POI embeddings')

# read trajectories
# Read data
def read_pois_r():
    Train_DATA = []
    Train_USER = []
    Test_DATA = []
    Test_USER = []
    T_DATA = {}
    fread_train = open('data/fq/FQ_Rtrain.dat', 'r')
    for lines in fread_train.readlines():
        line = lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Train_DATA.append(data_line)
        Train_USER.append(line[0])
        T_DATA.setdefault(line[0], []).append(data_line)
    fread_train = open('data/fq/FQ_Rtest.dat', 'r')
    for lines in fread_train.readlines():
        line = lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Test_DATA.append(data_line)
        Test_USER.append(line[0])
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
    fread_train = open('data/fq/FQ_finaltrain.txt', 'r')
    for lines in fread_train.readlines():
        line = lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Train_DATA.append(data_line)
        Train_USER.append(line[0])
        T_DATA.setdefault(line[0], []).append(data_line)
    fread_train = open('data/fq/FQ_finaltest.txt', 'r')
    for lines in fread_train.readlines():
        line = lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Test_DATA.append(data_line)
        Test_USER.append(line[0])
    return T_DATA, Train_DATA, Train_USER, Test_DATA, Test_USER


# @1 convert POI identity to a index
def extract_words_vocab():
    int_to_vocab = {idx: word for idx, word in enumerate(voc_poi)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int


# @2 filling the Mask
def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
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
        new_table_X[T[i_][j_]] = table_X[T[i_][j_]]  # new pois dictionary for dataset

# add additional characters
new_table_X['<GO>'] = table_X['<GO>']
new_table_X['<EOS>'] = table_X['<EOS>']
new_table_X['<PAD>'] = table_X['<PAD>']
for poi in new_table_X:
    voc_poi.append(poi)
print('lens', len(voc_poi))
int_to_vocab, vocab_to_int = extract_words_vocab()
print('Dictionary Length', len(int_to_vocab), 'real POI number', len(int_to_vocab) - 3)
TOTAL_POI = len(int_to_vocab)
print('Total check-ins', total_check, TOTAL_POI)
print('Total Users', user_number)

# graph
in_A = []
out_A = []
u_History = {}  #
u_unique_H = {}

for key in RH_DATA.keys():  # index char
    temp = RH_DATA[key]
    temp = flatten(temp)
    new_temp = []
    for i in temp:
        new_temp.append(vocab_to_int[i])
    u_History[key] = new_temp
    u_unique_H[key] = np.unique(new_temp)
Popularity_POI={}
for T in RTrain_DATA:  # index char
    for i in T:
        if i in Popularity_POI:
            value=Popularity_POI[i]
            Popularity_POI[i]=value+1
        else:
            Popularity_POI[i]=1

length = []
for key in u_History:
    unique_values = np.unique(u_History[key])
    length.append(len(unique_values))
print('avg_user_length', np.mean(length), np.max(length), np.min(length))

in_A = OrderedDict()
out_A = OrderedDict()

# personal graph
for key in RH_DATA.keys():
    temps = RH_DATA[key]
    temp_Auo = np.zeros([len(u_unique_H[key]), len(u_unique_H[key])])  # ,dtype=np.float
    value = u_unique_H[key]
    for subtemp in temps:
        for i in range(len(subtemp) - 1):
            v1 = np.where(value == vocab_to_int[subtemp[i]])[0][0]
            v2 = np.where(value == vocab_to_int[subtemp[i + 1]])[0][0]
            count = temp_Auo[v1][v2]
            temp_Auo[v1][v2] = count+1#
    #
    u_sum_in = np.sum(temp_Auo, 0)
    u_sum_in[np.where(u_sum_in == 0)] = 1
    u_A_in = np.divide(temp_Auo, u_sum_in)

    u_sum_out = np.sum(temp_Auo, 1)
    u_sum_out[np.where(u_sum_out == 0)] = 1
    u_A_out = np.divide(temp_Auo.transpose(), u_sum_out)
    out_A[key] = u_A_out  #outgoing
    in_A[key] = u_A_in  # incoming

# convert data
new_trainT = list()
fnew_trainT = list()
for i in range(len(RTrain_DATA)):  # TRAIN
    temp = list()
    ftemp = list()
    temp.append(vocab_to_int['<GO>'])
    ftemp.append(vocab_to_int['<GO>'])
    for j in range(len(RTrain_DATA[i])):
        temp.append(vocab_to_int[RTrain_DATA[i][j]])
        ftemp.append(vocab_to_int[FTrain_DATA[i][j]])
    new_trainT.append(temp)
    fnew_trainT.append(ftemp)

new_testT = list()
fnew_testT = list()
for i in range(len(RTest_DATA)):
    temp = list()
    ftemp = list()
    temp.append(vocab_to_int['<GO>'])
    ftemp.append(vocab_to_int['<GO>'])
    for j in range(len(RTest_DATA[i])):
        temp.append(vocab_to_int[RTest_DATA[i][j]])
        ftemp.append(vocab_to_int[FTest_DATA[i][j]])
    new_testT.append(temp)
    fnew_testT.append(ftemp)

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

# Creat dictionary embeddings
Pop_Scores=list()

def dic_em():
    dic_embeddings = list()
    for key in new_table_X:
        dic_embeddings.append(new_table_X[key])
        if key in Popularity_POI.keys():
            Pop_Scores.append(Popularity_POI[key])
        else:
            Pop_Scores.append(1.)
    return dic_embeddings

dic_embeddings = tf.constant(dic_em())

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

# deep learning function-----------------------------------------------------------
# position encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # sin（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # cos；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# mask
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, vocab_to_int['<PAD>']), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

#Attention
def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    #  matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)


    if mask is not None:
        scaled_attention_logits += (mask * -1e9)


    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

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
        """(num_heads, depth).
        to (batch_size, num_heads, seq_len, depth)
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
        attention_weights = tf.nn.softmax(matmul_qk, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, values)
        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.batt = BahdanauAttention(d_model)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dense_out = tf.keras.layers.Dense(d_model)
        self.dense_out2 = tf.keras.layers.Dense(d_model)
        self.dense_in = tf.keras.layers.Dense(d_model)
        self.dense_in2 = tf.keras.layers.Dense(d_model)

    def call(self, x, pad_Au_out, pad_Au_in, training, mask):  # encoder

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model) #
        ffn_output = self.dropout2(ffn_output, training=training)  #
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        pad_Au_out = tf.cast(pad_Au_out, dtype=tf.float32)
        pad_Au_in = tf.cast(pad_Au_in, dtype=tf.float32)
        gnn_out = tf.nn.relu(self.dense_out(pad_Au_out))#
        gnn_out2 = self.dense_out2(gnn_out)
        gnn_in =tf.nn.relu(self.dense_out(pad_Au_in))
        gnn_in2=self.dense_in2(gnn_in)
        gnn = tf.concat([gnn_in2,gnn_out2],axis=2)#
        # print(tf.shape(gnn))
        his_attn_output, _ = self.batt(out2, gnn)  # (batch_size, input_seq_len, d_model)#k,q
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

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)


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

        self.embedding = dic_embeddings
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.mlp = tf.keras.layers.Dense(d_model)

    def call(self, x, pad_Au_out, pad_Au_in, training, mask):
        seq_len = tf.shape(x)[1]
        x = tf.nn.embedding_lookup(self.embedding, x)
        x = self.mlp(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, pad_Au_out, pad_Au_in, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
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
        attention_weights = {}
        x = tf.nn.embedding_lookup(self.embedding, x)
        x = self.mlp(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)


        self.final_layer = tf.keras.layers.Dense(tf.shape(dic_embeddings)[0])  #  tf.shape(dic_embeddings)[1]


    def call(self, inp, tar,pad_Au_out, pad_Au_in, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, pad_Au_out, pad_Au_in, training, enc_padding_mask
                                )  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        Pop_Scores_rev = tf.math.reciprocal(Pop_Scores)
        Pop_Scores_rev =tf.expand_dims(tf.nn.softmax(Pop_Scores_rev), axis=0)
        Pop_Scores_rev=tf.tile(Pop_Scores_rev,[tf.shape(final_output)[1],1])
        final_output=(final_output+Pop_Scores_rev*0.001) #tf.nn.softmax
        return final_output, attention_weights


# hyper-paras
num_layers = 1
d_model_in = 256
d_model = 128
dff = 1024
num_heads = 2
input_vocab_size = len(int_to_vocab)
target_vocab_size = len(int_to_vocab)
dropout_rate = 0.1
#
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


# adam
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
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# loss
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
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if you want recoer your model you can use it。
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar, pad_Au_out, pad_Au_in):  # inp

    inp = tf.keras.preprocessing.sequence.pad_sequences(inp, padding='post', value=vocab_to_int[
        '<PAD>'])  # pad_sentence_batch(inp, vocab_to_int['<PAD>'])#

    # print('inp', inp)
    inp = inp[:, 1:]  # 移除GO符号
    tar = eos_sentence_batch(tar, vocab_to_int[
        '<EOS>'])
    tar = tf.keras.preprocessing.sequence.pad_sequences(tar, padding='post', value=vocab_to_int[
        '<PAD>'])  # pad_sentence_batch(tar_inp, vocab_to_int['<PAD>'])#
    tar_inp = tar[:, :-1]  # go,1,2,3->go,1,2,3,pad,pad

    tar_real = tar[:, 1:]  # go,1,2,3->1,2,3,eos,pad,pad

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

    return loss, tar_real, predictions


def evaluate(inp, tar, pad_Au_out, pad_Au_in):  # inp

    original = tar
    inp = tf.keras.preprocessing.sequence.pad_sequences(inp, padding='post', value=vocab_to_int[
        '<PAD>'])  # pad_sentence_batch(inp, vocab_to_int['<PAD>'])#
    inp = inp[:, 1:]  #
    tar = eos_sentence_batch(tar, vocab_to_int[
        '<EOS>'])  #

    tar = tf.keras.preprocessing.sequence.pad_sequences(tar, padding='post', value=vocab_to_int[
        '<PAD>'])  # pad_sentence_batch(tar_inp, vocab_to_int['<PAD>'])#
    tar_inp = tar[:, :-1]  # go,1,2,3->go,1,2,3,pad,pad

    tar_real = tar[:, 1:]  # go,1,2,3->1,2,3,eos,pad,pad

    decoder_input = tar_inp[:, 0]

    decoder_input = tf.expand_dims(decoder_input, 1)

    pred_result = []
    ACC = 0
    for t in range(0, tar_inp.shape[1]):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, decoder_input)

        predictions, _ = transformer(inp, decoder_input,pad_Au_out, pad_Au_in,
                                     False,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        predictions = predictions[:, -1:, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        pred_result.append(predicted_id.numpy().tolist())
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)
    pred_result = np.array(pred_result).T[0]

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

            else:
                temp.append(0.)
        f1 = f1_score(y, py, average='macro')
        batch_f1.append(f1)

        batch_recall.append(np.mean(temp))
    return np.mean(batch_recall), ACC, np.mean(batch_f1)


#
EPOCHS = 100

for epoch in range(EPOCHS):
    start = time.time()
    print('training epoch', epoch)

    train_loss.reset_states()
    train_accuracy.reset_states()
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

        historical_tra = []
        Au_out = []
        Au_in = []
        for u in range(len(input_u)):

            emb_fx=tf.nn.embedding_lookup(dic_embeddings, u_unique_H[input_u[u]])
            A1=tf.cast(out_A[input_u[u]],tf.float32)
            A2 = tf.cast(in_A[input_u[u]], tf.float32)
            context =tf.matmul(A1,emb_fx)
            context2 = tf.matmul(A2, emb_fx)

            Au_out.append(context)
            Au_in.append(context2)
            historical_tra.append(u_unique_H[input_u[u]])

        pad_historical_tra = tf.keras.preprocessing.sequence.pad_sequences(historical_tra, padding='post',
                                                                           value=vocab_to_int['<PAD>'])
        pad_Au_out=[]
        pad_Au_in=[]
        max_length = tf.shape(pad_historical_tra)[1]
        for index in range(tf.shape(pad_historical_tra)[0]):
            pad=tf.ones([max_length-len(Au_out[index]),d_model_in])*1e-9

            A1=tf.concat([Au_out[index],pad],axis=0)
            pad_Au_out.append(A1)
            A2=tf.concat([Au_in[index],pad],axis=0)
            pad_Au_in.append(A2)

        loss, ttar_real, tpredictions = train_step(finput_x, input_x, pad_Au_out, pad_Au_in)

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
    while step < len(testU) // batch_size:
        start_i = step * batch_size
        input_x = ftestT[start_i:start_i + batch_size]  #fake
        input_y = testT[start_i:start_i + batch_size]  # real
        input_u = testU[start_i:start_i + batch_size]
        historical_tra = []
        Au_out = []
        Au_in = []
        for u in range(len(input_u)):
            emb_fx = tf.nn.embedding_lookup(dic_embeddings, u_unique_H[input_u[u]])
            A1 = tf.cast(out_A[input_u[u]], tf.float32)
            A2 = tf.cast(in_A[input_u[u]], tf.float32)
            context = tf.matmul(A1, emb_fx)
            context2 = tf.matmul(A2, emb_fx)
            Au_out.append(context)
            Au_in.append(context2)
            historical_tra.append(u_unique_H[input_u[u]])
        pad_historical_tra = tf.keras.preprocessing.sequence.pad_sequences(historical_tra, padding='post',
                                                                           value=vocab_to_int['<PAD>'])
        pad_Au_out = []
        pad_Au_in = []
        max_length = tf.shape(pad_historical_tra)[1]
        for index in range(tf.shape(pad_historical_tra)[0]):
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
    ckpt_save_path = ckpt_manager.save()

print('finish!')
