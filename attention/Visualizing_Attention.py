# -*- coding:UTF-8 -*-
import numpy as np
from keras import backend as K

# cjy: code in this file is untested !

# 需要导入两个模型，分别是句子级别的和篇章级别的，以及预处理后的文本序列
def get_attention(sent_model, doc_model, sequences, topN=5):
    sent_before_att = K.function([sent_model.layers[0].input, K.learning_phase()],
                                 [sent_model.layers[2].output])
    cnt_reviews = sequences.shape[0]

    # 导出这个句子每个词的权重
    sent_att_w = sent_model.layers[3].get_weights()
    sent_all_att = []
    for i in range(cnt_reviews):
        sent_each_att = sent_before_att([sequences[i], 0])
        sent_each_att = cal_att_weights(sent_each_att, sent_att_w, model_name='HAN')
        sent_each_att = sent_each_att.ravel()
        sent_all_att.append(sent_each_att)
    sent_all_att = np.array(sent_all_att)

    doc_before_att = K.function([doc_model.layers[0].input, K.learning_phase()],
                                [doc_model.layers[2].output])
    # 找到重要的分句
    doc_att_w = doc_model.layers[3].get_weights()
    doc_sub_att = doc_before_att([sequences, 0])
    doc_att = cal_att_weights(doc_sub_att, doc_att_w, model_name='HAN')

    return sent_all_att, doc_att

# 使用numpy重新计算attention层的结果
def cal_att_weights(output, att_w, model_name):
    if model_name == 'HAN':
        eij = np.tanh(np.dot(output[0], att_w[0]) + att_w[1])
        eij = np.dot(eij, att_w[2])
        eij = eij.reshape((eij.shape[0], eij.shape[1]))
        ai = np.exp(eij)
        weights = ai / np.sum(ai)
        return weights