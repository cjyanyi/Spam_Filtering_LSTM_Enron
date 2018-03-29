# -*- coding:UTF-8 -*-
import os,sys
import numpy as np
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
#from keras.utils import to_categorical
from sklearn.externals import joblib
#from .object_json import objectDumps2File, objectLoadFromFile

BASE_DIR = '../'
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'enron')
MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

VOCABULARY_NAME = 'tfidf_vocabulary_eron'

def preprocessing(text_dir=TEXT_DATA_DIR):
    # second, prepare text samples and their labels
    print('Processing text dataset')

    texts = []  # list of text samples
    labels_index = {'ham': 0, 'spam': 1}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(text_dir)):
        path = os.path.join(text_dir, name)
        if os.path.isdir(path):
            # label_id = len(labels_index)
            # labels_index[name] = label_id
            for folder in sorted(os.listdir(path)):
                sndpath = os.path.join(path, folder)
                if os.path.isdir(sndpath):
                    for fname in sorted(os.listdir(sndpath)):
                        if fname is not None:
                            fpath = os.path.join(sndpath, fname)
                            args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                            with open(fpath, **args) as f:
                                t = f.read()
                                i = t.find('Subject:')  # skip header
                                if 0 < i:
                                    t = t[i:]
                                texts.append(t)
                            labels.append(0) if folder == 'ham' else labels.append(1)

    # print (labels[:5000])

    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=40)  # tf-idf特征抽取ngram_range=(1,2)
    features = vectorizer.fit_transform(texts)
    print('Found %s unique features.' % features.shape[1])
    print(vectorizer.get_feature_names()[:50]) #特征名展示
    #test_features = vectorizer.transform(d_test.title)
    #print("测试样本特征长度为：" + str(test_features.shape))

    data = features
    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # save
    filename = VOCABULARY_NAME + '.pkl'
    word_index = vectorizer.vocabulary_

    # Erorr: np.int32 cannot be serialized
    #with open(filename + '.json', 'w') as outfile:
        #json.dump(word_index, outfile)
    ## serializer
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(word_index, f)
    ## restore
    #with open('name.pkl', 'rb') as f:
    #aa = pickle.load(f)
    #print(aa)




    return data,labels

def create_model(data, labels, val_percent=VALIDATION_SPLIT, cost=0.99):
    # split the data into a training set and a validation set
    print('Begin create a TFIDF SVM model.')
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(val_percent * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    # 支持向量机
    # C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0
    svm = SVC(C=cost, kernel="linear", verbose=True)  # kernel：参数选择有rbf, linear, poly, Sigmoid, 默认的是"RBF";

    nn = svm.fit(x_train, y_train)
    print(nn)
    #save model
    joblib.dump(svm, "svm_model.m")

    preds = svm.predict(x_val)
    num = 0
    preds = preds.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(y_val[i]):
            num += 1
    print('precision_score:' + str(float(num) / len(preds)))

if __name__ == "__main__":
    data,labels = preprocessing()
    create_model(data, labels)

