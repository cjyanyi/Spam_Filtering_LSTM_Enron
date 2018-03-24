# -*- coding:UTF-8 -*-
from keras.models import load_model
import os
import sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import  json

from keras.preprocessing.text import text_to_word_sequence

BASE_DIR = ''
MODEL_DIR = os.path.join(BASE_DIR, 'model_lstm')
FOLDER_DIR = os.path.join(BASE_DIR, '')
MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 20000

# load dictionary
def load_dict(path):
    dict = {}
    with open(path, 'r') as f:
        dict = json.load(f)
    print('Open Word Dict, Length: %s' % len(dict))
    return dict


class SpamFilter(object):

    def __init__(self):
        self.model = load_model(os.path.join(MODEL_DIR, 'model_weights.hdf5'))
        self.dict_w = load_dict(os.path.join(BASE_DIR, 'dict_enron.json'))
        self.test_dir = os.path.join(BASE_DIR, 'test_mails')
        self.labels_index = {'ham':0, 'spam':1}
        self.labels = ('ham','spam')
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH

    def load_raw_text(self,name):
        #texts = [text1,...,text2]
        path = os.path.join(BASE_DIR, 'fake')
        fpath = os.path.join(path, name)
        args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
        with open(fpath, **args) as f:
            t = f.read()
            i = t.find('Subject:')  # skip header
            if 0 < i:
                t = t[i:]
            texts.append(t)

        print('Found %s Raw texts.' % len(texts))


    def load_test_mail(self,texts):
        path = os.path.join(BASE_DIR, 'fake')
        fpath = os.path.join(path, 'spam.txt')
        args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
        with open(fpath, **args) as f:
            t = f.read()
            i = t.find('Subject:')  # skip header
            if 0 < i:
                t = t[i:]
            texts.append(t)

        print('Found %s texts.' % len(texts))
        # print texts
        for str in texts:
            print([str[:80]])
        # covert texts to tensors
        return (self._words2seq(texts))

    def texts_list2labels(self,texts):
        data = self._words2seq(texts)
        predictions = self.predict(data)
        return (self.decode_predictions(predictions,verbose=False))

    def _words2seq(self,texts):
        # finally, vectorize the text samples into a 2D integer tensor
        dict = self.dict_w
        content = text_to_word_sequence(''.join(texts))
        data = []
        for word in content:
            if word in dict.keys() and dict[word] < MAX_NUM_WORDS:
                data.append(dict[word])
            else:
                data.append(0)

        data = np.array([data])
        data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', data.shape)
        return data

    def predict(self,data):
        # inference
        predictions = self.model.predict(data)
        print('Model Predict: ', predictions)
        return predictions

    def decode_predictions(self,predictions, verbose =True):
        labels = []
        for pre in predictions:
            indice = 0
            if pre[0]<pre[1]:
                indice = 1
            labels.append(indice)
            if verbose == True:
                print('This is a {} ! >> ham: {:-4}%  and spam: {:-4}%'.format(self.labels[indice], pre[0]*100,pre[1]*100))
        return labels

    def text_process(self,texts):
        dict = self.dict_w
        content = text_to_word_sequence(''.join(texts))
        data = []
        for word in content:
            if word in dict.keys() and dict[word] < MAX_NUM_WORDS:
                data.append(dict[word])
            else:
                data.append(0)

        return min(len(data),MAX_SEQUENCE_LENGTH),content,data



if __name__ == "__main__":
        texts = []
        filter = SpamFilter()
        data = filter.load_test_mail(texts)
        predictions = filter.predict(data)
        filter.decode_predictions(predictions)



