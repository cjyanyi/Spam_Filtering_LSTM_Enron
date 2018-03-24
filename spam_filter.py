# -*- coding:UTF-8 -*-
from keras.models import load_model
import os
import sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
import  json


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

def read_file(path, head='Subject:'):
    args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
    with open(path, **args) as f:
        t = f.read()
        i = t.find(head)  # skip header
        if 0 < i:
            t = t[i:]
    return t

class SpamFilter(object):

    def __init__(self, model_name='model_weights.hdf5', test_dir='test_mails'):
        try:
            self.model = load_model(os.path.join(MODEL_DIR, model_name))
            self.dict_w = load_dict(os.path.join(BASE_DIR, 'dict_enron.json'))
        except Exception as e:
            print(e.message)
        self.test_dir = os.path.join(BASE_DIR, test_dir)
        self.labels_index = {'ham':0, 'spam':1}
        self.labels = ('ham','spam')
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH

    def get_raw_text(self,name):
        #texts = [text1,...,text2]
        #path = os.path.join(BASE_DIR, 'fake')
        path = self.test_dir
        fpath = os.path.join(path, name)
        texts = []
        texts.append(read_file(fpath))
        print('Found 1 Raw text.')
        return texts


    def load_test_mail(self,path=None):
        #path = os.path.join(BASE_DIR, 'fake')
        if path == None:
            path = self.test_dir

        #read all files in path
        texts = []
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            texts.append(read_file(fpath))
        print('Found %s texts.' % len(texts))
        # print texts
        for str in texts:
            print([str[:80]])
        # covert texts to tensors
        return self._words2seq(texts),texts

    def texts_to_labels(self,texts, verb=False):
        data = self._words2seq(texts)
        predictions = self.predict(data)
        return (self.decode_predictions(predictions,verbose=verb))

    def _words2seq(self,texts):
        '''convert texts to input vectors'''
        # finally, vectorize the text samples into a 2D integer tensor
        dict = self.dict_w
        data = []
        for text in texts:
            #word split
            content = text_to_word_sequence(text)
            one_data = []
            #word to token
            for word in content:
                one_data=[]
                if word in dict.keys() and dict[word] < MAX_NUM_WORDS:
                    one_data.append(dict[word])
                else:
                    one_data.append(0)
            data.append(one_data)

        data = np.array(data)
        data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', data.shape)
        return data

    def predict(self,data):
        # inference, from input vectors to probability
        predictions = self.model.predict(data)
        print('Model Predict: ', predictions)
        return predictions

    def decode_predictions(self,predictions, verbose =True):
        '''to get labels of predictions'''
        labels = []
        for pre in predictions:
            indice = 0
            if pre[0]<pre[1]:
                indice = 1
            labels.append(indice)
            if verbose == True:
                print('This is a {} ! ** ham: {:-4}%  and spam: {:-4}%'.format(self.labels[indice], pre[0]*100,pre[1]*100))
        return labels

    def text_process(self,str):
        dict = self.dict_w
        #content = text_to_word_sequence(''.join(texts))
        content = text_to_word_sequence(str)
        data = []
        for word in content:
            if word in dict.keys() and dict[word] < MAX_NUM_WORDS:
                data.append(dict[word])
            else:
                data.append(0)

        return min(len(data),self.MAX_SEQUENCE_LENGTH),content,data


if __name__ == "__main__":
        filter = SpamFilter()
        data,texts = filter.load_test_mail()
        predictions = filter.predict(data)
        filter.decode_predictions(predictions)



