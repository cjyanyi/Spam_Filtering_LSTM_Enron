# Spam_Filtering_LSTM_Enron
a cnn-lstm classifer of spam filtering, in Keras, which is based on Enron dataset

### 1: what's needed?  
glove.6B &nbsp; http://nlp.stanford.edu/projects/glove/ <br />
Enron &nbsp; http://www2.aueb.gr/users/ion/data/enron-spam/

### 2: training set  
selected __32k__ (16k ham vs. 16k spam) as raw data set;    
used __25.6k__ (80%) samples as training set, and __6.4k__ samples as validation set;

### 3: project files  
train-cnn-lstm.py &nbsp; &nbsp; ##train a model  
spam_filter.py &nbsp; &nbsp; ##top layer APIs of the spam filter   
dict_enron.json &nbsp; &nbsp; ##generated tokens (specific words) in Enron dataset  
dict-pre-process.py &nbsp;&nbsp; ##to generate the 'dict_enron.json'

### 4: result  
I didnot tune the parameters precisely, it SEEMED that the best performance was `98.8%` of 'val_acc', after 9 epochs. 
