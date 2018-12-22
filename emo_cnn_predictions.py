# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:31:13 2018

@author: 63483
"""

import numpy as np
import pickle
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling1D,Conv1D
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
tf.set_random_seed(1)
import string
from keras.models import load_model



## word to vec
def func(st):
    st=st.split(' ')
    lst=[]
    for w in st:
        if(len(w) > 0):
            lst.append(w)
    return lst        

def PadAndAppend(sentence,maxlen):
    dist=maxlen-len(sentence)
    for i in range(dist):
        sentence.append('pad')
    sentence.append('eos')
    return sentence     

preprocess_test_data = pickle.load(open("C:\\Users\\63483\\Desktop\\EMO\\preprocess_test_data.p","rb")) 
print(len(preprocess_test_data))
list_of_sentences=[]
max_len=56 
for tp in preprocess_test_data:
        st2=func(tp[2])
        list_of_sentences.append(st2)

data_x=[]    
for sent in list_of_sentences:
    data_x.append(PadAndAppend(sent,max_len))   

print (len(data_x[0]))       
model=pickle.load(open("word2vec_emo.p","rb"))  
words = list(model.wv.vocab)
print (model)
print (len(words)) 
X_test=np.empty([len(data_x),len(data_x[0]),5])
for i,tp in enumerate(data_x):
        for j,w in enumerate(tp):
            if(w in words):
                X_test[i][j] = model[w]
            else:
                tmp="unk"
                X_test[i][j] = model[tmp]
print  ('----------',X_test.shape) 

lstm_model = Sequential()
lstm_model.add(keras.layers.InputLayer(input_shape=(57,5))) 
lstm_model.add(keras.layers.LSTM(units = 100, activation='relu',
    use_bias=True))
lstm_model.add(keras.layers.Dense(units=4, input_dim=57,activation='softmax'))
lstm_model.summary()
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


lstm_model.load_weights("model_lstm_emo.h5")
print("Loaded model from disk")
pred=lstm_model.predict_classes(X_test)
print (pred)
label_dict={}
label_dict[0]='angry'
label_dict[1]='happy'
label_dict[2]='others'
label_dict[3]='sad'
print (label_dict)
lstm_predictions=[]
h=0;
a=0
s=0
o=0
for p in pred:
    if(p == 0):
        a += 1
    elif(p == 1):
        h += 1
    elif(p == 2):
        o += 1
    elif(p == 3):
        s += 1
    lstm_predictions.append(label_dict[p])
pickle.dump(lstm_predictions,open("lsmt_predictions.p","wb"),protocol=2)   
print (len(lstm_predictions)) 
print(h,a,s,o)    

    

"""
#make predictions
model=pickle.load(open("model_cnn.p","rb"))
pred = model.predict_classes(X_cnn_test, verbose=2)
print (len(pred))

pred_labels=[]
for p in pred:
    if(p == 0):
        pred_labels.append('others')
    elif(p == 1):
        pred_labels.append('happy')
    elif(p == 2):
        pred_labels.append('sad')
    elif(p == 3):
        pred_labels.append('angry')

print (pred_labels)
pickle.dump(pred_labels,open("pred_labels_cnn.p","wb"))        
"""    








