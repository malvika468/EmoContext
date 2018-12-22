# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:58:33 2018

@author: 63483
"""

import numpy as np
import pickle
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import keras
import random
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import model_from_json
from collections import defaultdict
import operator
import string

def func(st):
    st=st.split(' ')
    lst=[]
    for w in st:
        if(all(c in string.digits for c in w)):
            continue
        if(len(w) > 0):
            lst.append(w)
    return lst      

def PadAndAppend(sentence,maxlen):
    dist=maxlen-len(sentence)
    for i in range(dist):
        sentence.append('pad')
    sentence.append('eos')
    return sentence     

preprocess_train_data = pickle.load(open("C:\\Users\\63483\\Desktop\\EMO\\preprocess_data.p","rb")) 
list_of_sentences=[]
list_of_labels=[]
max_len=0
for tp in preprocess_train_data:
    st1=func(tp[0])
    max_len=max(len(st1),max_len)
    list_of_sentences.append(st1)
    st2=func(tp[2])
    max_len=max(len(st2),max_len)
    list_of_sentences.append(st2)
    list_of_labels.append(tp[3])
    list_of_labels.append(tp[3])

print (max_len)
data_x=[]
for sent in list_of_sentences:
    data_x.append(PadAndAppend(sent,max_len))


    
freq_dict=defaultdict( int )
for tp in data_x:
    for w in tp:
        freq_dict[w] += 1
#print(freq_dict)
new_data_x_train=[]
for tp in data_x:
        sent=[]
        for w in tp:
            if(freq_dict[w] > 1):
                sent.append(w)
            elif(freq_dict[w] == 1):
                sent.append("unk")
        new_data_x_train.append(sent)   
print(len(new_data_x_train))
#print(new_data_x_train[1] , list_of_labels[1])
  
model = Word2Vec(new_data_x_train,size=5,window=2,min_count=1)   
pickle.dump(model,open("word2vec_emo.p","wb"))
print("saved word2vec model ")
words = list(model.wv.vocab)
print (model)
print (len(words))

X_rnn=np.empty([len(new_data_x_train),len(new_data_x_train[0]),5])
for i,tp in enumerate(new_data_x_train):
        for j,w in enumerate(tp):
            X_rnn[i][j] = model[w]

print (X_rnn.shape) 
label_dict={}
label_dict['angry']=0
label_dict['happy']=1
label_dict['others']=2
label_dict['sad']=3
print (label_dict)
encoded_y=[]
for y in list_of_labels:
    encoded_y.append(label_dict[y])
one_hot_encoded_y = to_categorical(encoded_y)
one_hot_encoded_y=np.array(one_hot_encoded_y)
print (one_hot_encoded_y.shape)


X_train, X_val, y_train, y_val = train_test_split(X_rnn, one_hot_encoded_y, test_size=0.2, random_state=1)

"""

#building a simple RNN model
model = Sequential()
model.add(keras.layers.InputLayer(input_shape=(5,5))) 
model.add(keras.layers.recurrent.SimpleRNN(units = 100, activation='relu',
    use_bias=True))


model.add(keras.layers.Dense(units=4, input_dim=5,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train , epochs=10, batch_size=120 ,validation_data=(X_val,y_val))
#pred=model.predict(X_train[6].reshape(1,5,5))
"""
## building a simple LSTM model

model = Sequential()
model.add(keras.layers.InputLayer(input_shape=(57,5))) 
model.add(keras.layers.LSTM(units = 100, activation='relu',
    use_bias=True))
model.add(keras.layers.Dense(units=4, input_dim=57,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train , epochs=6, batch_size=120)
score = model.evaluate(X_val, y_val, verbose=1)
print ("LSTM for Emo***",score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model_json = model.to_json()
with open("model_lstm_emo.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_lstm_emo.h5")
print("Saved model to disk")