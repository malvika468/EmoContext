# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:31:13 2018

@author: 63483
"""

import numpy as np
import pickle
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling1D,Conv1D
from keras.layers.normalization import BatchNormalization



## word to vec
def func(st):
    st=st.split(' ')
    lst=[]
    for w in st:
        lst.append(w)
    return lst      

preprocess_test_data = pickle.load(open("C:\\Users\\63483\\Desktop\\EMO\\preprocess_test_data.p","rb")) 
list_of_sentences=[]
max_len=56 
for tp in preprocess_test_data:
    if(len(tp[2]) > 0):
        st2=func(tp[2])
        list_of_sentences.append(st2)
    
    
print (len(list_of_sentences))    
model = Word2Vec(list_of_sentences,size=5,window=2,min_count=1)   
words = list(model.wv.vocab)
print (model)

#X_cnn_test=np.zeros((0,max_len,5))
"""
for sent in list_of_sentences:
    #print (sent)
    vec=np.zeros((0,5))
    for word in sent:
        tmp=model[word]
        vec=np.vstack([vec,np.array(tmp)])
    s=vec.shape[0]+1
    for i in range(s,max_len+1):
        vec=np.vstack([vec,np.zeros((1,5))])
    X_cnn_test=np.vstack([X_cnn_test,vec[np.newaxis]])
"""
X_cnn_test=[]
for sent in list_of_sentences:
    vec=[]
    c=0;
    for word in sent:
        tmp=model[word]
        c += 1
        vec.append(np.array(tmp))
    c += 1
    for i in range(c,max_len+1):
        listofzeros = [0] * 5
        vec.append(np.array(listofzeros))
    X_cnn_test.append(np.array(vec))

X_cnn_test=np.array(X_cnn_test)
print (X_cnn_test.shape)    
   
rows=max_len
cols=5  
X_cnn_test = X_cnn_test.reshape(X_cnn_test.shape[0],rows,cols,1)

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
      








