# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 13:48:50 2018

@author: 63483
"""

import numpy as np
import pickle
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Conv1D
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import accuracy_score

## word to vec
def func(st):
    st=st.split(' ')
    lst=[]
    for w in st:
        lst.append(w)
    return lst      

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
  
#print (list_of_sentences)    
model = Word2Vec(list_of_sentences,size=5,window=2,min_count=1)   
words = list(model.wv.vocab)
print (model)
print (len(words))
"""
X_cnn=np.zeros((0,max_len,5))
for sent in list_of_sentences:
    vec=np.zeros((0,5))
    for word in sent:
        tmp=model[word]
        vec=np.vstack([vec,np.array(tmp)])
    s=vec.shape[0]+1
    for i in range(s,max_len+1):
        vec=np.vstack([vec,np.zeros((1,5))])
    X_cnn=np.vstack([X_cnn,vec[np.newaxis]])
"""
X_cnn=[]
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
    X_cnn.append(np.array(vec))

X_cnn=np.array(X_cnn)
print (X_cnn.shape)    


Y_cnn=np.zeros((0,4))
label_vocab=['angry','others','happy','sad']
one_hot=[0]*len(label_vocab)
binary_bag=dict(zip(label_vocab,one_hot))
print (binary_bag)
for lab in list_of_labels:
    if binary_bag[lab] == 0:
        binary_bag[lab]=1
    vec=[]
    for v in binary_bag.values():
        vec.append(v)
    Y_cnn=np.vstack([Y_cnn,vec])

print (Y_cnn.shape)

rows=max_len
cols=5

X_train, X_val, y_train, y_val = train_test_split(X_cnn, Y_cnn, test_size=0.2, random_state=10)

X_train = X_train.reshape(X_train.shape[0],rows,cols, 1)
X_val = X_val.reshape(X_val.shape[0],rows,cols, 1)
    
print (X_train.shape)
 # bulding cnn model

batch_size = 256
num_classes = 4
epochs = 100

input_shape = (rows,cols,1)

model=Sequential()
model.add(Conv2D(16, kernel_size=(2,2) ,activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(2,2))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

print (model.summary())

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

score = model.evaluate(X_val, y_val, verbose=0)
pickle.dump(model,open("model_cnn.p","wb"))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#make predictions
"""
model=pickle.load(open("model_cnn.p","rb"))
pred = model.predict_classes(X_val, verbose=2)
print (len(X_val))
print (pred)
c1=0
c2=0
c3=0
c4=0
pred_labels=[]
for p in pred:
    if(p == 0):
        c1=c1+1
    elif(p == 1):
        c2 += 1
    elif(p == 2):
        c3 += 1
    elif (p == 3):
        c4 += 1
print (c1,c2,c3,c4)        

"""
     
     

    
    
   