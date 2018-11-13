# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:41:18 2018

@author: user
"""

import pickle
from nltk.stem import *
from nltk.stem.porter import *
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import random
import string
from nltk.stem import WordNetLemmatizer
import re
from nltk import bigrams
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re
import unicodedata

"""
data = open("D:\\EMO\\starterkitdata\\devwithoutlabels.txt","r").readlines()
structured_test_data=[]
for line in data[1:]:
    split_line=line.split("\t")
    structured_test_data.append([split_line[1],split_line[2],split_line[3]])

pickle.dump(structured_test_data,open("structured_test_data.p","wb"))   

def stemmingAndStopwords(sentences):
	stop_words = set(stopwords.words('english'))
	stemmed = []
	stemmer = PorterStemmer()
	for sent in sentences:
		#print "Sentence = ",sent
		try:
			word_tokens = word_tokenize(sent)
			stemmed_sent = [stemmer.stem(word) for word in word_tokens]
			filtered_sentence = [w for w in stemmed_sent if not w in stop_words]
			filtered_sentence = " ".join(filtered_sentence)
			stemmed.append(filtered_sentence)
		except UnicodeDecodeError:
			pass
	return stemmed 

"""
def lemmatizeAndStopWords(sentences):
    wnl = WordNetLemmatizer()
    stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 
                  'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 
                  'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 
                  's', 'am', 'or', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 
                  'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 
                  'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'at', 'any', 'before', 
                  'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 
                  'that', 'because', 'over', 'so', 'can', 'did', 'not', 'now', 'under', 
                  'he', 'you', 'herself', 'has', 'just', 'too', 'only', 'myself', 
                  'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against',
                  'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
    stemmed = []
    for sent in sentences:
        try:
            word_tokens=word_tokenize(sent)
            stemmed_sent = [wnl.lemmatize(word) for word in word_tokens]
            l=len(stemmed_sent)
            if l>1:
                filtered_sentence = [w for w in stemmed_sent if not w in stop_words]
                filtered_sentence = " ".join(filtered_sentence)
                stemmed.append(str(filtered_sentence))
            elif l == 1:
                stemmed.append(str(stemmed_sent[0]))
            elif l == 0:
                stemmed.append('')
        except UnicodeDecodeError:
			pass
    return stemmed   
            
def removeEmoticons(text):
    
    emoji_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)
    u = unicode(text, "utf-8")
    u=emoji_pattern.sub(r'', u) # no emoji
    u=unicodedata.normalize('NFKD', u).encode('ascii','ignore')
    print u
    return u

def removePunctuationAndConvertToLowercase(sentence):
    processed = list()
    for line in sentence:
        #print line
        line=removeEmoticons(line)
        for c in string.punctuation:
            line=line.replace(c,"")
        line=line.lower()
        processed.append(line)
    return processed    

"""
structured_data=pickle.load(open("structured_data.p","r"))
train=structured_data[67]
turn_data=train[:3]
preprocess_train=removePunctuationAndConvertToLowercase(turn_data)
print "after punctuation" , preprocess_train
preprocess_train=lemmatizeAndStopWords(preprocess_train)
print "after lemma" , preprocess_train
"""

"""
structured_test_data=pickle.load(open("structured_test_data.p","r"))
preprocess_test_data = []
for test in structured_test_data:
    turn_data=test[:3]
    #print turn_data
    preprocess_test=removePunctuationAndConvertToLowercase(turn_data)
    preprocess_test=lemmatizeAndStopWords(preprocess_test)
    preprocess_test_data.append(preprocess_test)    
    
pickle.dump(preprocess_test_data,open("preprocess_test_data.p","wb"))

"""
preprocessed_test_data = pickle.load(open("preprocess_test_data.p","rb"))

for training_point in preprocessed_test_data[:20] :
	print training_point
 
"""     
vocab_final=[]
def convertunigram(var):
    vocab=[]
    tk=nltk.word_tokenize(var)
    uni=ngrams(tk,1)
    c=Counter(uni)
    for x in c:
        x=str(x)
        x=x[2:-3]
        vocab.append(x)
    return vocab   
    
def convertbigram(var):
    vocab=[]
    token = nltk.word_tokenize(var)
    bigram = ngrams(token,2)
    c=Counter(bigram)
    for x in c:
        s=str(x)
        s=s[1:-1]
        s=s.split(',')
        #print s[0][1:-1] + ' '+  s[1][2:-1]
        vocab.append(s[0][1:-1] + ' '+  s[1][2:-1])
    return vocab    

def convertToVec(var):
    
    vocab = convertunigram(var)
    vocab += convertbigram(var)
    #print "vocab" , vocab
    hot_encoder = [0] * len(vocab_final)
    binary_bag = dict(zip(vocab_final, hot_encoder))
    for word in vocab:
        #print word
        if word in vocab_final and binary_bag[word] == 0:
            binary_bag[word]=1
        
    vec=[]
    for k,v in binary_bag.iteritems():
        vec.append(v)
    #print "Binary vector" , vec
    return vec
     
preprocessed_data = pickle.load(open("preprocess_data.p","rb"))
for training_point in preprocessed_data:
    training_point = training_point[:3]
    #print training_point
    for train in training_point:
             #print str(train)
             vocab_final += convertunigram(str(train))
             vocab_final += convertbigram(str(train))

pickle.dump(vocab_final,open("vocab_final.p","wb"))
print "vocabulary" , len(vocab_final)
"""
## create feature vectors

"""

vocab_final = pickle.load(open("vocab_final.p","rb"))
print len(vocab_final)
preprocessed_data = pickle.load(open("preprocess_data.p","rb"))

X_train=[]
for training_point in preprocessed_data:
    #print training_point
    try:
        label = training_point[3].rstrip()
        training_point = training_point[:3]
        #print training_point , label
        ls=convertToVec(str(training_point[0]))
        ls.append(label)
        X_train.append(ls)
        ls=convertToVec(str(training_point[2]))
        ls.append(label)
        X_train.append(ls)
    except:IndexError
    pass
    

pickle.dump(X_train,open("EMO_train_data2.p","wb"))
print len(X_train)    

   
X_train=pickle.load(open("EMO_train_data.p","rb"))   
print "data",len(X_train)       
      
"""       








