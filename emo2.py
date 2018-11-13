# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 15:13:26 2018

@author: user
"""

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import  GradientBoostingClassifier
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import string
"""
def load_preprocessed_data():
	return pickle.load(open("preprocess_data.p","rb"))

def get_list_of_sentences_and_label(preprocessed_data):
	list_of_sentences = []
	list_of_labels = []
	invalid_data = []
	for data_point in preprocessed_data :
		if len(data_point) == 4 : 
			#print " Data Point : ",data_point
			list_of_sentences.append(data_point[0]) # User 1 - Turn 1
			list_of_sentences.append(data_point[2]) # User 1 - Turn 3
			list_of_labels.append(data_point[3]) # label
			list_of_labels.append(data_point[3]) # same label for next sentence
		else : 
			invalid_data.append(data_point)
	return list_of_sentences, list_of_labels, invalid_data

def get_features_and_vectorized_sentences(list_of_sentences):
	vectorizer = TfidfVectorizer(max_features=None)
	vectorizer.fit(list_of_sentences)
	features = vectorizer.get_feature_names()
	vectorized_sentences = vectorizer.transform(list_of_sentences)
	return features, vectorized_sentences

def save_vectorized_sentences_and_labels(vectorized_sentences,list_of_labels):
	pickle.dump(vectorized_sentences,open("vectorized_sentences.p","wb"))
	pickle.dump(list_of_labels,open("list_of_labels.p","wb"))

preprocessed_data = load_preprocessed_data()
list_of_sentences, list_of_labels, invalid_data = get_list_of_sentences_and_label(preprocessed_data)
features, vectorized_sentences = get_features_and_vectorized_sentences(list_of_sentences)
save_vectorized_sentences_and_labels(vectorized_sentences,list_of_labels)
print vectorized_sentences.shape


def load_vectorized_sentences_and_labels():
	vectorized_sentences = pickle.load(open("vectorized_sentences.p","rb"))
	list_of_labels = pickle.load(open("list_of_labels.p","rb"))
	return vectorized_sentences, list_of_labels

def display_some_points(vectorized_sentences,list_of_labels):
	print "Number of features : ",vectorized_sentences.shape
	print "vectorized_sentences : ",vectorized_sentences[:10]
	print "list_of_labels : ",list_of_labels[:10]

def get_training_and_testing_data(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	return X_train, X_test, y_train, y_test


def train_model(X_train,y_train):
    clf=svm.SVC(decision_function_shape='ovo' , kernel='linear')
    clf=GradientBoostingClassifier(n_estimators=100,min_samples_leaf=10)
    #clf = MultinomialNB()
    clf.fit(X_train,y_train)
    return clf
    

def save_trained_model(trained_model):
	pickle.dump(trained_model,open("trained_model.p","wb"))

def get_accuracy_on_test_data(trained_model,X_test,y_test):
	return trained_model.score(X_test,y_test)


vectorized_sentences, list_of_labels = load_vectorized_sentences_and_labels()
#display_some_points(vectorized_sentences, list_of_labels)
X_train, X_test, y_train, y_test = get_training_and_testing_data(vectorized_sentences, list_of_labels)
trained_model = train_model(X_train,y_train)
save_trained_model(trained_model)
print "Accuracy : ",get_accuracy_on_test_data(trained_model,X_test,y_test)

"""




	      

### prediction on test data


"""
data = open("D:\\EMO\\starterkitdata\\devwithoutlabels.txt","r").readlines()
test_data=[]
for line in data:
    split_line=line.split("\t")
    test_data.append([split_line[1],split_line[2],split_line[3].rstrip("\n")])

pickle.dump(test_data,open("test_data.p","wb")) 


test_data=pickle.load(open("test_data.p","r"))

preprocess_test_data = []
for test in test_data:
    preprocess_test=removePunctuationAndConvertToLowercase(test)
    preprocess_test=lemmatizeAndStopWords(preprocess_test)
    preprocess_test_data.append(preprocess_test)    
 
#print preprocess_test_data    
pickle.dump(preprocess_test_data,open("preprocess_test_data.p","wb"))
"""

    
preprocess_test_data=pickle.load(open("preprocess_test_data.p","r"))
test_data_final=[]

for test_data in preprocess_test_data:
    test=[]
    

print test_data_final