import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

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
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	return clf

def save_trained_model(trained_model):
	pickle.dump(trained_model,open("trained_model.p","wb"))

def get_accuracy_on_test_data(trained_model,X_test,y_test):
	return trained_model.score(X_test,y_test)

def print_confusion_matrix(trained_model,X_test,y_test):
	y_pred = trained_model.predict(X_test)
	print confusion_matrix(y_test, y_pred,labels=["angry", "happy", "sad", "others"])



vectorized_sentences, list_of_labels = load_vectorized_sentences_and_labels()
#display_some_points(vectorized_sentences, list_of_labels)
X_train, X_test, y_train, y_test = get_training_and_testing_data(vectorized_sentences, list_of_labels)
trained_model = train_model(X_train,y_train)
save_trained_model(trained_model)
y_pred=[]
for test in X_test:
    p=trained_model.predict(test)
    y_pred.append(p)
print "Accuracy : ",get_accuracy_on_test_data(trained_model,X_test,y_test)
print "F_score:" ,f1_score(y_pred, y_test, average='micro') 