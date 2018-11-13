import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def load_preprocessed_data():
	return pickle.load(open("preprocessed_data_v2_by_malvika.p","rb"))

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

def print_some_information_about_data (list_of_sentences, list_of_labels, invalid_data):
	print "Total correct sentences : ",len(list_of_sentences)
	print "Length of Labels (should be same as number of sentences) : ",len(list_of_labels)
	print "Count of Angry sentences : ",list_of_labels.count("angry")
	print "Count of Sad sentences : ",list_of_labels.count("sad")
	print "Count of Happy sentences : ",list_of_labels.count("happy")
	print "Count of Others sentences : ",list_of_labels.count("others") 
	print "Total invalid sentences : ",len(invalid_data)

def get_vectorizer_and_vectorized_sentences(list_of_sentences):
	vectorizer = TfidfVectorizer(max_features=None)
	vectorizer.fit(list_of_sentences)
	features = vectorizer.get_feature_names()
	vectorized_sentences = vectorizer.transform(list_of_sentences)
	return vectorizer, vectorized_sentences

def save_vectorizer_and_vectorized_sentences_and_labels(vectorizer,vectorized_sentences,list_of_labels):
	pickle.dump(vectorizer,open("vectorizer.p","wb"))
	pickle.dump(vectorized_sentences,open("vectorized_sentences.p","wb"))
	pickle.dump(list_of_labels,open("list_of_labels.p","wb"))

if __name__ == "__main__":
	preprocessed_data = load_preprocessed_data()
	list_of_sentences, list_of_labels, invalid_data = get_list_of_sentences_and_label(preprocessed_data)
	print_some_information_about_data (list_of_sentences, list_of_labels, invalid_data)
	vectorizer, vectorized_sentences = get_vectorizer_and_vectorized_sentences(list_of_sentences)
	save_vectorizer_and_vectorized_sentences_and_labels(vectorizer, vectorized_sentences,list_of_labels)
