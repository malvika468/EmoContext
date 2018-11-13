import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def load_preprocessed_data(filename):
	return pickle.load(open(filename,"rb"))

#=============================================================================
def predict_on_test_data(preprocessed_data):
	predicted_label_list = []
	progress = 1
	for data_point in preprocessed_data :
		conversation_sentences = []
		if len(data_point) == 3 : 
			conversation_sentences = data_point 
			predicted_label = get_emotion(conversation_sentences)
			predicted_label_list.append(predicted_label)
			print "Progress : ",progress, " Predicted : ",predicted_label
		else : 
			print "<==== ERROR ===>"
			sys.exit()
			predicted_label_list.append("others")
		progress += 1
	return predicted_label_list

def save_predicted_list_on_test_data(predicted_label_list):
	pickle.dump(predicted_label_list,open("predicted_label_list_test_data.p","wb"))

#===============================================================================
def get_accuracy_on_complete_data(preprocessed_data):
	actual_label_list,predicted_label_list = predict_on_training_data_with_accuracy(preprocessed_data)
	accuracy,fscore = calculate_accuracy(actual_label_list,predicted_label_list)
	return accuracy,fscore

def predict_on_training_data_with_accuracy(preprocessed_data):
	actual_label_list = []
	predicted_label_list = []
	progress = 1
	correct_prediction_count = 0
	for data_point in preprocessed_data :
		conversation_sentences = []
		if len(data_point) == 4 : 
			conversation_sentences = data_point[:3] 
			actual_label = data_point[3]
			predicted_label = get_emotion(conversation_sentences)
			actual_label_list.append(actual_label)
			predicted_label_list.append(predicted_label)
			if actual_label == predicted_label :
				correct_prediction_count += 1
			print "Progress : ",progress, " Actual/Predicted : ",actual_label ,"/",predicted_label ,
			print  " Current Accuracy : ",float(correct_prediction_count)/progress
			progress += 1
		else : 
			pass
	return actual_label_list,predicted_label_list


def calculate_accuracy(actual_label_list,predicted_label_list):
	if len(actual_label_list) == len(predicted_label_list):
		correct_prediction_count = 0
		for index in range(len(actual_label_list)):
			if actual_label_list[index] == predicted_label_list[index] :
				correct_prediction_count += 1
		return float(correct_prediction_count)/len(actual_label_list),f1_score(actual_label_list, predicted_label_list, average='weighted')  
	else:
		print "Size of actual and prediction list must be same!"
    

def get_emotion(conversation_sentences):
	vectorizer = load_vectorizer()
	vectorized_conversation_sentences = vectorizer.transform(conversation_sentences)
	classifier = load_classifer()
	first_emotion = classifier.predict(vectorized_conversation_sentences[0])[0]
	second_emotion = classifier.predict(vectorized_conversation_sentences[1])[0]
	third_emotion = classifier.predict(vectorized_conversation_sentences[2])[0]
	return get_final_emotion(first_emotion,second_emotion,third_emotion)

def load_vectorizer():
	return pickle.load(open("vectorizer.p","rb"))

def load_classifer():
	return pickle.load(open("trained_model.p","rb"))

def get_final_emotion(first_emotion,second_emotion,third_emotion):
	return third_emotion

def info_about_predicted_labels_test_data(predicted_label_list):
	count_happy = predicted_label_list.count("happy")
	count_sad = predicted_label_list.count("sad")
	count_angry = predicted_label_list.count("angry")
	count_others = predicted_label_list.count("others")
	total = len(predicted_label_list)
	print "Total sentences : ",total
	print "Count of Happy : ",count_happy, "  Percentage of Happy : ",float(count_happy)/total
	print "Count of Sad : ",count_sad, "  Percentage of sad : ",float(count_sad)/total
	print "Count of angry : ",count_angry, "  Percentage of angry : ",float(count_angry)/total
	print "Count of others : ",count_others, "  Percentage of others : ",float(count_others)/total

three_emotions_to_final_emotion = {
	"happy,*,happy" : "happy",
	"sad,*,sad" : "sad",
	"angry,*,angry" : "angry",
}

#=========================================================================================

if __name__ == "__main__" :
	filename = "preprocessed_data_v2_by_malvika.p"
	#filename = "preprocess_test_data_v2.p"
	preprocessed_data = load_preprocessed_data(filename)
	print preprocessed_data[:50]
	accuracy,fscore = get_accuracy_on_complete_data(preprocessed_data) 
   #print fscore
	#predicted_label_list = predict_on_test_data(preprocessed_data)
	print accuracy , fscore
	#save_predicted_list_on_test_data(predicted_label_list)
	#info_about_predicted_labels_test_data(predicted_label_list)
	