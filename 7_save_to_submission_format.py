import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

def load_predicted_list_on_test_data(filename):
	return pickle.load(open(filename,"rb"))

def read_and_save_to_submission_format(inputtestfile, predicted_label_list):
	submission_filename = "test.txt"
	inputtestfilecontents = open(inputtestfile,"r").readlines()
	submission_file = open(submission_filename,"w")
	submission_file = open(submission_filename,"a")
	submission_file.write(inputtestfilecontents[0].strip()+"\tlabel\n")
	for index in range(0,len(predicted_label_list)):
		sentences = inputtestfilecontents[index+1].strip()
		#print "Sentence : ",sentences
		label = predicted_label_list[index]
		#print "Label : ",label
		output = sentences + "\t" + label.strip() + "\n"
		#print "Output : ",output
		submission_file.write(output)
	submission_file.close()

if __name__ == "__main__" :
	devwithoutlabel = "devwithoutlabels.txt"
	predictedlabelfile = "predicted_label_list_test_data.p"
	predicted_label_list = load_predicted_list_on_test_data(predictedlabelfile)
	read_and_save_to_submission_format(devwithoutlabel,predicted_label_list)

	