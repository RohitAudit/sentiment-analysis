import pandas as pd
import numpy as np
import os
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def readfile():
	filepath = []
	review = []
	sentiment = []
	#reading the files in pos folder
	for file in os.listdir("C:/Users/roaggarw/Documents/NLP/sentiment_analysis/sentiment-analysis/movie_reviews/pos"):
		if file.endswith(".txt"):
			p = os.path.join(file)
			filepath.append(p)
			path = 'C:/Users/roaggarw/Documents/NLP/sentiment_analysis/sentiment-analysis/movie_reviews/pos/%s'%file
			with open(path,"r+") as myfile:
				review.append(myfile.read())
			sentiment.append(1)
	#reading the file in neg folder
	for file in os.listdir("C:/Users/roaggarw/Documents/NLP/sentiment_analysis/sentiment-analysis/movie_reviews/neg"):
		if file.endswith(".txt"):
			p = os.path.join(file)
			filepath.append(p)
			path = 'C:/Users/roaggarw/Documents/NLP/sentiment_analysis/sentiment-analysis/movie_reviews/neg/%s'%file
			with open(path,"r") as myfile:
				review.append(myfile.read())
			sentiment.append(0)
	j = {'review':review,'sentiment':sentiment}
	return pd.DataFrame(j)

def plotROC(x,y):
	plt.title('Receiver Operating Characteristic')
	plt.plot(x,y)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()

def splitTrainTest(Dataframe):
	vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
	X = vectorizer.fit_transform(readfile().review)#frequency matrix
	y = readfile().sentiment
	X_train,X_test,y_train,y_test = train_test_split(X.toarray(), y, test_size=0.3, random_state=0)
	trainset={'review':X_train,'sentiment':y_train}
	testset={'review':X_test,'sentiment':y_test}
	return trainset,testset

	
def multimonialnaivebaiyes(test):
	mnb = MultinomialNB()#multimonial type naive bayes
	train = splitTrainTest(readfile())[0]
	y_pred_mnb = mnb.fit(train['review'],train['sentiment']).predict(test)#predicting sentiment of review with multimonial NB for test case
	return y_pred_mnb
	
def gaussiannaivebaiyes(test):
	gnb = GaussianNB()#gaussian type naive bayes
	train = splitTrainTest(readfile())[0]
	y_pred_gnb = gnb.fit(train['review'],train['sentiment']).predict(test)#predicting with gaussian NB
	return y_pred_gnb

def importantParam(predicted,original):
	cnf_matrix_gnb = confusion_matrix(original, predicted)
	fpr, tpr, threshold = roc_curve(original, predicted)
	AUC_score = np.trapz(fpr,tpr)
	return cnf_matrix_gnb,AUC_score , fpr, tpr

	
test = splitTrainTest(readfile())[1]
test_predict = multimonialnaivebaiyes(test['review'])
#test_predict = gaussiannaivebaiyes(test.review)
param = importantParam(test_predict,test['sentiment'])
print(param[1])

plotROC(param[2],param[3])
