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

q = []
s = []
l = []
#reading the files in pos folder
for file in os.listdir("c:/python27/IItb/movie_reviews/movie_reviews/pos"):
	if file.endswith(".txt"):
		p = os.path.join(file)
		q.append(p)
		path = 'c:/python27/IItb/movie_reviews/movie_reviews/pos/%s'%file
		with open(path,"r+") as myfile:
			r = myfile.read()
			s.append(r)
		k = 1
		l.append(k)
#reading the file in neg folder
for file in os.listdir("c:/python27/IItb/movie_reviews/movie_reviews/neg"):
	if file.endswith(".txt"):
		p = os.path.join(file)
		q.append(p)
		path = 'c:/python27/IItb/movie_reviews/movie_reviews/neg/%s'%file
		with open(path,"r") as myfile:
			r = myfile.read()
			s.append(r)
		k = 0
		l.append(k)


	
j = {'review':s,'sentiment':l}
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
p = pd.DataFrame(j,q)#panda DataFrame
X = vectorizer.fit_transform(p.review)#frequency matrix
y = p.sentiment

	
def multimonialnaivebaiyes(X,y):
	X_train,X_test,y_train,y_test = train_test_split(X.toarray(), y, test_size=0.3, random_state=0)
	mnb = MultinomialNB()#multimonial type naive bayes
	y_pred_mnb = mnb.fit(X_train,y_train).predict(X_test)#predicting sentiment of review with multimonial NB for test case
	y_pred_mnb2 = mnb.fit(X_test,y_test).predict(X_train)#predicting for training case
	cnf_matrix_mnb = confusion_matrix(y_test, y_pred_mnb)#confusion matrix for test case
	cnf_matrix_mnb2 = confusion_matrix(y_train, y_pred_mnb2)#confusion matrix fot training case
	fpr, tpr, threshold = roc_curve(y_train, y_pred_mnb2)#plotting ROC curve
	return y_pred_mnb,y_pred_mnb2,cnf_matrix_mnb,cnf_matrix_mnb2,fpr,tpr
	
def gaussiannaivebaiyes(X,y):
	X_train,X_test,y_train,y_test = train_test_split(X.toarray(), y, test_size=0.3, random_state=0)
	gnb = GaussianNB()#gaussian type naive bayes
	y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)#predicting with gaussian NB
	y_pred_gnb2 = gnb.fit(X_test,y_test).predict(X_train)#predicting for training case
	cnf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)#confusion matrix for test case
	cnf_matrix_gnb2 = confusion_matrix(y_train, y_pred_gnb2)#confusion matrix fot training case
	fpr, tpr, threshold = roc_curve(y_test, y_pred_gnb)
	return y_pred_gnb,y_pred_gnb2,cnf_matrix_gnb,cnf_matrix_gnb2,fpr,tpr


#plotting ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(multimonialnaivebaiyes(X,y)[4], multimonialnaivebaiyes(X,y)[5])
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


print(np.trapz(multimonialnaivebaiyes(X,y)[5], multimonialnaivebaiyes(X,y)[4]))	#printing ROC_auc_score
