import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import one_hot
sentences = ["To Sherlock Holmes she is always the woman.", "I have seldom heard him mention her under any other name."]
bigrams = []
for sentence in sentences:
    sequence = word_tokenize(sentence) 
    bigrams.extend(list(ngrams(sequence, 2)))
#print(bigrams)
freq_dist = nltk.FreqDist(bigrams)
prob_dist = nltk.MLEProbDist(freq_dist)
number_of_bigrams = freq_dist.N()
#Finding the unigram representation
from sklearn.feature_extraction.text import CountVectorizer
# vectorizer=CountVectorizer()
# unigram_training_words=vectorizer.fit_transform(bigrams)
# print( unigram_training_words.shape)
import pandas as pd
#df = pd.read_csv('Consumer_Complaints.csv')
# df=pd.read_csv('finaltext.csv',delimiter='\t',encoding='utf-8')
# print(df.head())
# print(df[text])
label=[]
text=[]
import csv
from sklearn.model_selection import train_test_split
with open('finaltext.csv') as myFile1:  
	reader = csv.reader(myFile1,delimiter=',')
	for row in reader:
		text1=row[1]
		label.append(text1)
print(label)
with open('finaltext.csv') as myFile1:  
	reader = csv.reader(myFile1,delimiter=',')
	for row in reader:
		text1=row[2]
		text.append(text1)
print(len(text))
# def tokens(x):
# return x.split(',')
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(text)
#print(features.shape)
print((features.shape[1]))
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features,label, test_size=0.2) # 70% training and 30% test
from sklearn import svm
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred,average='micro'))
