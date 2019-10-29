import nltk
from textblob import TextBlob
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
import csv
import nltk
from collections import Counter
first_pronouns=['i','me','we','us','my','mine','our','ours.']
nltk.download('averaged_perceptron_tagger')
text=[]
tot_counts=[]
first_pronouncount=[]
sentiment=[]
subjectivity=[]
with open('finaltext.csv') as myFile1:  
	reader = csv.reader(myFile1,delimiter=',')
	for row in reader:
		text1=row[2]
		#text.append(text1)
#print(text)
		blob1=TextBlob(text1)
		print(format(blob1.sentiment))
		print(blob1.sentiment.polarity)
		sentiment.append(blob1.sentiment.polarity)
		print(blob1.sentiment.subjectivity)
		subjectivity.append(blob1.sentiment.subjectivity)

#sentence = "My name is Jocelyn"
		tokens=nltk.word_tokenize(text1.lower())
		#print(tokens)
		# token.append(tokens)
		print(tokens)
		text=nltk.Text(tokens)
		#print(text)
		tags=nltk.pos_tag(tokens)
		print(tags)
		# counts=Counter(tag for word,tag in tags)
		# print(counts)
		# tot_counts.append(counts)
		#print(tot_counts)
		count=0
		for w in tokens:
			if w.lower() in first_pronouns:
				count += 1
		#print(count)
		first_pronouncount.append(count)
print(len(first_pronouncount))
print(len(sentiment))
print(len(subjectivity))
final=[]
for i in range(len(sentiment)):
	a=[]
	a.append(first_pronouncount[i])
	a.append(sentiment[i])
	a.append(subjectivity[i])
	final.append(a)
print(len(final))
label=[]
from sklearn.model_selection import train_test_split
with open('finaltext.csv') as myFile1:  
	reader = csv.reader(myFile1,delimiter=',')
	for row in reader:
		text1=row[1]
		label.append(text1)
print(len(label))
X_train, X_test, y_train, y_test = train_test_split(final,label, test_size=0.2) # 70% training and 30% test
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



