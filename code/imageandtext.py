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
ngram_label=[]
text=[]
import csv
from sklearn.model_selection import train_test_split
with open('combine.csv') as myFile1:  
	reader = csv.reader(myFile1,delimiter=',')
	next(reader)
	for row in reader:
		text1=row[1]

		ngram_label.append(text1)
print("ngram_label:: ",len(ngram_label))
with open('combine.csv') as myFile1:  
	reader = csv.reader(myFile1,delimiter=',')
	next(reader)
	for row in reader:
		text1=row[2]
		text.append(text1)
print(len(text))
# def tokens(x):
# return x.split(',')
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
ngram_features = tfidf.fit_transform(text)
#print(features.shape)
print("ngram_features :: ",(ngram_features.shape))
import numpy as np
import time
import urllib
import cv2
import csv 
import cv2
import numpy as np
import numpy as np
#import urllib
import cv2
import urllib.request
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import merge, Input
import os

from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.request import Request, urlopen
from operator import itemgetter
import csv
dict_={}
with open('instagram_experiment_1_annotations (1).tsv') as myFile:  
	reader = csv.reader(myFile,delimiter='\t')

	for row in reader:
		pid=row[0]
		#print(pid)
		if pid not in dict_:
			dict_[pid]=[row[3]]
			#print(len(dict_[pid]))
		elif pid in dict_ :
			dict_[pid].append(row[3])
dict1={}
for key in dict_:
	lst=dict_[key]
	label=max(lst,key=lst.count)
	dict1[key]=label
#print(dict1)
text1={}
with open('instagram_experiment_1_posts.tsv') as myFile1:  
	reader = csv.reader(myFile1,delimiter='\t')
	
	for rows in reader:
		k = rows[0]
		v = rows[2]
		w = rows[3]
		text1[k] = v
		#text[k] = w
#print(text1)
text2={}
with open('instagram_experiment_1_posts.tsv') as myFile1:  
	reader = csv.reader(myFile1,delimiter='\t')
	
	for rows in reader:
		k = rows[0]
		v = rows[2]
		w = rows[3]
		text2[k] = w
		#text[k] = w
print(len(text2))
with open('combine1.csv', 'w') as f:
	for key in dict1.keys():
		f.write("%s,%s,%s,%s\n"%(key,dict1[key],text1[key],text2[key]))
data=[]
post_id=[]
label=[]
list_Overallsentiment=[]
images=[]
with open('combine1.csv') as myFile:  
	reader = csv.DictReader(myFile,delimiter='\t')
	for row in reader:
		y=[]
		# y.append((row['postid']))
		# y.append((row['is_this_post_sarcastic']))
		y.append(rows[0])
		y.append(rows[1])
		data.append(y)
print("data:::",len(data))
pid=[]
import cv2
import os
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

list_image=[]
list_label=[]
folder="images"
list_filename=[]
for filename in os.listdir(folder):
	# print(filename)
	
	temp=filename.split(".")
	
	id_=temp[0]
	list_filename.append(id_)
	# print("id_:::",id_)
	# file_=folder+"/"+filename
	# img = cv2.imread(file_)
	# # print(img)
	# for i in range(len(data)):
	#  	if id_==data[i][0] :
	#  		list_image.append(img)
	#  		list_label.append(data[i][1])
count=0
n_count=1
xcount=0
for i in range(len(data)):
	# if(xcount>10):
	# 	break
	if data[i][0] in list_filename :
		xcount=xcount+1
		filename=data[i][0]+'.jpg'
		file_=folder+"/"+filename
		img = cv2.imread(file_)
		im = cv2.resize(img, (224, 224)).astype(np.float64)
		# print(img.shape)
		list_image.append(im)
		list_label.append(data[i][1])
		count=count+1

	else:
		n_count=n_count+1
		xcount=xcount+1
		img=np.zeros((224,224,3))
		# print(img.shape)
		im = cv2.resize(img, (224, 224)).astype(np.float64)
		list_image.append(img)
		list_label.append(data[i][1])


	# if img is not None:
	# 	print("apoo")
print("len of list_image",len(list_image))
print("len of list_label",len(list_label))
image_input = Input(shape=(224,224,3))
model_vgg = VGG16(include_top=False,weights="imagenet",input_tensor=image_input)
vgg16_feature_list = []

#for idx, dirname in enumerate(subdir):
    # get the directory names, i.e., 'dogs' or 'cats'
    # ...
#[[for x in range(len(images))] for x in range(len(images)) for x in range(len(images))   
count=0
for i in range(len(list_image)):
        # process the files under the directory 'dogs' or 'cats'
        # ...
        
	# img = image.load_img(images[i], target_size=(224, 224))
	# img_data = .img_to_array(img)
	im=list_image[i]
	count=count+1
	#imgUMat = cv2.UMat(im)
	#im=cv2.cvtColor(cv2.UMat(imUMat), cv2.COLOR_RGB2GRAY)
	img_data = np.expand_dims(im, axis=0)
	img_data = preprocess_input(img_data)
	vgg16_feature = model_vgg.predict(img_data)
	vgg16_feature_np = np.array(vgg16_feature)
	vgg16_feature_list.append(vgg16_feature_np.flatten())
print("count",count)
print("fet list :",len(vgg16_feature_list))
vgg16_feature_list_np = np.array(vgg16_feature_list)
# print(vgg16_feature_list_np)
print("vgg shape",vgg16_feature_list_np.shape)
#print((data))
import pandas as pd 
import numpy as np
from numpy import array
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
# for i in range(len(vgg16_feature_list_np)):
# 	ngram_features[i,:vgg16_feature_list_np[i]]=0
final_features=[]
final_label=[]
ngram_features=ngram_features.todense()
print(ngram_features.shape)
count=0
for i in range(len(vgg16_feature_list_np)) :

	list_=[]
	t=np.array(ngram_features[i])
	t=t.reshape(-1)	
	# print("ttt:::;",np.array(t).shape)
	t1=t.tolist()
	# print(type(t))
	
	s=np.array(vgg16_feature_list_np[i]).astype(np.float64)
	s1=s.tolist()
	# print(type(s))
	# print("sss:::;",np.array(s1).shape)
	x=[t1,s1]
	r=sum(x, [])
	# print("r ::::",np.array(r).shape)
	final_features.append(r)
	final_label.append(ngram_label[i])
	count=count+1
	
print(len(final_features))
print(len(final_label))
# final_label = np.asarray([np.asarray(row, dtype=float) for row in final_label], dtype=float)
import keras
from sklearn import svm
from keras.layers import Masking
from sklearn.svm import SVC
from keras.layers import concatenate

X_train, X_test, y_train, y_test = train_test_split(np.array(final_features),final_label, test_size=0.2)
print("x train",np.array(X_train).shape)
print("y train",np.array(y_train).shape)
from sklearn import svm
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred,average='micro'))

#y_tr = np.asarray([np.asarray(row, dtype=float) for row in y_tr], dtype=float)
