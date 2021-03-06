import time
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
# AA CHANGE: Added for debugging and making features
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kPAGE_FIELD = 'page'
kTROPE_FIELD = 'trope'
# IMDB Data Processing
kTITLE_FIELD = 'title'
kGENRE_FIELD = 'genre'
imdb_dict = list(DictReader(open("imdb_dict.csv", 'r')))
title_vec = []
genre_vec = []
for ii in imdb_dict:
	title_vec.append(ii[kTITLE_FIELD])
	genre_vec.append(ii[kGENRE_FIELD])

imdb_dict = dict(zip(title_vec, genre_vec))

global_count = 0
# Stop word and stemmer vocabulary
global_stop_list = ["a","am","an","any","is","it","he","she","they","i","her","him","we","then","out","he's","of","in","and","that","she's","i'm","the"]
global_dead_list = ["dead","death","die","dies","died"]
global_kill_list = ["kill","kills","killed","killing"]

# So the way that I actually look at different feature extraction methods are to uncomment the vectorizer I want in the featurizer class
# then uncomment the associated feature extraction calls in main. To pick and choose features in the custom analyzer, I simply commented out 
# certain sections. I know this isn't the best coding practice, but it was fast and effective.   
# The script requires imdb_dict.csv in addition to the training and test csv's. 
# This script now prints out computation time. In this configuration it's about 17 seconds. 
# The script is currently configured for combo processing and best observed performance. 

class Featurizer:
	def __init__(self):
		# Vecotrizers for built-in or combo processing		
		self.vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,8))# analyzer='word' 'char' 'char_wb', strip_accents = 'ascii', ngram_range=(1,8), stop_words = 'english', max_df = 0.999, min_df = 0.0001
		#self.vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,8))
		# Vectorizers for custom analyzer feature extraction		
		#self.vectorizer = TfidfVectorizer(analyzer=my_analyzer)
		#self.vectorizer = CountVectorizer(analyzer=my_analyzer)
	def train_feature(self, examples):
		return self.vectorizer.fit_transform(examples)

	def test_feature(self, examples):
		return self.vectorizer.transform(examples)

	def show_top10(self, classifier, categories):
		feature_names = np.asarray(self.vectorizer.get_feature_names())
		if len(categories) == 2:
			top10 = np.argsort(classifier.coef_[0])[-20:]
			bottom10 = np.argsort(classifier.coef_[0])[:20]
			print("Pos: %s" % " ".join(feature_names[top10]))
			print("Neg: %s" % " ".join(feature_names[bottom10]))
		else:
			for i, category in enumerate(categories):
				top10 = np.argsort(classifier.coef_[i])[-20:]
				print("%s: %s" % (category, " ".join(feature_names[top10])))


# AA CHANGE: Added from example script
def accuracy(classifier, x, y):
	predictions = classifier.predict(x)
	#cm = confusion_matrix(y, predictions)

	print("Accuracy: %f" % accuracy_score(y, predictions))
	
def my_analyzer(s):
	global global_stop_list
	global global_dead_list
	global global_kill_list
	global imdb_dict
	stop_val = False
	dead_val = False	
	kill_val = False
	name_val = False
	s1 = s[kTEXT_FIELD]
	# Turn it all lowercase for better processing
	s11 = s1.lower()
	s2 = s11.split()
	len_val = len(s2)
	# Stop word and stem processing
	for jj in range(0,len_val):
		s3 = s2[jj]
		for word in global_stop_list:
			if(word==s3):
				stop_val = True

		for word in global_kill_list:
			if(word==s3):
				kill_val = True

		for word in global_dead_list:
			if(word==s3):
				dead_val = True	
		
		if(stop_val):
			break
		elif(dead_val):
			yield "dead"
		elif(kill_val):
			yield "kill"
		else:
		# Yield the word, 2-gram and 3-gram
			yield s3

		if(jj>0):
			yield s2[jj-1]+" "+s3

		if(jj<(len_val-2)):
			yield s3+" "+s2[jj+1]

		if(jj>1):
			yield s2[jj-2]+" "+s2[jj-1]+" "+s3

		if(jj<(len_val-3)):
			yield s3+" "+s2[jj+1]+" "+s2[jj+2]	

	# Yield the whole sentence
	yield s1
	# NLTK Processing to yield parts of speech
	words = word_tokenize(s1)
	pos_vals = pos_tag(words)
	for ii in pos_vals:
		yield str(ii[1])
	# Yield Genres
	key_str = s[kPAGE_FIELD]
	genre_string = imdb_dict[key_str]
	genre_list = genre_string.split()
	for ii in genre_list:
		yield ii
	


if __name__ == "__main__":
	start_time = time.time()
	# Cast to list to keep it all in memory
	train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
	test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

	feat = Featurizer()

	labels = []
	for line in train:
		if not line[kTARGET_FIELD] in labels:
			labels.append(line[kTARGET_FIELD])

	print("Label set: %s" % str(labels))
  
	# Uncomment for custom analysis	
	#x_train = feat.train_feature(x for x in train)
	#x_test = feat.test_feature(x for x in test)
	# Uncomment for built-in analysis
	#x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
	#x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)
	# Uncomment for Combo analysis
	x_train = feat.train_feature(str(x[kTEXT_FIELD]+" "+str(imdb_dict[x[kPAGE_FIELD]])) for x in train)
	x_test = feat.test_feature(str(x[kTEXT_FIELD]+" "+str(imdb_dict[x[kPAGE_FIELD]])) for x in test)

	y_train = array(list(labels.index(x[kTARGET_FIELD])
						 for x in train))

	print(len(train), len(y_train))
	print(set(y_train))

	# Train classifier
	lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
	lr.fit(x_train, y_train)

	feat.show_top10(lr, labels)
	# AA CHANGE: Add debug accuracy 
	print("TRAIN\n-------------------------")
	accuracy(lr,x_train,y_train)
	predictions = lr.predict(x_test)
	o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
	o.writeheader()
	for ii, pp in zip([x['id'] for x in test], predictions):
		d = {'id': ii, 'spoiler': labels[pp]}
		o.writerow(d)

	end_time = time.time()
	print end_time-start_time
