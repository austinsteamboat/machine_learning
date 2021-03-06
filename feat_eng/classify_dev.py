from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
# AA CHANGE: Added for debugging and making features
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kPAGE_FIELD = 'page'
kTROPE_FIELD = 'trope'

class Featurizer:
    def __init__(self):
        #self.vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,3),stop_words='english')
	self.vectorizer = CountVectorizer()

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
	    print("Got this far...")
            top10 = np.argsort(classifier.coef_[0])[-10:]
	    print("Got this far as well")
            bottom10 = np.argsort(classifier.coef_[0])[:10]
	    print("Not this far...")
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))


# AA CHANGE: Added from example script
def accuracy(classifier, x, y):
    predictions = classifier.predict(x)
    #cm = confusion_matrix(y, predictions)

    print("Accuracy: %f" % accuracy_score(y, predictions))


if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))
    d_ut0 = (x[kTEXT_FIELD] for x in train)
    d_ut1 = (x[kTROPE_FIELD] for x in train)
    d_ut2 = (x[kPAGE_FIELD] for x in train)
    word_ngram_vec = TfidfVectorizer(analyzer='word',strip_accents='ascii',ngram_range=(1,4),stop_words='english')
    word_1gram_vec = CountVectorizer(analyzer='word',strip_accents='ascii',ngram_range=(1,1))
    char_vec = TfidfVectorizer(analyzer='char')
    V = DictVectorizer()
    x_train = V.fit_transform(train)
    for x in train[:1]:
	print x[kTEXT_FIELD]+" "+x[kPAGE_FIELD]+" "+x[kTROPE_FIELD]
        print x[kPAGE_FIELD]
	print type(x[kTEXT_FIELD])
    #vectorizer.fit_transform(examples)    
    #x_train = word_ngram_vec.fit_transform((x[kTEXT_FIELD]+" "+x[kPAGE_FIELD]+" "+x[kTROPE_FIELD]) for x in train)
    x_train = feat.train_feature((x[kTEXT_FIELD]+" "+x[kTROPE_FIELD]) for x in train)
    #
    x_train1 = char_vec.fit_transform(x[kTEXT_FIELD] for x in train)
    x_train2 = word_1gram_vec.fit_transform(x[kTROPE_FIELD] for x in train)
    x_train3 = word_1gram_vec.fit_transform(x[kPAGE_FIELD] for x in train)	
    x_train1 = feat.train_feature(x[kTEXT_FIELD] for x in train)

    #x_train2 = feat.train_feature(x[kPAGE_FIELD] for x in train)
    #x_train3 = feat.train_feature(x[kTROPE_FIELD] for x in train)
    

    #x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)
    #x_test = word_ngram_vec.fit_transform((x[kTEXT_FIELD]+" "+x[kPAGE_FIELD]+" "+x[kTROPE_FIELD]) for x in test)
    x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)
    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    print(len(train), len(y_train))
    print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)


    # AA CHANGE: Add debug accuracy 
    print("TRAIN\n-------------------------")
    accuracy(lr,x_train,y_train)
    feat.show_top10(lr, labels)
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'spoiler': labels[pp]}
        o.writerow(d)


