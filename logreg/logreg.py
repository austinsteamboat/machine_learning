import random
from numpy import zeros, sign, dot, inner
from math import exp, log
from collections import defaultdict
import matplotlib.pyplot as plt

import argparse

# AA COMMENTS: No libraries this time

kSEED = 1701 # AA COMMENT: seed the random number gen
kBIAS = "BIAS_CONSTANT" # Used to check that BIAS_CONSTANT is not in the paper

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)

# Think the class example just makes a structure and counts up words 
# in the feature set
# includes known labels
# think df is the file with the data
# No changes required
class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example

        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        #self.nonzero = {} # AA CHANGE
        self.nonzero = zeros(len(vocab))
        self.y = label
        self.x = zeros(len(vocab))
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document" # and that's because the bias is independent of the data...I think 
                self.x[vocab.index(word)] += float(count)
                #self.nonzero[vocab.index(word)] = word # AA CHANGE
                self.nonzero[vocab.index(word)] = 1
        self.x[0] = 1
        self.nonzero[0] = 1


class LogReg:
    def __init__(self, num_features, mu, step): # look into what that lambda fellow is doing =lambda x: 0.05
        """
        Create a logistic regression classifier

        :param num_features: The number of features (including bias)
        :param mu: Regularization parameter
        :param step: A function that takes the iteration as an argument (the default is a constant value)
        """
        # Creates structure for log reg
    # Look at step thingamajig
        # Sets up weights beta, mu, the regulation parameter and step size 

        self.beta = zeros(num_features)
        self.mu = mu
        self.step = step
        self.last_update = zeros(num_features) # defaultdict is a python call to fill up the default dictionary

        assert self.mu >= 0, "Regularization parameter must be non-negative"

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy

        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """
    # Yeah so this bad larry calculates the logprob and accuracy given
    # your current weights, input data and labels 
    # the first portion calculates the probability p(y|x,B)
    # the second portion checks if it's right by looking at the sigmoid
    # the sigmoid is bounded by 0,1 so if we're stricly less than half 
    # way, it's counted as correct and we incrament the number right
    # then the accuracy is just how many we got right over the total

        logprob = 0.0
        num_right = 0
        for ii in examples:
            p = sigmoid(self.beta.dot(ii.x))
            if ii.y == 1:
                logprob += log(p) # not clear here, this implies we know y, but aren't we trying to find it's probabiliy, ie prob of 0 or prob of 1 and which of those is better...
                  # well ok we could just say we're now checking for 1 so assume it's one then we're checking for 0 so find it for 0
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    # generally progress is straightforward as it's just calculating the probability 

    def sg_update(self, train_example, iteration, use_tfidf=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.

        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """

        # TODO: Implement updates in this function
        # AA COMMENT: so here's the first big thing we have to do, make it run the updates to give us a beta vector 
        # so iteration is iteration on the gradient, so are we doing this in a serial-stream or as a semi-real-time batch?
        # iteration suggests it's batchy, train example ought to be a vector
        # don't know what tfidf is... 
        # calculate pi_i
        #self.step = step_update(iteration)

        if(iteration==0):
            self.last_update = zeros(len(train_example.x))

        dot_pi = dot(self.beta,train_example.x) 
        exp_pi = exp(dot_pi)
        pi_i = exp_pi/(1+exp_pi)
        for ii in range(0,len(self.beta)):
            self.beta[ii] = self.beta[ii]+self.step*(train_example.y-pi_i)*train_example.x[ii]

        # Ok, so here, we have self.nonzero saying which index is non-zero
        # so all we have to do is update the values of last_update that 
        # are zero and if we are non-zero, reset it
        
        for jj in range(0,len(train_example.x)):
            if(train_example.x[jj]!=0):
                self.beta[jj] = self.beta[jj]*pow((1-2*self.step*self.mu),self.last_update[jj]+1)
		if(self.last_update[jj]>0):
		    print self.last_update[jj]+1

            if(train_example.nonzero[jj] == 1):
                self.last_update[jj] = 0
            else:
                self.last_update[jj] += 1
            
        return self.beta


def read_dataset(positive, negative, vocab, test_proportion=.1):
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """
    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab
    
    # AA COMMENT: so I think read_dataset is ok shouldn't need to change it


def step_update(iteration):
    # TODO (extra credit): Update this function to provide an
    # effective iteration dependent step size
    # AA COMMENT: Ok so I think we could call this from the gradient step 
    return 1.0

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mu", help="Weight of L2 regression",
                           type=float, default=0.00, required=False)
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.05, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../data/hockey_baseball/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../data/hockey_baseball/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/hockey_baseball/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

    args = argparser.parse_args()
    train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = LogReg(len(vocab), args.mu, lambda x: args.step)

    # Iterations
    update_number = 0
    t_vec = zeros(len(train))
    tp_vec = zeros(len(train))
    hp_vec = zeros(len(train))
    ta_vec = zeros(len(train))
    ha_vec = zeros(len(train))

    for pp in xrange(args.passes):
        for ii in train:
            beta_vec = lr.sg_update(ii, update_number)
	    train_lp, train_acc = lr.progress(train)
            ho_lp, ho_acc = lr.progress(test)
	    t_vec[update_number] = update_number
	    tp_vec[update_number] = train_lp
	    ta_vec[update_number] = train_acc
	    hp_vec[update_number] = ho_lp
	    ha_vec[update_number] = ho_acc
            update_number += 1	
            if update_number % 100 == 1:
		per_done = float(100*update_number)/float(len(train))
		#per_done = len(train)
                print("Percent done: %i\r" %
                      int(round(per_done)))

    
    
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(t_vec,ta_vec,t_vec,ha_vec)
    #plt.xlabel('Iteration Number')
    plt.ylabel('Percent Accuracy')
    plt.title('Accuracy')
    plt.legend(["Tennis","Hockey"])
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(t_vec,tp_vec,t_vec,hp_vec)
    plt.xlabel('Iteration Number')
    plt.ylabel('Loglikelihood')
    plt.title('Probability')
    plt.legend(["Tennis","Hockey"])
    plt.grid(True)
    plt.figure(2)
    plt.plot(beta_vec)
    plt.show()



