import argparse
from collections import Counter, defaultdict

import random
import numpy as np
from numpy import median
from sklearn.neighbors import BallTree
from scipy import stats

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to
        self._kdtree = BallTree(x)# our coord map
        self._y = y# our label map
        self._k = k# number of neighboors

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median value (as implemented in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"

        # Finish this function to return the most common y value for   ################################################################################
        # these indices
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html

        # AA CHANGE
        ar_vals = np.array(self._y[item_indices])
        count_ar = Counter(ar_vals)
        count_ar_vals = count_ar.values()
        max_val = max(count_ar_vals)
        count_ar_ordered = count_ar.most_common()
        b = []
        for i in range(len(count_ar_ordered)):
            m = count_ar_ordered[i]
            if m[1]==max_val:
                b.append(m[0])

        val_out = np.median(b)
        return val_out

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the       ############################################################################
        # majority function, and return the value.
        dist_dum, ind_dum = self._kdtree.query(example,self._k) # this is ok
        ind_vals = [n for m in ind_dum for n in m] # not sure about this thing, that is why, but it's returning a list of lists which fucks the count check right up
        return self.majority(ind_vals)

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the ############################################################################
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        # AA CHANGE: fix this section up next, it's more straightforward then you think, just check predictions against truth 

        d = defaultdict(dict)
        data_index = 0
        for xx, yy in zip(test_x, test_y):
            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))
            test_val = test_x[data_index-1]
            pred_y = self.classify(test_val)
            true_y = test_y[data_index-1]
            pred_y = int(pred_y)
            true_y = int(true_y)
            if d.get(true_y,{}).get(pred_y,None)==None:
                d[true_y][pred_y]=1
            else:
                d[true_y][pred_y]+=1
        return d

    @staticmethod
    def acccuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in xrange(10)))
    print("".join(["-"] * 90))
    for ii in xrange(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(10)))
    print("Accuracy: %f" % knn.acccuracy(confusion))
