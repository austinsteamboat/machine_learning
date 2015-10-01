from random import randint, seed
from collections import defaultdict
from math import atan, sin, cos, pi, ceil

from numpy import array, spacing, divide, correlate, dot, sign, sort, unique, empty, ones, append, mean
from numpy.linalg import norm

from bst import BST

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]

class Classifier:
    def correlation(self, data, labels):
        """
        Return the correlation between a label assignment and the predictions of
        the classifier

        Args:
          data: A list of datapoints
          labels: The list of labels we correlate against (+1 / -1)
        """

        assert len(data) == len(labels), \
            "Data and labels must be the same size %i vs %i" % \
            (len(data), len(labels))

        assert all(x == 1 or x == -1 for x in labels), "Labels must be binary"
        
        labels = array(labels)
        # Convert data from bool (0,1) to -1,+1
        data_lab = []
        for ii in data:
            data_lab.append(self.classify(ii))
        
        data_lab = array(data_lab)*2-1
        cor_val = float(dot(data_lab,labels))/float(len(data_lab)) 
        return cor_val


class PlaneHypothesis(Classifier):
    """
    A class that represents a decision boundary.
    """

    def __init__(self, x, y, b):
        """
        Provide the definition of the decision boundary's normal vector

        Args:
          x: First dimension
          y: Second dimension
          b: Bias term
        """
        self._vector = array([x, y])
        self._bias = b

    def __call__(self, point):
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return self(point) >= 0

    def __str__(self):
        return "x: x_0 * %0.2f + x_1 * %0.2f >= %f" % \
            (self._vector[0], self._vector[1], self._bias)


class OriginPlaneHypothesis(PlaneHypothesis):
    """
    A class that represents a decision boundary that must pass through the
    origin.
    """
    def __init__(self, x, y):
        """
        Create a decision boundary by specifying the normal vector to the
        decision plane.

        Args:
          x: First dimension
          y: Second dimension
        """
        PlaneHypothesis.__init__(self, x, y, 0)


class AxisAlignedRectangle(Classifier):
    """
    A class that represents a hypothesis where everything within a rectangle
    (inclusive of the boundary) is positive and everything else is negative.

    """
    def __init__(self, start_x, start_y, end_x, end_y):
        """

        Create an axis-aligned rectangle classifier.  Returns true for any
        points inside the rectangle (including the boundary)

        Args:
          start_x: Left position
          start_y: Bottom position
          end_x: Right position
          end_y: Top position
        """
        assert end_x >= start_x, "Cannot have negative length (%f vs. %f)" % \
            (end_x, start_x)
        assert end_y >= start_y, "Cannot have negative height (%f vs. %f)" % \
            (end_y, start_y)

        self._x1 = start_x
        self._y1 = start_y
        self._x2 = end_x
        self._y2 = end_y

    def classify(self, point):
        """
        Classify a data point

        Args:
          point: The point to classify
        """
        return (point[0] >= self._x1 and point[0] <= self._x2) and \
            (point[1] >= self._y1 and point[1] <= self._y2)

    def __str__(self):
        return "(%0.2f, %0.2f) -> (%0.2f, %0.2f)" % \
            (self._x1, self._y1, self._x2, self._y2)


class ConstantClassifier(Classifier):
    """
    A classifier that always returns true
    """

    def classify(self, point):
        return True


def constant_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over the single constant
    hypothesis possible.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    yield ConstantClassifier()


def origin_plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector.  The classification decision is
    the sign of the dot product between an input point and the classifier.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    # TODO: Complete this function
    eps = spacing(1)
    # Put the data into an numpy array
    data_array = array(dataset);
    # Break out X,Y vector components
    x_vec = data_array[:,0]
    y_vec = data_array[:,1]
    # Calculate Slopes of all the points
    m_vec = divide(y_vec,x_vec)
    # Sort the slope vector and remove duplicate entries
    m_vec = unique(sort(m_vec)) 
    # Initialize empty vector of interpolated boundaries for hypothesis 
    m_int_vec = empty(len(m_vec))
    for ii in range(0,len(m_vec)-1):
        m_int_vec[ii] = (m_vec[ii+1]+m_vec[ii])/2
    
    # The largest slope will get increased by machine precision and will be our final boundary
    m_int_vec[len(m_vec)-1] = m_vec[len(m_vec)-1]+eps
    # Then make a vector of tuples to return that are inverted
    # to get the normal vector 
    yy = -1*ones(len(m_int_vec))
    mm = empty([len(m_int_vec),2])
    mm[:,0] = m_int_vec
    mm[:,1] = yy
    mm = append(mm,-1*mm,axis=0)
    for ii in mm:
        yield OriginPlaneHypothesis(ii[0],ii[1])

def plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector and a bias.  The classification
    decision is the sign of the dot product between an input point and the
    classifier plus a bias.

    Args:
      dataset: The dataset to use to generate hypotheses

    """

    # Complete this for extra credit
    return


def axis_aligned_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are axis-aligned rectangles

    Args:
      dataset: The dataset to use to generate hypotheses
    """

    # TODO: complete this function
    yield AxisAlignedRectangle(0, 0, 0, 0)


def coin_tosses(number, random_seed=0):
    """
    Generate a desired number of coin tosses with +1/-1 outcomes.

    Args:
      number: The number of coin tosses to perform

      random_seed: The random seed to use
    """
    if random_seed != 0:
        seed(random_seed)

    return [randint(0, 1) * 2 - 1 for x in xrange(number)]


def rademacher_estimate(dataset, hypothesis_generator, num_samples=500,
                        random_seed=0):
    """
    Given a dataset, estimate the rademacher complexity

    Args:
      dataset: a sequence of examples that can be handled by the hypotheses
      generated by the hypothesis_generator

      hypothesis_generator: a function that generates an iterator over
      hypotheses given a dataset

      num_samples: the number of samples to use in estimating the Rademacher
      correlation
    """
    # TODO: complete this function
    # Max an empty vector to hold the correlation results for each step to take the mean 
    cor_mean_vec = []
    # Iterate over num_samples to get an expected value
    for hh in xrange(num_samples):
    	# Make some random values
        if random_seed != 0:
            rand_lab = coin_tosses(len(dataset), random_seed + hh)
        else:
            rand_lab = coin_tosses(len(dataset))
        
        # Make sure to regenerate the hypothesis set EACH time!        
        h_gen = hypothesis_generator(dataset);
        dat_cor = []
        for ii in h_gen:
            dat_cor.append(ii.correlation(dataset,rand_lab))
            
        #print dat_cor
        tt = max(dat_cor)
        cor_mean_vec.append(tt)
        
    #print cor_mean_vec
    mean_out = mean(array(cor_mean_vec));
    # return the correlation
    return mean_out#

if __name__ == "__main__":
    print("Rademacher correlation of constant classifier %f" %
          rademacher_estimate([(0, 0)], constant_hypotheses, num_samples=1000,random_seed=3))
         
    print("Rademacher correlation of constant classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, constant_hypotheses))      
    #print("Rademacher correlation of rectangle classifier %f" %
          #rademacher_estimate(kSIMPLE_DATA, axis_aligned_hypotheses))
    print("Rademacher correlation of plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, origin_plane_hypotheses))
