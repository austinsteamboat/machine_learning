from random import randint, seed
from collections import defaultdict
from math import atan, sin, cos, pi

from numpy import array, spacing, divide, correlate, dot, sign, sort, unique, empty, ones, append
from numpy.linalg import norm

from bst import BST

from rademacher import origin_plane_hypotheses, axis_aligned_hypotheses, rademacher_estimate, kSIMPLE_DATA as rad_data, PlaneHypothesis, constant_hypotheses


kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]
x = constant_hypotheses(kSIMPLE_DATA)
dat_lab = []
for ii in x:
	for jj in kSIMPLE_DATA:
		t = ii.classify(jj)
		dat_lab.append(t)
		
print dat_lab
print type(dat_lab)
print 2*array(dat_lab)-1			

