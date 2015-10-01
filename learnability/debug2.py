from random import randint, seed
from collections import defaultdict
from math import atan, sin, cos, pi
import numpy as np
from numpy import array, spacing, divide, correlate, dot, sign, sort, unique, empty, ones, append
from numpy.linalg import norm

from bst import BST
eps = spacing(1)
#eps = 0;
#eps = 0.1

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]
z = array(kSIMPLE_DATA)
x_vec = z[:,0]
y_vec = z[:,1]

R = BST();

for ii in x_vec:
	R.insert(ii);

for ii in y_vec:
	R.insert(ii);

print R


