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

#tup1 = array([1,1]);
#tup2 = array([2,2]);
#tup3 = array([3,0]);
#tup4 = array([4,2]);
tup1 = (1,1)
tup2 = (2,2)
tup3 = (3,0)
tup4 = (4,2)
kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]
z = array(kSIMPLE_DATA)
m_val = 1-eps;
m_val = -1/m_val
m_vec = (1,m_val)
m_vec = array(m_vec)


for jj in z:
	print jj
	print sign(dot(jj,m_vec))
print type(z[0,:])
print z[0,:]
x_vec = z[:,0]
y_vec = z[:,1]
c_vec = correlate(x_vec,y_vec,mode='full')
#print x_vec
#print y_vec
#print len(c_vec)
#print c_vec
m_vec = divide(y_vec,x_vec)
print m_vec
g = unique(sort(m_vec))
print g
gg = empty(len(g))
for ii in range(0,len(g)-1):
	gg[ii] = (g[ii+1]+g[ii])/2

gg[len(g)-1] = g[len(g)-1]+eps
print gg

yy = -1*ones(len(gg))
mm = empty([2,len(gg)])
mm[0,:] = gg
mm[1,:] = yy
print "This is the new yy:"
print mm[:,2]
print gg[2]
tt = mm[:,2]
for ii in range(0,len(z)):
	print sign(tt.dot(z[ii,:]))


