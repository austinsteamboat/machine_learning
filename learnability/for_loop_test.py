from random import randint, seed
from collections import defaultdict
from math import atan, sin, cos, pi, ceil

from numpy import array, spacing, divide, correlate, dot, sign, sort, unique, empty, ones, append, mean
from numpy.linalg import norm

from bst import BST

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]

for hh in xrange(len(kSIMPLE_DATA)):
        if hh != 0:
            print "a"
        else:
            print "b"
                
        for jj in xrange(3):
            print "c"
            for ii in xrange(2):
                print "d"       
                
                
# Backups:
    h_gen = hypothesis_generator(dataset);
    cor_mean_vec = []
    for hh in xrange(num_samples):
        if random_seed != 0:
            rand_lab = coin_tosses(len(dataset), random_seed + hh)
        else:
            rand_lab = coin_tosses(len(dataset))
                
        dat_lab = []
        dat_cor = []
        print rand_lab
        for ii in h_gen:
            print "inner loop 1"
            for jj in dataset:
                print "inner loop 2"
                dum_var = ii.classify(jj)
                dat_lab.append(dum_var)
                
            dat_cor.append(ii.correlation(dat_lab,rand_lab))
            
        #print dat_cor
        cor_mean_vec.append(0)
