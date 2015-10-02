from random import randint, seed
from collections import defaultdict
from math import atan, sin, cos, pi
import numpy as np
from numpy import array, single, spacing, divide, correlate, dot, sign, sort, unique, empty, ones, append
from numpy.linalg import norm
from itertools import combinations
from bst import BST
eps = spacing(single(1))
#eps = 0;
#eps = 0.1
x1 = xrange(4)
y1 = combinations(x1,1)
y2 = combinations(x1,2)
z1 = []
for ii in y1:
    z1.append(ii)
    
for ii in y2:
    z1.append(ii)
    
print z1    
kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]
k_sim = kSIMPLE_DATA
z = array(kSIMPLE_DATA)
x_vec = z[:,0]
y_vec = z[:,1]

X_BST = BST();
Y_BST = BST();
HYP_BST = BST();

for ii in x_vec:
	X_BST.insert(ii);

for ii in y_vec:
	Y_BST.insert(ii);
# candidate for BST accel
def make_a_rec(data_list,ind_list):
    eps = spacing(single(1))
    z = []
    for ii in ind_list:
        z.append(data_list[ii])
    
    z = array(z)
    x_vec = z[:,0]
    y_vec = z[:,1]
    x_min = min(x_vec)-eps
    y_min = min(y_vec)-eps
    x_max = max(x_vec)+eps
    y_max = max(y_vec)+eps
    min_max_ar = [x_min,y_min,x_max,y_max]
    min_max_ar = array(min_max_ar)
    return min_max_ar   
     
def make_null_rec(data_list):    
    eps = spacing(single(1))
    z = []
    for ii in ind_list:
        z.append(data_list[ii])
    
    z = array(z)
    x_vec = z[:,0]
    y_vec = z[:,1]
    x_min = max(x_vec)+eps
    y_min = max(y_vec)+eps
    x_max = max(x_vec)+2*eps
    y_max = max(y_vec)+2*eps
    min_max_ar = [x_min,y_min,x_max,y_max]
    min_max_ar = array(min_max_ar)
    return min_max_ar

# candidate for bst accel
def check_a_point(rec_ar,data_point):
    dx = float(data_point[0])
    dy = float(data_point[1])
    x_min = rec_ar[0]
    y_min = rec_ar[1]
    x_max = rec_ar[2]
    y_max = rec_ar[3]
    return ((x_min<dx) and (dx<x_max) and (y_min<dy) and (dy<y_max))

# candidate for BST accel    
def check_a_rec(rec_ar,data_list):
    bool_list = []
    for ii in data_list:
        bool_val = check_a_point(rec_ar,ii)
        bool_list.append(bool_val)
        
    return bool_list
    
# candidate for BST accel    
def check_a_rec_bst(rec_ar,data_list):
    z = array(z)
    x_vec = z[:,0]
    y_vec = z[:,1]
    X_BST = BST();
    Y_BST = BST();
    for ii in x_vec:
        X_BST.insert(ii);

    for ii in y_vec:
        Y_BST.insert(ii);
    
    bool_list = []
    for ii in data_list:
        bool_val = check_a_point(rec_ar,ii)
        bool_list.append(bool_val)
        
    return bool_list    
    
def check_a_hyp(bool_list,rec_ar,rec_dict,hypo_dict):
    if not(bool_list in hypo_dict.values()):
        key_num = len(hypo_dict)+1
        hypo_dict[key_num] = bool_list
        rec_dict[key_num] = rec_ar
    
    return hypo_dict, rec_dict
                    
def itter_gen(data_len):
    it_lo = []
    dum_list = list(xrange(data_len))
    for xx in xrange(data_len):
        dum_gen = combinations(dum_list,xx+1)
        for jj in dum_gen:
            it_lo.append(jj)
        
    return it_lo

z2 = itter_gen(4)
REC_DICT = defaultdict(array);        
HYP_DICT = defaultdict(list);
for ii in z2:    
    ind_list = ii
    rec1 =  make_a_rec(k_sim,ind_list)
    bool_out = check_a_rec(rec1,k_sim)
    HYP_DICT, REC_DICT = check_a_hyp(bool_out,rec1,REC_DICT,HYP_DICT)

null_rec = make_null_rec(k_sim)
bool_out = check_a_rec(null_rec,k_sim)
HYP_DICT, REC_DICT = check_a_hyp(bool_out,null_rec,REC_DICT,HYP_DICT)

#print bool_out
#print rec1
#print HYP_DICT
rec_vals = REC_DICT.values()
for ii in rec_vals:
    print type(float(ii[0]))





