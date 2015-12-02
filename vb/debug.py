import time
from numpy import array
from numpy import log, exp, ones, multiply
import numpy

import scipy
import scipy.misc
from scipy.special import psi as digam

def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:            
            memo[x] = f(x)
        return memo[x]
    return helper
    
def digam_mem(n):
	return digam(n);

digam_mem = memoize(digam_mem);

n0 = digam_mem(2.0);
print n0
n2 = digam_mem(2.0);
print n2

beta = array([[.26, .185, .185, .185, .185],
              [.185, .185, .26, .185, .185],
              [.185, .185, .185, .26, .185]])
gamma = array([2.0, 2.0, 2.0])
word = 0;
count = 2;
beta_word = beta[:,word];
gamma_sum = sum(gamma);
digam_sum = digam_mem(gamma_sum);
digam_gam = digam_mem(gamma);
exp_term = digam_gam-digam_sum;
exp_vec = exp(exp_term);
unnorm_out = multiply(beta_word,exp_vec);
norm_term = sum(unnorm_out)/count;
if count==0:
    count = 1;
            
phi = unnorm_out/norm_term;
prop = 0.27711205238850234
normalizer = sum(x * prop for x in beta[:, 0]) /2.0
x0 = beta[0][0] * prop / normalizer;
x1 = beta[1][0] * prop / normalizer;
x2 = beta[2][0] * prop / normalizer;
xout = array([x0,x1,x2])
print phi;
print xout;
# NEXT!
topic_counts = array([[5., 4., 3., 2., 1.],
                      [0., 2., 2., 4., 1.],
                      [1., 1., 1., 1., 1.]])

sum_vec = numpy.sum(topic_counts,1)
beta_mat = topic_counts;
count = 0;
for x in sum_vec:
    if x==0:
        beta_mat[count,:] = beta_mat[count,:]
    else:
        beta_mat[count,:] = beta_mat[count,:]/x
        
    count+=1
    
print beta_mat
