import numpy as np
from numpy import array, single, spacing, divide, correlate, dot, sign, sort, unique, empty, ones, append, mean, isinf

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.),(0.,1.),(0.,2.)]

dataset = kSIMPLE_DATA
# Machine precision
eps = spacing(single(1))
# Put the data into an numpy array
data_array = array(dataset);
# Break out X,Y vector components
x_vec = data_array[:,0]
y_vec = data_array[:,1]
# Calculate Slopes of all the points
m_vec = divide(y_vec,x_vec)
print m_vec
print sort(m_vec)
print unique(sort(m_vec))
# Sort the slope vector and remove duplicate entries
m_vec = unique(sort(m_vec)) 
# Initialize empty vector of interpolated boundaries for hypothesis 
m_int_vec = empty(len(m_vec))
for ii in range(0,len(m_vec)-1):
    if(isinf(m_vec[ii+1])):
        m_int_vec[ii] = 1.*m_vec[ii]+eps
    else:
        m_int_vec[ii] = 1.*(m_vec[ii+1]+m_vec[ii])/2.

# The largest slope will get increased by machine precision and will be our final boundary
m_int_vec[len(m_vec)-1] = m_vec[len(m_vec)-1]+eps
# Then make a vector of tuples to return that are inverted
# to get the normal vector 
yy = -1*ones(len(m_int_vec))
mm = empty([len(m_int_vec),2])
mm[:,0] = m_int_vec
mm[:,1] = yy

for jj in mm:
    print jj

# Run a quick check for inf case
for ii in xrange(len(mm)):
 if(isinf(mm[ii][0])):
     mm[ii][0] = 1.
     mm[ii][1] = eps

for jj in mm:
    print jj  
## Also append the inverse classifications
mm = append(mm,-1*mm,axis=0)

    
