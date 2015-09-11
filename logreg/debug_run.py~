import unittest
import operator
from numpy import zeros, sign, dot, inner
from numpy import linalg as LA
from logreg import LogReg, Example, sigmoid


kTOY_VOCAB = "BIAS_CONSTANT A B C D".split()
kPOS = Example(1, "A:4 B:3 C:1".split(), kTOY_VOCAB, None)
kPOS1 = Example(1, "A:1".split(), kTOY_VOCAB, None)
kNEG = Example(0, "B:1 C:3 D:4".split(), kTOY_VOCAB, None)
kOUT = LogReg(len(kPOS.x),0.0,1)


t1 = [0,1,2,3,4]
t2 = [0,1,2,3,4]
t3 = dot(t1,t2)
t4 = LA.norm(t1)
#print t3
#print t4 
#print kOUT.beta
debug_out = kOUT.sg_update(kPOS,0)
print kOUT.beta
print kOUT.last_update
debug_out = kOUT.sg_update(kPOS1,1)
print kOUT.beta
print kOUT.last_update
debug_out = kOUT.sg_update(kPOS1,2)
print kOUT.beta
print kOUT.last_update
debug_out = kOUT.sg_update(kPOS1,3)
print kOUT.beta
print kOUT.last_update
debug_out = kOUT.sg_update(kPOS,4)
print kOUT.beta
print kOUT.last_update
debug_out = kOUT.sg_update(kNEG,5)
print kOUT.beta
print kOUT.last_update
debug_out = kOUT.sg_update(kPOS,6)
print kOUT.beta
print kOUT.last_update
d = {}
for i in range(len(kTOY_VOCAB)):
    d[kTOY_VOCAB[i]] = kOUT.beta[i]


asc_x = sorted(d.items(), key=operator.itemgetter(1),reverse = True)
des_x = sorted(d.items(), key=operator.itemgetter(1))
print d
x_max = range(2)
x_min = range(2)
for i in range(2):
    x_max[i] = asc_x[i][0]
    x_min[i] = des_x[i][0]
		
print x_max
print x_min

