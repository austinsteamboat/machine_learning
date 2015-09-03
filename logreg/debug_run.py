import unittest
from numpy import zeros, sign, dot, inner
from logreg import LogReg, Example, sigmoid

kTOY_VOCAB = "BIAS_CONSTANT A B C D".split()
kPOS = Example(1, "A:4 B:3 C:1".split(), kTOY_VOCAB, None)
kNEG = Example(0, "B:1 C:3 D:4".split(), kTOY_VOCAB, None)
kOUT = LogReg(len(kPOS.x),0.1,lambda x: 1)
print kPOS.x
print kPOS.y
print kPOS.nonzero
print kNEG.x
print kNEG.y
print kNEG.nonzero
x1 = sigmoid(10,20)
x2 = sigmoid(0.1,20)
x3 = sigmoid(-0.1,20)
x4 = sigmoid(-10,20)
print x1
print x2
print x3
print x4
t1 = [0,1,2,3,4]
t2 = [0,1,2,3,4]
t3 = dot(t1,t2)
print t3
 
print kOUT.beta
debug_out = kOUT.sg_update(kPOS,0)
print kOUT.beta
print debug_out
