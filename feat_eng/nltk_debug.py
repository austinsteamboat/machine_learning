from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

words = word_tokenize("If he was alive he'd kill again   George manages to induce a partial werewolf transformation in Series 4 when he needs his werewolf strength to rescue his daughter.")
t2 = pos_tag(words)
print t2
print type(t2)
for ii in t2:
    print ii[1]
    print type(ii)
