import time
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

kTITLE_FIELD = 'title'
kGENRE_FIELD = 'genre'

imdb_dict = list(DictReader(open("imdb_dict.csv", 'r')))
title_vec = []
genre_vec = []
for ii in imdb_dict:
	title_vec.append(ii[kTITLE_FIELD])
	genre_vec.append(ii[kGENRE_FIELD])

imdb_dict = dict(zip(title_vec, genre_vec))
key_str = "Homeland"
print len(title_vec)
print len(genre_vec)
print genre_vec[1]
print imdb_dict[key_str]
print type(imdb_dict[key_str])

