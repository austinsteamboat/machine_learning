import time
from csv import DictReader, DictWriter

import numpy as np
from numpy import array
from imdb import IMDb
ia = IMDb()

kPAGE_FIELD = 'page'
start_time = time.time()
# Cast to list to keep it all in memory
train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
print type(train)
test_vec = []
in_list = False
count = 0
for x in train:
	if count==0:
		test_vec.append(x[kPAGE_FIELD])
		count = 1
	else:
		for y in test_vec:
			in_list = False

			if y==x[kPAGE_FIELD]:
				in_list = True
		
		if(not(in_list)):
			test_vec.append(x[kPAGE_FIELD])

in_list = False
count = 0
for x in test:
	if count==0:
		test_vec.append(x[kPAGE_FIELD])
		count = 1
	else:
		for y in test_vec:
			in_list = False

			if y==x[kPAGE_FIELD]:
				in_list = True
		
		if(not(in_list)):
			test_vec.append(x[kPAGE_FIELD])

genre_vec = []
for y in test_vec:
	str_val = str(y)
	movie_list = ia.search_movie(str_val)
	if(len(movie_list)>0):	
		first_match = movie_list[0]	
		fm = ia.get_movie(first_match.movieID)
		try:
			fm_g = fm['genre']
		except:
			fm_g = ""

		genre_str = ""
		for ii in fm_g:
			genre_str=genre_str+str(ii)+" "

	else:
		genre_str = ""

	genre_vec.append(genre_str)
	print(y+":"+" "+genre_str)


end_time = time.time()
print end_time-start_time
print len(test_vec)
o = DictWriter(open("imdb_dict_new.csv", 'w'), ["title", "genre"])
o.writeheader()
for ii in range(0,len(test_vec)):
	d = {'title': test_vec[ii], 'genre': genre_vec[ii]}
	o.writerow(d)
