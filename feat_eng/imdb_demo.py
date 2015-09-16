import time
from imdb import IMDb
ia = IMDb()

start_time = time.time()
str_val = "LizzieMcGuire"
movie_list = ia.search_movie(str_val)
first_match = movie_list[0]
found_time = time.time()
fm = ia.get_movie(first_match.movieID)
got_mov = time.time()
attrs = vars(fm)
print first_match['kind']
print first_match['year']

print fm['genre']
print len(fm['genre'])
t1 = fm['genre']
for ii in t1:
 print ii
print fm['director']
print found_time-start_time
print got_mov-found_time
