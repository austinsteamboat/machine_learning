if (global_count % 500 == 0):
		movie_list = ia.search_movie(s[kPAGE_FIELD])
		if(len(movie_list)>0):
			first_match = movie_list[0]
			yield first_match['year']
			try:
			   fm = ia.get_movie(first_match.movieID)
			   fm_g = fm['genre']
			   for ii in fm_g:
				   yield str(ii)
			except: 
				print("Hung")
