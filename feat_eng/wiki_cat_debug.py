import wikipedia as wiki
page_query = "Ogopogo"
try:
    p_ut = wiki.page(page_query)
    d_ut = p_ut.categories
    print d_ut
    print len(d_ut)
except:
    print "Couldn't fine the search"

