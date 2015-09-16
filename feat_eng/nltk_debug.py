from nltk.stem.porter import *
stemmer = PorterStemmer()
plurals = ['dead','kill','kills','dies','revealed','end','turns','finale','death','killed']
singles = [stemmer.stem(plural) for plural in plurals]
print(' '.join(singles))  # doctest: +NORMALIZE_WHITESPACEsingles
