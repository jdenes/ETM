from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='all')  # , remove=('headers', 'footers'))  # , 'quotes'
res = [preprocess_string(x, filters=[strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric]) for x in data.data]
res = [' '.join(x) for x in res]

with open('data/20ng/text_uncased.txt', 'w', encoding='utf-8') as f:
	for i in res:
		f.write("%s\n" % i)