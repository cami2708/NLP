import nltk
import random
import numpy as np

from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

wordNetLemmatizer = WordNetLemmatizer()

stopWords = set(w.rstrip() for w in open('stopwords.txt'))

#In this case, we are going to parse an xml file
#review_text is the only data require
positiveReviews = BeautifulSoup(open('Electronics/positive.xml'), "lxml") 
positiveReviews = positiveReviews.findAll('review_text')

negativeReviews = BeautifulSoup(open('Electronics/negative.xml'), "lxml")
negativeReviews = negativeReviews.findAll('review_text')

#Now we are going to balance the positive reviews
np.random.shuffle(positiveReviews)
positiveReviews = positiveReviews[:len(negativeReviews)]

trigrams = {}
for review in positiveReviews:
	s = review.text.lower()
	tokens = nltk.tokenize.word_tokenize(s)
	for i in xrange(len(tokens) - 2):
		k = (tokens[i], tokens[i+2])
		if k not in trigrams:
			trigrams[k] = []
		trigrams[k].append(tokens[i+1])

#Now we are going to create the probabilities 
trigramsProbability = {}
for k, words in trigrams.iteritems():
	if len(set(words))>1:
		d = {}
		n = 0
		for w in word:
			if w not in d:
				d[w] = 0
			d[w] += 1
			n += 1
		for w, c in d.iteritems():
			d[w] = float(c) / n
		trigramsProbability[k] = d


def randomSample(d):
	r = random.random()
	cumulative = 0
	for w,p in d.iteritems():
		cumulative += p
		if r < cumulative:
			return w

def testSpinner():
	review = random.choice(positiveReviews)
	s = review.text.lower()
	print "Original:", s
	tokens = nltk.tokenize.word_tokenize(s)
	for i in xrange(len(tokens) - 2):
		if random.random() < 0.2:
			k = (tokens[i], tokens[i + 2])
			if k in trigramsProbability:
				w = randomSample(trigramsProbability[k])
				tokens[i + 1] = w
	print "Spun:"			
	print " ".join(tokens.replace(" .", ".").replace(" '", "'")).replace(" ,", ",").replace("$ ", "$").replace(" !", "!")





