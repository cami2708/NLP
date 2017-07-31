
import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
#This is the parser for XML and HTML documents
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

#####Data preparation to remove stop words#######
#Defining function to remove stop words
def tokenizer(s):
	#All word to lowercase
	s = s.lower()
	tokens = nltk.tokenize.word_tokenize(s)
	tokens = [t for t in tokens if len(t) > 2]
	tokens = [wordNetLemmatizer.lemmatize(t) for t in tokens]
	tokens = [t for t in tokens if not stopWords]
	return tokens

wordIndexMap = {}
currentIndex = 0

#Word data
positiveTokenized = []
negativeTokenized = []

for review in positiveReviews:
	tokens = tokenizer(review.text)
	positiveTokenized.append(tokens)
	for token in tokens:
		if token not in wordIndexMap:
			wordIndexMap[token] = currentIndex
			currentIndex += 1


for review in negativeReviews:
	tokens = tokenizer(review.text)
	negativeTokenized.append(tokens)
	for token in tokens:
		if token not in wordIndexMap:
			wordIndexMap[token] = currentIndex
			currentIndex += 1

def tokensToVector(tokens, label):
	x = np.zeros(len(wordIndexMap) + 1)
	for t in tokens:
		i = wordIndexMap[t]
		x[i] += 1
	if x.sum() != 0:
		x = x/x.sum()
	x[-1] = label
	return x

tokenizedNumber = len(positiveTokenized) + len(negativeTokenized)		

data = np.zeros((tokenizedNumber, len(wordIndexMap) + 1))
i = 0
for tokens in positiveTokenized:
	xy = tokensToVector(tokens, 1)
	data[i,:] = xy
	i += 1

########Building the model#########
np.random.shuffle(data)



