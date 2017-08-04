import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD


wordNetLemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open('all_book_titles.txt')]

stopWords = set(w.rstrip() for w in open('stopwords.txt'))
stopWords = stopWords.union({
	'introduction', 'edition', 'series', 'application',
	'approach', 'card', 'access', 'package', 'plus', 'etext',
	'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
	'third', 'second', 'fourth',
	})


#####Data preparation to remove stop words#######
#Defining function to remove stop words
def tokenizer(s):
	#All word to lowercase
	s = s.lower()
	tokens = nltk.tokenize.word_tokenize(s)
	tokens = [t for t in tokens if len(t) > 2]
	tokens = [wordNetLemmatizer.lemmatize(t) for t in tokens]
	tokens = [t for t in tokens if t not in stopWords]
	tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
	return tokens

wordIndexMap = {}
currentIndex = 0
allTokens = []
allTitles = []
indexWordMap = []
for title in titles:
    try:
        title = title.encode('ascii', 'ignore') # this will throw exception if bad characters
        allTitles.append(title)
        tokens = tokenizer(title)
        allTokens.append(tokens)
        for token in tokens:
            if token not in wordIndexMap:
                wordIndexMap[token] = currentIndex
                currentIndex += 1
                indexWordMap.append(token)
    except:
        pass

print "Those are title values:", allTitles[:10]

#This is unsurpervised learning
def tokensToVector(tokens):
	x = np.zeros(len(wordIndexMap))
	for t in tokens:
		i = wordIndexMap[t]
		#We are using binary feature
		x[i] = 1
	return x

N = len(allTokens)
D = len(wordIndexMap)
X = np.zeros((D, N))
i = 0
for tokens in allTokens:
	X[:,i] = tokensToVector(tokens)
	i += 1

def main():
	svd = TruncatedSVD()
	z = svd.fit_transform(X)
	plt.scatter(z[:,0], z[:,1])
	for i in xrange(D):
		plt.annotate(s = indexWordMap[i], xy = (z[i,0], z[i,1]))
	plt.show()

if __name__ == '__main__':
	main()



