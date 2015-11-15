
# csdaiwei@foxmail.com
# naive bayes classifier (multinomial, bernoulli)

import pdb
import numpy as np
from time import time

from math import log
from math import factorial as ftl

class NaiveBayes:
	
	def __init__(self, model = 'multinomial'):
		self.labels = [] 		#unique class labels
		self.classprob = {}		#prior probilities, classprob[yi] indicates p(yi)
		self.condprob = {}		#conditonal probilities, condprob[yi][xi] indicates p(xi|yi)
		self.model = model
		assert model in ['multinomial', 'bernoulli']

	def fit(self, X, Y):

		#class labels
		self.labels = list(set(Y))

		#class probilities
		for l in self.labels:
			self.classprob[l] = (Y == l).sum() / float(len(Y))

		#conditional probilities
		if self.model == 'multinomial':
			for l in self.labels:
				self.condprob[l] = (X[Y==l].sum(axis=0) + 1)/ float(X[Y==l].sum()+ len(X[0]))  #add one smooth
		if self.model == 'bernoulli':
			for l in self.labels:
				self.condprob[l] = np.array([(((X[Y==l])[:, i] > 0).sum() + 1) for i in xrange(0, (X[Y==l]).shape[1])]) #numerator
				self.condprob[l] = self.condprob[l]/float((Y==l).sum() + X.shape[1]) 	#denominator, add one smooth
		

	def predict(self, X):
		p = []
		for x in X:
			px = np.array([self.__logposterior(x, y) for y in self.labels])
			p.append(self.labels[px.argmax()])
		return np.array(p)

	def __logposterior(self, x, y):		#log(p(x|y) * p(y)) (common denominator omitted)
		
		p = log(self.classprob[y])

		if self.model == 'multinomial':
			#p += log(float(ftl(x.sum()))/reduce(lambda x,y:x*y, map(ftl, x))) #discard by long int to float overflow
			for i in xrange(0, len(x)):
				if x[i] > 0:
					p += log(self.condprob[y][i]) * x[i]
			return p
		if self.model == 'bernoulli':
			for i in xrange(0, len(x)):
				if x[i] > 0:
					p += log(self.condprob[y][i]) 
				else :
					p += log(1 - self.condprob[y][i])
			return p

if __name__ == '__main__':
	
	train = np.load('train.npz')
	test = np.load('test.npz')
	train_samples_full, train_labels = train['samples'], train['labels']
	test_samples_full, test_labels = test['samples'], test['labels']

	for model in ['multinomial', 'bernoulli']:
		print '\n%s model'%model
		for dim in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
			train_samples = train_samples_full[:, 0:dim]
			test_samples = test_samples_full[:, 0:dim]

			start = time()
			nb = NaiveBayes(model=model)
			nb.fit(train_samples, train_labels)
			nb_predicts = nb.predict(test_samples)
			print 'naivebayes dim:%d\taccu:%.4f, time:%4f'%(dim, (nb_predicts == test_labels).mean(), time() - start)

	pdb.set_trace()





