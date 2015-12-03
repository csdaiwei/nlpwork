
# csdaiwei@foxmail.com
# k nearest neighbour classifier

import pdb
import numpy as np
from time import time
from math import sqrt


class KNN:
	def __init__(self, K = 5, dist = 'jaccard'):
		self.labels = []
		self.X = 0
		self.Y = 0
		self.K = K
		self.dist = dist
		assert dist in ['jaccard', 'cosine', 'euclid']

	def fit(self, X, Y):
		self.X = X
		self.Y = Y

	def predict(self, X):
		p = []
		for x in X:
			dists = np.array([self.__distance(x, y, self.dist) for y in self.X])
			knn_y = self.Y[dists.argpartition(self.K)[0:self.K]]
			p.append(self.__mode(knn_y))
		return p

	def __mode(self, y):		#most frequent element in list
		s = list(set(y))
		return s[np.array([(y == e).sum() for e in s]).argmax()]


	def __distance(self, x, y, opt):
		if opt == 'euclid':
			return sqrt(np.dot(x-y, x-y))
		if opt == 'cosine':
			if np.dot(x, y) == 0:
				return 1
			return 1 - np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
		if opt == 'jaccard':
			if all(x*y == 0):
				return 1
			return 1 - float((x*y!=0).sum())/(((x+y)!=0).sum())


if __name__ == '__main__':

	train = np.load('train.npz')
	test = np.load('test.npz')
	train_samples_full, train_labels = train['samples'], train['labels']
	test_samples_full, test_labels = test['samples'], test['labels']


	for k in [3, 5, 17]:
		for dim in [50, 100, 500]:
			for dist in ['jaccard', 'cosine', 'euclid']:

				train_samples = train_samples_full[:, 0:dim]
				test_samples = test_samples_full[:, 0:dim]

				start = time()
				knn = KNN(K = k, dist = dist)
				knn.fit(train_samples, train_labels)
				knn_predicts = knn.predict(test_samples)
				print '%s, k=%d, dim=%d knn \taccu:%.4f, time:%4f'%(dist, k, dim, (knn_predicts == test_labels).mean(), time() - start)

	#pdb.set_trace()
