
#csdaiwei@foxmail.com

#test compare methods by scikit-learn, totally 5 methods with all default parameter
#multinominal naive bayes classifier, knn classifier, logistic regression classifier
#decision tree classifier , svm classifier

import pdb
import numpy as np
from time import time

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#load data
train = np.load('train.npz')
test = np.load('test.npz')
train_samples_full, train_labels = train['samples'], train['labels']
test_samples_full, test_labels = test['samples'], test['labels']


#start compare test

for dim in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
	train_samples = train_samples_full[:, 0:dim]
	test_samples = test_samples_full[:, 0:dim]

	print '\nexecute compare test on data dim %d'%(train_samples.shape[1])

	start = time()
	bnb_model = BernoulliNB()
	bnb_model.fit(train_samples, train_labels)
	bnb_predicts = bnb_model.predict(test_samples)
	print 'sklearn bnb\taccu:%.4f, time:%4f'%((bnb_predicts == test_labels).mean(), time() - start)

	start = time()
	mnb_model = MultinomialNB()
	mnb_model.fit(train_samples, train_labels)
	mnb_predicts = mnb_model.predict(test_samples)
	print 'sklearn mnb\taccu:%.4f, time:%4f'%((mnb_predicts == test_labels).mean(), time() - start)


	start = time()
	knn_model = KNeighborsClassifier()
	knn_model.fit(train_samples, train_labels)
	knn_predicts = knn_model.predict(test_samples)
	print 'sklearn knn\taccu:%.4f, time:%4f'%((knn_predicts == test_labels).mean(), time() - start)


	start = time()
	sgdlr_model = SGDClassifier(loss='log', penalty='none')
	sgdlr_model.fit(train_samples, train_labels)
	sgdlr_predicts = sgdlr_model.predict(test_samples)
	print 'sklearn sgdlr\taccu:%.4f, time:%4f'%((sgdlr_predicts == test_labels).mean(), time() - start)


	start = time()
	dt_model = DecisionTreeClassifier()
	dt_model.fit(train_samples, train_labels)
	dt_predicts = dt_model.predict(test_samples)
	print 'sklearn dt\taccu:%.4f, time:%4f'%((dt_predicts == test_labels).mean(), time() - start)


	start = time()
	svm_model = SVC()
	svm_model.fit(train_samples, train_labels)
	svm_predicts = svm_model.predict(test_samples)
	print 'sklearn svm\taccu:%.4f, time:%4f'%((svm_predicts == test_labels).mean(), time() - start)


#pdb.set_trace()