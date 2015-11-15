
#csdaiwei@foxmail.com
#featurization text data in ./data folder and save as .npz file

#do not import this module

import os
import pdb
import logging
import numpy as np
from time import time

import jieba
import jieba.analyse

jieba.setLogLevel(logging.ERROR)

#########################################################
#### functions ##########################################
#########################################################

def load_data(type):
	assert type == 'train' or type == 'test'

	label_file = 'data/' + type +'2.rlabelclass'
	sample_path = 'data/' + type + '2'

	dataset = []	#list of data, each element is a tuple (filename, label, content)

	label_dict = {}
	lines = open(label_file).read().strip().split('\n')
	for l in lines:
		name = l.split(' ')[0]
		label = int(l.split(' ')[1])
		label_dict[name] = label

	names = os.listdir(sample_path)
	for name in names:
		if name.endswith('.txt'):
			content = open(sample_path + '/' + name).read().strip().decode('gbk')
			label = label_dict[name] 
			dataset.append((name, label, content))
	return dataset


def extract_tags(dataset, topK = 100):
	content = ''
	for d in dataset:
		content += d[2]
	tags = jieba.analyse.extract_tags(content, topK=topK)
	return tags

def featurize(dataset, tags):
	samples = []
	labels = []
	for d in dataset:
		content = jieba.cut(d[2])
		v = [0] * len(tags)
		for w in content:
			if w in tags:
				v[tags.index(w)] += 1
		samples.append(v)
		labels.append(d[1])
	return np.array(samples), np.array(labels)


#########################################################
#### main procedure #####################################
#########################################################

train_data = load_data('train')
test_data = load_data('test')

tags = extract_tags(train_data+test_data, 10000)

train_samples, train_labels = featurize(train_data, tags)
test_samples, test_labels = featurize(test_data, tags)

np.savez_compressed('train.npz', samples = train_samples, labels = train_labels)
np.savez_compressed('test.npz', samples = test_samples, labels = test_labels)

