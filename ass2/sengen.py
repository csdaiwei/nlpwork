#! /usr/bin/python
# -*- coding: utf-8 -*-

import pdb
import sys

from math import log
from time import time
from itertools import permutations

from nltk.corpus import brown			#main corpus
from nltk.util import ngrams			#ngrams extractor
from nltk.probability import FreqDist	#counter

#########################################################
####training models######################################
#########################################################

print 'training model, using brown corpus ...'
start = time()

unicount = FreqDist(ngrams(brown.words(), 1))	#word count
bicount = FreqDist()							#bigram count
for sent in brown.sents():				#bigram count need to be done with sentences
	bigrams = ngrams(sent, 2)			#while unigram count(word count) do not need
	for bigram in bigrams:
		bicount[bigram] += 1

print 'training finished in %.2f seconds'%(time()-start)

#########################################################
####functions############################################
#########################################################

def condprob(w1, w2=None):
	if w2 == None:
		return float(unicount[(w1, )])/unicount.N()
	else:
		return float(bicount[(w1, w2)])/unicount[(w1, )]

def seqprob(seq):
	p = log(condprob(seq[0]))
	for i in range(1, len(seq)):
		p += log(condprob(seq[i-1], seq[i]))
	return p

#########################################################
####testing##############################################
#########################################################

while 1:
	words = raw_input("Please input some words separated by blanks\n>>> ")
	if len(words) == 0:
		continue
	
	seq = words.split(' ')
	all_seqs = list(permutations(seq, len(seq)))
	all_probs = [seqprob(list(s)) for s in all_seqs]
	target = all_seqs[all_probs.index(max(all_probs))]
	
	print "The most proper sentence generated by these words is \n>>> "
	print target

	


