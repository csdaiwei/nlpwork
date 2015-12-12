
# csdaiwei@foxmail.com


from __future__ import division		#float division

import pdb
import codecs
import numpy as np

from time import time
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression

# training process in the end of file
# read code from there

######################################################################
#### parameters, definitions #########################################
######################################################################

path = 'BosonNLP_NER_6C/BosonNLP_NER_6C.txt'

NULL_STATE = u'n'			#symbol for non-ne state
BEGIN_STATE_PREFIX = u'b'	#symbol for begin character of a name entity
INTER_STATE_PREFIX = u'i'	#symbol for next character of a name entity

D = 10 ** 3			#hash trick size, aka dimension of features

######################################################################
#### functions, classes ##############################################
######################################################################

def extract_line(line):
	'''
		extrace sequence from a line of data
		each element of the seuqence is a tuple (observation, state)
	'''
	while line:
		# split a line into 3 parts as a{{b}}c
		(a, _, b) = line.partition('{{')
		(b, _, c) = b.partition('}}')
		
		# yield tuples
		if a:
			for ch in a:
				yield (ch, NULL_STATE)		#null state
		if b:
			(nertype, _, name) = b.partition(':')
			yield (name[0], BEGIN_STATE_PREFIX + u'_' + 'nertype')		#b ner state
			for ch in name[1:]:
				yield (ch, INTER_STATE_PREFIX + u'_' + 'nertype')		#i ner state
		
		# handle c recursively
		line = c


def extract_file(path):
	'''
		read input data file, 
		generate an observation sequence and a state sequence per line
	'''
	f = codecs.open(path, 'r', 'UTF-8')
	observe_seqs, state_seqs = [], []
	for line in f.readlines():
		o_seq, s_seq = zip(*list(extract_line(line)))	#unzip o and s from list of [(o1, s1), (o2, s2)...]
		observe_seqs.append(o_seq)		#sequences of observe tuples [(o1.1, o1.2, o1.3...), (o2.1, o2.2, ...),...]
		state_seqs.append(s_seq)		#sequences of state tuples
	return observe_seqs, state_seqs


class MEMM:
	"""
		Maximum Entropy Markov Models

		In short, 
		It is a discriminant model for finite state sequential data,
		it trains maxent models that represent the probability of a 
		state given an observation and previous state.

		Reference:
		http://cs.nju.edu.cn/dxy/mt.htm#assignment4
		http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf
	"""
	def __init__(self, D, INIT_STATE):
		self.D = D
		self.INIT_STATE = INIT_STATE
		self.models = {}
		self.states = []

	def fit(self, observe_seqs, state_seqs):
		'''	fit the model with multiple observation sequences, each 
			observation sequence has a corresponding state sequences
		'''
		# 
		self.states = list(reduce(lambda x,y: set(x)|set(y), [s for s in state_seqs]))	#unique states

		observe_list = self._reform_observe_seqs(observe_seqs)			#list of all observations
		state_list = self._reform_state_seqs(state_seqs)				#list of all corresponding states
		pstate_list = self._reform_pstate_seqs(state_seqs)				#list of all previous states

		# for each state, training a maxent model
		for state in self.states:
			indice = pstate_list == state
			x = self._observe_to_matrix(observe_list[indice])
			y = state_list[indice]
			self.models[state] = LogisticRegression(solver='lbfgs', multi_class='multinomial')
			self.models[state].fit(x, y)

		# training finished

	def predict(self, observe_seq):
		'''	predict the state sequence of the given observation sequence
		'''
		observe_list = self._reform_observe_seqs([observe_seq])
		prob, state_seq = self._viterbi(observe_list)
		prob2, state_seq2 = self._viterbi2(observe_list)

		pdb.set_trace()
		return tuple(state_seq)
		

	def _reform_observe_seqs(self, seqs):
		seq_list = []
		for s in seqs:
			s_list = list(s)
			seq_list += zip([None]+s_list[:-1], s_list, s_list[1:]+[None])	#enable 3-sized window
		return np.array(seq_list)

	def _reform_state_seqs(self, seqs):
		seq_list = []
		for s in seqs:
			seq_list += list(s)
		return np.array(seq_list)

	def _reform_pstate_seqs(self, seqs):
		seq_list = []
		for s in seqs:
			seq_list += [self.INIT_STATE] + list(s)[0:-1]
		return np.array(seq_list)

	# add some features in an observation
	def _extend_observe(self, ob):	
		eob = []
		for i, o in enumerate(ob):	#ob is an observation containing some raw feature strings
			if o:
				#assert(isinstance(o, unicode))
				eob += [unicode(i) + u'_' + o]
				if o.encode('utf-8').isalpha():
					eob += [unicode(i) + u'_is_alpha']
				if o.encode('utf-8').isdigit():
					eob += [unicode(i) + u'_is_dight']
				if o.encode('utf-8').isspace():
					eob += [unicode(i) + u'_is_space']
		return eob

	# one-hot encoding an observation using hash trick
	def _observe_to_vector(self, o):
		x = np.zeros(shape = (self.D, ), dtype = np.int8)
		for f in self._extend_observe(o):
			j = hash(f) % self.D
			x[j] = 1
		return x

	# make up a sample matrix of observation sequences
	def _observe_to_matrix(self, observe_list):
		x = np.zeros(shape = (len(observe_list), self.D), dtype = np.int8)
		for i, o in enumerate(observe_list):
			for f in self._extend_observe(o):
				j = hash(f) % D
				x[i, j] = 1
		return coo_matrix(x)

	# transfer probility dictionary of an observation by all state pairs,
	# use the dictionary as tpd[previous_state][state]
	def _transfer_prob_dict(self, observe):
		tpd = {}
		x = self._observe_to_vector(observe).reshape(1, -1)
		for ps in self.states:
			ps_prob = {}
			model_prob = self.models[ps].predict_proba(x)[0]
			for i, ts in enumerate(self.models[ps].classes_):
				ps_prob[ts] = model_prob[i]
			for ts in (set(self.states) - set(self.models[ps].classes_)):
				ps_prob[ts] = 0
			tpd[ps] = ps_prob
		return tpd

	# transfer probility dictionary of a list of observations by all state pairs,
	# tpd[i][ps][s] = p(s|ps, o[i])
	def _transfer_prob_dict2(self, observe_list):
		tpd = []
		model_prob = {}			#model_prob[ps][i] = [p1, p2, p3...]
		x = self._observe_to_matrix(observe_list)
		for s in self.states:
			model_prob[s] = self.models[s].predict_proba(x)

		for i in range(0, len(observe_list)):
			tpd.append({})
			for ps in self.states:
				tpd[i][ps] = {}
				for j, ts in enumerate(self.models[ps].classes_):
					tpd[i][ps][ts] = model_prob[ps][i][j]
				for ts in (set(self.states) - set(self.models[ps].classes_)):
					tpd[i][ps][ts] = 0
		return tpd

	#https://en.wikipedia.org/wiki/Viterbi_algorithm
	def _viterbi2(self, observe_list):
		v = []
		path = {}
		tpd = self._transfer_prob_dict2(observe_list)
		for t in range(0, len(observe_list)):
			v.append({})
			newpath = {}
			if t == 0:
				for s in self.states:
					v[t][s] = tpd[t][self.INIT_STATE][s]
					newpath[s] = [s]
			else:	
				for s in self.states:
					(prob, ps) = max((tpd[t][ps][s]*v[t-1][ps], ps) for ps in self.states)
					v[t][s] = prob
					newpath[s] = path[ps] + [s]
			path = newpath
	
		(prob, state) = max((v[len(observe_list)-1][s], s) for s in self.states)
		return prob, path[state]


	
		

	#https://en.wikipedia.org/wiki/Viterbi_algorithm
	def _viterbi(self, observe_list):
		v = []
		path = {}
		for t, o in enumerate(observe_list):
			v.append({})
			newpath = {}
			tpd = self._transfer_prob_dict(o)
			if t == 0:
				for s in self.states:
					v[t][s] = tpd[self.INIT_STATE][s]
					newpath[s] = [s]
			else:	
				for s in self.states:
					(prob, ps) = max((tpd[ps][s]*v[t-1][ps], ps) for ps in self.states)
					v[t][s] = prob
					newpath[s] = path[ps] + [s]
			path = newpath
	
		(prob, state) = max((v[len(observe_list)-1][s], s) for s in self.states)
		return prob, path[state]


######################################################################
#### training & testing ##############################################
######################################################################

# load data
observe_seqs, state_seqs = extract_file(path)

# start training
memm = MEMM(D, NULL_STATE)
memm.fit(observe_seqs, state_seqs)
memm.predict(observe_seqs[0])









