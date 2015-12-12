
# csdaiwei@foxmail.com

import pdb
import codecs
import numpy as np

from datetime import datetime
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

D = 10 ** 5		#hash trick size, aka dimension of features

######################################################################
#### functions, classes ##############################################
######################################################################

def extract_line(line):
	'''
		extrace sequence from a line of data
		each element of the seuqence is a tuple (observation, state)

		INPUT:
			one line of inputfile

		OUTPUT:
			yield pair of string, (observation, state)
	'''
	while line:
		# split a line into 3 parts by the first occurence of {{,  as a{{b}}c
		(a, _, b) = line.partition('{{')
		(b, _, c) = b.partition('}}')
		
		# yield tuples
		if a:
			for ch in a:
				yield (ch, NULL_STATE)		#null state
		if b:
			(nertype, _, name) = b.partition(':')
			yield (name[0], BEGIN_STATE_PREFIX + u'_' + nertype)		#b ner state
			for ch in name[1:]:
				yield (ch, INTER_STATE_PREFIX + u'_' + nertype)		#i ner state
		
		# handle c recursively
		line = c


def extract_file(path):
	'''
		read input data file, 
		generate an observation sequence and a state sequence per line

		INPUT:
			BosonNLP_NER_6C file path
		
		OUTPUT:
			oseqs:	[(o1.1, o1.2...), (o2.1, o2.2...), ...]
					list of tuples, each element is an observation 
					sequence gengrate from a line of inputfile
			
			sseqs:	 	similar as oseqs

	'''
	f = codecs.open(path, 'r', 'UTF-8')
	oseqs, sseqs = [], []
	for line in f.readlines():
		oseq, sseq = zip(*list(extract_line(line)))	#unzip o and s from list of [(o1, s1), (o2, s2)...]
		oseqs.append(oseq)
		sseqs.append(sseq)
	return oseqs, sseqs

def precision(py, y):		#y and py is np.ndarry
	n = ((py != NULL_STATE)*(y == py)).sum()
	d = (py != NULL_STATE).sum()
	return float(n)/d

def recall(py, y):
	n = ((py != NULL_STATE)*(y == py)).sum()
	d = (y != NULL_STATE).sum()
	return float(n)/d




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
		'''	
			fit the model with multiple observation sequences, each 
			observation sequence has a corresponding state sequence
			
			INPUT:	
				list of observation sequence 	[(oseq1), (oseq2), ...]
				list of state sequence 			[(sseq1), (sseq2), ...]
		'''
		self.states = list(reduce(lambda x,y: set(x)|set(y), [s for s in state_seqs]))	#unique states

		observe_list = self._reform_observe_seqs(observe_seqs)			#list of all observations
		state_list = self._reform_state_seqs(state_seqs)				#list of all corresponding states
		pstate_list = self._reform_pstate_seqs(state_seqs)				#list of all previous states

		# for each state, training a maxent model
		self.models = {}
		for state in self.states:
			indice = pstate_list == state
			x = self._observe_to_matrix(observe_list[indice])
			y = state_list[indice]
			self.models[state] = LogisticRegression(solver='lbfgs', multi_class='multinomial')
			self.models[state].fit(x, y)

		# training finished


	def predict(self, observe_seq):
		'''	
			predict the state sequence of the given observation sequence
			
			INPUT: 
				one observation sequence (o1, o2, ...)
				only 1 sequence!

			OUTPUT:
				one state sequence (s1, s2, ...)
		'''
		observe_list = self._reform_observe_seqs([observe_seq])
		prob, state_seq = self._viterbi(observe_list)
		return tuple(state_seq)
		

	# reform [(oseq1), (oseq2)] into [oseq1.1, oseq1.2, ..., oseq2.1, oseq2.2, ... ]
	# also,  a window of size 3 is add to each observation, that is oseq1.2 = (oseq1[1], oseq1[2], oseq1[3])
	def _reform_observe_seqs(self, seqs):
		seq_list = []
		for s in seqs:
			s_list = list(s)
			seq_list += zip([None]+s_list[:-1], s_list, s_list[1:]+[None])	#enable 3-sized window
		return np.array(seq_list)


	# simply joining each tuple(sequence) to a list
	def _reform_state_seqs(self, seqs):
		seq_list = []
		for s in seqs:
			seq_list += list(s)
		return np.array(seq_list)


	# joining each tuple(sequence) to a list, p means previous
	def _reform_pstate_seqs(self, seqs):
		seq_list = []
		for s in seqs:
			seq_list += [self.INIT_STATE] + list(s)[0:-1]	#add an INIT_STATE at the beginning of each sequence
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


	# make up a sample matrix of observation sequences
	# one-hot encoding is introduced by hash trick of size D
	def _observe_to_matrix(self, observe_list):
		x = np.zeros(shape = (len(observe_list), self.D), dtype = np.int8)
		for i, o in enumerate(observe_list):
			for f in self._extend_observe(o):
				j = hash(f) % D
				x[i, j] = 1
		return coo_matrix(x)


	# transfer probility dictionary of a list of observations by all state pairs,
	# tpd[i][ps][s] = p(s|ps, o[i])
	def _transfer_prob_dict(self, observe_list):
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
	def _viterbi(self, observe_list):
		v = []
		path = {}
		tpd = self._transfer_prob_dict(observe_list)
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

#end of class MEMM


def cross_validation(model, oseqs, sseqs, k=10):
	
	n = len(oseqs)	#assert len(oseq) == len(sseq)
	c = n/k			#remainder removed division
	for i in range(0, k):
		test_oseqs = oseqs[c*i : c*(i+1)]
		test_sseqs = sseqs[c*i : c*(i+1)]
		train_oseqs = oseqs[0 : c*i] + oseqs[c*(i+1):]
		train_sseqs = sseqs[0 : c*i] + sseqs[c*(i+1):]

		# training
		start = datetime.now()
		memm.fit(train_oseqs, train_sseqs)
		print '\nfold %d, training time: %s'%(i, str(datetime.now() - start))

		# testing on left data
		start = datetime.now()
		psseqs = []		#predicted state sequences
		for oseq in test_oseqs:
			psseq = memm.predict(oseq)
			psseqs += psseq

		print 'fold %d, testing time: %s'%(i, str(datetime.now() - start))

		py = np.array(psseqs)
		y = np.array(reduce(lambda x,y: list(x)+list(y), [s for s in test_sseqs]))

		print 'fold %d, precision: %.4f'%(i, precision(py, y))
		print 'fold %d, recall: %.4f'%(i, recall(py, y))

######################################################################
#### training & testing ##############################################
######################################################################

observe_seqs, state_seqs = extract_file(path)

print 'loaded data, with D:%d'%(D)
print 'cross validation...'

memm = MEMM(D, NULL_STATE)

cross_validation(memm, observe_seqs, state_seqs)

pdb.set_trace()









