# csdaiwei@foxmail.com

# maximun entropy markov models for name entity recognition

from __future__ import division		#float division

import pdb
import codecs
import numpy as np

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

D = 10 ** 3		#hash trick size, aka dimension of features

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

def get_sequence(path):
	'''
		read data and generate a sequence per line
	'''
	f = codecs.open(path, 'r', 'UTF-8')
	sequence = []
	for line in f.readlines():
		seq = list(extract_line(line))
		sequence.append(seq)
	return sequence


def reform_sequence(seqs):
	'''
		reshape input sequences data into 3 corresponding list 
			o_list:  current observation list 
			s_list:  current state list
			ps_list: previous state list
	'''
	o_list = []
	s_list = []
	ps_list = []
	for seq in seqs:
		o_seq = [ o for (o, s) in seq]	#observation sequence
		s_seq = [ s for (o, s) in seq]	#state sequence
		o_list += zip([None]+o_seq[:-1], o_seq, o_seq[1:]+[None])	#enable 3-sized window
		s_list += s_seq
		ps_list += [NULL_STATE] + s_seq[0:-1]
	return np.array(o_list), np.array(s_list), np.array(ps_list)


# add some features in an observation
def extend_observe(ob):
	eob = []
	for i, o in enumerate(ob):
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

# make up a feature vector of an observation using hash trick
def observe_to_vector(o):
	x = np.zeros(shape = (D, ), dtype = np.int8)
	for f in extend_observe(o):
		j = hash(f) % D
		x[j] = 1
	return x


# make up a sample matrix of observation sequences
def observe_to_matrix(o_list):
	x = np.zeros(shape = (len(o_list), D), dtype = np.int)
	for i, o in enumerate(o_list):
		for f in extend_observe(o):
			j = hash(f) % D
			x[i, j] = 1
	return coo_matrix(x)
	

# transfer probility dictionary of an observation by all state pairs,
# use the dictionary as tpd[previous_state][state]
def transfer_prob_dict(models, observe, states):
	tpd = {}
	x = observe_to_vector(observe).reshape(1, -1)
	for ps in states:
		ps_prob = {}
		model_prob = models[ps].predict_proba(x)[0]
		for i, ts in enumerate(models[ps].classes_):
			ps_prob[ts] = model_prob[i]
		for ts in (set(states) - set(models[ps].classes_)):
			ps_prob[ts] = 0
		tpd[ps] = ps_prob
	return tpd


#https://en.wikipedia.org/wiki/Viterbi_algorithm
def viterbi(models, o_list, states):
	v = []
	path = {}
	for t, o in enumerate(o_list):
		v.append({})
		newpath = {}
		tpd = transfer_prob_dict(models, o, states)
		if t == 0:
			for s in states:
				try:
					v[t][s] = tpd[NULL_STATE][s]
					newpath[s] = [s]
				except Exception, e:
					pdb.set_trace()

		else:
			for s in states:
				(prob, ps) = max((tpd[ps][s]*v[t-1][ps], ps) for ps in states)
				v[t][s] = prob
				newpath[s] = path[ps] + [s]
		path = newpath
	
	(prob, state) = max((v[len(o_list)-1][s], s) for s in states)
	return prob, path[state]



######################################################################
#### training & testing ##############################################
######################################################################

#load sequential data
seqs = get_sequence(path)
observe_list, state_list, pstate_list = reform_sequence(seqs)

#for each state, train a logistic regression model
models = {}
states = list(set(state_list))
for state in states:
	indice = pstate_list == state
	x = observe_to_matrix(observe_list[indice])
	y = state_list[indice]
	models[state] = LogisticRegression(solver='lbfgs', multi_class='multinomial')
	models[state].fit(x, y)

#full validation
predict_state_list = []
for seq in seqs:
	o, s, ps = reform_sequence([seq])
	prob, path = viterbi(models, o, states)
	predict_state_list += path

#precision & recall
sl = state_list
psl = np.array(predict_state_list)
print 'precision ', ((psl != NULL_STATE)*(sl != NULL_STATE)).sum()/((psl != NULL_STATE)).sum()
print 'recall ', ((psl != NULL_STATE)*(sl != NULL_STATE)).sum()/((sl != NULL_STATE).sum())




