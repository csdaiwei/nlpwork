
# csdaiwei@foxmail.com

import pdb
import codecs

ch_path = 'test.ch.txt'
en_path = 'test.en.txt'
align_path = 'test.align.txt'	# ch-en alignment
word_separator = ' '
align_separator = '-'

# return seqs is [(s0w0, s0w1, s0w2...),  (s1w0, s1w2...),  (), ...]
# seqs for sequences, s for sentence, w for word
def read_data(path):
	seqs = []
	f = codecs.open(path, 'r', 'utf-8')
	for l in f.readlines():
		seqs.append(tuple(l.strip().split(word_separator)))
	return seqs


def ibm_m1_train(ch_seqs, en_seqs, align_seqs):

	a_cnt = {}		# a_cnt[e][f] is the # of word e aligned to word f
	e_cnt = {}		# e_cnt[e] is the # of word e aligned to any French(Chinese) word

	# init a_cnt & e_cnt as 0
	ch_set = reduce(lambda x,y: set(x)|set(y), ch_seqs)
	en_set = reduce(lambda x,y: set(x)|set(y), en_seqs)
	for e in en_set:
		e_cnt[e] = 0
		a_cnt[e] = {}
		for c in ch_set:
			a_cnt[e][c] = 0

	# calc a_cnt & e_cnt
	n = len(ch_seqs)
	for k in range(0, n):
		ch, en, align = ch_seqs[k], en_seqs[k], align_seqs[k]
		for a in align:
			i, j = map(int, a.split(align_separator))
			c, e = ch[i], en[j]
			
			a_cnt[e][c] += 1
			e_cnt[e] += 1

	return a_cnt, e_cnt

def ibm_m1_test(a_cnt, e_cnt, ch_seq, en_seq):
	#t(f|e) = a_cnt[e][f] / e_cnt[e]
	align_seq = []
	for i in range(0, len(en_seq)):		# for each english word 
		maxp = 0
		maxj = 0
		e = en_seq[i]
		if e_cnt[e] == 0:
			break;
		for j in range(0, len(ch_seq)):		# find chinese word with best match
			c = ch_seq[j]
			p = a_cnt[e][c] / float(e_cnt[e])
			if p > maxp:
				maxp, maxj = p, j
		if maxp > 0:
			align_seq.append(str(maxj) + align_separator + str(i))
	return tuple(align_seq)

def precision(pseqs, seqs):
	tp = 0
	for i in range(0, len(pseqs)):
		p, y  = set(pseqs[i]), set(seqs[i])
		tp += len(p & y)
	d = len(reduce(lambda x, y: x+y, pseqs))
	return tp / float(d)
	

def recall(pseqs, seqs):
	tp = 0
	for i in range(0, len(pseqs)):
		p, y  = set(pseqs[i]), set(seqs[i])
		tp += len(p & y)
	d = len(reduce(lambda x, y: x+y, seqs))	#only difference here
	return tp / float(d)	



if __name__ == '__main__':
	ch_seqs = read_data(ch_path)
	en_seqs = read_data(en_path)
	align_seqs = read_data(align_path)
	a_cnt, e_cnt = ibm_m1_train(ch_seqs, en_seqs, align_seqs)
	palign = [ibm_m1_test(a_cnt, e_cnt, ch_seqs[i], en_seqs[i])  for i in range(0, len(ch_seqs))]

	print precision(palign, align_seqs)
	print recall(palign, align_seqs)


	pdb.set_trace()


