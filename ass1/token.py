#! /usr/bin/python
# -*- coding: utf-8 -*-

import pdb
import sys

dict_path = 'dic_ec_utf8.txt'

def read_dict(path):
    d = {}
    f = open(path, 'r')
    for l in f.read().split('\n'):
        s = [e for e in l.split('')]
        d[s[0]] = s[1:]
    return d

def word_reduce(dic, w):
    candidate = [] 
    
    #construct candidate using reduction rules
    if w.endswith('s'):
        candidate.append(w[:-1])                        #singular
        if w.endswith('es'):
            candidate.append(w[:-2])
        if w.endswith('ies'):
            candidate.append(w[:-3]+'y')
    if w.endswith('ing'):
        candidate.append(w[:-3])                        #ving
        candidate.append(w[:-3]+'e')
        if w.endswith('ying'):  #enjoy
            candidate.append(w[:-4]+'ie')
        if w[-4] == w[-5]: #&& w.endswith('ing')
            candidate.append(w[:-4])
    if w.endswith('ed'):
        candidate.append(w[:-2])                        #ven
        candidate.append(w[:-2]+'e')
        if w.endswith('ied'):
            candidate.append(w[:-3]+'y')
        if w[-3] == w[-4]: #&& w.endswith('ed')
            candidate.append(w[:-3])

    #no suitable reduction rules, return w itself 
    if len(candidate) == 0:
        return  w

    #else, choose best candidate by appearance in the dic
    for c in candidate:
        if dic.has_key(c):
            return c

    #no proper candidate
    return candidate[0]

    

if __name__ == '__main__':

    dic = read_dict(dict_path)  # key: word, value: [speech, meaning]

    if len(sys.argv) != 2:
        print 'run this script with a parameter, try:'
        print '$ python token.py tests'
        exit()

    w = sys.argv[1]
    if dic.has_key(w):
        print w, 'in the dictionary:'
        print w, dic[w][0], dic[w][1]                           #situation 1, no need to reduce
    else:
        rw = word_reduce(dic, w)
        print w, '->', rw
        if dic.has_key(rw):
            print rw, dic[rw][0], dic[rw][1]                    #situation 2, reduce success
        else:
            if w == rw:
                print 'no suitable reduction rule for', w       #situation 3, reduce fail
            else:
                print 'but there is no ', rw, 'in dictionary'   #situation 4, reduce a bad word

        