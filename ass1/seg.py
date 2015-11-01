#! /usr/bin/python
# -*- coding: utf-8 -*-

import pdb
import sys

dict_path = 'dic_ce_utf8.txt'

def read_dict(path):
    d = {}
    f = open(path, 'r')
    for l in f.read().split('\n'):
        s = [e for e in l.split(',', 1)]
        d[s[0]] = s[1:]
    return d        

def fmm(dic, u):        #forward maximun matching
    l = len(u)
    if l <= 0:
        return []
    for i in range(0, l):
        k = u[0:l-i].encode('utf-8')
        if dic.has_key(k):
            break
    return [k]+fmm(dic, u[len(k.decode('utf-8')):])


def rmm(dic, u):        #reverse maximun matching
    l = len(u)
    if l <= 0:
        return []
    for i in range(0, l):
        k = u[i:].encode('utf-8')
        if dic.has_key(k):
            break
    return rmm(dic, u[:l-len(k.decode('utf-8'))]) + [k]


def seg(dic, u):
    fs = fmm(dic, u)
    rs = rmm(dic, u)
    return fs if len(fs)<=len(rs) else rs   #less words, better segmentation

    

if __name__ == '__main__':

    dic = read_dict(dict_path)  #key: cwords(encode in utf8), value: corresponding ewords

    if len(sys.argv) != 2:
        print 'run this script with a parameter, try:'
        print '$ python seg.py 一阵春风吹过'
        exit()

    u = sys.argv[1].decode('utf-8')
    s = seg(dic, u)

    print ' | '.join(s)

