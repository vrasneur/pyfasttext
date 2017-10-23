#!/usr/bin/env python

from pyfasttext import FastText

from sys import argv

def print_subwords(fname):
    model = FastText(fname)
    maxn = model.args['maxn']
    res = {}
    
    for word in model.words:
        for subword, arr in zip(model.get_subwords(word), model.get_numpy_subword_vectors(word)):
            # real ngram, not the full word?
            if len(subword) > maxn:
                continue
            
            res[subword] = arr

    for key in sorted(res.keys()):
        print('{} {}'.format(key, ' '.join(str(val) for val in res[key])))

if __name__ == '__main__':
    print_subwords(argv[1])
