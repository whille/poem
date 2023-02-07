#!/usr/bin/env python
from t_poem import *


def test_sample():
    fname = './data/sample.txt'
    for s in gen_word(gen_sentence(gen_poem(fname))):
        print(s)
