#!/usr/bin/env python
import os
import re
# from itertools import cycle
# import jieba
from gensim.models.word2vec import Word2Vec, KeyedVectors


def gen_poem(fname):
    stop_words = set(('。', '，'))
    with open(fname) as f:
        txt = ''
        for line in f:
            line = line.strip()
            if not line or line[-1] not in stop_words:
                if txt:
                    yield txt
                    txt = ''
                continue
            else:
                txt += line


def gen_sentence(gen):
    for poem in gen:
        # print(poem)
        if len(poem) < 6 and len(poem) % 6 != 0 and len(poem) % 8 != 0:
            continue
        for s in re.split(r"。", poem):
            # print(s)
            if s:
                yield s


def gen_word(gen):
    for s in gen:
        N, n = len(s), (len(s)) // 2
        if s[n] != '，':
            continue
        for i in range(N):
            if i == n:
                continue
            lst = s[i]
            if i > 0 and i != n+1:
                lst += s[i-1]
            if i < N - 1 and i != n-1:
                lst += s[i + 1]
            p = n+1
            if i > n:
                p = -p
            try:
                lst += s[i + p]
            except Exception:
                # print(s, i, p)
                pass
            yield lst


class Actor():

    def __init__(self, f_corpse, model_name=None, fvec=None, vector_size=128, window=5, min_count=1):
        self.f_corpse = f_corpse
        self.fmode = model_name
        self.fvec = f"{fvec}_{vector_size}_{window}_{min_count}"
        self.wv = None
        self.vector_size=vector_size
        self.window=window
        self.min_count=min_count

    # https://radimrehurek.com/gensim/models/word2vec.html
    def train(self):
        if os.path.exists(self.fvec):
            return
        print("start train...")
        model = Word2Vec(list(gen_word(gen_sentence(gen_poem(self.f_corpse)))), workers=4,
                vector_size=self.vector_size, window=self.window, min_count=self.min_count)
        if self.fmode:
            model.save(self.mode)
        if self.fvec:
            model.wv.save(self.fvec)
        print("train done")

    def predict(self, word, topn=10):
        if not self.wv:
            self.train()
            print(f"{self.fvec} loading...")
            self.wv = KeyedVectors.load(self.fvec, mmap='r')
            print(f"{self.fvec} loaded")
        # Load back with memory-mapping = read-only, shared across processes.
        return self.wv.most_similar(word.strip(), topn=topn)


def main():
    fname = './data/全唐诗.txt'
    act = Actor(fname, fvec='./data/tang_poem.wv', min_count=5)
    act.train()
    while True:
        word = input('word?')
        if not word:
            break
        try:
            print(act.predict(word))
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
