import numpy as np
import torch
from gensim.models import Word2Vec
from torch import nn


class Embedding:
    def word_embedding(self, word):
        raise NotImplementedError()

    def update(self, sentences):
        pass

    def __call__(self, sentence):
        embeddings = []
        for word in sentence:
            embeddings.append(self.word_embedding(word))
        return np.stack(embeddings)


class RandomEmbedding(Embedding):
    def __init__(self, sentences, embedding_dim):
        words = set()
        for sentence in sentences:
            for word in sentence:
                words.add(word)

        word2id = {}
        for (i, word) in enumerate(words):
            word2id[word] = i

        self.word2id = word2id
        self.embedding = nn.Embedding(len(word2id) + 1, embedding_dim=embedding_dim)
        self.embedding.requires_grad_(False)

    def word_embedding(self, word):
        if word not in self.word2id:
            return self.embedding(torch.LongTensor(len(self.word2id))).numpy()

        return self.embedding(torch.LongTensor(self.word2id[word])).numpy()


class Word2VecEmbedding(Embedding):
    def __init__(self, sentences, embedding_dim):
        self.model = Word2Vec(sentences=sentences, size=embedding_dim, window=5, min_count=1, workers=2)

    def update(self, sentences):
        self.model.build_vocab(sentences, update=True)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)

    def word_embedding(self, word):
        return self.model.wv[word]
