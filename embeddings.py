from pathlib import Path
from sortedcontainers import SortedDict, SortedSet

import numpy as np
import torch
from gensim.models import Word2Vec
from torch import nn


class Embedding:
    def word_embedding(self, word):
        raise NotImplementedError()

    def update(self, sentences):
        pass

    def clone(self):
        return self

    def __call__(self, sentence):
        return np.stack(list(map(self.word_embedding, sentence)))


class RandomEmbedding(Embedding):
    def __init__(self, sentences, embedding_dim, load_name):
        words = SortedSet()
        for sentence in sentences:
            for word in sentence:
                words.add(word)

        word2id = SortedDict()
        for (i, word) in enumerate(words):
            word2id[word] = i

        self.word2id = word2id
        self.embedding = nn.Embedding(len(word2id) + 1, embedding_dim=embedding_dim)
        self.load(load_name)
        self.embedding.requires_grad_ = False

    def load(self, name):
        name = "text2company_embeddings" if name == "Topic" else "text2sentiment_embeddings"
        load_path = Path() / "learned_embeddings" / f"{name}.pt"
        if load_path.is_file():
            self.embedding.load_state_dict(torch.load(load_path))
            # self.embedding = nn.Embedding.from_pretrained(torch.load(load_path)['weight'])
        else:
            torch.save(self.embedding.state_dict(), load_path)

    def word_embedding(self, word):
        if word not in self.word2id:
            return self.embedding(torch.LongTensor([len(self.word2id)])).detach().numpy().squeeze()

        return self.embedding(torch.LongTensor([self.word2id[word]])).detach().numpy().squeeze()


class Word2VecEmbedding(Embedding):
    def __init__(self, sentences=None, embedding_dim=None):
        if sentences is not None and embedding_dim is not None:
            self.model = Word2Vec(size=embedding_dim, window=5, min_count=1, workers=2)
            self.model.build_vocab(sentences=sentences, progress_per=100)
            self.model.train(sentences=sentences, total_examples=len(sentences), epochs=32)

    def update(self, sentences):
        self.model.build_vocab(sentences, update=True)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)

    def word_embedding(self, word):
        return self.model.wv[word]

    def clone(self):
        self.model.save("learned_models/word2vec_tmp.model")
        model = Word2Vec.load("learned_models/word2vec_tmp.model")
        # model.load("learned_models/word2vec_tmp.model")
        embedding = Word2VecEmbedding()
        embedding.model = model

        return embedding
