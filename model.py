import torch
from torch import nn


class LSTMTwitClassifier(nn.Module):
    def __init__(self, targets_cnt, embedding_dim=10, hidden_dim=10):
        super(LSTMTwitClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.word2id = word2id

        self.unknown_words2embedding = {}

        # self.embedding = nn.Embedding(len(word2id), embedding_dim=embedding_dim)
        # self.embedding.requires_grad_ = False

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, targets_cnt)
        # self.linear2 = nn.Linear(hidden_dim, targets_cnt)

    def get_word_embedding(self, word):
        if word not in self.word2id:
            if word not in self.unknown_words2embedding:
                self.unknown_words2embedding[word] = torch.rand(self.embedding_dim)
            return self.unknown_words2embedding[word]

        return self.embedding(torch.LongTensor([self.word2id[word]]))

    def _get_sentence_embedding(self, sentence):
        embedded = [self.get_word_embedding(word) for word in sentence]
        return torch.stack(embedded)

    # only batch size 1 is supported
    def forward(self, sentence):
        # if len(sentence) == 1 and isinstance(sentence[0], list):
        #     sentence = sentence[0]

        # embedded = self._get_sentence_embedding(sentence)

        _, (h, _) = self.lstm(sentence.view(sentence.shape[1], 1, -1))
        h = self.linear1(h.view(-1))
        # h = self.linear2(h)

        return h
