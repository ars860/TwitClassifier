import torch
from torch import nn


class LSTMTwitClassifier(nn.Module):
    def __init__(self, targets_cnt, embedding_dim=10, hidden_dim=10, dropout=0, lstm_layers=1,
                 additional_one_hot_arg=False):
        super(LSTMTwitClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(lstm_layers * hidden_dim + (4 if additional_one_hot_arg else 0), targets_cnt)

    def get_word_embedding(self, word):
        if word not in self.word2id:
            return torch.zeros(self.embedding_dim)

        return self.embedding(torch.LongTensor([self.word2id[word]]))

    def _get_sentence_embedding(self, sentence):
        embedded = [self.get_word_embedding(word) for word in sentence]
        return torch.stack(embedded)

    # only batch size 1 is supported
    def forward(self, sentence, one_hot_stuff=None):
        _, (h, _) = self.lstm(sentence.view(sentence.shape[1], 1, -1))
        h = self.dropout(h)

        if one_hot_stuff is not None:
            h = torch.cat((h.view(-1), one_hot_stuff.view(-1)))

        h = self.linear1(h.view(-1))

        return h
