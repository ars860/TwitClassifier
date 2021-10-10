import re
from pathlib import Path
from typing import Union

import snowballstemmer
import pandas
import torch
from torch.utils.data import Dataset, DataLoader

from embeddings import RandomEmbedding, Word2VecEmbedding, Embedding

stemmer = snowballstemmer.stemmer("english")


def split_sentence(sentence: str) -> list[str]:
    sentence = re.sub(r"https?:\S+|http?:\S|[^A-Za-z0-9@#]+", ' ', sentence.lower())
    sentence = map(lambda w: stemmer.stemWord(w), filter(lambda w: w != '', sentence.split(' ')))
    return list(sentence)


def one_hot_company(company: str):
    if company == 'apple':
        return [1, 0, 0, 0]
    if company == 'google':
        return [0, 1, 0, 0]
    if company == 'microsoft':
        return [0, 0, 1, 0]
    if company == 'twitter':
        return [0, 0, 0, 1]


class TwitCompanyDataset(Dataset):
    def __init__(self, dataset_path, embedding_dim=10, embedding: Union[str, Embedding] = "random"):
        csv = pandas.read_csv(dataset_path)
        csv = csv[["TweetText", "Topic"]]

        sentences = []
        for (i, row) in csv.iterrows():
            sentence = split_sentence(row.TweetText)
            sentences.append(sentence)

        if isinstance(embedding, str):
            if embedding == "random":
                self.embedding = RandomEmbedding(sentences, embedding_dim)
            if embedding == "word2vec":
                self.embedding = Word2VecEmbedding(sentences, embedding_dim)
        else:
            # if isinstance(embedding, Word2VecEmbedding):
            self.embedding = embedding.clone()
            self.embedding.update(sentences)
            # else:
            #     raise NotImplementedError("Fuck you")

        self.rows = []
        for (i, row) in csv.iterrows():
            processed = split_sentence(row.TweetText)
            if len(processed) > 0:
                sentence = self.embedding(split_sentence(row.TweetText))
                self.rows.append((sentence, torch.Tensor(one_hot_company(row.Topic))))
            else:
                print(row.TweetText)

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


def get_twit_company_dataloaders(dataset_path: str = "dataset", workers: int = 1, batch_size: int = 1,
                                 embedding_dim: int = 10, embedding: str = "random"):
    train_dataset = TwitCompanyDataset(Path() / dataset_path / "Train.csv", embedding_dim=embedding_dim,
                                       embedding=embedding)
    test_dataset = TwitCompanyDataset(Path() / dataset_path / "Test.csv", embedding=train_dataset.embedding,
                                      embedding_dim=embedding_dim)

    return train_dataset, DataLoader(train_dataset, batch_size=batch_size, num_workers=workers), \
           test_dataset, DataLoader(test_dataset, batch_size=batch_size, num_workers=workers)
