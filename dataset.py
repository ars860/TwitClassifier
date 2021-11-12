import re
from pathlib import Path
from typing import Union

import pandas
import snowballstemmer
import torch
from torch.utils.data import Dataset, DataLoader

from embeddings import RandomEmbedding, Word2VecEmbedding, Embedding
from preprocessing import Clean, StopWords


def split_sentence(sentence: str) -> list[str]:
    sentence = re.sub(r"https?:\S+|http?:\S|[^A-Za-z0-9@#]+", ' ', sentence.lower())
    # sentence = map(lambda w: stemmer.stemWord(w), )
    return list(filter(lambda w: w != '', sentence.split(' ')))


def one_hot_company(company: str):
    one_hotted = None
    if company == 'apple':
        one_hotted = [1, 0, 0, 0]
    if company == 'google':
        one_hotted = [0, 1, 0, 0]
    if company == 'microsoft':
        one_hotted = [0, 0, 1, 0]
    if company == 'twitter':
        one_hotted = [0, 0, 0, 1]

    return torch.Tensor(one_hotted)


def one_hot_sentiment(sentiment: str):
    one_hotted = None
    if sentiment == 'positive':
        one_hotted = [1, 0, 0, 0]
    if sentiment == 'negative':
        one_hotted = [0, 1, 0, 0]
    if sentiment == 'neutral':
        one_hotted = [0, 0, 1, 0]
    if sentiment == 'irrelevant':
        one_hotted = [0, 0, 0, 1]

    return torch.Tensor(one_hotted)


class TwitDataset(Dataset):
    def __init__(self, dataset_path, keys, value, embedding_dim=10,
                 embedding: Union[str, Embedding] = "random", preprocessing: str = "simple",
                 use_stop_words: bool = False):
        value, value_processor = value

        self.stemmer = snowballstemmer.stemmer("english")

        csv = pandas.read_csv(dataset_path)
        csv = csv[["TweetText", *map(lambda k: k[0], keys), value]]

        if preprocessing == "simple":
            self.preprocess = split_sentence
        elif preprocessing == "tutorial":
            self.preprocess = Clean()
        else:
            raise AttributeError(f"Unknown preprocessing: {preprocessing}")

        self.rows = []
        for (i, row) in csv.iterrows():
            keys_processed = []

            processed = self.preprocess_with_stem(row["TweetText"])
            if len(processed) > 0:
                keys_processed.append(processed)
            else:
                print(f"Tweet ignored due to unreadability: {row.TweetText}")
                continue

            for key, processor in keys:
                keys_processed.append(processor(row[key]))

            self.rows.append((*keys_processed, value_processor(row[value])))

        sentences = [sv[0] for sv in self.rows]

        if use_stop_words:
            self.stop = StopWords(sentences, self.stemmer)
            self.rows = list(filter(lambda sv: len(sv[0]) > 0, map(lambda sv: (self.stop(sv[0]), *sv[1:]), self.rows)))

        sentences = [sv[0] for sv in self.rows]

        if isinstance(embedding, str):
            if embedding == "random":
                self.embedding = RandomEmbedding(sentences, embedding_dim, load_name=value)
            if embedding == "word2vec":
                self.embedding = Word2VecEmbedding(sentences, embedding_dim)
        else:
            self.embedding = embedding.clone()
            self.embedding.update(sentences)

        self.rows = list(filter(lambda sv: len(sv[0]) > 0, map(lambda sv: (self.embedding(sv[0]), *sv[1:]), self.rows)))

    def preprocess_with_stem(self, sentence):
        return list(map(lambda w: self.stemmer.stemWord(w), self.preprocess(sentence)))

    def process(self, sentence):
        sentence = self.preprocess_with_stem(sentence)
        sentence = self.stop(sentence)
        sentence = torch.Tensor(self.embedding(sentence))
        sentence = torch.unsqueeze(sentence, 0)

        return sentence

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


def twit2company_dataset(dataset_path: Path, embedding_dim: int, embedding: str, preprocessing: str,
                         use_stop_words: bool):
    return TwitDataset(dataset_path=dataset_path, keys=[], value=("Topic", one_hot_company),
                       embedding_dim=embedding_dim,
                       embedding=embedding,
                       preprocessing=preprocessing, use_stop_words=use_stop_words)


def twit2sentiment_dataset(dataset_path: Path, embedding_dim: int, embedding: str, preprocessing: str,
                           use_stop_words: bool):
    return TwitDataset(dataset_path=dataset_path, keys=[], value=("Sentiment", one_hot_sentiment),
                       embedding_dim=embedding_dim,
                       embedding=embedding,
                       preprocessing=preprocessing, use_stop_words=use_stop_words)


def twit_company2sentiment_dataset(dataset_path: Path, embedding_dim: int, embedding: str, preprocessing: str,
                                   use_stop_words: bool):
    return TwitDataset(dataset_path=dataset_path, keys=[("Topic", one_hot_company)],
                       value=("Sentiment", one_hot_sentiment), embedding_dim=embedding_dim,
                       embedding=embedding,
                       preprocessing=preprocessing, use_stop_words=use_stop_words)


def get_dataloaders(task, dataset_path: str = "dataset", workers: int = 1, batch_size: int = 1,
                    embedding_dim: int = 10, embedding: str = "random", preprocessing: str = "simple",
                    use_stop_words: bool = False):
    get_dataset = twit2company_dataset if task == "text2company" else twit2sentiment_dataset if task == "text2sentiment" else twit_company2sentiment_dataset

    train_dataset = get_dataset(Path() / dataset_path / "Train.csv", embedding_dim=embedding_dim,
                                embedding=embedding, preprocessing=preprocessing, use_stop_words=use_stop_words)
    test_dataset = get_dataset(Path() / dataset_path / "Test.csv", embedding=train_dataset.embedding,
                               embedding_dim=embedding_dim, preprocessing=preprocessing, use_stop_words=use_stop_words)

    return train_dataset, DataLoader(train_dataset, batch_size=batch_size, num_workers=workers), \
           test_dataset, DataLoader(test_dataset, batch_size=batch_size, num_workers=workers)


def get_twit_company_dataloaders(dataset_path: str = "dataset", workers: int = 1, batch_size: int = 1,
                                 embedding_dim: int = 10, embedding: str = "random", preprocessing: str = "simple",
                                 use_stop_words: bool = False):
    return get_dataloaders("text2company", dataset_path, workers, batch_size, embedding_dim, embedding, preprocessing,
                           use_stop_words=use_stop_words)


def get_twit_sentiment_dataloaders(dataset_path: str = "dataset", workers: int = 1, batch_size: int = 1,
                                   embedding_dim: int = 10, embedding: str = "random", preprocessing: str = "simple",
                                   use_stop_words: bool = False):
    return get_dataloaders("text2sentiment", dataset_path, workers, batch_size, embedding_dim, embedding, preprocessing,
                           use_stop_words=use_stop_words)


def get_twit_company_sentiment_dataloaders(dataset_path: str = "dataset", workers: int = 1, batch_size: int = 1,
                                           embedding_dim: int = 10, embedding: str = "random",
                                           preprocessing: str = "simple", use_stop_words: bool = False):
    return get_dataloaders("text_company2sentiment", dataset_path, workers, batch_size, embedding_dim, embedding,
                           preprocessing, use_stop_words=use_stop_words)
