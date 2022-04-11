import os
import re
import json
import itertools
import tldextract

import numpy as np
import pandas as pd
from collections import Counter
from urllib.parse import urlparse
from sklearn import metrics
from skmultilearn.model_selection import IterativeStratification

import torch
import torch.nn as nn


def load_data(path_data):
    """
    convert parquet files to dataframe
    """
    df_data = pd.DataFrame(columns=["url", "target", "day"])
    for filename in os.listdir(path_data):
        if filename.endswith(".parquet"):
            df_data = pd.concat(
                [
                    df_data,
                    pd.read_parquet(
                        os.path.join(path_data, filename), engine="pyarrow"
                    ),
                ],
                ignore_index=True,
            )
    return df_data


def preprocess_data(data, min_tag_freq=20):
    # Remove duplicated sample
    data = data[~data["url"].duplicated()].reset_index(drop=True)
    # clean url
    # data['url'] = data['url'].apply(lambda x : parse_url(x))
    # Filter tags that have fewer than <min_tag_freq> occurrences
    targets = [item for sublist in list(data["target"]) for item in sublist]
    d = Counter(targets)

    tags_above_freq = Counter(tag for tag in d if d[tag] >= min_tag_freq)

    include = list(tags_above_freq.keys())
    data["target"] = data["target"].apply(filter, include=include)
    # Remove sample with no more remaining tags
    data = data[data["target"].map(len) > 0]

    return data


class LabelEncoder(object):
    """Label encoder for labels."""

    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(list(itertools.chain.from_iterable(y)))
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        y_one_hot = np.zeros((len(y), len(self.class_to_index)), dtype=int)
        for i, item in enumerate(y):
            for class_ in item:
                y_one_hot[i][self.class_to_index[class_]] = 1
        return y_one_hot

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            indices = np.where(item == 1)[0]
            classes.append([self.index_to_class[index] for index in indices])
        return classes

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


def parse_url(url):
    domain_name = tldextract.extract(url)[1]
    full_path = urlparse(url).path
    first_tokens = re.split("[- _ % : , / \. \+ ]", full_path)
    tokens = []
    for token in first_tokens:
        tokens += re.split("\d+", token)
    # return unique elements
    final_sentence = list(dict.fromkeys([domain_name] + tokens))
    return " ".join(final_sentence)


def filter(l, include=[], exclude=[]):
    """Filter a list using inclusion and exclusion lists of items."""
    filtered = [item for item in l if item in include and item not in exclude]
    return filtered


def iterative_train_test_split(X, y, train_size):
    """
    Custom iterative train test split which
    maintains balanced representation with respect
    to order-th label combinations.'
    """
    stratifier = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[
            1.0 - train_size,
            train_size,
        ],
    )
    train_indices, test_indices = next(stratifier.split(X, y))
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


def get_data_splits(data, train_size=0.7):
    # Get data
    X = data.url.to_numpy()
    y = data.target

    # Binarize y
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y = label_encoder.encode(y)

    # Split
    X_train, X_, y_train, y_ = iterative_train_test_split(X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = iterative_train_test_split(X_, y_, train_size=0.5)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder


def log_metrics(preds, labels):
    """
    Function to calculate AUC
    """
    preds = torch.stack(preds)
    preds = preds.cpu().detach().numpy()
    labels = torch.stack(labels)
    labels = labels.cpu().detach().numpy()

    fpr_micro, tpr_micro, _ = metrics.roc_curve(labels.ravel(), preds.ravel())
    auc_micro = metrics.auc(fpr_micro, tpr_micro)

    return {"auc_micro": auc_micro}
