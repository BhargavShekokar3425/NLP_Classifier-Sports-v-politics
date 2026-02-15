"""
preprocess.py
=============
Text preprocessing and feature extraction:
  - Tokenisation with NLTK stopword removal
  - Bag-of-Words, TF-IDF, and N-gram vectorisers (from scratch)
"""

import math
import re
import string
from collections import Counter

# --------------- NLTK stopwords (library-based) ---------------
try:
    from nltk.corpus import stopwords as _sw
    import nltk

    try:
        STOPWORDS = set(_sw.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(_sw.words("english"))
except ImportError:
    # Fallback if NLTK is not installed
    STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now",
    }


# --------------- tokenisation ---------------

def tokenize(text, remove_stopwords=True):
    """Lowercase, strip punctuation, optionally remove stopwords."""
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# ===============================================================
# Feature extractors  (all implement fit / transform / fit_transform)
# ===============================================================

class BagOfWords:
    """Count-based Bag-of-Words vectoriser."""

    name = "BoW"

    def __init__(self, max_features=500):
        self.max_features = max_features
        self.vocab = {}

    def fit(self, documents):
        freq = Counter()
        for doc in documents:
            tokens = tokenize(doc)
            freq.update(set(tokens))
        most_common = freq.most_common(self.max_features)
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}

    def transform(self, documents):
        matrix = []
        for doc in documents:
            counts = Counter(tokenize(doc))
            vec = [0.0] * len(self.vocab)
            for word, count in counts.items():
                if word in self.vocab:
                    vec[self.vocab[word]] = float(count)
            matrix.append(vec)
        return matrix

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    @property
    def feature_names(self):
        names = [""] * len(self.vocab)
        for w, i in self.vocab.items():
            names[i] = w
        return names


class TFIDF:
    """TF-IDF vectoriser (from scratch)."""

    name = "TF-IDF"

    def __init__(self, max_features=500):
        self.max_features = max_features
        self.vocab = {}
        self.idf = {}

    def fit(self, documents):
        n_docs = len(documents)
        df = Counter()
        for doc in documents:
            tokens = set(tokenize(doc))
            df.update(tokens)

        if self.max_features:
            words = [w for w, _ in df.most_common(self.max_features)]
        else:
            words = sorted(df.keys())

        self.vocab = {w: i for i, w in enumerate(words)}
        self.idf = {
            w: math.log((1 + n_docs) / (1 + df[w])) + 1 for w in self.vocab
        }

    def transform(self, documents):
        matrix = []
        for doc in documents:
            tokens = tokenize(doc)
            tf = Counter(tokens)
            total = max(len(tokens), 1)
            vec = [0.0] * len(self.vocab)
            for word, count in tf.items():
                if word in self.vocab:
                    vec[self.vocab[word]] = (count / total) * self.idf[word]
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            matrix.append(vec)
        return matrix

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    @property
    def feature_names(self):
        names = [""] * len(self.vocab)
        for w, i in self.vocab.items():
            names[i] = w
        return names


class NGramVectoriser:
    """Word-level n-gram count vectoriser."""

    name = "Bigram"

    def __init__(self, n=2, max_features=500):
        self.n = n
        self.max_features = max_features
        self.vocab = {}

    def _extract(self, text):
        words = tokenize(text)
        return [
            " ".join(words[i : i + self.n])
            for i in range(len(words) - self.n + 1)
        ]

    def fit(self, documents):
        freq = Counter()
        for doc in documents:
            freq.update(set(self._extract(doc)))
        most_common = freq.most_common(self.max_features)
        self.vocab = {ng: idx for idx, (ng, _) in enumerate(most_common)}

    def transform(self, documents):
        matrix = []
        for doc in documents:
            counts = Counter(self._extract(doc))
            vec = [0.0] * len(self.vocab)
            for ng, count in counts.items():
                if ng in self.vocab:
                    vec[self.vocab[ng]] = float(count)
            matrix.append(vec)
        return matrix

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    @property
    def feature_names(self):
        names = [""] * len(self.vocab)
        for ng, i in self.vocab.items():
            names[i] = ng
        return names


# --------------- registry ---------------

FEATURE_EXTRACTORS = {
    "BoW": lambda: BagOfWords(max_features=500),
    "TF-IDF": lambda: TFIDF(max_features=500),
    "Bigram": lambda: NGramVectoriser(n=2, max_features=500),
}
