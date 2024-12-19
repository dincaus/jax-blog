import re
import nltk
import numpy as np

from math import sqrt
from tqdm.notebook import tqdm
from nltk.corpus import stopwords
from collections import Counter


nltk.download('punkt_tab')
nltk.download('stopwords')


def remove_stopwords_and_common_words(tokens, additional_common_words=None):
    stop_words = set(stopwords.words('english'))

    if additional_common_words:
        stop_words.update(additional_common_words)

    filtered_tokens = [word.strip() for word in tokens if word.lower() not in stop_words and len(word) > 1]

    return filtered_tokens


def subsample_tokens(tokens: list[str], subsample_threshold: float = 1e-3) -> list[str]:
    total_words = len(tokens)
    word_counts = Counter(tokens)
    word_frequencies = {word: count / total_words for word, count in word_counts.items()}

    # compute subsampling probabilities
    subsampling_probs = {
        word: (sqrt(freq / subsample_threshold) + 1) * (subsample_threshold / freq)
        if freq > subsample_threshold else 1.0
        for word, freq in word_frequencies.items()
    }

    sub_sampled_tokens = []
    for word in tqdm(tokens, desc="Subsampling tokens"):
        # Generate a random value between 0 and 1
        random_value = np.random.rand()
        if random_value < subsampling_probs[word]:
            sub_sampled_tokens.append(word)

    return sub_sampled_tokens


def create_vocabulary(text_dataset: str | list[str], top_k: int = 10_000) -> tuple[dict[str, int], Counter]:

    if type(text_dataset) == str:
        text_dataset_split = text_dataset.split(" ")
    else:
        text_dataset_split = text_dataset

    dataset_counter = Counter(text_dataset_split)

    vocab = {"<unk>": 0}
    vocab_idx = 1

    for word in dataset_counter.most_common(top_k - 1):
        w = word[0].strip()

        if dataset_counter[w] < 5:
            continue

        vocab[w] = vocab_idx
        vocab_idx += 1

    return vocab, dataset_counter


def generate_training_text(
        tokens: list[str],
        vocabulary: dict[str, int],
        window_size: int = 2,
        stride: int = 1,
        batch_size: int | None = None,
        to_ids: bool = False,
        shuffle: bool = False
):

    def __shuffle(_batch_context, _batch_positives):
        _indices = np.random.permutation(_batch_positives.shape[0])

        return _batch_context[_indices], _batch_positives[_indices]

    len_tokens = len(tokens)

    batch_context, batch_positives = [], []
    for token_idx in range(window_size + 1, len_tokens - window_size - 1, stride):
        left_context = tokens[token_idx - window_size:token_idx]
        right_context = tokens[token_idx + 1:token_idx + window_size + 1]
        if to_ids:
            left_context_vector = [vocabulary.get(word, vocabulary["<unk>"]) for word in left_context]
            right_context_vector = [vocabulary.get(word, vocabulary["<unk>"]) for word in right_context]
            target_vector = vocabulary.get(tokens[token_idx], vocabulary["<unk>"])

            context_vector = left_context_vector + right_context_vector
        else:
            left_context_vector = left_context
            right_context_vector = right_context
            target_vector = tokens[token_idx]

            context_vector = left_context_vector + right_context_vector


        if target_vector == vocabulary["<unk>"]:
            continue


        if batch_size is None:
            yield context_vector, [target_vector, ]
        else:
            batch_context.append(context_vector)
            batch_positives.append(target_vector)

            if len(batch_positives) == batch_size:
                batch_context = np.array(batch_context)
                batch_positives = np.array(batch_positives)

                if shuffle:
                    batch_context, batch_positives = __shuffle(batch_context, batch_positives)

                yield batch_context, batch_positives
                batch_context, batch_positives = [], []

    if len(batch_positives) > 0 or len(batch_context) > 0:
        batch_context = np.array(batch_context)
        batch_positives = np.array(batch_positives)

        if shuffle:
            batch_context, batch_positives = __shuffle(batch_context, batch_positives)

        yield batch_context, batch_positives


def generate_training_text_w_negative_samples(
        tokens: list[str],
        vocabulary: dict[str, int],
        window_size: int = 2,
        stride: int = 1,
        batch_size: int | None = None,
        to_ids: bool = False,
        number_of_negatives: int | None = 5,
        token_probabilities: np.ndarray | None = None,
        shuffle: bool = False
):
    len_tokens = len(tokens)
    range_len = len(range(window_size + 1, len_tokens - window_size - 1, stride))

    negative_samples_generated = None
    if number_of_negatives:
        negative_samples_generated = np.random.choice(
            list(vocabulary.keys()),
            size=(range_len, number_of_negatives),
            p=token_probabilities
        )

    batch_context, batch_positives, batch_negatives = [], [], []
    for token_idx in range(window_size + 1, len_tokens - window_size - 1, stride):
        left_context = tokens[token_idx - window_size:token_idx]
        right_context = tokens[token_idx + 1:token_idx + window_size + 1]

        negative_samples = []
        if number_of_negatives:
            negative_samples = negative_samples_generated[token_idx - window_size - 1, :] if negative_samples_generated is not None else []

            while tokens[token_idx] in negative_samples:
                negative_samples = np.random.choice(
                    list(vocabulary.keys()),
                    size=(number_of_negatives, ),
                    p=token_probabilities
                )

        if to_ids:
            left_context_vector = [vocabulary.get(word, vocabulary["<unk>"]) for word in left_context]
            right_context_vector = [vocabulary.get(word, vocabulary["<unk>"]) for word in right_context]
            target_vector = vocabulary.get(tokens[token_idx], vocabulary["<unk>"])

            context_vector = left_context_vector + right_context_vector
            negative_samples_vector = [vocabulary.get(word, vocabulary["<unk>"]) for word in negative_samples]
        else:
            left_context_vector = left_context
            right_context_vector = right_context
            target_vector = tokens[token_idx]

            context_vector = left_context_vector + right_context_vector
            negative_samples_vector = negative_samples

        if target_vector == vocabulary["<unk>"]:
            continue

        if batch_size is None:
            yield context_vector, [target_vector, ], negative_samples_vector
        else:
            batch_context.append(context_vector)
            batch_positives.append(target_vector)
            batch_negatives.append(negative_samples_vector)

            if len(batch_positives) == batch_size:
                batch_context = np.array(batch_context)
                batch_positives = np.array(batch_positives)
                batch_negatives = np.array(batch_negatives)

                if shuffle:
                    indices = np.random.permutation(len(batch_positives))
                    batch_context = batch_context[indices]
                    batch_positives = batch_positives[indices]
                    batch_negatives = batch_negatives[indices]

                yield batch_context, batch_positives, batch_negatives
                batch_context, batch_positives, batch_negatives = [], [], []

    if len(batch_positives) > 0 or len(batch_context) > 0 or len(batch_negatives) > 0:
        yield np.array(batch_context), np.array(batch_positives), np.array(batch_negatives)


def shuffle_dataset(dataset):
    np.random.shuffle(dataset)

    for (context_vector, target_vector) in dataset:
        yield context_vector, target_vector


def shuffle_dataset_w_negative_samples(dataset):
    np.random.shuffle(dataset)

    for (context_vector, target_vector, negative_samples) in dataset:
        yield context_vector, target_vector, negative_samples


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

