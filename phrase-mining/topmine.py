import math
from collections import Counter
import re
import string

import jieba
from zhon import hanzi


class PhraseMining:
    def __init__(
        self,
        sentences=None,
        min_support=5,
        max_phrase_size=7,
        alpha=3.0,
        delimiter="_",
    ):
        self.min_support = min_support
        self.max_phrase_size = max_phrase_size
        self.alpha = alpha
        self.delimiter = delimiter
        self.vocab = Counter()
        self.total_tokens = 0

        if sentences is not None:
            self.add_vocab(sentences)

    def add_vocab(self, sentences):
        vocab, total_tokens = self._frequent_phrase_mining(
            sentences,
            self.min_support,
            self.max_phrase_size,
            self.alpha,
            self.delimiter,
        )
        self.vocab += vocab
        self.total_tokens += total_tokens

    def _frequent_phrase_mining(
        self, sentences, min_support, max_phrase_size, alpha, delimiter
    ):
        vocab = Counter()
        active_indices = []

        n = 1

        while n <= max_phrase_size:
            new_sentences = []
            new_active_indices = []
            for sentence_i, sentence in enumerate(sentences):
                if n == 1:
                    new_indices = list(i for i in range(len(sentence)))
                else:
                    indices = active_indices[sentence_i]
                    new_indices = []
                    for idx in indices:
                        if idx + n - 1 < len(sentence):
                            key = delimiter.join(sentence[idx : idx + n - 1])
                            if vocab[key] >= min_support:
                                new_indices.append(idx)

                if len(new_indices) > 0:
                    new_sentences.append(sentence)
                    new_active_indices.append(new_indices)
                    for i, idx in enumerate(new_indices[:-1]):
                        if new_indices[i + 1] == idx + 1:
                            phrase = delimiter.join(sentence[idx : idx + n])
                            vocab[phrase] += 1

            sentences = new_sentences
            active_indices = new_active_indices
            n += 1

        vocab = Counter(x for x in vocab.elements() if vocab[x] >= min_support)
        total_tokens = sum(
            v for k, v in vocab.items() if len(k.split(delimiter)) == 1
        )
        return vocab, total_tokens

    def export_phrases(self, sentences):
        res = {}
        for sentence in sentences:
            if len(sentence) < 2:
                continue
            for phrase, score in self._analyze_sentence(sentence):
                if score is not None and score > self.alpha:
                    res[phrase] = score
        return res

    def _analyze_sentence(self, sentence):
        scores = {}
        phrases = sentence[:]
        while True:
            max_score = float("-inf")
            max_idx = -1
            max_phrase = None
            for idx, token in enumerate(phrases[:-1]):
                phrase = phrases[idx] + self.delimiter + phrases[idx + 1]
                if phrase not in scores:
                    score = self._score_candidate(
                        phrases[idx], phrases[idx + 1], phrase
                    )
                    scores[phrase] = score
                else:
                    score = scores[phrase]
                if score > max_score:
                    max_score = score
                    max_idx = idx
                    max_phrase = phrase
            if max_score < self.alpha:
                break
            phrases[max_idx] = max_phrase
            phrases.pop(max_idx + 1)
        res = [(phrase, scores.get(phrase)) for phrase in phrases]
        return res

    def _score_candidate(self, token1, token2, phrase):
        token1_cnt = self.vocab.get(token1, 0)
        if token1_cnt <= 0:
            return float("-inf")
        token2_cnt = self.vocab.get(token2, 0)
        if token2_cnt <= 0:
            return float("-inf")
        phrase_cnt = self.vocab.get(phrase, 0)
        if phrase_cnt <= 0:
            return float("-inf")
        numerator = token1_cnt * token2_cnt
        denominator = self.total_tokens * self.total_tokens
        independent_prob = numerator / denominator * 2
        expected_cnt = independent_prob * self.total_tokens
        score = (phrase_cnt - expected_cnt) / math.sqrt(
            max(phrase_cnt, expected_cnt)
        )
        return score


if __name__ == "__main__":
    input_fn = "/Users/linhx13/datasets/book/xiyouji_utf8.txt"
    # input_fn = "/Users/linhx13/Projects/open-source/topmine/input/dblp_5k.txt"

    def gen_sentences():
        with open(input_fn) as fin:
            for line in fin:
                text_list = re.split(
                    "[%s%s]" % (hanzi.punctuation, string.punctuation), line
                )
                for text in text_list:
                    text = text.strip()
                    if text:
                        # yield [
                        #     x.strip() for x in jieba.lcut(text) if x.strip()
                        # ]
                        yield list(text)

    pm = PhraseMining(gen_sentences())
    # for item in pm.vocab.most_common():
    #     print(item)

    phrases = Counter(pm.export_phrases(gen_sentences()))
    for item in phrases.most_common():
        print(item)
