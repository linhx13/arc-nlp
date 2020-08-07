from typing import Optional, List, Dict, Union
import json
import collections

from .constants import UNK_TOKEN, PAD_TOKEN
from .data.utils import DefaultLookupDict, count_tokens


class Vocab:
    """ Reference: gluonnlp vocab """

    def __init__(self, counter, max_size=None, min_freq=1,
                 unknown_token: Optional[str] = UNK_TOKEN,
                 reserved_tokens: Optional[List[str]] = [PAD_TOKEN, UNK_TOKEN],
                 token_to_idx: Optional[Dict[str, int]] = None):
        assert min_freq > 0, '`min_freq` must be set to a positie value.'

        special_tokens = []
        if reserved_tokens is not None:
            special_tokens.extend(reserved_tokens)
            special_tokens_set = set(special_tokens)
            assert len(special_tokens_set) == len(special_tokens), \
                "`reserved_tokens` cannot contain duplicate reserved tokens or " \
                "other special tokens."

        self._unknown_index = None
        if unknown_token is not None:
            if unknown_token in special_tokens:
                self._unknown_index = special_tokens.index(unknown_token)
            else:
                special_tokens.append(unknown_token)
                self._unknown_index = len(special_tokens) - 1
            self._token_to_idx = DefaultLookupDict(self._unknown_index)
        else:
            self._token_to_idx = {}

        self._idx_to_token = []

        if not special_tokens:
            self._reserved_tokens = None
        else:
            self._reserved_tokens = special_tokens
            self._idx_to_token.extend(special_tokens)

        self._token_to_idx.update((token, idx) for idx, token in enumerate(self._idx_to_token))

        if counter:
            self._index_counter_keys(counter, unknown_token, special_tokens, max_size, min_freq)

        if token_to_idx:
            self._sort_index_according_to_user_specification(token_to_idx)
            if unknown_token:
                self._token_to_idx._default = self._token_to_idx[unknown_token]

    def _index_counter_keys(self, counter, unk_token, special_tokens,
                            max_size, min_freq):
        unk_and_special_tokens = set(special_tokens) if special_tokens else set()
        if unk_token:
            unk_and_special_tokens.add(unk_token)

        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        token_cap = len(unk_and_special_tokens) + (
            len(counter) if not max_size else max_size)

        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) >= token_cap:
                break
            if token not in unk_and_special_tokens:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    def _sort_index_according_to_user_specification(self, token_to_idx):
        # Sanity checks
        if not set(token_to_idx.keys()).issubset(self.token_to_idx.keys()):
            raise ValueError('User-specified token_to_idx mapping can only contain '
                             'tokens that will be part of the vocabulary.')
        if len(set(token_to_idx.values())) != len(token_to_idx):
            raise ValueError('User-specified indices must not contain duplicates.')
        if min(token_to_idx.values()) < 0 or max(token_to_idx.values()) >= len(self.token_to_idx):
            raise ValueError('User-specified indices must not be < 0 or >= the number of tokens '
                             'that will be in the vocabulary. The current vocab contains {}'
                             'tokens.'.format(len(self.token_to_idx)))

        # Update index ordering
        for token, new_idx in token_to_idx.items():
            old_idx = self.token_to_idx[token]
            ousted_token = self.idx_to_token[new_idx]

            self.token_to_idx[token] = new_idx
            self.token_to_idx[ousted_token] = old_idx
            self.idx_to_token[old_idx] = ousted_token
            self.idx_to_token[new_idx] = token

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def reserved_tokens(self):
        return self._reserved_tokens

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def unknown_token(self) -> str:
        return self._unknown_token

    def __contains__(self, token:str) -> bool:
        return token in self._token_to_idx

    def __getitem__(self, tokens) -> Union[str, List[str]]:
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx[tokens]
        else:
            return [self._token_to_idx[token] for token in tokens]

    def __len__(self):
        return len(self._idx_to_token)

    def to_tokens(self, indices):
        """Convert token indices to tokens according to the vocabulary."""

        to_reduce = False
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
            to_reduce = True

        max_idx = len(self._idx_to_token) - 1

        tokens = []
        for idx in indices:
            if not isinstance(idx, int) or idx > max_idx:
                raise ValueError('Token index {} in the provided `indices` is invalid.'.format(idx))

            tokens.append(self._idx_to_token[idx])

        return tokens[0] if to_reduce else tokens

    def to_indices(self, tokens):
        return self[tokens]

    def __call__(self, tokens):
        return self[tokens]

    def __repr__(self):
        unk = '"{}"'.format(self._unknown_token) if self._unknown_token else 'None'
        reserved = '"{}"'.format(self._reserved_tokens) if self._reserved_tokens else 'None'
        return 'Vocab(size={}, unk={}, reserved={})'.format(len(self), unk, reserved)

    def to_json(self):
        vocab_dict = {}
        vocab_dict['idx_to_token'] = self._idx_to_token
        vocab_dict['token_to_idx'] = dict(self._token_to_idx)
        vocab_dict['reserved_tokens'] = self._reserved_tokens
        vocab_dict['unknown_token'] = self._unknown_token
        return json.dumps(vocab_dict, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str):
        vocab_dict = json.loads(json_str)
        token_to_idx = vocab_dict.get('token_to_idx')
        unknown_token = vocab_dict.get("unknown_token")
        reserved_tokens = vocab_dict.get('reserved_tokens')

        special_tokens = {unknown_token}
        if reserved_tokens is not None:
            reserved_tokens = [
                t for t in reserved_tokens if t not in special_tokens
            ]

        vocab = cls(
            counter=count_tokens(token_to_idx.keys()),
            unknown_token=unknown_token,
            reserved_tokens=reserved_tokens,
            token_to_idx=token_to_idx
        )

        return vocab
