# -*- coding: utf-8 -*-

from typing import Iterable, Any, Dict, Union
from itertools import chain
from collections import defaultdict

from .. import Field, Example, Dataset


class DataHandler(object):

    def __init__(self,
                 feature_fields: Dict[str, Union[Field, Dict[str, Field]]],
                 target_fields: Dict[str, Field],
                 sort_feature: str = None):
        self.features: Dict[str, Field] = self._create_fields(feature_fields)
        self.targets: Dict[str, Field] = self._create_fields(target_fields)
        self.example_fields = self._create_example_fields(feature_fields,
                                                          target_fields)
        self.sort_feature = sort_feature or list(self.features.keys())[0]
        self.vocabs = {}  # namespace -> vocab

    def build_dataset_from_path(self, path: str) -> Dataset:
        examples = list(self._read_from_path(path))
        return Dataset(examples, self.fields)

    def _read_from_path(self, path: str) -> Iterable[Example]:
        raise NotImplementedError

    def build_example(self, *inputs) -> Example:
        raise NotImplementedError

    def build_vocab(self, *datasets, **kwargs):
        datasets = [d for d in datasets if d is not None]
        specials = kwargs.pop('specials', {})
        ns_fs = defaultdict(list)
        for f in chain(self.features.values(), self.targets.values()):
            if not f.use_vocab:
                continue
            if not f.namespace:
                raise ValueError(
                    "Field namespace cannot be null when use_vocab")
            ns_fs[f.namespace].append(f)
        for ns, fs in ns_fs.items():
            f0_specials = [fs[0].pad_token, fs[0].unk_token,
                           fs[0].init_token, fs[0].eos_token]
            f0_specials = [x for f in fs[1:] for x in [f.pad_token, f.unk_token,
                                                       f.init_token, f.eos_token]
                           if x not in f0_specials]
            f0_specials.extend([x for x in specials.get(ns, [])
                                if x not in f0_specials])
            fs[0].build_vocab(*datasets, specials=f0_specials)
            for f in fs[1:]:
                f.vocab = fs[0].vocab
            self.vocabs[ns] = fs[0].vocab

    def sort_key(self, example: Example) -> Any:
        return len(getattr(example, self.sort_feature))

    @property
    def fields(self):
        return {**self.features, **self.targets}

    def _create_fields(self, fields):
        res = {}
        for key, val in fields.items():
            if isinstance(val, dict):
                for n, f in val.items():
                    res['%s.%s' % (key, n)] = f
            else:
                res[key] = val
        return res

    def _create_example_fields(self, feature_fields, target_fields):
        res = {}
        for key, val in chain(feature_fields.items(), target_fields.items()):
            if isinstance(val, dict):
                res[key] = [('%s.%s' % (key, n), f) for n, f in val.items()]
            else:
                res[key] = (key, val)
        return res
