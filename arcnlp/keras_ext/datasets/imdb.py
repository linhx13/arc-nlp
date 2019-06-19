# -*- coding: utf-8 -*-

from .. import data


class IMDB(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
