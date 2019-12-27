# -*- coding: utf-8 -*-

from collections import OrderedDict

from .dot_product import DotProductSimilarity
from .cosine import CosineSimilarity
from .linear import LinearSimilarity
from .bilinear import BilinearSimilarity

# The first item added here will be used as the default in some cases.
similarity_functions = OrderedDict()
similarity_functions['dot_product'] = DotProductSimilarity
similarity_functions['cosine'] = CosineSimilarity
similarity_functions['linear'] = LinearSimilarity
similarity_functions['bilinear'] = BilinearSimilarity
