#  -*- coding: utf-8 -*-
#    @package: __init__.py
#     @author: Guilherme N. Ramos (gnramos@unb.br)
#

import nltk

for path_to_package in ['corpora/stopwords', 'tokenizers/punkt']:
    try:
        if not nltk.data.find(path_to_package):
            raise LookupError()
    except LookupError as e:
        package = path_to_package.split('/')[-1]
        nltk.download(package)
