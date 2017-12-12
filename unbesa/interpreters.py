#  -*- coding: utf-8 -*-
#    @package: core.py
#     @author: Andrew Yuri, Guilherme N. Ramos (gnramos@unb.br)
#
# Conjunto de classes abstratas. O objetivo delas é criar o padrão para futuras
# extensões.
#

import abc
import collections
import math
import nltk
import pandas


class SemanticInterpreter(abc.ABC):
    '''Classe base para os Interpretadores Semânticos do UNB_ESA.'''
    def __init__(self, corpus):
        self.corpus = corpus

    @abc.abstractmethod
    def fitCorpusToModel(self, corpus):
        raise NotImplementedError()


class TFIDFSemanticInterpreter(SemanticInterpreter):
    class TFIDF:
        '''Classe que recebe o dicionário contendo os textos do Corpus e os
        modela como um Bag-of-Words.

        Attributes:
            documents: Dicionário que modela o corpus.
        '''
        def __init__(self, corpus):
            '''Inicia a classe TFIDF

                parameters:
                    tf: vetor contendo as frequencias dos termos
                    idf: vetor contendo os inverted document frequencias
                    corpus: Corpus recebido.
            '''
            self.corpus = corpus
            self.tf = {}
            self.idf = {}
            self.tfidfDict = {}

        def _term_frequency(self, document, normalize=True):
            '''Retorna vetor com a frequência dos termos em um documento.

            "The weight of a term that occurs in a document is simply
            proportional to the term frequency."
            Hans Peter Luhn (1957)

                parameters:
                    document: Modelo que representa o documento
                    normalize: Normaliza Term Frequency, dividindo pelo número
                                total de termos do documento
            '''
            tokens = nltk.word_tokenize(document)
            counter = collections.Counter(tokens)

            if normalize:
                num_tokens = len(counter)
                for word in counter:
                    counter[word] /= num_tokens

            return counter

        def _inverse_document_frequency(self):
            '''Retorna vetor com o 'Inverted Document Frequency' do vetor
            identificado pela chave especificada.

            "The specificity of a term can be quantified as an inverse function
            of the number of documents in which it occurs."
            Karen Spärck Jones (1972)

            parameters:

                key: Chave que identifica o documento cujo IDF será gerado
            '''
            idf = {}
            numberOfDocuments = len(self.corpus.getCorpusFiles())

            for document in self.corpus.getCorpusFiles():
                for token in self.tf[document]:
                    if (token not in idf.keys()):
                        occurrences = 0
                        for documentAux in self.corpus.getCorpusFiles():
                            if (token in self.tf[documentAux]):
                                occurrences += 1

                        idf[token] = math.log(numberOfDocuments / occurrences)
            return idf

        def getTfidfAsDataframe(self):
            return pandas.DataFrame(self.tfidfDict)

        def getTfidf(self, dataFrame=True):
            tfidf = {}
            for doc in self.corpus.getCorpusFiles():
                self.tf[doc] = self._term_frequency(self.corpus.documents[doc])

            self.idf = self._inverse_document_frequency()
            for doc in self.corpus.getCorpusFiles():
                tfidf = {token: self.idf[token] * self.tf[doc][token]
                         for token in self.tf[doc]}

                for token in self.idf:
                    if (token not in self.tf[doc]):
                        tfidf[token] = 0

                self.tfidfDict[doc] = tfidf

        def sortDictByValue(self, dictonary):
            return sorted(dictonary.items(), key=lambda x: x[1])

    def __init__(self, corpus):
        super().__init__(corpus)

    def fitCorpusToModel(self):
        tfidf = TFIDFSemanticInterpreter.TFIDF(self.corpus)
        tfidf.getTfidf()
        self.model = tfidf.getTfidfAsDataframe()
        # self.modelType = 'tfidf'
