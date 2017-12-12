#  -*- coding: utf-8 -*-
#    @package: classifiers.py
#     @author: Guilherme N. Ramos (gnramos@unb.br)


import abc
import math
import operator
# import unbesa.core


class Classifier(abc.ABC):
    '''Classe base para os classificadores do UNB_ESA.'''
    @abc.abstractmethod
    def similarity(self, semanticInterpreter, doc1, doc2):
        '''
        Método abstrato que recebe dois documentos e um interpretador semântico
        e mede a similaridade entre os documentos.

        parameter:
            semanticInterpreter: Instância de um Interpretador Semântico,
                                 contendo o Corpus textual e o modelo
                                 representativo do mesmo.
            doc1: Nome de um dos documentos que serão comparados. Deve ser uma
                  chave para o dicionário presente no Intepretador Semântico.
            doc1: Nome de um dos documentos que serão comparados. Deve ser uma
                  chave para o dicionário presente no Intepretador Semântico.

        return: Número real que mede a similaridade entre o doc1 e doc2
        '''
        raise NotImplementedError()

    def similarities(self, semanticInterpreter, concept_files, text_file):
        return {concept_file: self.similarity(text_file, concept_file,
                                              semanticInterpreter)
                for concept_file in concept_files}

    def classify(self, interpreter, concept_files, text_file):
        '''
        Método abstrato que recebe dois documentos e um interpretador semântico
        e mede a similaridade entre os documentos.

        parameter:
            interpreter: Instância de um Interpretador Semântico,
                                 contendo o Corpus textual e o modelo
                                 representativo do mesmo.
            concept_files: Lista contendo os nomes dos arquivos

        return: Matriz de similaridades
        '''
        concept_files.pop(text_file)
        return self.similarities(interpreter, concept_files, text_file)


class CosineSingleLabelClassifier(Classifier):
    '''Implementação de um classificador usando o cosseno como a métrica de
    similaridade entre vetores.
    '''
    def similarity(self, doc1, doc2, interpreter):
        dot_product = interpreter.model[doc1][:].dot(interpreter.model[doc2][:])
        mag_doc1 = math.sqrt(sum([val**2 for val in interpreter.model[doc1][:]]))
        mag_doc2 = math.sqrt(sum([val**2 for val in interpreter.model[doc2][:]]))
        magnitude = mag_doc1 * mag_doc2
        return dot_product / magnitude if magnitude else 0

    def classify(self, interpreter, concept_files, text_file):
        similarities = self.similarities(interpreter, concept_files,
                                         text_file)
        del similarities[text_file]

        return max(similarities, key=lambda key: similarities[key])
