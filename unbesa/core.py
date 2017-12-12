#  -*- coding: utf-8 -*-
#    @package: core.py
#     @author: Andrew Yuri, Guilherme N. Ramos (gnramos@unb.br)
#
# Conjunto de classes abstratas. O objetivo delas é criar o padrão para futuras
# extensões.
#

import nltk
import string
import os
import unicodedata


def list_files(path):
    return [root + '/' + f
            for root, dirs, files in os.walk(path)
            for f in files]


class Corpus:
    '''
    Classe que modela um corpus

    Classe que modela um Corpus de documentos textuais localizados em um
    determinado diretório. A modelagem é feita através de um estrutura de
    Dicionário, onde o índice é o nome dos arquivos e os elementos são os seus
    respectivos textos.

    Attributes:
        documents: Dicionário que armazena o nome dos arquivos e o seus
                    respectivos conteúdos.
        source: Diretório onde os arquivos de texto estão localizados. Deve ser
                um subdiretório do Pai do diretório Classes. Ou uma lista de
                arquivos
        listConcepts: Lista contendo o nome dos arquivos que serão incluídos no
                      Corpus
    '''
    def __init__(self, directory, method='all', listCorpus={},
                 encoding='ISO-8859-1'):
        '''
        Inicia a classe Corpus com o seu caminho e cria o dicionário. Por
        padrão é feito o stem em português e removidas as stopwords em
        português.
        '''
        self.documents = {}
        if method == 'all':
            self.listFiles = os.listdir(directory)
        elif method == 'list':
            self.listFiles = listCorpus
        for file in self.listFiles:
            fileTemp = open(directory + '/' + file, encoding=encoding)
            self.documents[file] = fileTemp.read()

    def add_document(self, file_name, directory, encoding='ISO-8859-1'):
        with open(directory + '/' + file_name, encoding=encoding) as f:
            self.documents[file_name] = f.read()
        self.listFiles.append(file_name)

    def remove_document(self, file_name):
        self.documents.pop(file_name)

    def getCorpusFiles(self):
        '''
        Retorna uma lista de strings, onde cada elemento é o nome de um arquivo
        do Corpus
        '''
        return(self.documents.keys())

    def preprocess(self, language='portuguese', numbers=True, accents=True,
                   punctuation=True, stop_words=True, to_lower=True,
                   stem=True):
        '''
        Método que realiza a limpeza dos documentos presentes no corpus.

        parameters:
            language: Língua na qual o processo de stemming e remoção
            de stopwords será realizado. Por padrão será português.
            punctuation: Variável booleana que define se as pontuações
            serão retiradas. As pontuação são aquelas presentes na bilbioteca
            'string'. São elas: !'#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
            stop_words: Variável booleana que define se as stopwords serão
            retiradas. A linguagem é a mesma definida na variável 'language'.
        '''
        if stop_words:
            language_stopwords = nltk.corpus.stopwords.words(language)

        if stem:
            stemmer = nltk.stem.snowball.SnowballStemmer(language)

        for doc in self.documents:
            if to_lower:
                self.documents[doc] = self.documents[doc].lower()

            if accents:
                normalized = unicodedata.normalize('NFKD',
                                                   self.documents[doc])
                encoded = normalized.encode('ASCII', 'ignore')
                self.documents[doc] = encoded.decode('ASCII')

            if punctuation:
                translator = str.maketrans('','', string.punctuation)
                self.documents[doc] = self.documents[doc].translate(translator)

            if numbers:
                translator = str.maketrans('','', string.digits)
                self.documents[doc] = self.documents[doc].translate(translator)

            if stop_words:
                tokenized = nltk.word_tokenize(self.documents[doc])
                self.documents[doc] = ' '.join(w for w in tokenized
                                               if w not in language_stopwords)

            if stem:
                tokens = nltk.word_tokenize(self.documents[doc])
                self.documents[doc] = ' '.join(stemmer.stem(token)
                                               for token in tokens)


def run(concept_directory, text_directory, classifier, interpreter_cls,
        verbose=False):
    print('Iniciando a Execução')

    texts = os.listdir(text_directory)
    concepts = os.listdir(concept_directory)

    resultado = {}
    for text in texts:
        print('Processando Arquivo ' + text)
        list_corpus_temp = concepts
        corpus = Corpus(method='list', directory=concept_directory,
                        listCorpus=list_corpus_temp)
        corpus.add_document(text, text_directory)
        corpus.preprocess()

        interpreter = interpreter_cls(corpus)
        interpreter.fitCorpusToModel()

        classificacao = classifier.classify(interpreter, concepts, text)
        resultado[text] = classificacao

        concepts.remove(text)
        corpus.remove_document(text)
    return resultado
