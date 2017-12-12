#  -*- coding: utf-8 -*-
#    @package: mwe.py3
#     @author: Andrew Yuri, Guilherme N. Ramos (gnramos@unb.br)
# @disciplina: Algoritmos e Programação de Computadores
#
#

import unbesa.core
import unbesa.classifiers
import unbesa.interpreters

if __name__ == '__main__':
    # Diretório com os conceitos. Os conceitos são identificados pelo nome do
    # apenas arquivo e há um arquivo por conceito.
    concept_directory = 'conceitos'

    # Diretório com os textos a serem classificados. Os textos são
    # identificados pelo nome do apenas arquivo.
    text_directory = 'textos'

    # Define como atribuir os conceitos aos textos.
    classifier = unbesa.classifiers.CosineSingleLabelClassifier()

    # Define como interpretar o conteúdo.
    interpreter = unbesa.interpreters.TFIDFSemanticInterpreter

    # Executa o algoritmo
    result = unbesa.core.run(concept_directory, text_directory,
                             classifier, interpreter)
    # import unbesa_bkp
    # result = unbesa_bkp.run(concept_directory, text_directory, classifier)

    for text_file_name in sorted(result):
        print('{} -> {}'.format(text_file_name, result[text_file_name]))
