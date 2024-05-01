'''
    @Author: Paulina Alva
    @Date: 2024/04/20
    @Description:
        Este script se encarga de realizar las pruebas unitarias de la implementación de la fase 2.
'''

import unittest

# Importar funciones de procesamiento de texto
from main import clean_text, remove_stopwords, stem_text, lemmatize_text

class TestTextProcessingFunctions(unittest.TestCase):

    def test_clean_text(self):
        # Prueba para clean_text
        text = "This is test text with some special characters!" # Texto de prueba 
        cleaned_text = clean_text(text) # Llamar a la función clean_text
        expected_cleaned_text = "this is test text with some special characters" # Resultado esperado
        self.assertEqual(cleaned_text.strip(), expected_cleaned_text)


    def test_remove_stopwords(self):
        # Prueba para remove_stopwords
        text = "This is a test text with some stopwords" # Texto de prueba
        text_without_stopwords = remove_stopwords(text) # Llamar a la función remove_stopwords
        expected_text_without_stopwords = "This test text stopwords" # Resultado esperado
        self.assertEqual(text_without_stopwords, expected_text_without_stopwords)

    def test_stem_text(self):
        # Prueba para stem_text
        text = "running run ran" # Texto de prueba
        stemmed_text = stem_text(text) # Llamar a la función stem_text
        expected_stemmed_text = "run run ran" # Resultado esperado
        self.assertEqual(stemmed_text, expected_stemmed_text)


    def test_lemmatize_text(self):
        # Prueba para lemmatize_text
        text = "running run ran" # Texto de prueba
        lemmatized_text = lemmatize_text(text) # Llamar a la función lemmatize_text
        expected_lemmatized_text = "running run ran"   # Resultado esperado
        self.assertEqual(lemmatized_text, expected_lemmatized_text) 

if __name__ == '__main__':
    unittest.main()