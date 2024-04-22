'''
    @Author: Paulina Alva
    @Date: 2024/04/20
    @Description:
        Este script se encarga de realizar las pruebas unitarias de la implementación de la fase 2.
'''

import unittest
import os
import sys

# Ruta del directorio principal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar tu módulo principal
from main import check_plagiarism_file_to_files, check_plagiarism_files_to_files, check_plagiarism_directory

class TestPlagiarismChecker(unittest.TestCase):

    def setUp(self):
        # Archivos de prueba
        self.file_path = 'test_file.txt'
        self.files_path = ['test_file1.txt', 'test_file2.txt']
        self.dir_path = 'test_directory'
        self.create_test_files()

    def create_test_files(self):
        # Contenido de prueba para los archivos
        content = "This is some sample text for testing plagiarism detection."
        # Crear archivos de prueba
        with open(self.file_path, 'w') as file:
            file.write(content)
        for file_path in self.files_path:
            with open(file_path, 'w') as file:
                file.write(content)
        # Crear directorio de prueba si no existe
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
            for file_path in self.files_path:
                with open(os.path.join(self.dir_path, os.path.basename(file_path)), 'w') as file:
                    file.write(content)

    def test_plagiarism_from_one_file_to_multiple_files(self):
        similarity_scores = check_plagiarism_file_to_files(self.file_path, self.files_path)
        self.assertEqual(len(similarity_scores), len(self.files_path))  # Debería haber un score de similitud para cada archivo
        for similarity_score in similarity_scores:
            self.assertAlmostEqual(similarity_score, 1.0, places=2)  # Verificar similitud esperada

    def test_plagiarism_from_multiple_files_to_multiple_files(self):
        similarity_scores = check_plagiarism_files_to_files(self.files_path, self.files_path)
        self.assertEqual(len(similarity_scores), len(self.files_path) * len(self.files_path))  # Debería haber un score de similitud para cada combinación de archivos
        for similarity_score in similarity_scores:
            self.assertAlmostEqual(similarity_score, 1.0, places=2)  # Verificar similitud esperada

    def tearDown(self):
        # Eliminar archivos de prueba
        os.remove(self.file_path)
        for file_path in self.files_path:
            os.remove(file_path)
        # Eliminar archivos dentro del directorio de prueba
        for file_name in os.listdir(self.dir_path):
            file_path = os.path.join(self.dir_path, file_name)
            os.remove(file_path)
        # Eliminar directorio de prueba
        os.rmdir(self.dir_path)

if __name__ == '__main__':
    unittest.main()
