"""
    @author: @Lunaespindola
    @date: 2024/04/17
    @description:
        Este script se encarga de realizar las pruebas unitarias de la implementación de la fase 2.
"""

import unittest
import os
import sys

# Añadir la ruta del directorio principal al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ahora puedes importar tu módulo principal
from main import check_plagiarism, check_plagiarism_directory

class TestPlagiarismChecker(unittest.TestCase):

    def setUp(self):
        # Crear archivos de prueba
        self.file1_path = 'test_file1.txt'
        self.file2_path = 'test_file2.txt'
        self.dir_path = 'test_directory'
        self.create_test_files()

    def create_test_files(self):
        # Contenido de prueba para los archivos
        content1 = "This is some sample text for testing plagiarism detection."
        content2 = "This is a similar text used to test the plagiarism checker."
        # Crear archivos de prueba
        with open(self.file1_path, 'w') as file:
            file.write(content1)
        with open(self.file2_path, 'w') as file:
            file.write(content2)
        # Crear directorio de prueba si no existe
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
            with open(os.path.join(self.dir_path, 'file1.txt'), 'w') as file:
                file.write(content1)
            with open(os.path.join(self.dir_path, 'file2.txt'), 'w') as file:
                file.write(content2)

    def test_similarity_between_two_files(self):
        similarity_score = check_plagiarism(self.file1_path, self.file2_path)
        self.assertAlmostEqual(similarity_score,  0.5477225575051662, places=2)  # Verificar similitud esperada

    def test_similarity_in_directory(self):
        similarity_scores, _, _, _ = check_plagiarism_directory(self.dir_path)
        self.assertEqual(len(similarity_scores), 1)  # Debería haber solo una comparación en este caso
        self.assertAlmostEqual(similarity_scores[0],  0.5477225575051662, places=2)  # Verificar similitud esperada

    def tearDown(self):
        # Eliminar archivos de prueba
        os.remove(self.file1_path)
        os.remove(self.file2_path)
        # Eliminar archivos dentro del directorio de prueba
        for file in os.listdir(self.dir_path):
            file_path = os.path.join(self.dir_path, file)
            os.remove(file_path)
        # Eliminar directorio de prueba
        os.rmdir(self.dir_path)

if __name__ == '__main__':
    unittest.main()