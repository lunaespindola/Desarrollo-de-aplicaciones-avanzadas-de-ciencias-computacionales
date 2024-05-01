'''
    @Author: Paulina Alva
    @Date: 2024/04/20
    @Description:
        Este script se encarga de realizar las pruebas unitarias de la implementación de la fase 2.
'''

import unittest
import os

# Ruta del directorio principal
from main import check_plagiarism_directory

class TestPlagiarismChecker(unittest.TestCase):

    def setUp(self):
        # Archivos de prueba
        self.dir_path = 'test_directory'
        self.files = ['file1.txt', 'file2.txt', 'file3.txt']
        self.create_test_files()

    def create_test_files(self):
        # Directorio de prueba si no existe
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
            # Crear archivos de prueba
            for file_name in self.files:
                with open(os.path.join(self.dir_path, file_name), 'w') as file:
                    file.write("Sample text for testing plagiarism detection.")

    def test_check_plagiarism_directory(self):
        similarity_scores, true_labels, files_1, files_2 = check_plagiarism_directory(self.dir_path)
        
        # Verificar que se hayan generado puntuaciones de similitud
        self.assertIsNotNone(similarity_scores)
        # Verificar que las listas tengan la misma longitud
        self.assertEqual(len(similarity_scores), len(true_labels))
        self.assertEqual(len(similarity_scores), len(files_1))
        self.assertEqual(len(similarity_scores), len(files_2))

        # Verificar que las puntuaciones de similitud estén dentro del rango [0, 1]
        for score in similarity_scores:
            self.assertTrue(0 <= score <= 1)

        # Verificar que los nombres de archivo en files_1 y files_2 correspondan a archivos en el directorio
        for file_name in files_1 + files_2:
            self.assertIn(file_name, self.files)

    def tearDown(self):
        # Eliminar archivos de prueba 
        for file_name in self.files:
            os.remove(os.path.join(self.dir_path, file_name))
        # Eliminar directorio de prueba
        os.rmdir(self.dir_path)

if __name__ == '__main__':
    unittest.main()
