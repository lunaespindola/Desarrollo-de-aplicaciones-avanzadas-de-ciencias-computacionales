'''
    @Author: Paulina Alva
    @Date: 2024/04/20
    @Description:
        Este script se encarga de realizar las pruebas unitarias de la implementación de la fase 2.
'''


import unittest

# importamos la función check_accuracy
from main import check_accuracy

class TestCheckAccuracy(unittest.TestCase):

    def test_check_accuracy(self):
        true_labels = [1, 0, 1, 0, 1] # Etiquetas verdaderas
        predicted_labels = [1, 0, 1, 1, 0] # Etiquetas predichas
        accuracy = check_accuracy(true_labels, predicted_labels) # Llamar a la función check_accuracy
        expected_accuracy = 0.6 # Resultado esperado
        self.assertEqual(accuracy, expected_accuracy)

if __name__ == '__main__':
    unittest.main()
