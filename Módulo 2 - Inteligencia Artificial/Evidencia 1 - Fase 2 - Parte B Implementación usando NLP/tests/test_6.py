'''
    @Author: Paulina Alva
    @Date: 2024/04/20
    @Description:
        Este script se encarga de realizar las pruebas unitarias de la implementación de la fase 2.
'''


import unittest

# Importar la función get_y_pred_y_true
from main import get_y_pred_y_true

class TestGetYpredYTrue(unittest.TestCase):

    def test_get_y_pred_y_true(self):
        # Datos de ejemplo
        similarity_scores = [0.8, 0.6, 0.3, 0.9, 0.2]  # Similarity scores obtenidos
        threshold = 0.5  # Umbral para clasificar como plagiarizado o no
        true_labels = [1, 1, 0, 1, 0]  # Etiquetas verdaderas (1 para plagiarizado, 0 para no plagiarizado)

        # Calcular las predicciones y las etiquetas verdaderas esperadas manualmente
        expected_y_pred = [1, 1, 0, 1, 0]  # Los scores por encima del umbral se clasifican como 1, de lo contrario, 0
        expected_y_true = [1, 1, 0, 1, 0]  # Las etiquetas verdaderas no cambian

        # Obtener las predicciones y las etiquetas verdaderas usando la función get_y_pred_y_true
        y_pred, y_true = get_y_pred_y_true(similarity_scores, threshold)

        # Verificar si las predicciones y las etiquetas verdaderas obtenidas coinciden con las esperadas
        self.assertEqual(y_pred, expected_y_pred)
        self.assertEqual(y_true, expected_y_true)

if __name__ == '__main__':
    unittest.main()