import unittest
import numpy as np
from kalmanTracker import initialize_kalman, predict_future_position
import cv2
class TestKalmanTracker(unittest.TestCase):

    def test_initialize_kalman(self):
        kf = initialize_kalman()
        self.assertIsInstance(kf, cv2.KalmanFilter)
        np.testing.assert_array_equal(kf.statePost, np.zeros((4, 1), dtype=np.float32))

    def test_predict_future_position(self):
        kf = initialize_kalman()
        future_position = predict_future_position(kf, 5)
        
        # Verifica se o retorno tem o tamanho correto (x, y)
        self.assertEqual(future_position.shape, (2, 1))
        
        # Considerando que o estado inicial é zero, a previsão futura deve manter zero
        np.testing.assert_array_equal(future_position, np.zeros((2, 1), dtype=np.float32))

    def test_prediction_with_known_state(self):
        kf = initialize_kalman()
        kf.statePost = np.array([[0.], [0.], [1.], [1.]], dtype=np.float32)  # Configura uma velocidade de 1 unidade por passo
        future_position = predict_future_position(kf, 5)
        
        # Com uma velocidade de 1 unidade por passo, a posição prevista após 5 passos deve ser (5, 5)
        expected_position = np.array([[5.], [5.]], dtype=np.float32)
        np.testing.assert_array_almost_equal(future_position, expected_position)

if __name__ == '__main__':
    unittest.main()
