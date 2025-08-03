"""
Test model functionality
"""
import unittest
import numpy as np
from src.models.neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    
    def setUp(self):
        self.model = NeuralNetwork()
        
    def test_model_prediction(self):
        """Test model prediction"""
        # Sample input data (120 timesteps, 38 features)
        sample_input = np.random.randn(1, 120, 38)
        
        prediction = self.model.predict(sample_input)
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape[0], 1)
        
    def test_model_architecture(self):
        """Test model architecture"""
        model = self.model.build_model()
        self.assertIsNotNone(model)
        
        # Check input shape
        expected_input_shape = (None, 120, 38)
        self.assertEqual(model.input_shape, expected_input_shape)

if __name__ == '__main__':
    unittest.main()
