"""
Test training functionality
"""
import unittest
import numpy as np
from src.training.trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    
    def setUp(self):
        self.trainer = ModelTrainer()
        
    def test_data_preparation(self):
        """Test training data preparation"""
        # Sample data
        sample_data = np.random.randn(1000, 38)
        
        X, y = self.trainer.prepare_data(sample_data)
        self.assertEqual(X.shape[1:], (120, 38))  # 120 timesteps, 38 features
        self.assertEqual(len(X), len(y))
        
    def test_model_compilation(self):
        """Test model compilation"""
        model = self.trainer.compile_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.optimizer)

if __name__ == '__main__':
    unittest.main()
