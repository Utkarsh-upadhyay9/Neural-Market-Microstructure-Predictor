"""
Test data processing functionality
"""
import unittest
import numpy as np
import pandas as pd
from src.data.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = DataProcessor()
        
    def test_data_validation(self):
        """Test data validation"""
        # Sample data
        sample_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        result = self.processor.validate_data(sample_data)
        self.assertTrue(result)
        
    def test_feature_engineering(self):
        """Test feature engineering"""
        sample_data = pd.DataFrame({
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 5000, 100)
        })
        
        features = self.processor.engineer_features(sample_data)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 2)

if __name__ == '__main__':
    unittest.main()
