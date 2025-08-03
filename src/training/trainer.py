"""
Model training module with comprehensive training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from loguru import logger
import yaml
import os
import json
from datetime import datetime

from ..models.lstm_model import LSTMModel
from ..models.cnn_model import CNNModel
from ..models.attention_model import AttentionModel
from ..data.preprocessor import DataPreprocessor


class ModelTrainer:
    """Comprehensive model trainer for market prediction models."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the model trainer."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.training_results = {}
        self.preprocessor = DataPreprocessor(self.config)
        
    def prepare_data(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Prepare data for training.
        
        Args:
            data: Raw market data
            test_size: Proportion for testing
        
        Returns:
            Dictionary with prepared data
        """
        try:
            logger.info(" Preparing data for training...")
            
            # Use preprocessor pipeline
            sequence_length = self.config['models']['lstm']['sequence_length']
            target_column = self.config['data']['target']
            
            processed_data = self.preprocessor.preprocess_pipeline(
                data=data,
                sequence_length=sequence_length,
                target_column=target_column,
                test_size=test_size
            )
            
            if not processed_data:
                logger.error("Data preparation failed")
                return {}
            
            # Split validation data from training data
            X_train = processed_data['X_train']
            y_train = processed_data['y_train']
            
            # Create validation split
            val_split = self.config['training']['validation_split']
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=val_split, random_state=42, shuffle=False
            )
            
            result = {
                'X_train': X_train_split,
                'X_val': X_val,
                'X_test': processed_data['X_test'],
                'y_train': y_train_split,
                'y_val': y_val,
                'y_test': processed_data['y_test'],
                'scaler': processed_data['scaler'],
                'feature_columns': processed_data['feature_columns'],
                'original_data': processed_data['original_data']
            }
            
            logger.info(f" Data prepared successfully:")
            logger.info(f"   Training: {X_train_split.shape}")
            logger.info(f"   Validation: {X_val.shape}")
            logger.info(f"   Testing: {processed_data['X_test'].shape}")
            logger.info(f"  🔢 Features: {len(processed_data['feature_columns'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return {}
    
    def train_lstm_model(self, data: Dict) -> Dict:
        """Train LSTM model."""
        try:
            logger.info(" Training LSTM model...")
            
            # Initialize model
            lstm_config = self.config['models']['lstm']
            lstm_model = LSTMModel(lstm_config)
            
            # Build model
            input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
            model = lstm_model.build_model(input_shape)
            
            if model is None:
                logger.error("Failed to build LSTM model")
                return {}
            
            # Train model
            training_config = self.config['training']
            history = lstm_model.train(
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],
                y_val=data['y_val'],
                epochs=training_config['epochs'],
                batch_size=training_config['batch_size']
            )
            
            # Evaluate model
            metrics = lstm_model.evaluate(data['X_test'], data['y_test'])
            
            # Save model
            os.makedirs('models', exist_ok=True)
            lstm_model.save_model('models/lstm_model.h5')
            
            self.models['lstm'] = lstm_model
            
            result = {
                'model': lstm_model,
                'history': history,
                'metrics': metrics,
                'model_type': 'LSTM'
            }
            
            logger.info(f" LSTM training completed:")
            logger.info(f"   RMSE: {metrics.get('rmse', 0):.4f}")
            logger.info(f"   MAE: {metrics.get('mae', 0):.4f}")
            logger.info(f"   Directional Accuracy: {metrics.get('directional_accuracy', 0):.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {}
    
    def train_cnn_model(self, data: Dict) -> Dict:
        """Train CNN model."""
        try:
            logger.info(" Training CNN model...")
            
            # Initialize model
            cnn_config = self.config['models']['cnn']
            cnn_model = CNNModel(cnn_config)
            
            # Build model
            input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
            model = cnn_model.build_model(input_shape)
            
            if model is None:
                logger.error("Failed to build CNN model")
                return {}
            
            # Train model
            training_config = self.config['training']
            history = cnn_model.train(
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],
                y_val=data['y_val'],
                epochs=training_config['epochs'],
                batch_size=training_config['batch_size']
            )
            
            # Evaluate model
            metrics = cnn_model.evaluate(data['X_test'], data['y_test'])
            
            # Save model
            cnn_model.save_model('models/cnn_model.h5')
            
            self.models['cnn'] = cnn_model
            
            result = {
                'model': cnn_model,
                'history': history,
                'metrics': metrics,
                'model_type': 'CNN'
            }
            
            logger.info(f" CNN training completed:")
            logger.info(f"   RMSE: {metrics.get('rmse', 0):.4f}")
            logger.info(f"   MAE: {metrics.get('mae', 0):.4f}")
            logger.info(f"   Directional Accuracy: {metrics.get('directional_accuracy', 0):.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training CNN model: {e}")
            return {}
    
    def train_attention_model(self, data: Dict) -> Dict:
        """Train Attention model."""
        try:
            logger.info(" Training Attention model...")
            
            # Initialize model
            attention_config = self.config['models']['attention']
            attention_model = AttentionModel(attention_config)
            
            # Build model
            input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
            model = attention_model.build_model(input_shape)
            
            if model is None:
                logger.error("Failed to build Attention model")
                return {}
            
            # Train model
            training_config = self.config['training']
            history = attention_model.train(
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data['X_val'],
                y_val=data['y_val'],
                epochs=training_config['epochs'],
                batch_size=training_config['batch_size']
            )
            
            # Evaluate model
            metrics = attention_model.evaluate(data['X_test'], data['y_test'])
            
            # Save model
            attention_model.save_model('models/attention_model.h5')
            
            self.models['attention'] = attention_model
            
            result = {
                'model': attention_model,
                'history': history,
                'metrics': metrics,
                'model_type': 'Attention'
            }
            
            logger.info(f" Attention training completed:")
            logger.info(f"   RMSE: {metrics.get('rmse', 0):.4f}")
            logger.info(f"   MAE: {metrics.get('mae', 0):.4f}")
            logger.info(f"   Directional Accuracy: {metrics.get('directional_accuracy', 0):.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training Attention model: {e}")
            return {}
    
    def train_all_models(self, data: pd.DataFrame, models: List[str] = None) -> Dict:
        """
        Train all specified models.
        
        Args:
            data: Market data for training
            models: List of models to train ['lstm', 'cnn', 'attention']
        
        Returns:
            Dictionary with all training results
        """
        if models is None:
            models = ['lstm', 'cnn', 'attention']
        
        logger.info(f" Starting training pipeline for models: {models}")
        logger.info(f" User: Utkarsh-upadhyay9")
        logger.info(f" Time: {datetime.now()}")
        
        # Prepare data
        prepared_data = self.prepare_data(data)
        if not prepared_data:
            logger.error("Data preparation failed")
            return {}
        
        results = {
            'data_info': {
                'train_samples': len(prepared_data['X_train']),
                'val_samples': len(prepared_data['X_val']),
                'test_samples': len(prepared_data['X_test']),
                'features': len(prepared_data['feature_columns']),
                'sequence_length': prepared_data['X_train'].shape[1]
            },
            'models': {}
        }
        
        # Train each model
        for model_name in models:
            logger.info(f"\n Training {model_name.upper()} model...")
            
            if model_name == 'lstm':
                result = self.train_lstm_model(prepared_data)
            elif model_name == 'cnn':
                result = self.train_cnn_model(prepared_data)
            elif model_name == 'attention':
                result = self.train_attention_model(prepared_data)
            else:
                logger.warning(f"Unknown model type: {model_name}")
                continue
            
            if result:
                results['models'][model_name] = result
                self.training_results[model_name] = result
        
        # Save training results
        self.save_training_results(results)
        
        # Compare models
        self.compare_models(results)
        
        logger.info(" Training pipeline completed!")
        return results
    
    def save_training_results(self, results: Dict):
        """Save training results to file."""
        try:
            os.makedirs('results', exist_ok=True)
            
            # Prepare results for JSON serialization
            json_results = {
                'timestamp': datetime.now().isoformat(),
                'user': 'Utkarsh-upadhyay9',
                'data_info': results['data_info'],
                'models': {}
            }
            
            for model_name, model_result in results['models'].items():
                json_results['models'][model_name] = {
                    'metrics': model_result['metrics'],
                    'model_type': model_result['model_type']
                }
            
            # Save to JSON
            with open('results/training_results.json', 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(" Training results saved to results/training_results.json")
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
    
    def compare_models(self, results: Dict):
        """Compare model performances."""
        try:
            logger.info("\n MODEL COMPARISON")
            logger.info("=" * 50)
            
            comparison_data = []
            
            for model_name, model_result in results['models'].items():
                metrics = model_result['metrics']
                comparison_data.append({
                    'Model': model_name.upper(),
                    'RMSE': f"{metrics.get('rmse', 0):.4f}",
                    'MAE': f"{metrics.get('mae', 0):.4f}",
                    'MAPE': f"{metrics.get('mape', 0):.2f}%",
                    'Dir. Acc.': f"{metrics.get('directional_accuracy', 0):.2f}%"
                })
            
            # Print comparison table
            if comparison_data:
                # Find best model by RMSE
                best_model = min(results['models'].items(), 
                               key=lambda x: x[1]['metrics'].get('rmse', float('inf')))
                
                for data in comparison_data:
                    status = "🥇" if data['Model'] == best_model[0].upper() else "  "
                    logger.info(f"{status} {data['Model']:<10} | RMSE: {data['RMSE']:<8} | MAE: {data['MAE']:<8} | Dir. Acc: {data['Dir. Acc.']}")
                
                logger.info(f"\n🏆 Best model: {best_model[0].upper()} (lowest RMSE)")
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
