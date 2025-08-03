"""
Attention-based model implementation for market prediction.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger


class AttentionModel:
    """Attention-based model for time series prediction."""
    
    def __init__(self, config: Dict):
        """
        Initialize Attention model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int], output_dim: int = 1) -> tf.keras.Model:
        """
        Build Attention model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            output_dim: Output dimension
        
        Returns:
            Compiled Keras model
        """
        try:
            # Model parameters
            d_model = self.config.get('d_model', 128)
            num_heads = self.config.get('num_heads', 8)
            num_layers = self.config.get('num_layers', 4)
            dropout_rate = self.config.get('dropout', 0.1)
            
            # Input layer
            inputs = Input(shape=input_shape)
            
            # Project input features to d_model dimensions
            x = Dense(d_model, activation='relu')(inputs)
            x = LayerNormalization()(x)
            x = Dropout(dropout_rate)(x)
            
            # Multiple attention layers
            for i in range(num_layers):
                # Multi-head attention
                attention_output = MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=d_model // num_heads,
                    dropout=dropout_rate
                )(x, x)
                
                # Add & norm
                x = LayerNormalization()(x + attention_output)
                
                # Feed forward
                ff_output = Dense(d_model * 2, activation='relu')(x)
                ff_output = Dropout(dropout_rate)(ff_output)
                ff_output = Dense(d_model)(ff_output)
                
                # Add & norm
                x = LayerNormalization()(x + ff_output)
                x = Dropout(dropout_rate)(x)
            
            # Global average pooling
            x = GlobalAveragePooling1D()(x)
            
            # Output layers
            x = Dense(64, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            outputs = Dense(output_dim, activation='linear')(x)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            optimizer = Adam(learning_rate=self.config.get('learning_rate', 0.0001))
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mape']
            )
            
            self.model = model
            logger.info(f"Attention model built with input shape: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error building Attention model: {e}")
            return None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the Attention model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training history
        """
        try:
            if self.model is None:
                logger.error("Model not built. Call build_model() first.")
                return {}
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    'models/best_attention_model.h5',
                    monitor='val_loss' if validation_data else 'loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.history = history
            logger.info("Attention model training completed")
            return history.history
            
        except Exception as e:
            logger.error(f"Error training Attention model: {e}")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features
        
        Returns:
            Predictions
        """
        try:
            if self.model is None:
                logger.error("Model not trained. Call train() first.")
                return np.array([])
            
            predictions = self.model.predict(X, verbose=0)
            logger.info(f"Generated predictions for {len(X)} samples")
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Evaluation metrics
        """
        try:
            if self.model is None:
                logger.error("Model not trained. Call train() first.")
                return {}
            
            # Get predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics
            mse = np.mean((y_test - predictions) ** 2)
            mae = np.mean(np.abs(y_test - predictions))
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            # Directional accuracy
            actual_direction = np.sign(np.diff(y_test))
            predicted_direction = np.sign(np.diff(predictions))
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'directional_accuracy': float(directional_accuracy)
            }
            
            logger.info(f"Model evaluation completed: RMSE={rmse:.4f}, MAE={mae:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return
            
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
