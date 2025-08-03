"""
Fixed LSTM model with updated Keras format.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger
import os


class LSTMModel:
    """LSTM model for time series prediction."""
    
    def __init__(self, config: Dict):
        """Initialize LSTM model with configuration."""
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int], output_dim: int = 1) -> tf.keras.Model:
        """Build LSTM model architecture."""
        try:
            model = Sequential([
                LSTM(
                    units=self.config.get('hidden_units', 50),
                    return_sequences=True,
                    input_shape=input_shape,
                    dropout=self.config.get('dropout', 0.2),
                    recurrent_dropout=self.config.get('dropout', 0.2)
                ),
                BatchNormalization(),
                
                LSTM(
                    units=self.config.get('hidden_units', 50),
                    return_sequences=self.config.get('layers', 2) > 2,
                    dropout=self.config.get('dropout', 0.2),
                    recurrent_dropout=self.config.get('dropout', 0.2)
                ),
                BatchNormalization(),
            ])
            
            # Add additional LSTM layers if specified
            for i in range(2, self.config.get('layers', 2)):
                model.add(LSTM(
                    units=self.config.get('hidden_units', 50),
                    return_sequences=i < self.config.get('layers', 2) - 1,
                    dropout=self.config.get('dropout', 0.2),
                    recurrent_dropout=self.config.get('dropout', 0.2)
                ))
                model.add(BatchNormalization())
            
            # Output layers
            model.add(Dropout(self.config.get('dropout', 0.2)))
            model.add(Dense(units=25, activation='relu'))
            model.add(Dropout(self.config.get('dropout', 0.2)))
            model.add(Dense(units=output_dim, activation='linear'))
            
            # Compile model
            optimizer = Adam(learning_rate=self.config.get('learning_rate', 0.001))
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']  # Simplified metrics
            )
            
            self.model = model
            logger.info(f"LSTM model built with input shape: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            return None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train the LSTM model."""
        try:
            if self.model is None:
                logger.error("Model not built. Call build_model() first.")
                return {}
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # Define callbacks with updated format
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
                    'models/best_lstm_model.keras',  # Updated format
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
            logger.info("LSTM model training completed")
            return history.history
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
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
        """Evaluate model performance."""
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
        """Save the trained model in Keras format."""
        try:
            if self.model is None:
                logger.error("No model to save")
                return
            
            # Ensure .keras extension
            if not filepath.endswith('.keras'):
                filepath = filepath.replace('.h5', '.keras')
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
            # Try both .keras and .h5 formats
            if os.path.exists(filepath):
                self.model = tf.keras.models.load_model(filepath, compile=False)
                # Recompile with simplified metrics
                self.model.compile(
                    optimizer=Adam(learning_rate=self.config.get('learning_rate', 0.001)),
                    loss='mse',
                    metrics=['mae']
                )
                logger.info(f"Model loaded from {filepath}")
            elif os.path.exists(filepath.replace('.h5', '.keras')):
                new_path = filepath.replace('.h5', '.keras')
                self.model = tf.keras.models.load_model(new_path, compile=False)
                self.model.compile(
                    optimizer=Adam(learning_rate=self.config.get('learning_rate', 0.001)),
                    loss='mse',
                    metrics=['mae']
                )
                logger.info(f"Model loaded from {new_path}")
            else:
                logger.error(f"Model file not found: {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
