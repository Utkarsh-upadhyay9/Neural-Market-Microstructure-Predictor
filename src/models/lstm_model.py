"""LSTM model implementation for market prediction."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import Dict, Tuple, Optional
from loguru import logger


class LSTMModel:
	"""LSTM model for time series prediction."""

	def __init__(self, config: Dict):
		self.config = config
		self.model: Optional[tf.keras.Model] = None
		self.history = None

	def build_model(self, input_shape: Tuple[int, int], output_dim: int = 1) -> Optional[tf.keras.Model]:
		try:
			hidden = self.config.get('hidden_units', 64)
			layers = self.config.get('layers', 2)
			dropout = self.config.get('dropout', 0.2)

			model = Sequential()
			model.add(LSTM(hidden, return_sequences=(layers > 1), input_shape=input_shape, dropout=dropout))
			model.add(BatchNormalization())

			for i in range(1, layers):
				is_last = (i == layers - 1)
				model.add(LSTM(hidden, return_sequences=(not is_last), dropout=dropout))
				model.add(BatchNormalization())

			model.add(Dropout(dropout))
			model.add(Dense(units=64, activation='relu'))
			model.add(Dropout(dropout))
			model.add(Dense(units=output_dim, activation='linear'))

			optimizer = Adam(learning_rate=self.config.get('learning_rate', 1e-3))
			model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

			self.model = model
			logger.info(f"LSTM model built with input shape: {input_shape}")
			return model
		except Exception as e:
			logger.error(f"Error building LSTM model: {e}")
			return None

	def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, epochs: int = 50, batch_size: int = 32) -> Dict:
		try:
			if self.model is None:
				logger.error("Model not built. Call build_model() first.")
				return {}

			validation = (X_val, y_val) if X_val is not None and y_val is not None else None

			callbacks = [
				EarlyStopping(monitor='val_loss' if validation else 'loss', patience=10, restore_best_weights=True),
				ReduceLROnPlateau(monitor='val_loss' if validation else 'loss', factor=0.5, patience=5, min_lr=1e-7),
				ModelCheckpoint('models/best_lstm_model.h5', save_best_only=True)
			]

			history = self.model.fit(X_train, y_train, validation_data=validation, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
			self.history = history
			logger.info("LSTM training completed")
			return history.history
		except Exception as e:
			logger.error(f"Error training LSTM model: {e}")
			return {}

	def predict(self, X: np.ndarray) -> np.ndarray:
		try:
			if self.model is None:
				logger.error("Model not available")
				return np.array([])
			preds = self.model.predict(X, verbose=0)
			return preds.flatten()
		except Exception as e:
			logger.error(f"Error making predictions: {e}")
			return np.array([])

	def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
		try:
			preds = self.predict(X_test)
			mse = np.mean((y_test - preds) ** 2)
			mae = np.mean(np.abs(y_test - preds))
			rmse = np.sqrt(mse)
			mape = np.mean(np.abs((y_test - preds) / np.maximum(np.abs(y_test), 1e-6))) * 100
			directional = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(preds))) * 100 if len(y_test) > 1 else 0.0
			metrics = {'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape), 'directional_accuracy': float(directional)}
			logger.info(f"Evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}")
			return metrics
		except Exception as e:
			logger.error(f"Error evaluating model: {e}")
			return {}

	def save_model(self, filepath: str):
		try:
			if self.model is None:
				logger.error("No model to save")
				return
			self.model.save(filepath)
			logger.info(f"Model saved to {filepath}")
		except Exception as e:
			logger.error(f"Error saving model: {e}")

	def load_model(self, filepath: str):
		try:
			self.model = tf.keras.models.load_model(filepath)
			logger.info(f"Model loaded from {filepath}")
		except Exception as e:
			logger.error(f"Error loading model: {e}")