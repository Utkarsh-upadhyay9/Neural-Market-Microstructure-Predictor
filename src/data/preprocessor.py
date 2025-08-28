"""Data preprocessing utilities for market data."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger

try:
	import ta
except Exception:
	ta = None


class DataPreprocessor:
	"""Preprocesses market data for machine learning models."""

	def __init__(self, config: Dict = None):
		self.config = config or {}
		self.scaler = None
		self.feature_columns: List[str] = []

	def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
		try:
			df = data.copy()
			df = df.drop_duplicates()
			df = df.fillna(method='ffill').fillna(method='bfill')

			numeric_columns = df.select_dtypes(include=[np.number]).columns
			for col in numeric_columns:
				Q1 = df[col].quantile(0.25)
				Q3 = df[col].quantile(0.75)
				IQR = Q3 - Q1
				lower = Q1 - 1.5 * IQR
				upper = Q3 + 1.5 * IQR
				df[col] = df[col].clip(lower, upper)

			if 'datetime' in df.columns:
				df['datetime'] = pd.to_datetime(df['datetime'])
			elif 'date' in df.columns:
				df['date'] = pd.to_datetime(df['date'])

			logger.info(f"Data cleaned: {len(df)} rows")
			return df
		except Exception as e:
			logger.error(f"Error cleaning data: {e}")
			return data

	def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
		df = data.copy()
		required = ['open', 'high', 'low', 'close', 'volume']
		if not all(col in df.columns for col in required):
			logger.warning("Missing required columns for technical indicators; skipping TA features")
			return df

		if ta is None:
			logger.warning("`ta` library not available; skipping technical indicators")
			return df

		try:
			df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
			df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
			df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
			df['macd'] = ta.trend.macd(df['close'])
			df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
			df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
			df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
			df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
			logger.info("Technical indicators added")
			return df
		except Exception as e:
			logger.error(f"Error computing technical indicators: {e}")
			return df

	def create_sequences(self, data: pd.DataFrame, sequence_length: int = 60, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
		try:
			exclude = ['datetime', 'date', 'symbol', target_column]
			feature_columns = [c for c in data.columns if c not in exclude and np.issubdtype(data[c].dtype, np.number)]
			self.feature_columns = feature_columns

			features = data[feature_columns].values
			target = data[target_column].values

			X, y = [], []
			for i in range(sequence_length, len(features)):
				X.append(features[i-sequence_length:i])
				y.append(target[i])

			X = np.array(X)
			y = np.array(y)
			logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
			return X, y
		except Exception as e:
			logger.error(f"Error creating sequences: {e}")
			return np.array([]), np.array([])

	def scale_features(self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None, scaler_type: str = 'standard'):
		try:
			if scaler_type == 'standard':
				self.scaler = StandardScaler()
			else:
				self.scaler = MinMaxScaler()

			orig_shape = X_train.shape
			Xr = X_train.reshape(-1, X_train.shape[-1])
			Xs = self.scaler.fit_transform(Xr).reshape(orig_shape)

			X_test_s = None
			if X_test is not None:
				tshape = X_test.shape
				Xtr = X_test.reshape(-1, X_test.shape[-1])
				X_test_s = self.scaler.transform(Xtr).reshape(tshape)

			return Xs, X_test_s
		except Exception as e:
			logger.error(f"Error scaling features: {e}")
			return X_train, X_test

	def preprocess_pipeline(self, data: pd.DataFrame, sequence_length: int = 60, target_column: str = 'close', test_size: float = 0.2) -> Dict:
		try:
			cleaned = self.clean_data(data)
			enhanced = self.add_technical_indicators(cleaned)
			enhanced = enhanced.dropna()

			if len(enhanced) < sequence_length + 10:
				logger.error("Insufficient data after preprocessing")
				return {}

			X, y = self.create_sequences(enhanced, sequence_length, target_column)
			if len(X) == 0:
				return {}

			split = int(len(X) * (1 - test_size))
			X_train, X_test = X[:split], X[split:]
			y_train, y_test = y[:split], y[split:]

			X_train_s, X_test_s = self.scale_features(X_train, X_test)

			result = {
				'X_train': X_train_s,
				'X_test': X_test_s,
				'y_train': y_train,
				'y_test': y_test,
				'feature_columns': self.feature_columns,
				'scaler': self.scaler,
				'original_data': enhanced
			}
			logger.info("Preprocessing pipeline completed")
			return result
		except Exception as e:
			logger.error(f"Error in preprocessing pipeline: {e}")
			return {}