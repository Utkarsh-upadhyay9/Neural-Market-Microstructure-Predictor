"""Enhanced data collection module with proper error handling."""

import os
import time
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger
import yaml


class DataCollector:
	"""Collects market data from multiple sources.

	The collector reads API configuration from a YAML file by default
	(config/config.yaml). The config format is kept backward compatible with
	the original project.
	"""

	def __init__(self, config_path: str = "config/config.yaml"):
		with open(config_path, "r") as f:
			self.config = yaml.safe_load(f)

		self.api_keys = self.config.get('api', {})
		self.symbols = self.config.get('data', {}).get('symbols', [])
		self.timeframe = self.config.get('data', {}).get('timeframe', '1d')

	def get_yahoo_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
		"""Fetch data from Yahoo Finance using yfinance."""
		try:
			ticker = yf.Ticker(symbol)
			data = ticker.history(period=period)
			# Standardize column names
			data.columns = [col.lower().replace(' ', '_') for col in data.columns]
			data = data.reset_index()
			data['symbol'] = symbol
			logger.info(f"Successfully fetched Yahoo Finance data for {symbol}")
			return data
		except Exception as e:
			logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
			return pd.DataFrame()

	def get_alpha_vantage_data(self, symbol: str, interval: str = "daily") -> pd.DataFrame:
		"""Fetch data from Alpha Vantage API. Uses a configured base_url and key.

		This method has defensive checks for rate-limits and unexpected payloads.
		"""
		try:
			av = self.api_keys.get('alpha_vantage', {})
			api_key = av.get('key')
			base_url = av.get('base_url', 'https://www.alphavantage.co/query')

			if interval == '1min':
				function = 'TIME_SERIES_INTRADAY'
				params = {'function': function, 'symbol': symbol, 'interval': '1min', 'apikey': api_key, 'outputsize': 'compact'}
				time_series_key = 'Time Series (1min)'
			else:
				function = 'TIME_SERIES_DAILY'
				params = {'function': function, 'symbol': symbol, 'apikey': api_key, 'outputsize': 'compact'}
				time_series_key = 'Time Series (Daily)'

			resp = requests.get(base_url, params=params, timeout=30)
			data = resp.json()

			if time_series_key in data:
				df = pd.DataFrame(data[time_series_key]).T
				# Clean column names
				df.columns = [col.split('. ')[1].lower().replace(' ', '_') for col in df.columns]
				df.index = pd.to_datetime(df.index)
				df = df.sort_index().reset_index().rename(columns={'index': 'datetime'})
				df['symbol'] = symbol
				# Convert numeric columns
				for col in ['open', 'high', 'low', 'close', 'volume']:
					if col in df.columns:
						df[col] = pd.to_numeric(df[col], errors='coerce')
				logger.info(f"Successfully fetched Alpha Vantage data for {symbol}")
				return df

			# handle errors
			if isinstance(data, dict) and 'Note' in data:
				logger.warning(f"Alpha Vantage Note: {data['Note']}")
			elif isinstance(data, dict) and 'Error Message' in data:
				logger.error(f"Alpha Vantage Error: {data['Error Message']}")
			else:
				logger.error(f"Unexpected Alpha Vantage response keys: {list(data.keys())}")
			return pd.DataFrame()

		except Exception as e:
			logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
			return pd.DataFrame()

	def get_nasdaq_data(self, dataset_code: str, symbol: Optional[str] = None) -> pd.DataFrame:
		"""Fetch data from Nasdaq Data Link (formerly Quandl).

		Tries a list of endpoints and returns the first valid DataFrame.
		"""
		try:
			ndl = self.api_keys.get('nasdaq_data_link', {})
			api_key = ndl.get('key')

			endpoints = [
				f"https://data.nasdaq.com/api/v3/datasets/{dataset_code}/data.json",
			]

			for url in endpoints:
				try:
					params = {'api_key': api_key, 'limit': 100}
					resp = requests.get(url, params=params, timeout=30)
					if resp.status_code != 200:
						logger.warning(f"Non-200 response from {url}: {resp.status_code}")
						continue
					payload = resp.json()
					if 'dataset_data' in payload:
						dd = payload['dataset_data']
						cols = dd.get('column_names', [])
						rows = dd.get('data', [])
						df = pd.DataFrame(rows, columns=cols)
						# try to find a date column
						for c in df.columns:
							if c.lower() in ('date', 'datetime', 'time'):
								df[c] = pd.to_datetime(df[c])
								df = df.sort_values(c).reset_index(drop=True)
								break
						if symbol:
							df['symbol'] = symbol
						logger.info(f"Successfully fetched Nasdaq dataset {dataset_code}: {len(df)} rows")
						return df
				except requests.RequestException as e:
					logger.warning(f"RequestException for {url}: {e}")
					continue
			logger.error("All Nasdaq endpoints failed or returned no data")
			return pd.DataFrame()

		except Exception as e:
			logger.error(f"Error fetching Nasdaq data: {e}")
			return pd.DataFrame()

	def get_news_data(self, query: str = "financial markets", days_back: int = 7) -> pd.DataFrame:
		try:
			news = self.api_keys.get('newsapi', {})
			api_key = news.get('key')
			base_url = news.get('base_url', 'https://newsapi.org/v2')

			end_date = datetime.utcnow()
			start_date = end_date - timedelta(days=days_back)
			params = {
				'q': query,
				'from': start_date.strftime('%Y-%m-%d'),
				'to': end_date.strftime('%Y-%m-%d'),
				'language': 'en',
				'sortBy': 'publishedAt',
				'apiKey': api_key,
				'pageSize': 50,
			}
			url = f"{base_url}/everything"
			resp = requests.get(url, params=params, timeout=30)
			payload = resp.json()
			if payload.get('status') == 'ok' and payload.get('articles'):
				df = pd.DataFrame(payload['articles'])
				df['publishedAt'] = pd.to_datetime(df['publishedAt'])
				df = df.sort_values('publishedAt', ascending=False).reset_index(drop=True)
				logger.info(f"Fetched {len(df)} news articles")
				return df
			logger.warning(f"No news articles or unexpected response: {payload}")
			return pd.DataFrame()
		except Exception as e:
			logger.error(f"Error fetching news data: {e}")
			return pd.DataFrame()

	def get_real_time_data(self, symbol: str) -> Dict:
		try:
			ticker = yf.Ticker(symbol)
			info = ticker.info or {}
			return {
				'symbol': symbol,
				'current_price': info.get('currentPrice', 0),
				'previous_close': info.get('previousClose', 0),
				'open': info.get('open', 0),
				'day_high': info.get('dayHigh', 0),
				'day_low': info.get('dayLow', 0),
				'volume': info.get('volume', 0),
				'market_cap': info.get('marketCap', 0),
				'timestamp': datetime.utcnow(),
			}
		except Exception as e:
			logger.error(f"Error fetching real-time data for {symbol}: {e}")
			return {}

	def collect_multiple_symbols(self, symbols: List[str], source: str = "yahoo") -> Dict[str, pd.DataFrame]:
		data_dict: Dict[str, pd.DataFrame] = {}
		for symbol in symbols:
			logger.info(f"Collecting data for {symbol}")
			if source == 'yahoo':
				df = self.get_yahoo_data(symbol)
			elif source == 'alpha_vantage':
				df = self.get_alpha_vantage_data(symbol, interval='daily')
			else:
				logger.error(f"Unknown source: {source}")
				continue

			if not df.empty:
				data_dict[symbol] = df
				logger.info(f"Collected {len(df)} records for {symbol}")
			else:
				logger.warning(f"No data for {symbol}")

			# simple rate limiting
			if source == 'alpha_vantage':
				time.sleep(12)
			else:
				time.sleep(1)

		return data_dict

	def save_data(self, data: pd.DataFrame, filename: str):
		try:
			os.makedirs(os.path.dirname(filename), exist_ok=True)
			if filename.endswith('.csv'):
				data.to_csv(filename, index=False)
			elif filename.endswith('.parquet'):
				data.to_parquet(filename, index=False)
			else:
				data.to_pickle(filename)
			logger.info(f"Data saved to {filename}")
		except Exception as e:
			logger.error(f"Error saving data to {filename}: {e}")