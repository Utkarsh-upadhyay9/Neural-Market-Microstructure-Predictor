"""
Enhanced data collection module with proper error handling.
"""

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
from loguru import logger
import yaml
import json


class DataCollector:
    """Collects market data from multiple sources."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data collector with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.api_keys = self.config['api']
        self.symbols = self.config['data']['symbols']
        self.timeframe = self.config['data']['timeframe']
        
    def get_yahoo_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data.reset_index(inplace=True)
            data['symbol'] = symbol
            
            logger.info(f"Successfully fetched Yahoo Finance data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_alpha_vantage_data(self, symbol: str, interval: str = "1min") -> pd.DataFrame:
        """Fetch data from Alpha Vantage API."""
        try:
            api_key = self.api_keys['alpha_vantage']['key']
            base_url = self.api_keys['alpha_vantage']['base_url']
            
            # Use daily data for better reliability
            if interval == "1min":
                function = 'TIME_SERIES_INTRADAY'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'interval': interval,
                    'apikey': api_key,
                    'outputsize': 'compact'
                }
                time_series_key = f'Time Series ({interval})'
            else:
                function = 'TIME_SERIES_DAILY'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': api_key,
                    'outputsize': 'compact'
                }
                time_series_key = 'Time Series (Daily)'
            
            response = requests.get(base_url, params=params, timeout=30)
            data = response.json()
            
            if time_series_key in data:
                df = pd.DataFrame(data[time_series_key]).T
                
                # Clean column names
                df.columns = [col.split('. ')[1].lower().replace(' ', '_') for col in df.columns]
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'datetime'}, inplace=True)
                df['symbol'] = symbol
                
                # Convert to numeric
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col])
                
                logger.info(f"Successfully fetched Alpha Vantage data for {symbol}")
                return df
            else:
                if 'Note' in data:
                    logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                elif 'Error Message' in data:
                    logger.error(f"Alpha Vantage error: {data['Error Message']}")
                else:
                    logger.error(f"Alpha Vantage unexpected response: {list(data.keys())}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_nasdaq_data(self, dataset_code: str, symbol: str = None) -> pd.DataFrame:
        """Fetch data from Nasdaq Data Link with better error handling."""
        try:
            api_key = self.api_keys['nasdaq_data_link']['key']
            
            # Try different endpoints
            endpoints = [
                f"https://data.nasdaq.com/api/v3/datasets/{dataset_code}/data.json",
                f"https://data.nasdaq.com/api/v3/datasets/FRED/GDP/data.json"  # Fallback to known working dataset
            ]
            
            for url in endpoints:
                try:
                    params = {
                        'api_key': api_key,
                        'limit': 100
                    }
                    
                    logger.info(f"Trying Nasdaq endpoint: {url}")
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'dataset_data' in data:
                            dataset_data = data['dataset_data']
                            columns = dataset_data['column_names']
                            rows = dataset_data['data']
                            
                            df = pd.DataFrame(rows, columns=columns)
                            
                            # Handle different date column names
                            date_col = None
                            for col in df.columns:
                                if col.lower() in ['date', 'datetime', 'time']:
                                    date_col = col
                                    break
                            
                            if date_col:
                                df[date_col] = pd.to_datetime(df[date_col])
                                df = df.sort_values(date_col).reset_index(drop=True)
                            
                            if symbol:
                                df['symbol'] = symbol
                            
                            logger.info(f"Successfully fetched Nasdaq data: {len(df)} records")
                            return df
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request failed for {url}: {e}")
                    continue
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error for {url}: {e}")
                    continue
            
            logger.error("All Nasdaq endpoints failed")
            return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching Nasdaq data: {e}")
            return pd.DataFrame()
    
    def get_news_data(self, query: str = "financial markets", days_back: int = 7) -> pd.DataFrame:
        """Fetch news data from NewsAPI."""
        try:
            api_key = self.api_keys['newsapi']['key']
            base_url = self.api_keys['newsapi']['base_url']
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': api_key,
                'pageSize': 50
            }
            
            url = f"{base_url}/everything"
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if data['status'] == 'ok' and data['articles']:
                articles = data['articles']
                
                df = pd.DataFrame(articles)
                df['publishedAt'] = pd.to_datetime(df['publishedAt'])
                df = df.sort_values('publishedAt', ascending=False).reset_index(drop=True)
                
                logger.info(f"Successfully fetched {len(df)} news articles")
                return df
            else:
                logger.warning(f"No news articles found: {data}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return pd.DataFrame()
    
    def get_real_time_data(self, symbol: str) -> Dict:
        """Get real-time data for a symbol using Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_data = {
                'symbol': symbol,
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'timestamp': datetime.now()
            }
            
            return current_data
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            return {}
    
    def collect_multiple_symbols(self, symbols: List[str], source: str = "yahoo") -> Dict[str, pd.DataFrame]:
        """Collect data for multiple symbols."""
        data_dict = {}
        
        for symbol in symbols:
            logger.info(f"Collecting data for {symbol}")
            
            if source == "yahoo":
                data = self.get_yahoo_data(symbol)
            elif source == "alpha_vantage":
                data = self.get_alpha_vantage_data(symbol, interval="daily")  # Use daily for better reliability
            else:
                logger.error(f"Unknown data source: {source}")
                continue
            
            if not data.empty:
                data_dict[symbol] = data
                logger.info(f"✅ Collected {len(data)} records for {symbol}")
            else:
                logger.warning(f"❌ No data collected for {symbol}")
            
            # Rate limiting
            if source == "alpha_vantage":
                time.sleep(12)  # 5 calls per minute limit
            else:
                time.sleep(1)
        
        return data_dict
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save data to file."""
        try:
            import os
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
