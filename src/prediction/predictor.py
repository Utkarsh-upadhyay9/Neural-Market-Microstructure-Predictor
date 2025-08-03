"""
Market prediction module for real-time and batch predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import tensorflow as tf
from datetime import datetime, timedelta
from loguru import logger
import yaml
import os

from ..data.collector import DataCollector
from ..data.preprocessor import DataPreprocessor
from ..models.lstm_model import LSTMModel
from ..models.cnn_model import CNNModel
from ..models.attention_model import AttentionModel


class MarketPredictor:
    """Market prediction system for real-time and batch predictions."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the market predictor."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.preprocessor = DataPreprocessor(self.config)
        self.data_collector = DataCollector(config_path)
        self.prediction_cache = {}
        
    def load_models(self, model_types: List[str] = None) -> bool:
        """
        Load trained models.
        
        Args:
            model_types: List of model types to load
        
        Returns:
            True if models loaded successfully
        """
        if model_types is None:
            model_types = ['lstm', 'cnn', 'attention']
        
        try:
            logger.info("🔄 Loading trained models...")
            
            for model_type in model_types:
                model_path = f"models/{model_type}_model.h5"
                
                if os.path.exists(model_path):
                    if model_type == 'lstm':
                        model = LSTMModel(self.config['models']['lstm'])
                        model.load_model(model_path)
                        self.models['lstm'] = model
                    elif model_type == 'cnn':
                        model = CNNModel(self.config['models']['cnn'])
                        model.load_model(model_path)
                        self.models['cnn'] = model
                    elif model_type == 'attention':
                        model = AttentionModel(self.config['models']['attention'])
                        model.load_model(model_path)
                        self.models['attention'] = model
                    
                    logger.info(f"✅ Loaded {model_type.upper()} model")
                else:
                    logger.warning(f"⚠️  Model file not found: {model_path}")
            
            if self.models:
                logger.info(f"🎉 Successfully loaded {len(self.models)} models")
                return True
            else:
                logger.error("❌ No models loaded")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def prepare_prediction_data(self, symbol: str, lookback_days: int = 100) -> Optional[np.ndarray]:
        """
        Prepare data for prediction.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
        
        Returns:
            Prepared data array or None
        """
        try:
            logger.info(f"📊 Preparing prediction data for {symbol}")
            
            # Get recent data
            data = self.data_collector.get_yahoo_data(symbol, period=f"{lookback_days}d")
            
            if data.empty:
                logger.error(f"No data available for {symbol}")
                return None
            
            # Add technical indicators
            enhanced_data = self.preprocessor.add_technical_indicators(data)
            enhanced_data = enhanced_data.dropna()
            
            if len(enhanced_data) < self.config['models']['lstm']['sequence_length']:
                logger.error(f"Insufficient data for {symbol}")
                return None
            
            # Create sequences
            sequence_length = self.config['models']['lstm']['sequence_length']
            target_column = self.config['data']['target']
            
            # Select feature columns
            exclude_columns = ['datetime', 'date', 'symbol', target_column]
            feature_columns = [col for col in enhanced_data.columns 
                             if col not in exclude_columns and enhanced_data[col].dtype in ['float64', 'int64']]
            
            # Get the latest sequence
            features = enhanced_data[feature_columns].values
            if len(features) >= sequence_length:
                latest_sequence = features[-sequence_length:].reshape(1, sequence_length, -1)
                
                # Scale the data (you would need the scaler from training)
                # For now, we'll use the data as-is
                logger.info(f"✅ Prepared prediction data: {latest_sequence.shape}")
                return latest_sequence
            else:
                logger.error(f"Not enough data points for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error preparing prediction data: {e}")
            return None
    
    def predict_single_symbol(self, symbol: str, models: List[str] = None) -> Dict:
        """
        Make predictions for a single symbol.
        
        Args:
            symbol: Stock symbol
            models: List of models to use for prediction
        
        Returns:
            Dictionary with predictions
        """
        if models is None:
            models = list(self.models.keys())
        
        try:
            logger.info(f"🔮 Making predictions for {symbol}")
            
            # Prepare data
            X = self.prepare_prediction_data(symbol)
            if X is None:
                return {}
            
            predictions = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'models': {}
            }
            
            # Get current price for reference
            current_data = self.data_collector.get_real_time_data(symbol)
            current_price = current_data.get('current_price', 0)
            predictions['current_price'] = current_price
            
            # Make predictions with each model
            for model_name in models:
                if model_name in self.models:
                    try:
                        model_pred = self.models[model_name].predict(X)
                        if len(model_pred) > 0:
                            predicted_price = float(model_pred[0])
                            price_change = predicted_price - current_price
                            price_change_pct = (price_change / current_price) * 100 if current_price > 0 else 0
                            
                            predictions['models'][model_name] = {
                                'predicted_price': predicted_price,
                                'price_change': price_change,
                                'price_change_pct': price_change_pct,
                                'direction': 'UP' if price_change > 0 else 'DOWN'
                            }
                            
                            logger.info(f"  {model_name.upper()}: ${predicted_price:.2f} ({price_change_pct:+.2f}%)")
                    except Exception as e:
                        logger.error(f"Error with {model_name} prediction: {e}")
            
            # Ensemble prediction (average of all models)
            if predictions['models']:
                model_prices = [pred['predicted_price'] for pred in predictions['models'].values()]
                ensemble_price = np.mean(model_prices)
                ensemble_change = ensemble_price - current_price
                ensemble_change_pct = (ensemble_change / current_price) * 100 if current_price > 0 else 0
                
                predictions['ensemble'] = {
                    'predicted_price': float(ensemble_price),
                    'price_change': float(ensemble_change),
                    'price_change_pct': float(ensemble_change_pct),
                    'direction': 'UP' if ensemble_change > 0 else 'DOWN',
                    'confidence': self.calculate_confidence(predictions['models'])
                }
                
                logger.info(f"  🎯 ENSEMBLE: ${ensemble_price:.2f} ({ensemble_change_pct:+.2f}%) [Confidence: {predictions['ensemble']['confidence']:.1f}%]")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions for {symbol}: {e}")
            return {}
    
    def predict_multiple_symbols(self, symbols: List[str], models: List[str] = None) -> Dict:
        """
        Make predictions for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            models: List of models to use
        
        Returns:
            Dictionary with all predictions
        """
        logger.info(f"🔮 Making predictions for {len(symbols)} symbols")
        logger.info(f"👤 User: Utkarsh-upadhyay9")
        logger.info(f"📅 Time: {datetime.now()}")
        
        all_predictions = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {}
        }
        
        for symbol in symbols:
            logger.info(f"\n📈 Predicting {symbol}...")
            prediction = self.predict_single_symbol(symbol, models)
            
            if prediction:
                all_predictions['symbols'][symbol] = prediction
            else:
                logger.warning(f"⚠️  No prediction generated for {symbol}")
        
        # Summary
        successful_predictions = len(all_predictions['symbols'])
        logger.info(f"\n✅ Completed predictions for {successful_predictions}/{len(symbols)} symbols")
        
        return all_predictions
    
    def calculate_confidence(self, model_predictions: Dict) -> float:
        """
        Calculate prediction confidence based on model agreement.
        
        Args:
            model_predictions: Dictionary of model predictions
        
        Returns:
            Confidence score (0-100)
        """
        try:
            if len(model_predictions) < 2:
                return 50.0  # Low confidence with single model
            
            # Check direction agreement
            directions = [pred['direction'] for pred in model_predictions.values()]
            direction_agreement = directions.count(directions[0]) / len(directions)
            
            # Check price prediction variance
            prices = [pred['predicted_price'] for pred in model_predictions.values()]
            price_std = np.std(prices)
            mean_price = np.mean(prices)
            price_cv = price_std / mean_price if mean_price > 0 else 1.0
            
            # Calculate confidence (higher agreement and lower variance = higher confidence)
            confidence = (direction_agreement * 50) + ((1 - min(price_cv, 1.0)) * 50)
            
            return min(max(confidence, 0), 100)  # Clamp between 0-100
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 50.0
    
    def get_prediction_summary(self, predictions: Dict) -> pd.DataFrame:
        """
        Generate a summary DataFrame of predictions.
        
        Args:
            predictions: Predictions dictionary
        
        Returns:
            Summary DataFrame
        """
        try:
            summary_data = []
            
            for symbol, pred_data in predictions.get('symbols', {}).items():
                if 'ensemble' in pred_data:
                    ensemble = pred_data['ensemble']
                    summary_data.append({
                        'Symbol': symbol,
                        'Current Price': f"${pred_data.get('current_price', 0):.2f}",
                        'Predicted Price': f"${ensemble['predicted_price']:.2f}",
                        'Change': f"{ensemble['price_change']:+.2f}",
                        'Change %': f"{ensemble['price_change_pct']:+.2f}%",
                        'Direction': ensemble['direction'],
                        'Confidence': f"{ensemble['confidence']:.1f}%"
                    })
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            logger.error(f"Error creating prediction summary: {e}")
            return pd.DataFrame()
    
    def save_predictions(self, predictions: Dict, filename: str = None):
        """
        Save predictions to file.
        
        Args:
            predictions: Predictions dictionary
            filename: Output filename
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"predictions/predictions_{timestamp}.json"
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            import json
            with open(filename, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
            
            logger.info(f"💾 Predictions saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
