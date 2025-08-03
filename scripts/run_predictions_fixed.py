#!/usr/bin/env python3
"""
Fixed prediction script with better error handling.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import pickle

from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from utils import setup_logging
import tensorflow as tf


def load_saved_scaler_and_features():
    """Load the scaler and feature columns from training."""
    try:
        with open('models/training_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return metadata.get('scaler'), metadata.get('feature_columns')
    except:
        logger.warning("Could not load training metadata")
        return None, None


def prepare_prediction_data_simple(symbol: str, sequence_length: int = 60):
    """Simplified data preparation that matches training."""
    try:
        logger.info(f" Preparing data for {symbol}")
        
        # Get more data to ensure we have enough after processing
        collector = DataCollector()
        data = collector.get_yahoo_data(symbol, period="1y")  # Get 1 year instead of 100 days
        
        if data.empty:
            logger.error(f"No data for {symbol}")
            return None
        
        logger.info(f"Downloaded {len(data)} records for {symbol}")
        
        # Use basic features only (matching training)
        basic_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Add simple derived features
        data['returns'] = data['close'].pct_change()
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        data['volume_change'] = data['volume'].pct_change()
        
        # Select features (should match training: 6 features)
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'returns']
        
        # Clean data
        data = data[feature_columns].dropna()
        
        logger.info(f"After cleaning: {len(data)} records with {len(feature_columns)} features")
        
        if len(data) < sequence_length + 10:
            logger.error(f"Not enough data: {len(data)} < {sequence_length + 10}")
            return None
        
        # Create the latest sequence
        features = data.values
        latest_sequence = features[-sequence_length:].reshape(1, sequence_length, len(feature_columns))
        
        logger.info(f" Created sequence: {latest_sequence.shape}")
        return latest_sequence, data['close'].iloc[-1]  # Return sequence and current price
        
    except Exception as e:
        logger.error(f"Error preparing data for {symbol}: {e}")
        return None


def make_simple_predictions(symbols: list, models: list):
    """Make predictions with simplified approach."""
    logger.info("🔮 Starting simplified predictions...")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'symbols': {}
    }
    
    # Load models
    loaded_models = {}
    for model_name in models:
        try:
            model_path = f"models/{model_name}_model.keras"
            if not os.path.exists(model_path):
                model_path = f"models/{model_name}_model.h5"
            
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path, compile=False)
                loaded_models[model_name] = model
                logger.info(f" Loaded {model_name} model")
            else:
                logger.warning(f"⚠  Model not found: {model_path}")
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
    
    if not loaded_models:
        logger.error("No models loaded successfully")
        return results
    
    # Make predictions for each symbol
    for symbol in symbols:
        logger.info(f"\n Predicting {symbol}...")
        
        # Prepare data
        pred_data = prepare_prediction_data_simple(symbol)
        if pred_data is None:
            logger.warning(f"Skipping {symbol} - data preparation failed")
            continue
        
        X, current_price = pred_data
        
        # Get real-time price for comparison
        try:
            collector = DataCollector()
            real_time_data = collector.get_real_time_data(symbol)
            if real_time_data:
                current_price = real_time_data.get('current_price', current_price)
        except:
            pass
        
        symbol_results = {
            'current_price': float(current_price),
            'models': {}
        }
        
        # Make predictions with each model
        for model_name, model in loaded_models.items():
            try:
                prediction = model.predict(X, verbose=0)
                predicted_price = float(prediction[0][0])
                
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100 if current_price > 0 else 0
                
                symbol_results['models'][model_name] = {
                    'predicted_price': predicted_price,
                    'price_change': price_change,
                    'price_change_pct': price_change_pct,
                    'direction': 'UP' if price_change > 0 else 'DOWN'
                }
                
                logger.info(f"  {model_name.upper()}: ${predicted_price:.2f} ({price_change_pct:+.2f}%)")
                
            except Exception as e:
                logger.error(f"Error with {model_name} prediction: {e}")
        
        # Calculate ensemble if multiple models
        if len(symbol_results['models']) > 1:
            model_prices = [pred['predicted_price'] for pred in symbol_results['models'].values()]
            ensemble_price = np.mean(model_prices)
            ensemble_change = ensemble_price - current_price
            ensemble_change_pct = (ensemble_change / current_price) * 100 if current_price > 0 else 0
            
            symbol_results['ensemble'] = {
                'predicted_price': float(ensemble_price),
                'price_change': float(ensemble_change),
                'price_change_pct': float(ensemble_change_pct),
                'direction': 'UP' if ensemble_change > 0 else 'DOWN',
                'confidence': 75.0  # Simplified confidence
            }
            
            logger.info(f"   ENSEMBLE: ${ensemble_price:.2f} ({ensemble_change_pct:+.2f}%)")
        
        results['symbols'][symbol] = symbol_results
    
    return results


def display_results(results):
    """Display prediction results."""
    if not results.get('symbols'):
        logger.error("No predictions to display")
        return
    
    logger.info("\n PREDICTION RESULTS")
    logger.info("=" * 60)
    
    # Create summary table
    summary_data = []
    for symbol, data in results['symbols'].items():
        if 'ensemble' in data:
            ensemble = data['ensemble']
        elif data['models']:
            # Use first model if no ensemble
            first_model = list(data['models'].values())[0]
            ensemble = first_model
        else:
            continue
        
        direction_icon = "" if ensemble['direction'] == "UP" else "📉"
        
        summary_data.append({
            'Symbol': symbol,
            'Current': f"${data['current_price']:.2f}",
            'Predicted': f"${ensemble['predicted_price']:.2f}",
            'Change': f"{ensemble['price_change_pct']:+.2f}%",
            'Direction': f"{direction_icon} {ensemble['direction']}"
        })
        
        logger.info(f"{direction_icon} {symbol}: ${data['current_price']:.2f} → ${ensemble['predicted_price']:.2f} "
                   f"({ensemble['price_change_pct']:+.2f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions/simple_predictions_{timestamp}.json"
    
    try:
        os.makedirs('predictions', exist_ok=True)
        import json
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f" Results saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Make market predictions (simplified)')
    parser.add_argument('--symbols', nargs='+', 
                       default=['AAPL', 'GOOGL', 'MSFT'],
                       help='Stock symbols to predict')
    parser.add_argument('--models', nargs='+',
                       default=['lstm'],
                       choices=['lstm', 'cnn', 'attention'],
                       help='Models to use for prediction')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("🔮 Simple Neural Market Predictor")
    logger.info("=" * 50)
    logger.info(f" User: Utkarsh-upadhyay9")
    logger.info(f" Time: {datetime.now()}")
    logger.info(f" Symbols: {args.symbols}")
    logger.info(f"🤖 Models: {args.models}")
    
    try:
        # Make predictions
        results = make_simple_predictions(args.symbols, args.models)
        
        # Display results
        display_results(results)
        
        logger.info("\n Predictions completed!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
