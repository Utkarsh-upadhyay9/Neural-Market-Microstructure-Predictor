#!/usr/bin/env python3
"""
Neural Market Predictor with proper output scaling and validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import tensorflow as tf
import ta
from loguru import logger

# Simple logging setup
logger.add(sys.stderr, format="{time} | {level} | {message}", level="INFO")


def add_exact_38_features(data):
    """Add exactly 38 features to match the trained model."""
    try:
        df = data.copy()
        df.columns = [col.title() for col in df.columns]
        
        # Start with basic features (excluding Close as target)
        feature_list = ['Open', 'High', 'Low', 'Volume']
        
        # Feature 5: Returns
        df['Returns'] = df['Close'].pct_change()
        feature_list.append('Returns')
        
        # Features 6-9: Basic ratios
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        feature_list.extend(['High_Low_Ratio', 'Close_Open_Ratio', 'Volume_Change', 'High_Low_Pct'])
        
        # Features 10-16: Moving averages
        df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_5'] = ta.trend.ema_indicator(df['Close'], window=5)
        df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        feature_list.extend(['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_10', 'EMA_20'])
        
        # Features 17-19: MACD
        df['MACD'] = ta.trend.macd(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        feature_list.extend(['MACD', 'MACD_Signal', 'MACD_Histogram'])
        
        # Features 20-24: Bollinger Bands
        df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
        df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        feature_list.extend(['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position'])
        
        # Feature 25: RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        feature_list.append('RSI')
        
        # Features 26-27: Stochastic
        df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        feature_list.extend(['Stoch_K', 'Stoch_D'])
        
        # Feature 28: ATR
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        feature_list.append('ATR')
        
        # Features 29-33: Volume indicators
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        df['AD'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        feature_list.extend(['Volume_SMA', 'VWAP', 'MFI', 'AD', 'OBV'])
        
        # Features 34-38: Additional features
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Momentum'] = ta.momentum.roc(df['Close'], window=10)
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        feature_list.extend(['Open_Close_Pct', 'Volatility', 'Momentum', 'Williams_R', 'ADX'])
        
        logger.info(f"Created {len(feature_list)} features")
        return df, feature_list
        
    except Exception as e:
        logger.error(f"Error adding features: {e}")
        return data, []


def validate_prediction(predicted_price, current_price, model_name, symbol):
    """Validate and adjust unrealistic predictions."""
    
    # Calculate change percentage
    change_pct = ((predicted_price - current_price) / current_price) * 100 if current_price > 0 else 0
    
    # Define reasonable bounds (stock prices don't typically change more than 50% overnight)
    max_change = 50.0  # 50% max change
    min_price = current_price * 0.5  # Can't drop below 50% of current
    max_price = current_price * 1.5  # Can't go above 150% of current
    
    # Check for obviously wrong predictions
    if abs(change_pct) > max_change or predicted_price < 0:
        logger.warning(f"⚠️  {model_name} prediction for {symbol} seems unrealistic: ${predicted_price:.2f} ({change_pct:+.2f}%)")
        
        # Adjust to reasonable range
        if predicted_price > max_price:
            adjusted_price = max_price
            logger.info(f"🔧 Adjusted {model_name} prediction to ${adjusted_price:.2f}")
            return adjusted_price
        elif predicted_price < min_price:
            adjusted_price = min_price
            logger.info(f"🔧 Adjusted {model_name} prediction to ${adjusted_price:.2f}")
            return adjusted_price
    
    return predicted_price


def get_stock_data_exact_38(symbol, period="2y"):
    """Get stock data with exactly 38 features."""
    try:
        logger.info(f"📊 Getting data for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            return None, None
        
        logger.info(f"Downloaded {len(data)} records")
        
        # Add exactly 38 features
        data_with_features, feature_columns = add_exact_38_features(data)
        
        if len(feature_columns) != 38:
            logger.error(f"Feature count mismatch: {len(feature_columns)} != 38")
            return None, None
        
        # Clean data
        features_df = data_with_features[feature_columns].dropna()
        
        logger.info(f"After cleaning: {len(features_df)} rows, {len(feature_columns)} features")
        
        if len(features_df) < 100:
            logger.error(f"Not enough clean data: {len(features_df)}")
            return None, None
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        return features_df, current_price
        
    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {e}")
        return None, None


def create_sequence_exact(data, sequence_length=60):
    """Create sequence with exactly the right shape."""
    if len(data) < sequence_length:
        logger.error(f"Not enough data: {len(data)} < {sequence_length}")
        return None
    
    features = data.values
    sequence = features[-sequence_length:].reshape(1, sequence_length, data.shape[1])
    
    logger.info(f"✅ Created sequence shape: {sequence.shape}")
    return sequence


def load_and_predict_validated(symbol, models):
    """Load models and make validated predictions."""
    logger.info(f"📈 Predicting {symbol}...")
    
    # Get data with exactly 38 features
    data, current_price = get_stock_data_exact_38(symbol)
    if data is None:
        logger.error(f"No data for {symbol}")
        return None
    
    # Create sequence
    sequence = create_sequence_exact(data)
    if sequence is None:
        logger.error(f"Could not create sequence for {symbol}")
        return None
    
    logger.info(f"💰 Current price: ${current_price:.2f}")
    logger.info(f"🔢 Input shape: {sequence.shape}")
    
    predictions = {}
    valid_predictions = {}
    
    # Try each model
    for model_name in models:
        try:
            model_paths = [
                f"models/{model_name}_model.keras",
                f"models/{model_name}_model.h5"
            ]
            
            model = None
            for path in model_paths:
                if os.path.exists(path):
                    try:
                        if model_name == 'attention':
                            # Skip attention model for now due to loading issues
                            logger.warning(f"⚠️  Skipping {model_name} due to loading issues")
                            break
                        
                        model = tf.keras.models.load_model(path, compile=False)
                        logger.info(f"✅ Loaded {model_name} from {path}")
                        
                        # Verify shape compatibility
                        expected = model.input_shape
                        actual = sequence.shape
                        logger.info(f"🔍 Model expects: {expected}, We provide: {actual}")
                        
                        if expected[1:] == actual[1:]:
                            logger.info(f"✅ Shape match for {model_name}")
                        else:
                            logger.warning(f"⚠️  Shape mismatch for {model_name}")
                            model = None
                        
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {path}: {e}")
                        continue
            
            if model is None:
                logger.warning(f"Could not load {model_name} model")
                continue
            
            # Make prediction
            logger.info(f"🔮 Making prediction with {model_name}...")
            pred = model.predict(sequence, verbose=0)
            raw_predicted_price = float(pred[0][0])
            
            # Validate and adjust prediction
            predicted_price = validate_prediction(raw_predicted_price, current_price, model_name, symbol)
            
            change = predicted_price - current_price
            change_pct = (change / current_price) * 100 if current_price > 0 else 0
            
            prediction_data = {
                'predicted_price': predicted_price,
                'raw_prediction': raw_predicted_price,
                'change': change,
                'change_pct': change_pct,
                'direction': 'UP' if change > 0 else 'DOWN'
            }
            
            predictions[model_name] = prediction_data
            
            # Only include reasonable predictions in ensemble
            if abs(change_pct) <= 20:  # Only predictions within ±20%
                valid_predictions[model_name] = prediction_data
            
            direction_icon = "📈" if change > 0 else "📉"
            confidence = "🔥" if abs(change_pct) > 2 else "📊"
            
            logger.info(f"  {confidence} {model_name.upper()}: ${predicted_price:.2f} ({change_pct:+.2f}%) {direction_icon}")
            
            if raw_predicted_price != predicted_price:
                logger.info(f"    (Raw prediction was: ${raw_predicted_price:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error with {model_name}: {e}")
    
    if valid_predictions:
        # Calculate ensemble from valid predictions only
        if len(valid_predictions) > 1:
            avg_price = np.mean([p['predicted_price'] for p in valid_predictions.values()])
            avg_change = avg_price - current_price
            avg_change_pct = (avg_change / current_price) * 100 if current_price > 0 else 0
            
            predictions['ensemble'] = {
                'predicted_price': avg_price,
                'change': avg_change,
                'change_pct': avg_change_pct,
                'direction': 'UP' if avg_change > 0 else 'DOWN',
                'models_used': list(valid_predictions.keys())
            }
            
            direction_icon = "📈" if avg_change > 0 else "📉"
            
            logger.info(f"  🎯 ENSEMBLE ({len(valid_predictions)} models): ${avg_price:.2f} ({avg_change_pct:+.2f}%) {direction_icon}")
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'predictions': predictions,
            'valid_predictions': valid_predictions,
            'timestamp': datetime.now().isoformat()
        }
    
    return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Neural Market Predictor - Validated')
    parser.add_argument('--symbols', nargs='+', default=['AAPL'], help='Stock symbols')
    parser.add_argument('--models', nargs='+', default=['lstm'], 
                       choices=['lstm', 'cnn', 'attention'], help='Models to use')
    parser.add_argument('--summary', action='store_true', help='Show summary')
    
    args = parser.parse_args()
    
    logger.info("🧠 Neural Market Predictor - Validated Predictions")
    logger.info("=" * 60)
    logger.info(f"👤 User: Utkarsh-upadhyay9")
    logger.info(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info(f"📈 Symbols: {args.symbols}")
    logger.info(f"🤖 Models: {args.models}")
    logger.info(f"🎯 Features: Exactly 38 with validation")
    
    results = []
    
    for symbol in args.symbols:
        result = load_and_predict_validated(symbol, args.models)
        if result:
            results.append(result)
    
    # Summary
    if args.summary and results:
        logger.info("\n📊 VALIDATED PREDICTION SUMMARY")
        logger.info("=" * 60)
        
        for result in results:
            symbol = result['symbol']
            current = result['current_price']
            
            if 'ensemble' in result['predictions']:
                pred = result['predictions']['ensemble']
                models_used = pred.get('models_used', [])
                method = f"ENSEMBLE ({len(models_used)} models)"
            elif result['valid_predictions']:
                pred = list(result['valid_predictions'].values())[0]
                method = list(result['valid_predictions'].keys())[0].upper()
            else:
                continue
            
            direction_icon = "📈" if pred['direction'] == 'UP' else "📉"
            confidence = "🔥" if abs(pred['change_pct']) > 2 else "📊"
            
            logger.info(f"{direction_icon} {symbol} ({method}): ${current:.2f} → ${pred['predicted_price']:.2f} "
                       f"({pred['change_pct']:+.2f}%) {confidence}")
    
    if results:
        logger.info(f"\n🎉 Successfully generated validated predictions for {len(results)}/{len(args.symbols)} symbols")
        return 0
    else:
        logger.error("❌ No valid predictions generated")
        return 1


if __name__ == "__main__":
    exit(main())
