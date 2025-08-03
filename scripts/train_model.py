#!/usr/bin/env python3
"""
Model training script for the Neural Market Microstructure Predictor.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from datetime import datetime
from loguru import logger

from src.training.trainer import ModelTrainer
from src.data.collector import DataCollector
from utils import setup_logging, load_config


def load_training_data(symbols: list, source: str = "yahoo") -> pd.DataFrame:
    """Load training data from specified source."""
    logger.info(f" Loading training data for symbols: {symbols}")
    
    collector = DataCollector()
    all_data = []
    
    for symbol in symbols:
        logger.info(f"   Loading {symbol}...")
        
        if source == "yahoo":
            data = collector.get_yahoo_data(symbol, period="2y")  # 2 years of data
        elif source == "alpha_vantage":
            data = collector.get_alpha_vantage_data(symbol, interval="daily")
        else:
            logger.error(f"Unknown source: {source}")
            continue
        
        if not data.empty:
            all_data.append(data)
            logger.info(f"     Loaded {len(data)} records")
        else:
            logger.warning(f"    ⚠  No data for {symbol}")
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f" Combined dataset: {len(combined_data)} total records")
        return combined_data
    else:
        logger.error(" No training data loaded")
        return pd.DataFrame()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train neural network models')
    parser.add_argument('--models', nargs='+', default=['lstm', 'cnn'], 
                       choices=['lstm', 'cnn', 'attention'],
                       help='Models to train')
    parser.add_argument('--symbols', nargs='+', 
                       default=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
                       help='Stock symbols for training')
    parser.add_argument('--source', choices=['yahoo', 'alpha_vantage'], 
                       default='yahoo', help='Data source')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info(" Neural Market Predictor - Model Training")
    logger.info("=" * 60)
    logger.info(f" User: Utkarsh-upadhyay9")
    logger.info(f" Time: {datetime.now()}")
    logger.info(f"🤖 Models: {args.models}")
    logger.info(f" Symbols: {args.symbols}")
    logger.info(f" Source: {args.source}")
    
    try:
        # Load training data
        training_data = load_training_data(args.symbols, args.source)
        
        if training_data.empty:
            logger.error("No training data available")
            return 1
        
        # Initialize trainer
        trainer = ModelTrainer(args.config)
        
        # Override epochs if specified
        if args.epochs:
            config = load_config(args.config)
            config['training']['epochs'] = args.epochs
            logger.info(f" Using {args.epochs} epochs")
        
        # Train models
        results = trainer.train_all_models(training_data, args.models)
        
        if results:
            logger.info("\n Training completed successfully!")
            logger.info(" Check the following directories:")
            logger.info("  🤖 models/ - Trained model files")
            logger.info("   results/ - Training results and metrics")
            
            # Show quick summary
            if 'models' in results:
                logger.info("\n TRAINING SUMMARY:")
                for model_name, model_result in results['models'].items():
                    metrics = model_result['metrics']
                    logger.info(f"  {model_name.upper()}: RMSE={metrics.get('rmse', 0):.4f}, "
                               f"Dir.Acc={metrics.get('directional_accuracy', 0):.1f}%")
            
            return 0
        else:
            logger.error("Training failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⚠  Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
