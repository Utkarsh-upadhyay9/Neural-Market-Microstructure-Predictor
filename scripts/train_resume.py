#!/usr/bin/env python3
"""
Training script optimized for powerful hardware with resume capability.
"""

import sys
import os
import json
from datetime import datetime
import tensorflow as tf
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_gpu():
    """Setup GPU for optimal training."""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logger.info(f"🔥 Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision for RTX series
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("✅ Mixed precision enabled for RTX optimization")
            
            return True
        else:
            logger.warning("⚠️  No GPU found, using CPU")
            return False
    except Exception as e:
        logger.error(f"GPU setup error: {e}")
        return False

def save_training_state(epoch, model_path, history_path=None):
    """Save training state for resume."""
    state = {
        'last_epoch': epoch,
        'model_path': model_path,
        'history_path': history_path,
        'timestamp': datetime.now().isoformat(),
        'user': 'Utkarsh-upadhyay9'
    }
    
    os.makedirs('models/checkpoints', exist_ok=True)
    
    with open('models/checkpoints/training_state.json', 'w') as f:
        json.dump(state, f, indent=2)
    
    logger.info(f"💾 Training state saved: epoch {epoch}")

def load_training_state():
    """Load previous training state."""
    state_path = 'models/checkpoints/training_state.json'
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
        logger.info(f"📁 Found previous state: epoch {state['last_epoch']}")
        return state
    return None

def main():
    """Main resume function."""
    logger.info("🔥 POWERFUL HARDWARE TRAINING - RESUME MODE")
    logger.info(f"👤 User: Utkarsh-upadhyay9")
    logger.info(f"📅 Date: {datetime.now()}")
    
    # Setup GPU
    gpu_available = setup_gpu()
    
    # Check for existing training state
    state = load_training_state()
    
    if state:
        logger.info(f"🔄 Can resume from epoch {state['last_epoch']}")
        logger.info(f"📁 Model path: {state['model_path']}")
    else:
        logger.info("🆕 No previous training state found")
        logger.info("💡 Run train_extreme_model.py to start fresh training")
    
    # Check existing models
    model_files = [
        'models/extreme_heavy_model.keras',
        'models/lstm_model.h5',
        'models/cnn_model.h5',
        'models/attention_model.h5'
    ]
    
    logger.info("\n📊 Available models:")
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024*1024)
            logger.info(f"  ✅ {model_file} ({size:.1f} MB)")
        else:
            logger.info(f"  ❌ {model_file} (not found)")
    
    # Check dataset
    if os.path.exists('data/extreme_dataset.csv'):
        import pandas as pd
        df = pd.read_csv('data/extreme_dataset.csv')
        logger.info(f"\n📊 Dataset: {df.shape[0]} records, {df.shape[1]} features")
    else:
        logger.info("\n❌ No extreme dataset found")
    
    logger.info("\n🚀 Ready for deployment to powerful hardware!")
    
    return 0

if __name__ == "__main__":
    exit(main())
