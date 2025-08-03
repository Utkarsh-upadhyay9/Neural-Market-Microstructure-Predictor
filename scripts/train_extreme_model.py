#!/usr/bin/env python3
"""
Train the extreme heavy model with massive data and resume capability.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from loguru import logger
import pandas as pd
import numpy as np

# Setup logging
logger.add(sys.stderr, format="{time} | {level} | {message}", level="INFO")

logger.info('🔥 EXTREME MODEL TRAINING INITIATED')
logger.info(f'👤 User: Utkarsh-upadhyay9')
logger.info(f'📅 Time: {datetime.now()}')


def check_existing_checkpoint():
    """Check if we can resume from existing checkpoint."""
    checkpoint_paths = [
        'models/extreme_heavy_model.keras',
        'models/extreme_heavy_final.keras'
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            logger.info(f'📁 Found existing checkpoint: {path}')
            return path
    
    return None


def load_existing_dataset():
    """Load existing dataset if available."""
    dataset_path = 'data/extreme_dataset.csv'
    
    if os.path.exists(dataset_path):
        logger.info(f'📁 Loading existing dataset: {dataset_path}')
        try:
            dataset = pd.read_csv(dataset_path)
            logger.info(f'✅ Loaded dataset: {dataset.shape}')
            return dataset
        except Exception as e:
            logger.error(f'Error loading dataset: {e}')
            return None
    
    return None


def create_extreme_dataset_simple():
    """Create a heavy dataset with existing tools."""
    try:
        # First check if dataset already exists
        existing_dataset = load_existing_dataset()
        if existing_dataset is not None:
            return existing_dataset
        
        # Import our existing modules
        from src.data.collector import DataCollector
        from src.data.preprocessor import DataPreprocessor
        
        # Massive symbol list
        MEGA_SYMBOLS = [
            # Tech giants
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            # Financial
            'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
            # Healthcare  
            'JNJ', 'PFE', 'UNH', 'ABBV',
            # Industrial
            'BA', 'CAT', 'GE', 'MMM',
            # Consumer
            'KO', 'PEP', 'WMT', 'PG',
            # Energy
            'XOM', 'CVX'
        ]
        
        logger.info(f'📊 Collecting data for {len(MEGA_SYMBOLS)} symbols')
        
        collector = DataCollector()
        preprocessor = DataPreprocessor()
        
        all_data = []
        
        for i, symbol in enumerate(MEGA_SYMBOLS):
            try:
                logger.info(f'📈 Processing {symbol} ({i+1}/{len(MEGA_SYMBOLS)})...')
                
                # Get 5 years of data for each symbol
                data = collector.get_yahoo_data(symbol, period="5y")
                
                if not data.empty:
                    # Add comprehensive technical indicators
                    enhanced_data = preprocessor.add_technical_indicators(data)
                    enhanced_data['symbol'] = symbol
                    
                    all_data.append(enhanced_data)
                    logger.info(f'  ✅ {symbol}: {len(enhanced_data)} records with {len(enhanced_data.columns)} features')
                else:
                    logger.warning(f'  ⚠️  No data for {symbol}')
                    
            except Exception as e:
                logger.error(f'  ❌ Error with {symbol}: {e}')
                continue
        
        if all_data:
            # Combine all data
            combined_dataset = pd.concat(all_data, ignore_index=True)
            logger.info(f'🎉 EXTREME dataset created: {len(combined_dataset)} records, {len(combined_dataset.columns)} features')
            
            # Ensure data directory exists
            os.makedirs('data', exist_ok=True)
            
            # Save the dataset
            combined_dataset.to_csv('data/extreme_dataset.csv', index=False)
            logger.info('💾 Dataset saved to data/extreme_dataset.csv')
            
            return combined_dataset
        else:
            logger.error('❌ No data collected')
            return None
            
    except Exception as e:
        logger.error(f'Error creating extreme dataset: {e}')
        return None


def create_heavy_model(input_shape):
    """Create a heavy model architecture."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Concatenate, Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        
        logger.info(f'🧠 Building HEAVY model with input shape: {input_shape}')
        
        # Input layer
        input_layer = Input(shape=input_shape)
        
        # LSTM branch
        lstm_branch = LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(input_layer)
        lstm_branch = BatchNormalization()(lstm_branch)
        lstm_branch = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(lstm_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
        lstm_branch = LSTM(64, return_sequences=False, dropout=0.3)(lstm_branch)
        lstm_output = Dense(128, activation='relu')(lstm_branch)
        
        # CNN branch
        cnn_branch = Conv1D(128, 3, activation='relu', padding='same')(input_layer)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(2)(cnn_branch)
        
        cnn_branch = Conv1D(256, 5, activation='relu', padding='same')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(2)(cnn_branch)
        
        cnn_branch = Conv1D(512, 7, activation='relu', padding='same')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = Flatten()(cnn_branch)
        cnn_output = Dense(128, activation='relu')(cnn_branch)
        
        # Combine branches
        combined = Concatenate()([lstm_output, cnn_output])
        
        # Heavy fully connected layers
        x = Dense(1024, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        output = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
            loss='mse',
            metrics=['mae']
        )
        
        param_count = model.count_params()
        logger.info(f'✅ HEAVY model built: {param_count:,} parameters')
        
        return model
        
    except Exception as e:
        logger.error(f'Error building heavy model: {e}')
        return None


def train_heavy_model(dataset, resume_from_checkpoint=True):
    """Train the heavy model with resume capability."""
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from sklearn.preprocessing import StandardScaler
        
        logger.info('🚀 Training HEAVY model...')
        
        # Prepare data
        from src.data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        
        # Use the preprocessing pipeline
        processed_data = preprocessor.preprocess_pipeline(dataset, sequence_length=120, target_column='close')
        
        if not processed_data:
            logger.error('❌ Data preprocessing failed')
            return None
        
        X_train = processed_data['X_train']
        X_test = processed_data['X_test'] 
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        
        logger.info(f'📊 Training data: {X_train.shape}')
        logger.info(f'📊 Test data: {X_test.shape}')
        
        # Create validation split
        val_split = int(len(X_train) * 0.8)
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train = X_train[:val_split]
        y_train = y_train[:val_split]
        
        # Check for existing checkpoint
        checkpoint_path = None
        if resume_from_checkpoint:
            checkpoint_path = check_existing_checkpoint()
        
        if checkpoint_path:
            logger.info(f'🔄 RESUMING from checkpoint: {checkpoint_path}')
            try:
                heavy_model = tf.keras.models.load_model(checkpoint_path)
                logger.info('✅ Successfully loaded checkpoint model')
            except Exception as e:
                logger.warning(f'⚠️  Failed to load checkpoint: {e}')
                logger.info('🔄 Creating new model instead...')
                input_shape = (X_train.shape[1], X_train.shape[2])
                heavy_model = create_heavy_model(input_shape)
        else:
            logger.info('🆕 Creating new model...')
            input_shape = (X_train.shape[1], X_train.shape[2])
            heavy_model = create_heavy_model(input_shape)
        
        if heavy_model is None:
            return None
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=15,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/extreme_heavy_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        logger.info('🔥 Starting/Resuming training...')
        history = heavy_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate - FIXED BUG HERE
        logger.info('📊 Evaluating model...')
        test_loss, test_mae = heavy_model.evaluate(X_test, y_test, verbose=0)  # Fixed: removed .model
        
        logger.info(f'✅ HEAVY model training completed!')
        logger.info(f'📈 Test Loss: {test_loss:.4f}')
        logger.info(f'📊 Test MAE: {test_mae:.4f}')
        
        # Save final model
        heavy_model.save('models/extreme_heavy_final.keras')
        logger.info('💾 Final model saved to models/extreme_heavy_final.keras')
        
        return heavy_model, history.history
        
    except Exception as e:
        logger.error(f'Error training heavy model: {e}')
        return None


def main():
    """Main function."""
    try:
        logger.info('🔥 STARTING EXTREME TRAINING PROCESS')
        
        # Step 1: Create massive dataset (or load existing)
        logger.info('\n📊 STEP 1: Preparing massive dataset...')
        dataset = create_extreme_dataset_simple()
        
        if dataset is None:
            logger.error('❌ Failed to create/load dataset')
            return 1
        
        # Step 2: Train heavy model (with resume capability)
        logger.info('\n🧠 STEP 2: Training heavy model...')
        result = train_heavy_model(dataset, resume_from_checkpoint=True)
        
        if result is None:
            logger.error('❌ Failed to train model')
            return 1
        
        model, history = result
        
        logger.info('\n🎉 EXTREME TRAINING COMPLETED SUCCESSFULLY!')
        logger.info('🚀 Your extreme model is ready for predictions!')
        
        return 0
        
    except KeyboardInterrupt:
        logger.info('\n⚠️  Training interrupted by user')
        logger.info('💡 You can resume training by running the script again!')
        return 1
    except Exception as e:
        logger.error(f'Training failed: {e}')
        return 1


if __name__ == "__main__":
    exit(main())
