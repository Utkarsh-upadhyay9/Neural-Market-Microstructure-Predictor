""" Training script for 3 Billion Parameter Massive MoE Model. """
import asyncio
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.massive_collector import MassiveDataCollector
from src.models.massive_moe_model import MassiveMoEModel, MassiveDataPreprocessor

class MassiveModelTrainer:
    """Trainer for the 3 Billion Parameter Massive MoE Model."""

    def __init__(self, config_path: str = "config/massive_config.yaml"):
        """Initialize the massive model trainer."""
        self.config = self._load_config(config_path)
        self.data_collector = MassiveDataCollector(self.config)
        self.preprocessor = MassiveDataPreprocessor(self.config)
        self.model = None

        logger.info("🚀 Massive Model Trainer initialized")
        logger.info(f"   Target: 3 Billion Parameters")
        logger.info(f"   Dataset: Massive multi-asset collection")
        logger.info(f"   Training: Distributed MoE architecture")

    def _load_config(self, config_path: str) -> Dict:
        """Load massive training configuration."""
        default_config = {
            'model': {
                'num_experts': 128,
                'expert_hidden_size': 4096,
                'num_layers': 48,
                'num_heads': 32,
                'd_model': 8192,
                'dropout_rate': 0.1,
                'max_seq_length': 2048
            },
            'training': {
                'epochs': 100,
                'batch_size': 8,
                'learning_rate': 1e-4,
                'mixed_precision': True,
                'gradient_clip_norm': 1.0,
                'weight_decay': 0.01,
                'warmup_steps': 10000,
                'max_steps': 500000
            },
            'data': {
                'collection_years': 10,
                'max_symbols_per_source': 5000,
                'parallel_requests': 50,
                'sequence_length': 2048,
                'test_size': 0.1,
                'val_size': 0.1
            },
            'distributed': {
                'num_gpus': -1,  # Use all available GPUs
                'mixed_precision': True,
                'xla_jit': True
            },
            'logging': {
                'log_dir': 'logs/massive_training',
                'checkpoint_dir': 'checkpoints/massive_model',
                'save_steps': 1000,
                'eval_steps': 500
            }
        }

        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as file:
                file_config = yaml.safe_load(file)
                self._merge_configs(default_config, file_config)

        return default_config

    def _merge_configs(self, base_config: Dict, override_config: Dict):
        """Merge configuration dictionaries."""
        for key, value in override_config.items():
            if isinstance(value, dict) and key in base_config:
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value

    async def collect_massive_dataset(self) -> Dict[str, pd.DataFrame]:
        """Collect massive dataset from all sources."""
        logger.info("📊 Starting massive dataset collection...")

        # Create data directory
        data_dir = Path("massive_data")
        data_dir.mkdir(exist_ok=True)

        # Collect data from all sources
        massive_data = await self.data_collector.collect_massive_dataset(str(data_dir))

        # Get dataset statistics
        stats = self.data_collector.get_dataset_stats(str(data_dir))

        logger.info("📈 Dataset Statistics:")
        logger.info(f"   Total Symbols: {stats.get('total_symbols', 0):,}")
        logger.info(f"   Total Rows: {stats.get('total_rows', 0):,}")
        logger.info(f"   Total Features: {stats.get('total_features', 0)}")
        logger.info(f"   Date Range: {stats.get('date_range', {})}")

        return massive_data

    def preprocess_massive_data(self, massive_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Preprocess the massive dataset."""
        logger.info("🔧 Preprocessing massive dataset...")

        sequence_length = self.config['data']['sequence_length']
        processed_data = self.preprocessor.preprocess_massive_dataset(
            massive_data, sequence_length
        )

        return processed_data

    def build_massive_model(self, input_shape: Tuple[int, int]) -> MassiveMoEModel:
        """Build the 3 Billion Parameter Massive MoE Model."""
        logger.info("🏗️ Building 3 Billion Parameter Massive MoE Model...")

        model_config = self.config['model']
        self.model = MassiveMoEModel(model_config)

        model = self.model.build_massive_moe_architecture(input_shape, output_dim=1)

        logger.info("✅ Massive MoE Model built successfully!")
        return model

    def setup_distributed_training(self):
        """Setup distributed training environment."""
        logger.info("🔥 Setting up distributed training...")

        # Enable mixed precision
        if self.config['distributed']['mixed_precision']:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("   ✅ Mixed precision enabled")

        # Enable XLA JIT compilation
        if self.config['distributed']['xla_jit']:
            tf.config.optimizer.set_jit(True)
            logger.info("   ✅ XLA JIT compilation enabled")

        # Setup distributed strategy
        num_gpus = self.config['distributed']['num_gpus']
        if num_gpus == -1:
            num_gpus = len(tf.config.list_physical_devices('GPU'))

        if num_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f"   ✅ Distributed training with {num_gpus} GPUs")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            logger.info("   ✅ Single GPU training")

        return strategy

    def train_massive_model(self, processed_data: Dict[str, np.ndarray]):
        """Train the massive 3B parameter model."""
        logger.info("🎯 Starting massive model training...")

        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']

        logger.info("📊 Training Data Shapes:")
        logger.info(f"   X_train: {X_train.shape}")
        logger.info(f"   y_train: {y_train.shape}")
        logger.info(f"   X_val: {X_val.shape}")
        logger.info(f"   y_val: {y_val.shape}")

        # Setup distributed training
        strategy = self.setup_distributed_training()

        with strategy.scope():
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = self.build_massive_model(input_shape)

            # Training configuration
            training_config = self.config['training']

            # Train the model
            history = self.model.train_massive_model(
                X_train, y_train, X_val, y_val,
                epochs=training_config['epochs'],
                batch_size=training_config['batch_size'],
                use_mixed_precision=training_config['mixed_precision']
            )

        # Save the trained model
        self.save_massive_model()

        logger.info("✅ Massive model training completed!")
        return history

    def save_massive_model(self):
        """Save the massive trained model."""
        logger.info("💾 Saving massive model...")

        # Create checkpoints directory
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = checkpoint_dir / f"massive_moe_3b_{timestamp}.keras"

        self.model.save_massive_model(str(model_path))

        # Save training configuration
        config_path = checkpoint_dir / f"training_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

        logger.info(f"✅ Massive model saved to {model_path}")

    def load_massive_model(self, model_path: str):
        """Load a previously trained massive model."""
        logger.info(f"📂 Loading massive model from {model_path}")

        self.model = MassiveMoEModel(self.config['model'])
        self.model.load_massive_model(model_path)

        return self.model

    def evaluate_massive_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the massive model performance."""
        logger.info("📊 Evaluating massive model...")

        predictions = self.model.predict_massive(X_test)

        # Calculate metrics
        metrics = {}

        # Price prediction metrics
        for horizon in [1, 5, 15, 60, 240, 1440]:
            pred_key = f'price_horizon_{horizon}'
            if pred_key in predictions:
                pred = predictions[pred_key].flatten()
                actual = y_test

                # Calculate metrics
                mse = np.mean((pred - actual) ** 2)
                mae = np.mean(np.abs(pred - actual))
                rmse = np.sqrt(mse)

                # Directional accuracy
                pred_direction = np.sign(pred[1:] - pred[:-1])
                actual_direction = np.sign(actual[1:] - actual[:-1])
                directional_acc = np.mean(pred_direction == actual_direction)

                metrics[f'horizon_{horizon}'] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'directional_accuracy': directional_acc
                }

        logger.info("📈 Evaluation Results:")
        for horizon, horizon_metrics in metrics.items():
            logger.info(f"   {horizon}:")
            logger.info(f"     RMSE: {horizon_metrics['rmse']:.6f}")
            logger.info(f"     MAE: {horizon_metrics['mae']:.6f}")
            logger.info(f"     Directional Acc: {horizon_metrics['directional_accuracy']:.4f}")

        return metrics

    def run_massive_training_pipeline(self):
        """Run the complete massive training pipeline."""
        logger.info("🚀 Starting Massive 3B Parameter Model Training Pipeline")

        try:
            # Step 1: Collect massive dataset
            logger.info("Step 1/5: Collecting massive dataset...")
            massive_data = asyncio.run(self.collect_massive_dataset())

            # Step 2: Preprocess data
            logger.info("Step 2/5: Preprocessing data...")
            processed_data = self.preprocess_massive_data(massive_data)

            # Step 3: Build model
            logger.info("Step 3/5: Building massive model...")
            # Model is built during training

            # Step 4: Train model
            logger.info("Step 4/5: Training massive model...")
            history = self.train_massive_model(processed_data)

            # Step 5: Evaluate model
            logger.info("Step 5/5: Evaluating massive model...")
            if 'X_val' in processed_data and 'y_val' in processed_data:
                metrics = self.evaluate_massive_model(
                    processed_data['X_val'],
                    processed_data['y_val']
                )

            logger.info("🎉 Massive training pipeline completed successfully!")

            return {
                'status': 'success',
                'data_stats': self.data_collector.get_dataset_stats(),
                'training_history': history,
                'evaluation_metrics': metrics if 'metrics' in locals() else None
            }

        except Exception as e:
            logger.error(f"❌ Massive training pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

def main():
    """Main function to run massive model training."""
    # Setup logging
    logger.add(
        "logs/massive_training.log",
        rotation="10MB",
        retention="1 week",
        level="INFO"
    )

    # Initialize trainer
    trainer = MassiveModelTrainer()

    # Run massive training pipeline
    results = trainer.run_massive_training_pipeline()

    # Save results
    results_path = Path("results/massive_training_results.json")
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"📄 Results saved to {results_path}")

    if results['status'] == 'success':
        logger.info("🎉 Massive 3B parameter model training completed successfully!")
    else:
        logger.error(f"❌ Training failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()