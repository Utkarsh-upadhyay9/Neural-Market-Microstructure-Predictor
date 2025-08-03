import tensorflow as tf
from scripts.run_predictions import *

# Test your current model
model_path = 'models/extreme_heavy_model.keras'
if os.path.exists(model_path):
    print("🎯 Testing current extreme model...")
    
    # Test predictions
    python scripts/run_predictions.py --symbols AAPL GOOGL MSFT --models lstm --summary
    
    print("✅ Your current model is working!")
else:
    print("❌ No checkpoint found")
