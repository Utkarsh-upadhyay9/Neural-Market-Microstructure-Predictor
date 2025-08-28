"""Neural Market Microstructure Predictor

A sophisticated deep learning system for predicting market microstructure using
LSTM, CNN, and Attention-based neural networks.

Author: Utkarsh-upadhyay9
Date: August 2025
"""

__version__ = "1.0.0"
__author__ = "Utkarsh-upadhyay9"

# Import only the modules that exist and work. Some submodules may have heavy
# optional dependencies; swallow import errors so simple imports still work.
try:
	from . import data, models, training, prediction
except Exception:
	# best-effort import; specific components import their dependencies lazily
	pass

__all__ = ["data", "models", "training", "prediction"]