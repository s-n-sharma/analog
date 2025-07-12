# 📁 cutoff_predictor/config.py

import os

# This file defines the shared configuration for the cutoff predictor module.

# The absolute path to the directory containing this config file (e.g., .../pipeline/cutoff_predictor)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The path where trained models will be saved and loaded from.
# This ensures train.py and predictor.py always use the same location.
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "trained_models")
