
import torch
import numpy as np
import os

# Use relative imports
from .model import CNN1D
from pipeline.cutoff_predictor.config import MODEL_SAVE_DIR

class CutoffPredictor:
    def __init__(self, model_type='lowpass'):
        """
        Initializes and loads a trained CNN model for cutoff frequency prediction.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN1D(output_size=1).to(self.device)
        
        # Construct the path to the trained model using the shared config
        model_path = os.path.join(MODEL_SAVE_DIR, f"{model_type}_cutoff_predictor.pth")
        
        if not os.path.exists(model_path):
            # This error is now much more reliable and clearer
            raise FileNotFoundError(f"Model not found at the expected location: {model_path}\nPlease ensure you have run the training script first")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded {model_type} predictor model from {model_path}")

    def predict(self, filter_response_db):
        """
        Predicts the cutoff frequency from a given filter response curve.
        """
        input_tensor = torch.from_numpy(filter_response_db.astype(np.float32))
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            log_fc_pred = self.model(input_tensor).squeeze().item()
            
        fc_pred = 10**log_fc_pred
        return fc_pred
