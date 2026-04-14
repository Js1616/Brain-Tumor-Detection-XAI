import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
import timm
import sys

from backend.utils.explainable import GradCAMPP, load_model, preprocess_image, visualize_and_save

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_model.pth')
CLASS_LABELS = ['glioma', 'meningioma', 'pituitary', 'notumor']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InferenceEngine:
    def __init__(self, model_path=MODEL_PATH):
        print(f"[INFO] Initializing Inference Engine with model: {model_path}")
        self.model = load_model(model_path)
        self.gradcam = GradCAMPP(self.model)
        
    def predict(self, image_path, output_dir):
        """
        Runs prediction and generates XAI heatmap.
        Returns: {label, confidence, heatmap_path}
        """
        input_tensor, orig_img = preprocess_image(image_path)
        
        with torch.no_grad():
            output = self.model(input_tensor.to(DEVICE))
            probs = F.softmax(output, dim=1)
            pred_class = probs.argmax(dim=1).item()
            pred_conf = probs[0, pred_class].item()
            pred_label = CLASS_LABELS[pred_class]
            
        heatmap, _ = self.gradcam.generate_cam(input_tensor, target_class=pred_class)
        save_path = visualize_and_save(heatmap, orig_img, pred_conf, pred_label, image_path, output_dir)
        
        return {
            "tumor_type": pred_label.capitalize(),
            "confidence": f"{pred_conf*100:.1f}%",
            "heatmap_path": save_path,
            "description": self._get_description(pred_label),
            "suggestion": self._get_suggestion(pred_label)
        }

    def _get_description(self, label):
        descriptions = {
            "glioma": "Gliomas are tumors that start in the glial cells of the brain or the spine.",
            "meningioma": "A meningioma is a tumor that arises from the meninges — the membranes that surround your brain.",
            "pituitary": "Pituitary tumors are abnormal growths that develop in your pituitary gland.",
            "notumor": "No signs of tumor were detected in the provided MRI scan."
        }
        return descriptions.get(label, "Unknown tumor type detected.")

    def _get_suggestion(self, label):
        if label == "notumor":
            return "Scan regular check-ups are recommended, but no immediate action is required."
        else:
            return "Please consult with a neurosurgeon or oncologist for a detailed diagnosis and treatment plan."

# Singleton instance
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = InferenceEngine()
    return engine
