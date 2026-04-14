import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import timm
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for server-side generation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import os
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms

CLASS_LABELS = ['glioma', 'meningioma', 'pituitary', 'notumor']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GradCAMPP:
    """
    Grad-CAM++ Implementation for generating pixel-wise importance heatmaps.
    """
    
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.model.eval()
        self.feature_maps = None
        self.gradients = None
        self.hooks = []
        
        if target_layer_name:
            self.target_layer = self._get_layer_by_name(target_layer_name)
            print(f"[INFO] Using specified target layer: {target_layer_name}")
        else:
            self.target_layer = self._find_last_conv_layer()
            print(f"[INFO] Auto-detected target layer: {self.target_layer}")
            
        self._register_hooks()
    
    def _find_last_conv_layer(self):
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise RuntimeError("No Conv2d layer found in the model.")
        return last_conv
    
    def _get_layer_by_name(self, name):
        for n, module in self.model.named_modules():
            if n == name:
                return module
        raise ValueError(f"Layer '{name}' not found in model.")
    
    def _forward_hook(self, module, input, output):
        self.feature_maps = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def _register_hooks(self):
        self.hooks.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(self._backward_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def generate_cam(self, input_tensor, target_class=None):
        if input_tensor.dim() == 4 and input_tensor.size(0) > 1:
            raise ValueError("Grad-CAM++ supports single image inference (batch_size=1).")
        
        self.feature_maps = None
        self.gradients = None
        
        output = self.model(input_tensor.to(DEVICE))
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)
        
        if self.feature_maps is None or self.gradients is None:
            raise RuntimeError("Failed to capture feature maps or gradients.")
            
        feature_maps = self.feature_maps.detach()
        gradients = self.gradients.detach()
        
        # Grad-CAM++ Weight Computation
        alpha_num = torch.sum(gradients, dim=(2, 3))
        alpha_den = torch.sum(gradients * feature_maps, dim=(2, 3)) + 1e-7
        alpha = alpha_num / alpha_den
        
        batch_size, num_channels, height, width = feature_maps.shape
        grad_cam_map = torch.zeros((height, width), device=feature_maps.device)
        
        for k in range(num_channels):
            grad_cam_map += alpha[0, k] * feature_maps[0, k]
            
        grad_cam_map = F.relu(grad_cam_map)
        grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-7)
        
        return grad_cam_map.cpu().numpy(), target_class


def load_model(model_path, num_classes=4):
    """Load trained brain tumor classification model."""
    # Update this if you used a different architecture
    model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=num_classes)
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model


def preprocess_image(image_path, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor, img


def visualize_and_save(heatmap, original_image, confidence, pred_label, 
                       image_path, output_dir, alpha=0.5):

    heatmap = np.clip(heatmap, 0, 1)

    orig_np = np.array(original_image)
    h, w = orig_np.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if len(orig_np.shape) == 2:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(orig_np, 1 - alpha, heatmap_colored, alpha, 0)

    title_color = "green" if pred_label == "notumor" else "red"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(orig_np)
    axes[0].set_title("Original MRI", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title(f"Interpretability overlay\nPred: {pred_label} | Conf: {confidence:.3f}", 
                      fontsize=12, color=title_color)
    axes[1].axis('off')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem
    save_path = os.path.join(output_dir, f"{base_name}_comparison.png")

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def sample_dataset_images(data_dir, mode='class_balanced', num_samples=4, seed=42):
    rng = random.Random(seed)
    image_paths = []
    
    print(f"Scanning folder: {data_dir}")
    
    if mode == 'class_balanced':
        for cls in CLASS_LABELS:
            # Try to find the folder (handles typos like 'giloma' vs 'glioma' loosely)
            # Check for exact match first
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                # Check subdirectories for a match
                found = False
                for subdir in os.listdir(data_dir):
                    if cls in subdir.lower():
                        cls_dir = os.path.join(data_dir, subdir)
                        found = True
                        print(f"Found class folder: {subdir} (mapped to {cls})")
                        break
                if not found:
                    print(f" Could not find folder for class '{cls}'")
                    continue

            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            if files:
                image_paths.append(rng.choice(files))
    else:
        files = []
        for root, _, fnames in os.walk(data_dir):
            for f in fnames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    files.append(os.path.join(root, f))
        if not files:
            raise FileNotFoundError(f"No valid images found in {data_dir}")
        image_paths = rng.sample(files, min(num_samples, len(files)))
        
    return image_paths


def process_batch(model, gradcam, image_paths, output_dir):
    results = []
    for img_path in image_paths:
        try:
            input_tensor, orig_img = preprocess_image(img_path)
            
            with torch.no_grad():
                output = model(input_tensor.to(DEVICE))
                probs = F.softmax(output, dim=1)
                pred_class = probs.argmax(dim=1).item()
                pred_conf = probs[0, pred_class].item()
                pred_label = CLASS_LABELS[pred_class]
                
            heatmap, _ = gradcam.generate_cam(input_tensor, target_class=pred_class)
            save_path = visualize_and_save(heatmap, orig_img, pred_conf, pred_label, 
                                           img_path, output_dir)
            
            results.append({
                'path': img_path,
                'pred_label': pred_label,
                'confidence': pred_conf,
                'heatmap': heatmap,
                'overlay_path': save_path
            })
            print(f" {Path(img_path).name:20} -> {pred_label:10} ({pred_conf:.3f})")
        except Exception as e:
            print(f" Failed on {img_path}: {e}")
    return results


def generate_comparison_report(results, output_dir):
    tumor_cases = [r for r in results if r['pred_label'] != 'notumor']
    nontumor_cases = [r for r in results if r['pred_label'] == 'notumor']
    
    print("INTERPRETABILITY ANALYSIS: TUMOR vs NO_TUMOR")
    
    if tumor_cases:
        avg_conf_t = np.mean([r['confidence'] for r in tumor_cases])
        print(f"• TUMOR CASES: {len(tumor_cases)} | Avg Confidence: {avg_conf_t:.3f}")
        
    if nontumor_cases:
        avg_conf_nt = np.mean([r['confidence'] for r in nontumor_cases])
        print(f"• NO_TUMOR CASES: {len(nontumor_cases)} | Avg Confidence: {avg_conf_nt:.3f}")
        
    all_cases = tumor_cases + nontumor_cases
    n = min(len(all_cases), 4)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    for i in range(n):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        overlay = plt.imread(all_cases[i]['overlay_path'])
        ax.imshow(overlay)
        
        color = 'red' if all_cases[i]['pred_label'] != 'notumor' else 'green'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            
        ax.set_title(f"{all_cases[i]['pred_label'].upper()}\nConf: {all_cases[i]['confidence']:.2%}", 
                     fontsize=12, fontweight='bold')
        ax.axis('off')
        
    plt.suptitle("Model Interpretability: Tumor (Red) vs No-Tumor (Green) Focus", fontsize=16, fontweight='bold', y=0.98)
    
    save_path = os.path.join(output_dir, "tumor_vs_notumor_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n Comparison visualization saved: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM++ Explainability for Brain Tumor MRI')
    
    # DEFAULT PATHS SET FOR YOUR STRUCTURE
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to model (default: same folder)')
    parser.add_argument('--data_dir', type=str, default='../data/balanced_dataset/Testing', help='Path to data folder')
    
    parser.add_argument('--output_dir', type=str, default='./gradcam_output', help='Save directory')
    parser.add_argument('--sample_mode', type=str, default='class_balanced', choices=['class_balanced', 'random'])
    parser.add_argument('--num_samples', type=int, default=8, help='Number of images for random mode')
    parser.add_argument('--target_layer', type=str, default=None, help='Explicit target conv layer name')
    parser.add_argument('--alpha', type=float, default=0.5, help='Overlay transparency')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224], help='Input size H W')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    model = load_model(args.model_path)
    gradcam = GradCAMPP(model, target_layer_name=args.target_layer)
    
    try:
        print(f"\nSampling images ({args.sample_mode} mode)...")
        image_paths = sample_dataset_images(args.data_dir, args.sample_mode, args.num_samples, args.seed)
        print(f"[INFO] Selected {len(image_paths)} images for analysis.\n")
        
        results = process_batch(model, gradcam, image_paths, args.output_dir)
        
        if results:
            generate_comparison_report(results, args.output_dir)
            
    finally:
        gradcam.remove_hooks()
        print("\n Cleanup complete. Hooks removed.")


if __name__ == '__main__':
    main()