import torch

def inspect_model(model_path):
    print(f"Inspecting PyTorch model: {model_path}")
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        if isinstance(checkpoint, dict):
            print("Detected a state dictionary or checkpoint dictionary.")
            print(f"Keys in dictionary: {list(checkpoint.keys())}")
            
            # If it's a state dict directly
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            print(f"\nNumber of parameter tensors: {len(state_dict)}")
            print("\nSample of layer names (first 10):")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                print(f"  {key}: {state_dict[key].shape}")
        else:
            print("Detected a full model object (jit or pickled object).")
            print(f"Type: {type(checkpoint)}")
            if hasattr(checkpoint, 'eval'):
                print("\nModel Architecture:")
                print(checkpoint)

    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    inspect_model(r'e:\Brain Tumor\src\models\best_model.pth')
