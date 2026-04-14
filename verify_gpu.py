import torch

def verify_gpu():
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("CUDA is NOT available. PyTorch is running on CPU.")

if __name__ == "__main__":
    verify_gpu()
