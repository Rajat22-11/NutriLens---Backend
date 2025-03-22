import torch
import sys

def test_cuda_availability():
    print("\n===== PyTorch CUDA Configuration Test =====\n")
    
    # Basic PyTorch version info
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # CUDA version info
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")
        
        # cuDNN info
        cudnn_available = torch.backends.cudnn.is_available()
        print(f"cuDNN available: {cudnn_available}")
        if cudnn_available:
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        # GPU info
        device_count = torch.cuda.device_count()
        print(f"GPU count: {device_count}")
        
        for i in range(device_count):
            print(f"\nGPU {i} info:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            
            # Memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            print(f"  Total memory: {total_memory:.2f} GB")
            
        # Test tensor creation on GPU
        print("\nCreating test tensor on GPU...")
        try:
            test_tensor = torch.ones(10, 10).cuda()
            print(f"Test tensor device: {test_tensor.device}")
            print("✅ Successfully created tensor on GPU!")
        except Exception as e:
            print(f"❌ Error creating tensor on GPU: {str(e)}")
    else:
        print("\n⚠️ CUDA is not available. Please check your PyTorch installation and CUDA setup.")
        print("Make sure you have installed the correct PyTorch version for your CUDA version.")
        print("For CUDA 12.4, use: pip install torch==2.2.0+cu124 torchvision==0.17.0+cu124 torchaudio==2.2.0+cu124 --index-url https://download.pytorch.org/whl/cu124")
    
    print("\n===== Test Complete =====\n")

if __name__ == "__main__":
    test_cuda_availability()