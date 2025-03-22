# PyTorch CUDA Setup for NutriLens

## Configuration

The NutriLens backend has been configured to use PyTorch with CUDA 12.4 support. This configuration allows the YOLOv5 model to leverage GPU acceleration for faster inference.

## Requirements

- NVIDIA GPU with CUDA capability
- CUDA 12.4 or compatible version installed on your system
- cuDNN library installed (compatible with your CUDA version)

## Installation

The `requirements.txt` file has been updated to use PyTorch with CUDA 12.4 support. To install the dependencies, run:

```bash
pip install -r requirements.txt
```

This will install PyTorch, TorchVision, and TorchAudio with CUDA 12.4 support from the PyTorch index URL.

## Verifying CUDA Setup

You can verify that PyTorch can access your GPU by running the test script:

```bash
python test_cuda.py
```

This script will display information about your PyTorch installation, CUDA version, and GPU availability.

## Troubleshooting

If you encounter issues with CUDA:

1. Ensure your NVIDIA drivers are up to date
2. Verify that CUDA 12.4 is installed correctly
3. Check that the PyTorch version matches your CUDA version

For CUDA 12.4/12.8, you may need to modify the requirements.txt file to use a compatible PyTorch version:

```
torch==2.2.0+cu121
torchvision==0.17.0+cu121
torchaudio==2.2.0+cu121
--index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with the appropriate CUDA version if needed.

## Code Configuration

The application is configured to use CUDA if available in `app.py`:

```python
# Select Device (GPU/CPU)
DEVICE = select_device("cuda:0" if torch.cuda.is_available() else "cpu")

# Enable half precision for faster inference if using CUDA
half = DEVICE.type != "cpu"  # half precision only supported on CUDA
model = DetectMultiBackend(MODEL_WEIGHTS, device=DEVICE, dnn=False, fp16=half)

# Clear CUDA cache to ensure clean start
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

This ensures that the model uses GPU acceleration when available, falling back to CPU if necessary.
