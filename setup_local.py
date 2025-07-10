#!/usr/bin/env python3
"""
Setup script for local ComfyUI development
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=check, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")


def check_cuda():
    """Check CUDA availability"""
    try:
        result = run_command("nvidia-smi", check=False)
        if result.returncode == 0:
            print("NVIDIA GPU detected")
            return True
        else:
            print("No NVIDIA GPU detected - will use CPU")
            return False
    except:
        print("nvidia-smi not found - will use CPU")
        return False


def install_pytorch(cuda_available=True):
    """Install PyTorch with appropriate CUDA support"""
    print("\nInstalling PyTorch...")

    if cuda_available:
        # Install CUDA version
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        # Install CPU version
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    run_command(cmd)


def install_requirements():
    """Install required packages"""
    print("\nInstalling requirements...")

    requirements = [
        "aiohttp>=3.8.0",
        "aiofiles",
        "pillow>=9.0.0",
        "numpy>=1.20.0",
        "safetensors>=0.3.0",
        "transformers>=4.20.0",
        "accelerate",
        "xformers",
        "opencv-python",
        "scipy",
        "psutil",
        "requests",
        "pyyaml",
        "tqdm",
        "omegaconf",
        "diffusers",
        "controlnet-aux",
        "segment-anything",
        "timm",
        "spandrel",
    ]

    for req in requirements:
        run_command(f"pip install {req}")


def setup_directories():
    """Create necessary directories"""
    print("\nSetting up directories...")

    directories = [
        "models/checkpoints",
        "models/clip",
        "models/vae",
        "models/loras",
        "models/unet",
        "models/controlnet/sd15",
        "models/controlnet/sdxl",
        "models/embeddings",
        "models/diffusion_models",
        "models/upscale_models",
        "models/style_models",
        "output",
        "input",
        "temp",
        "custom_nodes",
        "web/extensions",
        "user",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")


def create_launch_scripts():
    """Create convenient launch scripts"""
    print("\nCreating launch scripts...")

    # Windows batch file
    if platform.system() == "Windows":
        with open("start_comfyui.bat", "w") as f:
            f.write(
                """@echo off
echo Starting ComfyUI Local Server...
python local_comfyui.py %*
pause
"""
            )
        print("Created: start_comfyui.bat")

        with open("start_comfyui_public.bat", "w") as f:
            f.write(
                """@echo off
echo Starting ComfyUI Local Server (Public Access)...
python local_comfyui.py --public --auto-launch %*
pause
"""
            )
        print("Created: start_comfyui_public.bat")

    # Unix shell script
    else:
        with open("start_comfyui.sh", "w") as f:
            f.write(
                """#!/bin/bash
echo "Starting ComfyUI Local Server..."
python3 local_comfyui.py "$@"
"""
            )
        os.chmod("start_comfyui.sh", 0o755)
        print("Created: start_comfyui.sh")

        with open("start_comfyui_public.sh", "w") as f:
            f.write(
                """#!/bin/bash
echo "Starting ComfyUI Local Server (Public Access)..."
python3 local_comfyui.py --public --auto-launch "$@"
"""
            )
        os.chmod("start_comfyui_public.sh", 0o755)
        print("Created: start_comfyui_public.sh")


def create_example_workflow():
    """Create an example workflow file"""
    print("\nCreating example workflow...")

    example_workflow = {
        "1": {
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
            "class_type": "EmptyLatentImage",
        },
        "2": {
            "inputs": {"text": "a beautiful landscape", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
        },
        "3": {"inputs": {"text": "", "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "4": {
            "inputs": {"ckpt_name": "model.safetensors"},
            "class_type": "CheckpointLoaderSimple",
        },
        "5": {
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["1", 0],
            },
            "class_type": "KSampler",
        },
        "6": {
            "inputs": {"samples": ["5", 0], "vae": ["4", 2]},
            "class_type": "VAEDecode",
        },
        "7": {
            "inputs": {"filename_prefix": "ComfyUI", "images": ["6", 0]},
            "class_type": "SaveImage",
        },
    }

    import json

    with open("example_workflow.json", "w") as f:
        json.dump(example_workflow, f, indent=2)
    print("Created: example_workflow.json")


def main():
    print("ComfyUI Local Setup")
    print("===================")

    # Check Python version
    check_python_version()

    # Check CUDA
    cuda_available = check_cuda()

    # Install PyTorch
    install_pytorch(cuda_available)

    # Install other requirements
    install_requirements()

    # Setup directories
    setup_directories()

    # Create launch scripts
    create_launch_scripts()

    # Create example workflow
    create_example_workflow()

    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nTo start ComfyUI:")

    if platform.system() == "Windows":
        print("  python local_comfyui.py")
        print("  or double-click start_comfyui.bat")
    else:
        print("  python3 local_comfyui.py")
        print("  or ./start_comfyui.sh")

    print("\nFor public access (accessible from other machines):")
    print("  python local_comfyui.py --public")

    print("\nNotes:")
    print("- Place your models in the models/ subdirectories")
    print("- Access the web interface at http://localhost:8188")
    print("- Check the log file: comfyui_local.log")
    print("- Modify local_config.yaml for custom settings")


if __name__ == "__main__":
    main()
