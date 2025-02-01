# render.py - Place this file in your ComfyUI root directory
import os
import subprocess
import requests
from main import start_comfyui


def download_model(model_url, save_path):
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    if not os.path.exists(save_path):
        print(f"Attempting to download model from: {model_url}")
        # Use streaming to handle large files with less memory
        response = requests.get(model_url, stream=True)
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                # Download in chunks to reduce memory usage
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Model downloaded and saved to {save_path}")
        else:
            print(f"Failed to download model: {response.status_code}")
    else:
        print("Model already exists, skipping download.")


def clone_repository(repo_url, destination):
    if not os.path.exists(destination):
        try:
            print(f"Cloning repository from: {repo_url} to {destination}")
            # Use --depth 1 and --single-branch to minimize downloaded data
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--single-branch",
                    "--no-tags",
                    "--filter=blob:none",
                    repo_url,
                    destination,
                ],
                check=False,
            )
            print("Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
    else:
        print(f"Repo already exists at {destination}")


def textToAnim():

    repositories = [
        (
            "https://github.com/Fannovel16/comfyui_controlnet_aux/",
            "./custom_nodes/controlnet",
        ),
        ("https://github.com/rgthree/rgthree-comfy.git", "./custom_nodes/rgthree"),
    ]

    for repo_url, destination in repositories:
        clone_repository(repo_url, destination)

    # Download models sequentially
    models = [
        (
            "https://civitai.com/api/download/models/28100?type=Model&format=SafeTensor&size=full&fp=fp16",
            "./models/checkpoints/animePastelDream_softBakedVae.safetensors",
        ),
        (
            "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth?download=true",
            "./models/controlnet/sd15/control_v11p_sd15_canny.pth",
        ),
        (
            "https://civitai.com/api/download/models/229782?type=Model&format=SafeTensor",
            "./models/loras/SD15/hyperdetailer_v095.safetensors",
        ),
        (
            "https://civitai.com/api/download/models/223773?type=Model&format=SafeTensor",
            "./models/loras/SD15/hyperrefiner_v090.safetensors",
        ),
    ]

    for model_url, save_path in models:
        download_model(model_url, save_path)

    # Get event loop and start the server
    loop, server, start_server = start_comfyui()

    try:
        loop.run_until_complete(server.start(HOST, PORT))
        loop.run_forever()
    except KeyboardInterrupt:
        print("Stopping server")
    finally:
        loop.close()


def textToAnimFromPreprocessedImgs():

    repositories = [
        (
            "https://github.com/Fannovel16/comfyui_controlnet_aux/",
            "./custom_nodes/controlnet",
        ),
        ("https://github.com/rgthree/rgthree-comfy.git", "./custom_nodes/rgthree"),
    ]

    for repo_url, destination in repositories:
        clone_repository(repo_url, destination)

    # Download models sequentially
    models = [
        (
            "https://civitai.com/api/download/models/28100?type=Model&format=SafeTensor&size=full&fp=fp16",
            "./models/checkpoints/animePastelDream_softBakedVae.safetensors",
        ),
        (
            "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-8steps-lora.safetensors?download=true",
            "./models/loras/HyperSD/SD15/Hyper-SD15-8steps-lora.safetensors",
        ),
        (
            "https://civitai.com/api/download/models/28609?type=Model&format=SafeTensor&size=full&fp=fp16",
            "./models/loras/SD15/animetarotV51.safetensors",
        ),
    ]

    for model_url, save_path in models:
        download_model(model_url, save_path)

    # Get event loop and start the server
    loop, server, start_server = start_comfyui()

    try:
        loop.run_until_complete(server.start(HOST, PORT))
        loop.run_forever()
    except KeyboardInterrupt:
        print("Stopping server")
    finally:
        loop.close()


# Render specific configuration
PORT = int(os.environ.get("PORT", 8188))
HOST = "0.0.0.0"

if __name__ == "__main__":
    # Clone repositories sequentially to avoid memory spikes

    textToAnimFromPreprocessedImgs()
