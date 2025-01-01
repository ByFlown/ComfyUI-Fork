# render.py - Place this file in your ComfyUI root directory
import os
import subprocess
import requests
from main import start_comfyui


def download_model(model_url, save_path):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    if not os.path.exists(save_path):
        print(f"Attempting to download model from: {model_url}")
        response = requests.get(model_url)
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Model downloaded and saved to {save_path}")
        else:
            print(f"Failed to download model: {response.status_code}")
    else:
        print("Model already exists, skipping download.")


def clone_repository(repo_url, destination):

    if not os.path.exists(destination):

        try:
            print(f"Cloning repository from: {repo_url} to {destination}")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, destination], check=False
            )
            print("Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
    else:
        print(f"Repo already exist at {destination}")


# Render specific configuration
PORT = int(os.environ.get("PORT", 8188))
HOST = "0.0.0.0"  # Listen on all available interfaces

if __name__ == "__main__":

    clone_repository(
        "https://github.com/Fannovel16/comfyui_controlnet_aux/",
        "./custom_nodes/controlnet",  # Specify the destination directory
    )

    clone_repository(
        "https://github.com/rgthree/rgthree-comfy.git",
        "./custom_nodes/rgthree",
    )

    download_model(
        "https://civitai.com/api/download/models/28100?type=Model&format=SafeTensor&size=full&fp=fp16",
        os.path.join(
            "./models/checkpoints", "animePastelDream_softBakedVae.safetensors"
        ),
    )

    download_model(
        "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth?download=true",
        os.path.join("./models/controlnet/sd15", "control_v11p_sd15_canny.pth"),
    )

    download_model(
        "https://civitai.com/api/download/models/229782?type=Model&format=SafeTensor",
        os.path.join("./models/loras/SD15", "hyperdetailer_v095.safetensors"),
    )

    download_model(
        "https://civitai.com/api/download/models/223773?type=Model&format=SafeTensor",
        os.path.join("./models/loras/SD15", "hyperrefiner_v090.safetensors"),
    )

    # Get event loop and start the server
    loop, server, start_server = start_comfyui()

    try:
        # Run the server with Render-specific host/port
        loop.run_until_complete(server.start(HOST, PORT))
        loop.run_forever()
    except KeyboardInterrupt:
        print("Stopping server")
    finally:
        loop.close()
