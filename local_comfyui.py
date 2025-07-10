#!/usr/bin/env python3
"""
Local ComfyUI Server
A simplified version for local development with CUDA support
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path

# Add ComfyUI to path if needed
COMFYUI_ROOT = Path(__file__).parent
sys.path.insert(0, str(COMFYUI_ROOT))

# Try to import ComfyUI components
try:
    # First try to import from main if it has start_comfyui function
    from main import start_comfyui

    MAIN_IMPORT_SUCCESS = True
except (ImportError, AttributeError):
    MAIN_IMPORT_SUCCESS = False
    print("Warning: Could not import start_comfyui from main. Using fallback method.")

# Import ComfyUI components directly
try:
    import comfy.model_management
    import server
    import execution
    import nodes
    import folder_paths
    from app.logger import setup_logger
    from comfy.cli_args import args
except ImportError as e:
    print(f"Error importing ComfyUI modules: {e}")
    print(
        "Make sure you're running this from the ComfyUI directory with all dependencies installed."
    )
    sys.exit(1)


class LocalComfyUIServer:
    def __init__(self, host="127.0.0.1", port=8188, auto_launch=False):
        self.host = host
        self.port = port
        self.auto_launch = auto_launch
        self.loop = None
        self.server = None

    def setup_cuda(self):
        """Setup CUDA if available"""
        if comfy.model_management.is_device_available():
            device = comfy.model_management.get_torch_device()
            device_name = comfy.model_management.get_torch_device_name(device)
            logging.info(f"CUDA Device detected: {device_name}")
            logging.info(f"Using device: {device}")

            # Get memory info
            if device.type == "cuda":
                vram_total, torch_vram_total = comfy.model_management.get_total_memory(
                    device, torch_total_too=True
                )
                vram_free, torch_vram_free = comfy.model_management.get_free_memory(
                    device, torch_free_too=True
                )
                logging.info(f"VRAM Total: {vram_total / (1024**3):.1f} GB")
                logging.info(f"VRAM Free: {vram_free / (1024**3):.1f} GB")
        else:
            logging.warning("No CUDA device detected. Running on CPU.")

    def setup_logging(self, level=logging.INFO):
        """Setup logging for local development"""
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("comfyui_local.log"),
            ],
        )

    async def start_server(self):
        """Start the ComfyUI server"""
        try:
            # Setup CUDA
            self.setup_cuda()

            if MAIN_IMPORT_SUCCESS:
                # Use the main.py start_comfyui function
                self.loop, self.server, start_all_func = start_comfyui(
                    asyncio.get_event_loop()
                )
                await self.server.setup()
                await self.server.start_multi_address(
                    [(self.host, self.port)],
                    call_on_start=self._on_server_start,
                    verbose=True,
                )
                await self.server.publish_loop()
            else:
                # Fallback: Start ComfyUI manually
                await self.start_comfyui_manual()

        except KeyboardInterrupt:
            logging.info("Server stopped by user")
        except Exception as e:
            logging.error(f"Server error: {e}")
            raise

    async def start_comfyui_manual(self):
        """Manual ComfyUI startup when main import fails"""
        logging.info("Starting ComfyUI manually...")

        # Setup logging
        setup_logger(log_level="INFO", use_stdout=True)

        # Initialize nodes
        nodes.init_extra_nodes(init_custom_nodes=True)

        # Create server and queue
        self.loop = asyncio.get_event_loop()
        self.server = server.PromptServer(self.loop)
        q = execution.PromptQueue(self.server)

        # Add routes
        self.server.add_routes()

        # Setup server
        await self.server.setup()

        # Start prompt worker thread
        import threading

        def prompt_worker_thread():
            self.prompt_worker(q, self.server)

        threading.Thread(target=prompt_worker_thread, daemon=True).start()

        # Start server
        await self.server.start_multi_address(
            [(self.host, self.port)], call_on_start=self._on_server_start, verbose=True
        )

        # Start publish loop
        await self.server.publish_loop()

    def prompt_worker(self, q, server_instance):
        """Simplified prompt worker"""
        import time
        import gc

        e = execution.PromptExecutor(server_instance)

        while True:
            try:
                queue_item = q.get(timeout=1.0)
                if queue_item is not None:
                    item, item_id = queue_item
                    prompt_id = item[1]
                    server_instance.last_prompt_id = prompt_id

                    e.execute(item[2], prompt_id, item[3], item[4])

                    q.task_done(
                        item_id,
                        e.history_result,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str="success" if e.success else "error",
                            completed=e.success,
                            messages=e.status_messages,
                        ),
                    )

                    if server_instance.client_id is not None:
                        server_instance.send_sync(
                            "executing",
                            {"node": None, "prompt_id": prompt_id},
                            server_instance.client_id,
                        )

                # Handle flags
                flags = q.get_flags()
                if flags.get("unload_models", False):
                    comfy.model_management.unload_all_models()
                if flags.get("free_memory", False):
                    e.reset()
                    gc.collect()
                    comfy.model_management.soft_empty_cache()

            except Exception as e:
                logging.error(f"Error in prompt worker: {e}")
                continue

    def _on_server_start(self, scheme, address, port):
        """Callback when server starts"""
        if self.auto_launch:
            import webbrowser

            if address == "0.0.0.0":
                address = "127.0.0.1"
            webbrowser.open(f"{scheme}://{address}:{port}")

        logging.info(f"ComfyUI server running at {scheme}://{address}:{port}")
        logging.info("Press Ctrl+C to stop the server")

    def run(self):
        """Run the server"""
        try:
            asyncio.run(self.start_server())
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        except Exception as e:
            logging.error(f"Failed to start server: {e}")
            sys.exit(1)


def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import torch

        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA version: {torch.version.cuda}")
            logging.info(f"CUDA device count: {torch.cuda.device_count()}")
    except ImportError as e:
        logging.error(f"Missing PyTorch: {e}")


def setup_directories():
    """Ensure required directories exist"""
    directories = [
        "models/checkpoints",
        "models/clip",
        "models/vae",
        "models/loras",
        "models/unet",
        "models/controlnet",
        "models/embeddings",
        "output",
        "input",
        "temp",
        "custom_nodes",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logging.info("Directory structure verified")


def create_config_file():
    """Create a local configuration file if it doesn't exist"""
    config_path = Path("local_config.yaml")
    if not config_path.exists():
        config_content = """
# Local ComfyUI Configuration
server:
  host: "127.0.0.1"
  port: 8188
  auto_launch: false

cuda:
  device: "auto"  # auto, cpu, cuda:0, cuda:1, etc.
  memory_fraction: 0.8  # Fraction of VRAM to use

paths:
  models: "./models"
  output: "./output"
  input: "./input"
  temp: "./temp"
  custom_nodes: "./custom_nodes"

logging:
  level: "INFO"
  file: "comfyui_local.log"
"""
        with open(config_path, "w") as f:
            f.write(config_content.strip())
        logging.info(f"Created default config file: {config_path}")


def setup_directories():
    """Ensure required directories exist"""
    directories = [
        "models/checkpoints",
        "models/clip",
        "models/vae",
        "models/loras",
        "models/unet",
        "models/controlnet",
        "models/embeddings",
        "output",
        "input",
        "temp",
        "custom_nodes",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logging.info("Directory structure verified")


def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import torch
        import torchvision
        import torchaudio

        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA version: {torch.version.cuda}")
            logging.info(f"CUDA device count: {torch.cuda.device_count()}")
    except ImportError as e:
        logging.error(f"Missing dependency: {e}")
        sys.exit(1)


def create_config_file():
    """Create a local configuration file if it doesn't exist"""
    config_path = Path("local_config.yaml")
    if not config_path.exists():
        config_content = """
# Local ComfyUI Configuration
server:
  host: "127.0.0.1"
  port: 8188
  auto_launch: false

cuda:
  device: "auto"  # auto, cpu, cuda:0, cuda:1, etc.
  memory_fraction: 0.8  # Fraction of VRAM to use

paths:
  models: "./models"
  output: "./output"
  input: "./input"
  temp: "./temp"
  custom_nodes: "./custom_nodes"

logging:
  level: "INFO"
  file: "comfyui_local.log"
"""
        with open(config_path, "w") as f:
            f.write(config_content.strip())
        logging.info(f"Created default config file: {config_path}")


def main():
    # Handle ComfyUI's argument parsing first
    import comfy.options

    comfy.options.enable_args_parsing()

    parser = argparse.ArgumentParser(description="Local ComfyUI Server", add_help=False)
    parser.add_argument("--local-host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--local-port", type=int, default=8188, help="Port to bind to")
    parser.add_argument(
        "--local-auto-launch", action="store_true", help="Auto-launch browser"
    )
    parser.add_argument(
        "--local-public", action="store_true", help="Allow public access (0.0.0.0)"
    )
    parser.add_argument("--local-verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--help", "-h", action="help", help="Show this help message and exit"
    )

    # Parse only known args to avoid conflicts with ComfyUI args
    try:
        local_args, unknown = parser.parse_known_args()
    except:
        # Fallback to defaults if argument parsing fails
        class DefaultArgs:
            local_host = "127.0.0.1"
            local_port = 8188
            local_auto_launch = False
            local_public = False
            local_verbose = False

        local_args = DefaultArgs()

    # Setup logging
    log_level = logging.DEBUG if local_args.local_verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Set host for public access
    if local_args.local_public:
        local_args.local_host = "0.0.0.0"
        logging.warning("Running in public mode - accessible from other machines!")

    # Try to override ComfyUI's default listen and port if possible
    try:
        if hasattr(args, "listen") and local_args.local_host != "127.0.0.1":
            args.listen = local_args.local_host
        if hasattr(args, "port") and local_args.local_port != 8188:
            args.port = local_args.local_port
    except:
        pass  # Ignore if args modification fails

    # Check dependencies
    check_dependencies()

    # Setup directories
    setup_directories()

    # Create config file
    create_config_file()

    # Start server
    server_instance = LocalComfyUIServer(
        host=local_args.local_host,
        port=local_args.local_port,
        auto_launch=local_args.local_auto_launch,
    )

    logging.info("Starting local ComfyUI server...")
    server_instance.run()


if __name__ == "__main__":
    main()
