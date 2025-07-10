#!/usr/bin/env python3
"""
ComfyUI API Client
Easy API integration for controlling ComfyUI workflows
"""

import asyncio
import aiohttp
import json
import time
import base64
from typing import Dict, Any, Optional, List
from io import BytesIO
from PIL import Image


class ComfyUIAPI:
    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def queue_prompt(
        self, prompt: Dict[str, Any], client_id: str = "api_client"
    ) -> Dict[str, Any]:
        """Queue a workflow prompt for execution"""
        if not self.session:
            raise RuntimeError(
                "Use async context manager: async with ComfyUIAPI() as api:"
            )

        data = {"prompt": prompt, "client_id": client_id}

        async with self.session.post(f"{self.base_url}/prompt", json=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(
                    f"Failed to queue prompt: {response.status} - {error_text}"
                )

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        async with self.session.get(f"{self.base_url}/queue") as response:
            return await response.json()

    async def get_history(self, prompt_id: Optional[str] = None) -> Dict[str, Any]:
        """Get execution history"""
        url = f"{self.base_url}/history"
        if prompt_id:
            url += f"/{prompt_id}"
        async with self.session.get(url) as response:
            return await response.json()

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        async with self.session.get(f"{self.base_url}/system_stats") as response:
            return await response.json()

    async def interrupt_processing(self) -> None:
        """Interrupt current processing"""
        async with self.session.post(f"{self.base_url}/interrupt") as response:
            return response.status == 200

    async def clear_queue(self) -> None:
        """Clear the execution queue"""
        data = {"clear": True}
        async with self.session.post(f"{self.base_url}/queue", json=data) as response:
            return response.status == 200

    async def wait_for_completion(
        self, prompt_id: str, timeout: int = 300
    ) -> Dict[str, Any]:
        """Wait for a specific prompt to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            history = await self.get_history(prompt_id)

            if prompt_id in history:
                result = history[prompt_id]
                if result.get("status", {}).get("completed", False):
                    return result
                elif result.get("status", {}).get("status_str") == "error":
                    raise Exception(
                        f"Workflow failed: {result.get('status', {}).get('messages', [])}"
                    )

            await asyncio.sleep(1)

        raise TimeoutError(
            f"Workflow {prompt_id} did not complete within {timeout} seconds"
        )

    async def download_image(
        self, filename: str, subfolder: str = "", type: str = "output"
    ) -> bytes:
        """Download an image from ComfyUI"""
        params = {"filename": filename, "type": type}
        if subfolder:
            params["subfolder"] = subfolder

        async with self.session.get(f"{self.base_url}/view", params=params) as response:
            if response.status == 200:
                return await response.read()
            else:
                raise Exception(f"Failed to download image: {response.status}")


# Predefined workflow templates
class WorkflowTemplates:
    @staticmethod
    def text_to_image_basic(
        prompt: str = "a beautiful landscape",
        negative_prompt: str = "",
        model_name: str = "model.safetensors",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 8.0,
        seed: int = -1,
    ) -> Dict[str, Any]:
        """Basic text-to-image workflow"""
        return {
            "1": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage",
            },
            "2": {
                "inputs": {"text": prompt, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
            },
            "3": {
                "inputs": {"text": negative_prompt, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"ckpt_name": model_name},
                "class_type": "CheckpointLoaderSimple",
            },
            "5": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
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

    @staticmethod
    def text_to_image_with_lora(
        prompt: str = "a beautiful landscape",
        negative_prompt: str = "",
        model_name: str = "model.safetensors",
        lora_name: str = "lora.safetensors",
        lora_strength: float = 1.0,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 8.0,
        seed: int = -1,
    ) -> Dict[str, Any]:
        """Text-to-image workflow with LoRA"""
        return {
            "1": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage",
            },
            "2": {
                "inputs": {"text": prompt, "clip": ["8", 1]},
                "class_type": "CLIPTextEncode",
            },
            "3": {
                "inputs": {"text": negative_prompt, "clip": ["8", 1]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"ckpt_name": model_name},
                "class_type": "CheckpointLoaderSimple",
            },
            "5": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["8", 0],
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
            "8": {
                "inputs": {
                    "model": ["4", 0],
                    "clip": ["4", 1],
                    "lora_name": lora_name,
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength,
                },
                "class_type": "LoraLoader",
            },
        }


class ComfyUIMonitor:
    """Monitor ComfyUI execution with WebSocket"""

    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        self.ws_url = (
            base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        )
        self.callbacks = {}

    def on_progress(self, callback):
        """Register progress callback"""
        self.callbacks["progress"] = callback

    def on_executing(self, callback):
        """Register execution callback"""
        self.callbacks["executing"] = callback

    def on_executed(self, callback):
        """Register executed callback"""
        self.callbacks["executed"] = callback

    async def monitor(self, client_id: str = "monitor"):
        """Start monitoring"""
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"{self.ws_url}?clientId={client_id}") as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        event_type = data.get("type")

                        if event_type in self.callbacks:
                            await self.callbacks[event_type](data.get("data", {}))
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"WebSocket error: {ws.exception()}")
                        break


# Simple usage examples
async def simple_text_to_image(prompt: str, model_name: str = None):
    """Generate an image from text prompt"""
    # Get available models if none specified
    if not model_name:
        async with ComfyUIAPI() as api:
            # This would require additional API call to get models
            # For now, use a default
            model_name = "model.safetensors"

    # Create workflow
    workflow = WorkflowTemplates.text_to_image_basic(
        prompt=prompt, model_name=model_name, width=512, height=512, steps=20, cfg=8.0
    )

    async with ComfyUIAPI() as api:
        # Queue the workflow
        result = await api.queue_prompt(workflow)
        prompt_id = result["prompt_id"]
        print(f"✓ Queued workflow: {prompt_id}")

        # Wait for completion
        print("⏳ Waiting for completion...")
        history = await api.wait_for_completion(prompt_id)

        # Get output images
        outputs = history.get("outputs", {})
        images = []

        for node_id, output in outputs.items():
            if "images" in output:
                for img_info in output["images"]:
                    filename = img_info["filename"]
                    image_data = await api.download_image(filename)
                    images.append(image_data)
                    print(f"✓ Downloaded: {filename}")

        return images


async def monitor_progress_example():
    """Example of monitoring workflow progress"""
    monitor = ComfyUIMonitor()

    @monitor.on_progress
    async def on_progress(data):
        value = data.get("value", 0)
        max_val = data.get("max", 1)
        percentage = (value / max_val) * 100 if max_val > 0 else 0
        print(f"Progress: {percentage:.1f}% ({value}/{max_val})")

    @monitor.on_executing
    async def on_executing(data):
        node = data.get("node")
        if node:
            print(f"Executing node: {node}")

    # Start monitoring
    await monitor.monitor()


# Main example usage
async def main():
    """Example usage"""
    print("🚀 ComfyUI API Client Example")

    # Test connection
    async with ComfyUIAPI() as api:
        stats = await api.get_system_stats()
        print(f"✓ Connected to ComfyUI")
        print(f"  VRAM Free: {stats['devices'][0]['vram_free'] / (1024**3):.1f} GB")

        queue_status = await api.get_queue_status()
        print(f"  Queue: {len(queue_status['queue_pending'])} pending")

    # Generate an image
    print("\n🎨 Generating image...")
    images = await simple_text_to_image(
        prompt="a beautiful sunset over mountains, highly detailed, 8k",
        model_name="RealVisXL_V4.0.safetensors",  # Replace with your model
    )

    if images:
        print(f"✅ Generated {len(images)} image(s)")

        # Save first image
        with open("generated_image.png", "wb") as f:
            f.write(images[0])
        print("💾 Saved as: generated_image.png")


if __name__ == "__main__":
    asyncio.run(main())
