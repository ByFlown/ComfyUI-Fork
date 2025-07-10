#!/usr/bin/env python3
"""
Utility script for ComfyUI local development
Provides model management, system diagnostics, and maintenance tools
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Optional
import logging


class ComfyUILocalUtils:
    def __init__(self):
        self.base_path = Path.cwd()
        self.models_path = self.base_path / "models"
        self.output_path = self.base_path / "output"
        self.temp_path = self.base_path / "temp"

    def setup_logging(self, verbose=False):
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def scan_models(self) -> Dict[str, List[str]]:
        """Scan all models and return organized info"""
        models = {}

        model_dirs = {
            "checkpoints": ["*.safetensors", "*.ckpt", "*.pth"],
            "vae": ["*.safetensors", "*.ckpt", "*.pth"],
            "loras": ["*.safetensors", "*.ckpt"],
            "controlnet": ["*.safetensors", "*.pth"],
            "clip": ["*.safetensors"],
            "unet": ["*.safetensors"],
            "embeddings": ["*.safetensors", "*.pt", "*.bin"],
        }

        for model_type, extensions in model_dirs.items():
            model_dir = self.models_path / model_type
            if model_dir.exists():
                files = []
                for ext in extensions:
                    files.extend(model_dir.rglob(ext))
                models[model_type] = [f.name for f in files]
            else:
                models[model_type] = []

        return models

    def get_model_info(self, model_path: Path) -> Optional[Dict]:
        """Get detailed information about a model file"""
        if not model_path.exists():
            return None

        info = {
            "name": model_path.name,
            "size": model_path.stat().st_size,
            "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
            "modified": model_path.stat().st_mtime,
            "extension": model_path.suffix,
        }

        # Calculate MD5 hash for smaller files
        if info["size"] < 100 * 1024 * 1024:  # 100MB
            info["md5"] = self.calculate_md5(model_path)

        return info

    def calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logging.error(f"Error calculating MD5 for {file_path}: {e}")
            return "error"

    def clean_temp_files(self):
        """Clean temporary files"""
        if self.temp_path.exists():
            try:
                shutil.rmtree(self.temp_path)
                self.temp_path.mkdir(exist_ok=True)
                logging.info("Cleaned temporary files")
            except Exception as e:
                logging.error(f"Error cleaning temp files: {e}")

    def check_disk_space(self) -> Dict[str, int]:
        """Check available disk space"""
        total, used, free = shutil.disk_usage(self.base_path)
        return {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "used_percent": round((used / total) * 100, 1),
        }

    def check_system_info(self) -> Dict:
        """Get system information"""
        info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "disk_space": self.check_disk_space(),
        }

        # Check CUDA
        try:
            import torch

            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_names"] = [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
        except ImportError:
            info["torch_installed"] = False

        return info

    def validate_workflow(self, workflow_file: str) -> bool:
        """Validate a workflow JSON file"""
        try:
            with open(workflow_file, "r") as f:
                workflow = json.load(f)

            # Basic validation
            if not isinstance(workflow, dict):
                logging.error("Workflow must be a JSON object")
                return False

            # Check for required structure
            for node_id, node_data in workflow.items():
                if not isinstance(node_data, dict):
                    logging.error(f"Node {node_id} must be an object")
                    return False

                if "class_type" not in node_data:
                    logging.error(f"Node {node_id} missing class_type")
                    return False

            logging.info(f"Workflow {workflow_file} is valid")
            return True

        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {workflow_file}: {e}")
            return False
        except Exception as e:
            logging.error(f"Error validating workflow: {e}")
            return False

    def backup_models(self, backup_dir: str):
        """Create backup of important models"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)

        important_dirs = ["checkpoints", "loras", "vae"]

        for dir_name in important_dirs:
            src_dir = self.models_path / dir_name
            dst_dir = backup_path / dir_name

            if src_dir.exists():
                logging.info(f"Backing up {dir_name}...")
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        logging.info(f"Backup completed to {backup_path}")

    def optimize_models(self):
        """Optimize model organization"""
        # Remove duplicate models based on hash
        models = self.scan_models()

        for model_type, model_list in models.items():
            if not model_list:
                continue

            model_dir = self.models_path / model_type
            file_hashes = {}
            duplicates = []

            for model_file in model_list:
                model_path = model_dir / model_file
                if model_path.stat().st_size > 1024 * 1024:  # Only check files > 1MB
                    file_hash = self.calculate_md5(model_path)

                    if file_hash in file_hashes:
                        duplicates.append(model_path)
                        logging.info(f"Duplicate found: {model_file}")
                    else:
                        file_hashes[file_hash] = model_path

            # Optionally remove duplicates (commented out for safety)
            # for duplicate in duplicates:
            #     duplicate.unlink()

        return duplicates

    def test_server_connection(self, host="localhost", port=8188):
        """Test connection to ComfyUI server"""
        try:
            response = requests.get(f"http://{host}:{port}/system_stats", timeout=5)
            if response.status_code == 200:
                logging.info("Server is running and responding")
                return True
            else:
                logging.error(f"Server responded with status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logging.error(f"Cannot connect to server: {e}")
            return False

    def generate_model_report(self, output_file="model_report.json"):
        """Generate comprehensive model report"""
        report = {
            "scan_time": __import__("datetime").datetime.now().isoformat(),
            "system_info": self.check_system_info(),
            "models": {},
        }

        models = self.scan_models()

        for model_type, model_list in models.items():
            report["models"][model_type] = {"count": len(model_list), "files": []}

            model_dir = self.models_path / model_type
            total_size = 0

            for model_file in model_list:
                model_path = model_dir / model_file
                model_info = self.get_model_info(model_path)
                if model_info:
                    report["models"][model_type]["files"].append(model_info)
                    total_size += model_info["size"]

            report["models"][model_type]["total_size_mb"] = round(
                total_size / (1024 * 1024), 2
            )

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logging.info(f"Model report saved to {output_file}")
        return report


def main():
    parser = argparse.ArgumentParser(description="ComfyUI Local Utilities")
    parser.add_argument(
        "command",
        choices=[
            "scan",
            "info",
            "clean",
            "validate",
            "backup",
            "optimize",
            "test",
            "report",
        ],
        help="Command to execute",
    )
    parser.add_argument("--file", help="File path for specific commands")
    parser.add_argument("--backup-dir", default="./backup", help="Backup directory")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8188, help="Server port")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    utils = ComfyUILocalUtils()
    utils.setup_logging(args.verbose)

    if args.command == "scan":
        models = utils.scan_models()
        print("\nModel Scan Results:")
        for model_type, model_list in models.items():
            print(f"  {model_type}: {len(model_list)} files")

    elif args.command == "info":
        info = utils.check_system_info()
        print("\nSystem Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    elif args.command == "clean":
        utils.clean_temp_files()

    elif args.command == "validate":
        if not args.file:
            print("Error: --file required for validate command")
            sys.exit(1)
        utils.validate_workflow(args.file)

    elif args.command == "backup":
        utils.backup_models(args.backup_dir)

    elif args.command == "optimize":
        duplicates = utils.optimize_models()
        print(f"Found {len(duplicates)} duplicate files")

    elif args.command == "test":
        utils.test_server_connection(args.host, args.port)

    elif args.command == "report":
        report = utils.generate_model_report()
        print(
            f"Generated model report with {sum(len(m['files']) for m in report['models'].values())} models"
        )


if __name__ == "__main__":
    main()
