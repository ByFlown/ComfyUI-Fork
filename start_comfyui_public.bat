@echo off
echo Starting ComfyUI Local Server (Public Access)...
python local_comfyui.py --public --auto-launch %*
pause
