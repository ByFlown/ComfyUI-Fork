# render.py - Place this file in your ComfyUI root directory
import os
from main import start_comfyui

# Render specific configuration
PORT = int(os.environ.get("PORT", 8188))
HOST = "0.0.0.0"  # Listen on all available interfaces

if __name__ == "__main__":
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
