import shutil
import folder_paths
import os

print("### Loading: PixelArtDetector")

comfy_path = os.path.dirname(folder_paths.__file__)

def setup_js():
   webp_path = os.path.dirname(__file__)
   js_dest_path = os.path.join(comfy_path, "web", "extensions", "pixelArtDetector")
   js_src_path = os.path.join (webp_path, "pixelArtDetector")
     
   print("Copying PixelArtDetector JS files for Workflow loading")
   shutil.copytree (js_src_path, js_dest_path, dirs_exist_ok=True)        

setup_js()

from .PixelArtDetector import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']