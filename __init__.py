import os
import shutil
import subprocess
import sys

import folder_paths

from .PixelArtDetector import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    from pyclustering.cluster import kmeans
    import cv2
except ImportError:
    print(f"## PixelArtDetector: installing dependencies")
    my_path = os.path.dirname(__file__)
    requirements_path = os.path.join(my_path, "requirements.txt")
    try:
        subprocess.check_call([sys.executable, '-s', '-m', 'pip', 'install', '-r', requirements_path])
    except:
        try:
            subprocess.check_call(
                [sys.executable, '-s', '-m', 'pip', 'install', '-r', requirements_path, '--use-pep517'])
        except:
            print(f"## [ERROR] PixelArtDetector: Could not install the needed libraries.")
    print(f"## PixelArtDetector: installing dependencies done.")

print("### Loading: PixelArtDetector")

comfy_path = os.path.dirname(folder_paths.__file__)

WEB_DIRECTORY = "./web/js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
