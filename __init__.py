import os
import platform
import shutil
import subprocess
import sys

import folder_paths


def install_pyclustering_wheel():
    """Install pyclustering from prebuilt wheel if not available or if installation fails from PyPI"""
    try:
        # Try to import pyclustering first
        from pyclustering.cluster import kmeans
        print("## PixelArtDetector: pyclustering already available")
        return True
    except ImportError:
        pass

    print("## PixelArtDetector: pyclustering not found, attempting to install from prebuilt wheel...")

    # Get Python version info
    PY_MAJOR = sys.version_info.major
    PY_MINOR = sys.version_info.minor
    PY_VER = f"{PY_MAJOR}.{PY_MINOR}"
    OS = platform.system().lower()
    ARCH = platform.machine().lower()
    TAG = "0.10.1.4"

    # Map Python version to the available wheels
    # Available wheels: cp310, cp311, cp312
    if PY_MAJOR == 3 and PY_MINOR in [10, 11, 12]:
        python_version = f"cp{PY_MAJOR}{PY_MINOR}"
    else:
        print(f"## [WARNING] Python {PY_VER} not directly supported. Using fallback approach.")
        # For other Python versions, try the closest available
        if PY_MINOR >= 12:
            python_version = "cp312"
        elif PY_MINOR >= 11:
            python_version = "cp311"
        else:
            python_version = "cp310"

    # Determine architecture - only x86_64 wheels are available in the provided links
    if ARCH in ["x86_64", "amd64"]:
        arch = "x86_64"
    else:
        print(
            f"## [WARNING] Architecture {ARCH} not directly supported. Using x86_64 wheels which may not work on your system.")
        arch = "x86_64"  # Use x86_64 as fallback since only these are provided

    # Determine OS-specific wheel
    if OS == "windows":
        wheel_arch = "win_amd64"
    elif OS == "linux":
        wheel_arch = "manylinux_2_28_x86_64"
    elif OS == "darwin":  # macOS
        wheel_arch = "macosx_11_0_x86_64"
    else:
        print(f"## [WARNING] Unsupported OS: {OS}. Attempting to continue with generic approach.")
        # Default to the most common format
        if ARCH in ["x86_64", "amd64"]:
            if OS == "darwin":
                wheel_arch = "macosx_11_0_x86_64"
            else:
                wheel_arch = "manylinux_2_28_x86_64"
        else:
            # For unsupported configurations, try the generic approach
            print(
                "## [WARNING] Unsupported platform configuration, installation may fail.")
            return False

    # Build the wheel URL with the correct Python version and architecture
    wheel = f"https://github.com/dimtoneff/pyclustering/releases/download/{TAG}/pyclustering-{TAG}-{python_version}-none-{wheel_arch}.whl"

    print(f"## PixelArtDetector: Installing pyclustering from wheel: {wheel}")
    print(f"## Platform info: OS={OS}, ARCH={ARCH}, Python={PY_VER}, Wheel_arch={wheel_arch}")

    try:
        # Try to install with pip
        result = subprocess.run([sys.executable, "-m", "pip", "install", wheel],
                                capture_output=True, text=True, check=True)
        print("## PixelArtDetector: pyclustering installed successfully from wheel")
        return True
    except subprocess.CalledProcessError as e:
        print(f"## [ERROR] PixelArtDetector: Failed to install pyclustering from wheel: {e}")
        print(f"## [ERROR] stderr: {e.stderr}" if hasattr(e, 'stderr') else "")
        print(f"## [ERROR] stdout: {e.stdout}" if hasattr(e, 'stdout') else "")
        return False


try:
    from pyclustering.cluster import kmeans
    import cv2
except ImportError as e:
    print(f"## PixelArtDetector: installing dependencies")
    my_path = os.path.dirname(__file__)
    requirements_path = os.path.join(my_path, "requirements.txt")
    # First, try to install pyclustering from a prebuilt wheel
    install_pyclustering_wheel()
    try:
        subprocess.check_call(
            [sys.executable, '-s', '-m', 'pip', 'install', '-r', requirements_path])
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
from .PixelArtDetector import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
