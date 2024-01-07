"""

Custom nodes for SDXL in ComfyUI

MIT License

Copyright (c) 2023 dimtoneff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

# implementation of https://github.com/Astropulse/pixeldetector to a ComfyUI extension node + other goodies
# by dimtoneff
from PIL import Image, ImageOps
import numpy as np
import hashlib
import nodes

import torch
from pathlib import Path
from comfy.cli_args import args
from enum import Enum

import os, json, time, folder_paths
from datetime import datetime
from .pixelUtils import *


class GRID_SETTING(Enum):
    FONT_SIZE = "font_size"
    FONT_COLOR = "font_color"
    BACKGROUND_COLOR = "background_color"
    COLS_NUM = "cols_num"
    ADD_BORDER = "grid_add_border"
    BORDER_WIDTH = "grid_border_width"


class SETTINGS(Enum):
    # it will resize the image if user settings are above this treshold
    MIN_RESIZE_TRESHOLD: int = 64


class PixelArtLoadPalettes(nodes.LoadImage):
    """
    A node that scans images in a directory and returns the palette for the seleced image or for all images to display in a Grid
    """
    # Set the directory where we get the palettes from
    INPUT_DIR = "1x/"
    CATEGORY = "image/PixelArt🕹️"
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("paletteList",)
    FUNCTION = "load_image"

    @classmethod
    def INPUT_TYPES(s):
        files = scanFilesInDir(os.path.normpath(os.path.join(getPalettesPath(), s.INPUT_DIR)))
        return {"required": {
            "image": (files,),
            "render_all_palettes_in_grid": (
                "BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            "grid_settings": ("STRING", {"multiline": True,
                                         "default": "Grid settings. The values will be forwarded to the 'PixelArt Palette Converter to render the grid with all palettes from this node.'"}),
            "paletteList_grid_font_size": ("INT", {"default": 40, "min": 14, "max": 120, "step": 1},),
            "paletteList_grid_font_color": ("STRING", {"multiline": False, "default": "#f40e12"}),
            "paletteList_grid_background": ("STRING", {"multiline": False, "default": "#fff"}),
            "paletteList_grid_cols": ("INT", {"default": 6, "min": 1, "max": 20, "step": 1},),
            "paletteList_grid_add_border": (
                "BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            "paletteList_grid_border_width": ("INT", {"default": 3, "min": 1, "max": 30, "step": 1},),
        },
        }

    def load_image(self, image, render_all_palettes_in_grid, grid_settings, paletteList_grid_font_size,
                   paletteList_grid_font_color, paletteList_grid_background,
                   paletteList_grid_cols, paletteList_grid_add_border, paletteList_grid_border_width
                   ):
        def _getImagePalette(imgName):
            image_path = os.path.normpath(os.path.join(getPalettesPath(), self.INPUT_DIR, imgName))
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("P")
            return image.getpalette()

        def _generateGridUserSettings():
            return {
                GRID_SETTING.FONT_SIZE: paletteList_grid_font_size,
                GRID_SETTING.FONT_COLOR: paletteList_grid_font_color,
                GRID_SETTING.BACKGROUND_COLOR: paletteList_grid_background,
                GRID_SETTING.COLS_NUM: paletteList_grid_cols,
                GRID_SETTING.ADD_BORDER: paletteList_grid_add_border,
                GRID_SETTING.BORDER_WIDTH: paletteList_grid_border_width,
            }

        palettes = list()
        if (render_all_palettes_in_grid):
            files = scanFilesInDir(os.path.normpath(os.path.join(getPalettesPath(), self.INPUT_DIR)))
            palettes = [
                {"p": _getImagePalette(file), "a": Path(file).stem, "grid_settings": _generateGridUserSettings()} for
                file in files]
        else:
            palettes.append({"p": _getImagePalette(image), "a": Path(image).stem})

        return (palettes,)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = os.path.normpath(os.path.join(getPalettesPath(), s.INPUT_DIR, image))
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        image_path = os.path.normpath(os.path.join(getPalettesPath(), s.INPUT_DIR, image))
        if not Path(image_path).is_file():
            return "Invalid image file: {}".format(image)

        return True


class PixelArtAddDitherPattern():
    """
    Add an ordered dither pattern to image.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "pattern_type": (["bayer", "halftone", "none"], {"default": "bayer"}),
            "pattern_order": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
            "amount": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.0005}),
        },
            "optional": {
                "custom_pattern": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    CATEGORY = "image/PixelArt🕹️"

    def _bayer_pattern_normalized(order):
        def _normalized_bayer(level):
            if level == 0:
                return np.zeros((1, 1), "float32")
            else:
                q = 4 ** level
                m = q * _normalized_bayer(level - 1)
                return np.bmat(((m - 1.5, m + 0.5), (m + 1.5, m - 0.5))) / q

        return torch.from_numpy(_normalized_bayer(order))

    def _halftone_45_degrees_pattern(order):
        size = 2 ** order
        pattern = np.full((size, size), -1, "int")  # Mark cells initially unset.

        v = 0

        # Plot simultaneously at two dot origins. Input (xf,yf) is 1-bit fixed point.
        def plot(xf, yf):
            nonlocal v
            for o in (0, size):
                x, y = ((o + xf) // 2) % size, ((o + yf) // 2) % size
                if pattern[y][x] == -1:
                    pattern[y][x] = v
                    v += 1

        # Concentric Bresenham circles.
        for r in range(1, int(1.41421 * size)):
            t1 = r * 2 // 16  # To 1-bit fixed point, i.e. increment radius by 0.5 per iteration.
            x, y = r, 0
            while x >= y:
                # Order is relevant, scatters threshold increments more evenly.
                plot(+y, +x)
                plot(+x, -y - 1)
                plot(-y - 1, -x - 1)
                plot(-x - 1, +y)
                plot(-y - 1, +x)
                plot(+x, +y)
                plot(+y, -x - 1)
                plot(-x - 1, -y - 1)
                y += 1
                t1 = t1 + y
                t2 = t1 - x
                if t2 >= 0:
                    t1 = t2
                    x -= 1

        pattern = torch.tensor(pattern, dtype=torch.float32)
        return pattern

    # Input pattern values must be >= 0.
    def _normalize_pattern(pattern):
        pattern = pattern / pattern.amax(dim=(0, 1), keepdim=True) - 0.5  # Normalize to range [-0.5,0.5].
        order = math.log2(math.sqrt(pattern.shape[0] * pattern.shape[1]))  # Calculate order from sides.
        pattern = pattern * (1 - 0.5 ** (2 * order))  # Rescale range to order.
        return pattern

    def process(self, image, pattern_type, pattern_order, amount, custom_pattern=None):
        if custom_pattern != None:
            pattern = custom_pattern.squeeze(0)
            pattern = PixelArtAddDitherPattern._normalize_pattern(pattern)
        else:
            if pattern_type == "bayer":
                pattern = PixelArtAddDitherPattern._bayer_pattern_normalized(pattern_order)
            elif pattern_type.startswith("halftone"):
                pattern = PixelArtAddDitherPattern._halftone_45_degrees_pattern(pattern_order)
                pattern = PixelArtAddDitherPattern._normalize_pattern(pattern)
            else:
                pattern = torch.zeros((4, 4), dtype=torch.float32)  # Zero pattern.

        pattern = pattern * amount

        tw = math.ceil(image.shape[1] / pattern.shape[0])
        th = math.ceil(image.shape[2] / pattern.shape[1])
        tiled_pattern = pattern.tile(tw, th).unsqueeze(-1).unsqueeze(0)

        result = (image + tiled_pattern[:, :image.shape[1], :image.shape[2]]).clamp_(0, 1)
        return (result,)


class PixelArtDetectorConverter():
    """
    A node that can convert images to some fan favorite palettes: NES, GAME BOY etc.
    """

    def __init__(self):
        self.CGREEN = '\033[92m'
        self.CYELLOW = '\033[93m'
        self.CEND = '\033[0m'
        self.GAME_BOY_PALETTE_TUPLES = [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)]  # ,(202,220,159)
        self.NES_PALETTE_TUPLES = [(124, 124, 124), (0, 0, 252), (0, 0, 188), (68, 40, 188), (148, 0, 132),
                                   (168, 0, 32), (168, 16, 0), (136, 20, 0), (80, 48, 0), (0, 120, 0), (0, 104, 0),
                                   (0, 88, 0), (0, 64, 88), (0, 0, 0), (0, 0, 0), (0, 0, 0), (188, 188, 188),
                                   (0, 120, 248), (0, 88, 248), (104, 68, 252), (216, 0, 204), (228, 0, 88),
                                   (248, 56, 0),
                                   (228, 92, 16), (172, 124, 0), (0, 184, 0), (0, 168, 0), (0, 168, 68), (0, 136, 136),
                                   (0, 0, 0), (0, 0, 0), (0, 0, 0), (248, 248, 248), (60, 188, 252), (104, 136, 252),
                                   (152, 120, 248), (248, 120, 248), (248, 88, 152), (248, 120, 88), (252, 160, 68),
                                   (248, 184, 0), (184, 248, 24), (88, 216, 84), (88, 248, 152), (0, 232, 216),
                                   (120, 120, 120), (0, 0, 0), (0, 0, 0), (252, 252, 252), (164, 228, 252),
                                   (184, 184, 248),
                                   (216, 184, 248), (248, 184, 248), (248, 164, 192), (240, 208, 176), (252, 224, 168),
                                   (248, 216, 120), (216, 248, 120), (184, 248, 184), (184, 248, 216), (0, 252, 252),
                                   (248, 216, 248), (0, 0, 0), (0, 0, 0)
                                   ]
        self.GAME_BOY = [15, 56, 15, 48, 98, 48, 139, 172, 15, 155, 188, 15]
        self.NES = [
            124, 124, 124, 0, 0, 252, 0, 0, 188, 68, 40, 188, 148, 0, 132, 168, 0, 32, 168, 16, 0, 136, 20, 0, 80, 48,
            0, 0, 120, 0, 0, 104, 0, 0, 88, 0, 0, 64, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 188, 188, 188, 0, 120, 248,
            0, 88, 248, 104, 68, 252, 216, 0, 204, 228, 0, 88, 248, 56, 0, 228, 92, 16, 172, 124, 0, 0, 184, 0, 0, 168,
            0, 0, 168, 68, 0, 136, 136, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 248, 248, 60, 188, 252, 104, 136, 252,
            152, 120, 248, 248, 120, 248, 248, 88, 152, 248, 120, 88, 252, 160, 68, 248, 184, 0, 184, 248, 24, 88, 216,
            84, 88, 248, 152, 0, 232, 216, 120, 120, 120, 0, 0, 0, 0, 0, 0, 252, 252, 252, 164, 228, 252,
            184, 184, 248, 216, 184, 248, 248, 184, 248, 248, 164, 192, 240, 208, 176, 252, 224, 168, 248, 216, 120,
            216, 248, 120, 184, 248, 184, 184, 248, 216, 0, 252, 252, 248, 216, 248, 0, 0, 0, 0, 0, 0
        ]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "palette": (["NES", "GAMEBOY"], {"default": "GAMEBOY"}),
            "pixelize": (
                ["Image.quantize", "Grid.pixelate", "NP.quantize", "OpenCV.kmeans.reduce"],
                {"default": "Image.quantize"}),
            "grid_pixelate_grid_scan_size": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1},),
            "resize_w": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1},),
            "resize_h": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1},),
            "reduce_colors_before_palette_swap": (
                "BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            "reduce_colors_max_colors": ("INT", {"default": 128, "min": 1, "max": 256, "step": 1},),
            "apply_pixeldetector_max_colors": (
                "BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            "image_quantize_reduce_method": (["MAXCOVERAGE", "MEDIANCUT", "FASTOCTREE"], {"default": "MAXCOVERAGE"}),
            "opencv_settings": (
                "STRING", {"multiline": True, "default": "OpenCV.kmeans: only when reducing is enabled.\n" +
                                                         "RANDOM_CENTERS: Fast but doesn't guarantee same labels for the same image.\n" +
                                                         "PP_CENTERS: Slow but will yield optimum and consistent results for same input image.\n" +
                                                         "attempts: to run criteria_max_iterations so it gets the best labels. Increasing this value will slow down the runtime a lot, but improves the colors!\n"
                           }),
            "opencv_kmeans_centers": (["RANDOM_CENTERS", "PP_CENTERS"], {"default": "RANDOM_CENTERS"}),
            "opencv_kmeans_attempts": ("INT", {"default": 10, "min": 1, "max": 150, "step": 1},),
            "opencv_criteria_max_iterations": ("INT", {"default": 10, "min": 1, "max": 150, "step": 1},),
            "cleanup": ("STRING", {"multiline": True,
                                   "default": "Clean up colors: Iterate and eliminate pixels while there was none left covering less than the 'cleanup_pixels_threshold' of the image.\n" +
                                              "Optionally, enable the 'reduce colors' option, which runs before this cleanup. Good cleanup_threshold values: between .01 & .05"
                                   }),
            "cleanup_colors": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            "cleanup_pixels_threshold": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 1.0, "step": 0.001}),
            "dither": (["none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"],),
        },
            "optional": {
                "paletteList": ("LIST", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    CATEGORY = "image/PixelArt🕹️"
    OUTPUT_IS_LIST = (True,)

    def process(self, images, palette, pixelize, grid_pixelate_grid_scan_size, resize_w, resize_h,
                reduce_colors_before_palette_swap, reduce_colors_max_colors, apply_pixeldetector_max_colors,
                image_quantize_reduce_method, opencv_settings, opencv_kmeans_centers, opencv_kmeans_attempts,
                opencv_criteria_max_iterations, cleanup, cleanup_colors, cleanup_pixels_threshold, dither, paletteList=None
                ):
        isGrid = (paletteList is not None and len(paletteList) > 1)

        # Add a default palette
        if palette == "NES":
            palette = self.NES
        else:
            palette = self.GAME_BOY

        # Non grid input
        if paletteList is not None and not isGrid and len(paletteList):
            palette = paletteList[0].get("p")

        results = list()
        for image in images:
            pilImage = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")
            # resize(Upscale) if image size is less than the user given size
            resizeBefore = (pilImage.width < resize_w and pilImage.height < resize_h)

            # resize if image needs upscale
            if resizeBefore and resize_w >= SETTINGS.MIN_RESIZE_TRESHOLD.value and resize_h >= SETTINGS.MIN_RESIZE_TRESHOLD.value:
                pilImage = pilImage.resize((resize_w, resize_h), resample=Image.Resampling.NEAREST)
                print(f"### {self.CGREEN}[PixelArtDetectorConverter]{self.CEND} Image resized before reducing and quantizing!")

            if reduce_colors_before_palette_swap and not isGrid:
                # Start timer
                start = round(time.time() * 1000)
                best_k = determine_best_k(pixel_detect(pilImage),
                                          reduce_colors_max_colors) if apply_pixeldetector_max_colors else reduce_colors_max_colors
                if pixelize == "Image.quantize":
                    pilImage = pilImage.quantize(colors=best_k, dither=Image.Dither.NONE, kmeans=best_k, method=getQuantizeMethod(image_quantize_reduce_method)).convert('RGB')
                    print(
                        f"### {self.CGREEN}[PixelArtDetectorConverter]{self.CEND} Image colors reduced with {self.CYELLOW}Image.quantize{self.CEND} in {self.CYELLOW}{round(time.time() * 1000) - start}{self.CEND} milliseconds. Quantize method: {self.CYELLOW}{image_quantize_reduce_method}{self.CEND}. KMeans/Best_K: {self.CYELLOW}{best_k}{self.CEND}")
                else:
                    # Use OpenCV to reduce the colors of the image
                    cv2 = convert_from_image_to_cv2(pilImage)
                    cv2 = cv2_quantize(cv2, best_k, get_cv2_kmeans_flags(opencv_kmeans_centers), opencv_kmeans_attempts,
                                       opencv_criteria_max_iterations)
                    pilImage = convert_from_cv2_to_image(cv2)
                    print(
                        f"### {self.CGREEN}[PixelArtDetectorConverter]{self.CEND} Image colors reduced with {self.CYELLOW}OpenCV.kmeans{self.CEND} in {self.CYELLOW}{round(time.time() * 1000) - start}{self.CEND} milliseconds. Best_K: {self.CYELLOW}{best_k}{self.CEND}")

            if cleanup_colors and not isGrid:
                # Start timer
                start = round(time.time() * 1000)
                pilImage = cleanupColors(pilImage, cleanup_pixels_threshold, reduce_colors_max_colors, getQuantizeMethod(image_quantize_reduce_method))
                print(f"### {self.CGREEN}[PixelArtDetectorConverter]{self.CEND} Pixels clean up finished in {self.CYELLOW}{round(time.time() * 1000) - start}{self.CEND} milliseconds.")

            # Start timer
            start = round(time.time() * 1000)

            if not isGrid and dither.startswith("bayer"):
                order = int(dither.split('-')[-1])
                pilImage = ditherBayer(pilImage, transformPalette(palette, "image"), order)
                print(f"### {self.CGREEN}[PixelArtDetectorConverter]{self.CEND} Dither {dither} applied to the image!")
            elif not isGrid and dither == "floyd-steinberg" and not (pixelize == "Image.quantize" or pixelize == "OpenCV.kmeans.reduce"):
                # we need to pass a palette to enable dithering. this does NOT quantize
                pilImage = pilImage.quantize(palette=transformPalette(palette, "image"), dither=Image.Dither.FLOYDSTEINBERG).convert('RGB')
                print(f"### {self.CGREEN}[PixelArtDetectorConverter]{self.CEND} Dither {dither} applied to the image!")

            if isGrid:
                PILOutput = self.genImagesForGrid(pilImage, paletteList)
            # Swap palette
            else:
                if pixelize == "Image.quantize" or pixelize == "OpenCV.kmeans.reduce":
                    PILOutput = pilImage.quantize(palette=transformPalette(palette, "image"),
                                                  dither=(Image.Dither.FLOYDSTEINBERG if dither == "floyd-steinberg" else Image.Dither.NONE)).convert('RGB')
                    if dither == "floyd-steinberg":
                        print(f"### {self.CGREEN}[PixelArtDetectorConverter]{self.CEND} Dither {dither} applied to the image before quantization!")
                elif pixelize == "NP.quantize":
                    PILOutput = npQuantize(
                        pilImage,
                        transformPalette(palette, "tuple"))
                else:
                    PILOutput = pixelate(pilImage, grid_pixelate_grid_scan_size, transformPalette(palette, "tuple"))

            print(
                f"### {self.CGREEN}[PixelArtDetectorConverter]{self.CEND} Image converted in {self.CYELLOW}{round(time.time() * 1000) - start}{self.CEND} milliseconds.")

            # resize if image needs downscale
            if not resizeBefore and not isGrid and resize_w >= SETTINGS.MIN_RESIZE_TRESHOLD.value and resize_h >= SETTINGS.MIN_RESIZE_TRESHOLD.value:
                PILOutput = PILOutput.resize((resize_w, resize_h), resample=Image.Resampling.NEAREST)

            # Convert to torch.Tensor
            PILOutput = np.array(PILOutput).astype(np.float32) / 255.0
            PILOutput = torch.from_numpy(PILOutput)[None,]
            results.append(PILOutput)

        return (results,)

    def genImagesForGrid(self, image: Image, paletteList: list[dict], fontSize: int = 40, fontColor: str = "#f40e12",
                         gridBackground: str = "#fff", gridCols: int = 6, addBorder: bool = True,
                         borderWidth: int = 3) -> Image:
        def _parseGridUserSettings(g: dict):
            return g.get(GRID_SETTING.FONT_SIZE, fontSize), g.get(GRID_SETTING.FONT_COLOR, fontColor), g.get(
                GRID_SETTING.BACKGROUND_COLOR, gridBackground), g.get(GRID_SETTING.COLS_NUM, gridCols), g.get(
                GRID_SETTING.ADD_BORDER, addBorder), g.get(GRID_SETTING.BORDER_WIDTH, borderWidth)

        print(
            f"### {self.CGREEN}[PixelArtDetectorConverter]{self.CEND} Creating a grid with {self.CYELLOW}Image.quantized{self.CEND} converted images!")

        fontSize, fontColor, gridBackground, gridCols, addBorder, borderWidth = _parseGridUserSettings(
            paletteList[0].get("grid_settings", {}))
        images = list()

        for d in paletteList:
            palette = d.get("p")
            annotation = d.get("a")
            img = image.quantize(palette=transformPalette(palette, "image"), dither=Image.Dither.NONE).convert('RGB')
            drawTextInImage(img, annotation, fontSize, fontColor, strokeColor=gridBackground)
            images.append(img)

        return smart_grid_image(images=images, cols=gridCols, add_border=addBorder, border_color=gridBackground,
                                border_width=borderWidth)


class PixelArtDetectorToImage:
    """
    A node that can output the processed PixelArt image to a torchTensor (IMAGE) for furhter processing
    """

    def __init__(self):
        self.CGREEN = '\033[92m'
        self.CYELLOW = '\033[93m'
        self.CEND = '\033[0m'

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "reduce_palette": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            "reduce_palette_max_colors": ("INT", {"default": 128, "min": 1, "max": 256, "step": 1},),
        },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    CATEGORY = "image/PixelArt🕹️"
    OUTPUT_IS_LIST = (True,)

    def process(self, images, reduce_palette, reduce_palette_max_colors):
        results = list()
        for image in images:
            pilImage = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")

            # Start timer
            start = round(time.time() * 1000)

            # Find 1:1 pixel scale
            downscale = pixel_detect(pilImage)

            print(
                f"### {self.CGREEN}[PixelArtDetectorToImage]{self.CEND} Size detected and reduced from {self.CYELLOW}{pilImage.width}{self.CEND}x{self.CYELLOW}{pilImage.height}{self.CEND} to {self.CYELLOW}{downscale.width}{self.CEND}x{self.CYELLOW}{downscale.height}{self.CEND} in {self.CYELLOW}{round(time.time() * 1000) - start}{self.CEND} milliseconds")

            PILOutput = downscale

            if reduce_palette:
                print(
                    f"### {self.CGREEN}[PixelArtDetectorToImage]{self.CEND} Reduce pallete max_colors: {self.CYELLOW}{reduce_palette_max_colors}{self.CEND}")
                # Start timer
                start = round(time.time() * 1000)
                # Reduce color palette using elbow method
                PILOutput, best_k = reducePalette(downscale, reduce_palette_max_colors)
                print(
                    f"### {self.CGREEN}[PixelArtDetectorToImage]{self.CEND} Palette reduced to {self.CYELLOW}{best_k}{self.CEND} colors in {self.CYELLOW}{round(time.time() * 1000) - start}{self.CEND} milliseconds")

            PILOutput = np.array(PILOutput).astype(np.float32) / 255.0
            PILOutput = torch.from_numpy(PILOutput)[None,]
            results.append(PILOutput)

        return (results,)


class PixelArtDetectorSave:
    """
    A node that can save the processed PixelArt to different formats (WEBP, JPEG etc.)
    """

    def __init__(self):
        self.type = "output"
        self.CGREEN = '\033[92m'
        self.CYELLOW = '\033[93m'
        self.CEND = '\033[0m'

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "%date%/PixelArt"}),
                "reduce_palette": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "reduce_palette_max_colors": ("INT", {"default": 128, "min": 1, "max": 256, "step": 1},),
                "webp_mode": (["lossy", "lossless"],),
                "compression": ("INT", {"default": 80, "min": 1, "max": 100, "step": 1},),
                "save_jpg": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "save_exif": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "resize_w": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1},),
                "resize_h": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1},),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "process"

    OUTPUT_NODE = True

    CATEGORY = "image/PixelArt🕹️"

    def process(self, images, reduce_palette, reduce_palette_max_colors, filename_prefix, webp_mode, compression,
                resize_w, resize_h, prompt=None, extra_pnginfo=None, save_jpg=False, save_exif=True):

        results = list()
        for image in images:
            # Convert to PIL Image
            pilImage = tensor2pil(image)

            # Start timer
            start = round(time.time() * 1000)

            # Find 1:1 pixel scale
            downscale = pixel_detect(pilImage)

            print(
                f"### {self.CGREEN}[PixelArtDetectorSave]{self.CEND} Size detected and reduced from {self.CYELLOW}{pilImage.width}{self.CEND}x{self.CYELLOW}{pilImage.height}{self.CEND} to {self.CYELLOW}{downscale.width}{self.CEND}x{self.CYELLOW}{downscale.height}{self.CEND} in {self.CYELLOW}{round(time.time() * 1000) - start}{self.CEND} milliseconds")

            PILOutput = downscale

            if reduce_palette:
                print(
                    f"### {self.CGREEN}[PixelArtDetectorSave]{self.CEND} Reduce pallete max_colors: {self.CYELLOW}{reduce_palette_max_colors}{self.CEND}")
                # Start timer
                start = round(time.time() * 1000)
                PILOutput, best_k = reducePalette(downscale, reduce_palette_max_colors)
                print(
                    f"### {self.CGREEN}[PixelArtDetectorSave]{self.CEND} Palette reduced to {self.CYELLOW}{best_k}{self.CEND} colors in {self.CYELLOW}{round(time.time() * 1000) - start}{self.CEND} milliseconds")

            # resize
            if resize_w >= SETTINGS.MIN_RESIZE_TRESHOLD.value and resize_h >= SETTINGS.MIN_RESIZE_TRESHOLD.value:
                PILOutput = PILOutput.resize((resize_w, resize_h), resample=Image.Resampling.NEAREST)

            results.append(self.saveImage(
                PILOutput,
                filename_prefix,
                prompt,
                webp_mode,
                save_exif,
                save_jpg,
                extra_pnginfo,
                compression
            ))

        return {"ui": {"images": results}}

    def saveImage(self, output, filename_prefix, prompt, webp_mode, save_exif, save_jpg, extra_pnginfo, compression):
        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)

        def compute_vars(input):
            input = input.replace("%date%", datetime.now().strftime("%Y-%m-%d"))
            return input

        output_dir = folder_paths.get_output_directory()
        filename_prefix = compute_vars(filename_prefix)
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(output_dir, subfolder)

        try:
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                                 map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1

        workflowmetadata = str()
        promptstr = str()
        imgexif = output.getexif()  # get the (empty) Exif data of the generated Picture

        if not args.disable_metadata and save_exif:
            if prompt is not None:
                promptstr = "".join(json.dumps(prompt))  # prepare prompt String
                imgexif[0x010f] = "Prompt:" + promptstr  # Add PromptString to EXIF position 0x010f (Exif.Image.Make)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    workflowmetadata += "".join(json.dumps(extra_pnginfo[x]))
            imgexif[
                0x010e] = "Workflow:" + workflowmetadata  # Add Workflowstring to EXIF position 0x010e (Exif.Image.ImageDescription)

        file = f"{filename}_{counter:05}_"

        if webp_mode == "lossless":
            boolloss = True
        if webp_mode == "lossy":
            boolloss = False

        output.save(os.path.join(full_output_folder, file + ".webp"), method=6, exif=imgexif, lossless=boolloss,
                    quality=compression)  # Save as webp - options to be determined
        if save_jpg:
            output.save(os.path.join(full_output_folder, file + ".jpeg"), exif=imgexif,
                        quality=compression)  # Save as jpeg

        print(
            f"### {self.CGREEN}[PixelArtDetectorSave]{self.CEND} Saving file to {self.CYELLOW}{full_output_folder}{self.CEND} Filename: {self.CYELLOW}{file}{self.CEND}")

        return {
            "filename": file + ".webp",
            "subfolder": subfolder,
            "type": self.type
        }


from sklearn.cluster import KMeans

class PixelArtPaletteGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("INT", {"default": 16, "min": 8, "max": 256, "step": 1}),
                "mode": (["Chart", "back_to_back"],),
            },
        }

    RETURN_TYPES = ("IMAGE","LIST")
    RETURN_NAMES = ("image","color_palettes")
    FUNCTION = "image_generate_palette"

    CATEGORY = "image/PixelArt🕹️"

    # Tensor to PIL
    def tensor2pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    # PIL to Tensor
    def pil2tensor(self, image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def generate_palette(self, img, n_colors=16, cell_size=128, padding=0, font_path=None, font_size=15, mode='chart'):
        img = img.resize((img.width // 2, img.height // 2), resample=Image.BILINEAR)
        pixels = np.array(img)
        pixels = pixels.reshape((-1, 3))
        kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init='auto').fit(pixels)
        cluster_centers = np.uint8(kmeans.cluster_centers_)

        # Get the sorted indices based on luminance
        luminance = np.sqrt(np.dot(cluster_centers, [0.299, 0.587, 0.114]))
        sorted_indices = np.argsort(luminance)

        # Rearrange the cluster centers and luminance based on sorted indices
        cluster_centers = cluster_centers[sorted_indices]
        luminance = luminance[sorted_indices]

        # Group colors by their individual types
        reds = []
        greens = []
        blues = []
        others = []

        for i in range(n_colors):
            color = cluster_centers[i]
            color_type = np.argmax(color)  # Find the dominant color component

            if color_type == 0:
                reds.append((color, luminance[i]))
            elif color_type == 1:
                greens.append((color, luminance[i]))
            elif color_type == 2:
                blues.append((color, luminance[i]))
            else:
                others.append((color, luminance[i]))

        # Sort each color group by luminance
        reds.sort(key=lambda x: x[1])
        greens.sort(key=lambda x: x[1])
        blues.sort(key=lambda x: x[1])
        others.sort(key=lambda x: x[1])

        # Combine the sorted color groups
        sorted_colors = reds + greens + blues + others

        if mode == 'back_to_back':
            # Calculate the size of the palette image based on the number of colors
            palette_width = n_colors * cell_size
            palette_height = cell_size
        else:
            # Calculate the number of rows and columns based on the number of colors
            num_rows = int(np.sqrt(n_colors))
            num_cols = int(np.ceil(n_colors / num_rows))

            # Calculate the size of the palette image based on the number of rows and columns
            palette_width = num_cols * cell_size
            palette_height = num_rows * cell_size

        palette_size = (palette_width, palette_height)

        palette = Image.new('RGB', palette_size, color='white')
        draw = ImageDraw.Draw(palette)
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        hex_palette = []
        for i, (color, _) in enumerate(sorted_colors):
            if mode == 'back_to_back':
                cell_x = i * cell_size
                cell_y = 0
            else:
                row = i % num_rows
                col = i // num_rows
                cell_x = col * cell_size
                cell_y = row * cell_size

            cell_width = cell_size
            cell_height = cell_size

            color = tuple(color)

            cell = Image.new('RGB', (cell_width, cell_height), color=color)
            palette.paste(cell, (cell_x, cell_y))

            if mode != 'back_to_back':
                text_x = cell_x + (cell_width / 2)
                text_y = cell_y + cell_height + padding

                draw.text((text_x + 1, text_y + 1), f"R: {color[0]} G: {color[1]} B: {color[2]}", font=font, fill='black', anchor='ms')
                draw.text((text_x, text_y), f"R: {color[0]} G: {color[1]} B: {color[2]}", font=font, fill='white', anchor='ms')

            hex_palette.append('#%02x%02x%02x' % color)

        return palette, '\n'.join(hex_palette)

    def image_generate_palette(self, image, colors=16, mode="chart"):
        cn_root = os.path.dirname(os.path.abspath(__file__))
        res_dir = os.path.join(cn_root, 'res')
        font = os.path.join(res_dir, 'font.ttf')

        if not os.path.exists(font):
            font = None
        else:
            if mode == "Chart":
                cstr(f'Found font at `{font}`').msg.print()

        if len(image) > 1:
            palette_list = []
            for img in image:
                img = self.tensor2pil(img)
                palette_image, hex_palette = self.generate_palette(img, colors, 128, 10, font, 15, mode.lower())
                palette_bytes = self.hex_palette_to_bytes(hex_palette)
                palette_dict = {"p": palette_bytes, "a": "Arbitrary identifier or annotation."}
                palette_list.append(palette_dict)
            return (torch.cat([self.pil2tensor(palette_image) for palette_image, _ in palette_list], dim=0), palette_list)
        else:
            image = self.tensor2pil(image[0])
            palette_image, hex_palette = self.generate_palette(image, colors, 128, 10, font, 15, mode.lower())
            palette_bytes = self.hex_palette_to_bytes(hex_palette)
            palette_dict = {"p": palette_bytes, "a": "Arbitrary identifier or annotation."}
            return (self.pil2tensor(palette_image), [palette_dict,])

    def hex_palette_to_bytes(self, hex_palette):
        """Convert a hex color palette to bytes."""
        hex_colors = hex_palette.split('\n')
        palette_bytes = []
        for hex_color in hex_colors:
            # Remove the '#' character and convert to integer RGB values.
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
            palette_bytes.extend(rgb)
        return bytes(palette_bytes)


NODE_CLASS_MAPPINGS = {
    "PixelArtAddDitherPattern": PixelArtAddDitherPattern,
    "PixelArtDetectorSave": PixelArtDetectorSave,
    "PixelArtDetectorToImage": PixelArtDetectorToImage,
    "PixelArtDetectorConverter": PixelArtDetectorConverter,
    "PixelArtLoadPalettes": PixelArtLoadPalettes,
    "PixelArtPaletteGenerator": PixelArtPaletteGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelArtAddDitherPattern": "🕹️PixelArt Add Dither Pattern",
    "PixelArtDetectorSave": "🕹️PixelArt Detector (+Save)",
    "PixelArtDetectorToImage": "🕹️PixelArt Detector (Image->)",
    "PixelArtDetectorConverter": "🎨PixelArt Palette Converter",
    "PixelArtLoadPalettes": "🎨PixelArt Palette Loader",
    "PixelArtPaletteGenerator": "🎨PixelArt Palette Generator"
}
