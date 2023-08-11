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

# implementation of https://github.com/Astropulse/pixeldetector to a ComfyUI extension node
# by dimtoneff
from PIL import Image, ImageOps
import numpy as np
import hashlib
import nodes

import torch
from pathlib import Path
from comfy.cli_args import args

import os, json, time, folder_paths
from datetime import datetime
from .pixelUtils import *

class PixelArtLoadPalettes(nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.normpath(os.path.join(getPalettesPath(), "1x/"))
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), )},
                }

    CATEGORY = "image/PixelArt"
    RETURN_TYPES = ("LIST",)

    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = os.path.normpath(os.path.join(getPalettesPath(), "1x/", image))
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        
        image = i.convert("P")
        palette = image.getpalette()
        #print(palette)
        return (palette,)
    
    @classmethod
    def IS_CHANGED(s, image):
        image_path = os.path.normpath(os.path.join(getPalettesPath(), "1x/", image))
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
    
    @classmethod
    def VALIDATE_INPUTS(s, image):
        image_path = os.path.normpath(os.path.join(getPalettesPath(), "1x/", image))
        if not Path(image_path).is_file():
            return "Invalid image file: {}".format(image)

        return True

class PixelArtDetectorConverter():
    """
    A node that can convert images to some fan favorite palettes: NES, GAME BOY etc.
    """
    
    def __init__(self):
        self.CGREEN = '\033[92m'
        self.CYELLOW = '\033[93m'
        self.CEND = '\033[0m'
        self.GAME_BOY_PALETTE_TUPLES = [(15,56,15),(48,98,48),(139,172,15),(155,188,15)]#,(202,220,159)
        self.NES_PALETTE_TUPLES      = [(124,124,124),(0,0,252),(0,0,188),(68,40,188),(148,0,132),(168,0,32),(168,16,0),(136,20,0),(80,48,0),(0,120,0),(0,104,0),(0,88,0),
                                        (0,64,88),(0,0,0),(0,0,0),(0,0,0),(188,188,188),(0,120,248),(0,88,248),(104,68,252),(216,0,204),(228,0,88),(248,56,0),(228,92,16),
                                        (172,124,0),(0,184,0),(0,168,0),(0,168,68),(0,136,136),(0,0,0),(0,0,0),(0,0,0),(248,248,248),(60,188,252),(104,136,252),(152,120,248),
                                        (248,120,248),(248,88,152),(248,120,88),(252,160,68),(248,184,0),(184,248,24),(88,216,84),(88,248,152),(0,232,216),(120,120,120),
                                        (0,0,0),(0,0,0),(252,252,252),(164,228,252),(184,184,248),(216,184,248),(248,184,248),(248,164,192),(240,208,176),(252,224,168),
                                        (248,216,120),(216,248,120),(184,248,184),(184,248,216),(0,252,252),(248,216,248),(0,0,0),(0,0,0)
                                       ]
        self.GAME_BOY = [15,56,15,48,98,48,139,172,15,155,188,15]
        self.NES = [
            124,124,124,0,0,252,0,0,188,68,40,188,148,0,132,168,0,32,168,16,0,136,20,0,80,48,0,0,120,0,0,104,0,0,88,0,0,64,88,0,0,0,0,0,0,0,0,0,188,188,188,0,120,248,
            0,88,248,104,68,252,216,0,204,228,0,88,248,56,0,228,92,16,172,124,0,0,184,0,0,168,0,0,168,68,0,136,136,0,0,0,0,0,0,0,0,0,248,248,248,60,188,252,104,136,252,
            152,120,248,248,120,248,248,88,152,248,120,88,252,160,68,248,184,0,184,248,24,88,216,84,88,248,152,0,232,216,120,120,120,0,0,0,0,0,0,252,252,252,164,228,252,
            184,184,248,216,184,248,248,184,248,248,164,192,240,208,176,252,224,168,248,216,120,216,248,120,184,248,184,184,248,216,0,252,252,248,216,248,0,0,0,0,0,0
        ]
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "palette": (["NES", "GAMEBOY"], {"default": "GAMEBOY"}),
                    "pixelize": (["Image.quantize", "Grid.pixelate"], {"default": "Image.quantize"}),
                    "grid_size":("INT", {"default": 2, "min": 1, "max": 32, "step": 1},),
                    "resize_w":("INT", {"default": 512, "min": 128, "max": 2048, "step": 1},),
                    "resize_h":("INT", {"default": 512, "min": 128, "max": 2048, "step": 1},),
                    },
                "optional": {
                    "paletteList": ("LIST", {"forceInput": True}),
                    },                
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    CATEGORY = "image/PixelArt"
    OUTPUT_IS_LIST = (True,)

    def process(self, images, palette, pixelize, grid_size, resize_w, resize_h, paletteList=None):

        if (palette == "NES"):
            palette = self.NES
        else:
            palette = self.GAME_BOY

        if paletteList is not None:
            palette = paletteList

        palIm = Image.new('P', (1,1))
        palIm.putpalette(palette)
        #print(palIm.getpalette())
        
        results = list()
        for image in images:
            pilImage = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")
            
            # Start timer
            start = round(time.time()*1000)
            
            if (pixelize == "Image.quantize"):
                PILOutput = pilImage.quantize(palette=palIm, dither=Image.Dither.NONE).convert('RGB')
            else:
                PILOutput = pixelate(pilImage, grid_size, paletteToTuples(palette, 3))

            print(f"### {self.CGREEN}[PixelArtDetector]{self.CEND} Image converted in {self.CYELLOW}{round(time.time()*1000)-start}{self.CEND} milliseconds")
            
            # resize
            if resize_w >= 128 and resize_h >= 128:
                PILOutput = PILOutput.resize((resize_w, resize_h), resample=Image.Resampling.NEAREST)

            # Convert to torch.Tensor            
            PILOutput = np.array(PILOutput).astype(np.float32) / 255.0
            PILOutput = torch.from_numpy(PILOutput)[None,]
            results.append(PILOutput)
                
        return (results,)


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
                    "reduce_palette": (["enabled", "disabled"], {"default": "disabled"}),
                    "reduce_palette_max_colors":("INT", {"default": 128, "min": 1, "max": 256, "step": 1},),
                    },
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    CATEGORY = "image/PixelArt"
    OUTPUT_IS_LIST = (True,)

    def process(self, images, reduce_palette, reduce_palette_max_colors):
        results = list()
        for image in images:
            pilImage = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")
            
            # Start timer
            start = round(time.time()*1000)
            
            # Find 1:1 pixel scale
            downscale = pixel_detect(pilImage)
            
            print(f"### {self.CGREEN}[PixelArtDetector]{self.CEND} Size detected and reduced from {self.CYELLOW}{pilImage.width}{self.CEND}x{self.CYELLOW}{pilImage.height}{self.CEND} to {self.CYELLOW}{downscale.width}{self.CEND}x{self.CYELLOW}{downscale.height}{self.CEND} in {self.CYELLOW}{round(time.time()*1000)-start}{self.CEND} milliseconds")
                
            PILOutput = downscale
            
            if reduce_palette =="enabled":
                print(f"### {self.CGREEN}[PixelArtDetector]{self.CEND} Reduce pallete max_colors: {self.CYELLOW}{reduce_palette_max_colors}{self.CEND}")
                # Start timer
                start = round(time.time()*1000)
                # Reduce color palette using elbow method
                best_k = determine_best_k(downscale, reduce_palette_max_colors)
                PILOutput = downscale.quantize(colors=best_k, method=1, kmeans=best_k, dither=0).convert('RGB')
                print(f"### {self.CGREEN}[PixelArtDetector]{self.CEND} Palette reduced to {self.CYELLOW}{best_k}{self.CEND} colors in {self.CYELLOW}{round(time.time()*1000)-start}{self.CEND} milliseconds")
                
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
                "reduce_palette": (["enabled", "disabled"], {"default": "disabled"}),
                "reduce_palette_max_colors":("INT", {"default": 128, "min": 1, "max": 256, "step": 1},),
                "webp_mode":(["lossy","lossless"],),
                "compression":("INT", {"default": 80, "min": 1, "max": 100, "step": 1},),
                "save_jpg": (["disabled", "enabled"], {"default": "disabled"}),
                "save_exif": (["disabled", "enabled"], {"default": "enabled"}),
                "resize_w":("INT", {"default": 512, "min": 128, "max": 2048, "step": 1},),
                "resize_h":("INT", {"default": 512, "min": 128, "max": 2048, "step": 1},),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ()
    FUNCTION = "process"
    
    OUTPUT_NODE = True
    
    CATEGORY = "image/PixelArt"
    

    def process(self, images, reduce_palette, reduce_palette_max_colors, filename_prefix, webp_mode , compression, resize_w, resize_h, prompt=None, extra_pnginfo=None, save_jpg="disabled", save_exif="enabled"):
        
        results = list()
        for image in images:
            # Convert to PIL Image
            pilImage = tensor2pil(image)
                
            # Start timer
            start = round(time.time()*1000)
                
            # Find 1:1 pixel scale
            downscale = pixel_detect(pilImage)
                
            print(f"### {self.CGREEN}[PixelArtDetector]{self.CEND} Size detected and reduced from {self.CYELLOW}{pilImage.width}{self.CEND}x{self.CYELLOW}{pilImage.height}{self.CEND} to {self.CYELLOW}{downscale.width}{self.CEND}x{self.CYELLOW}{downscale.height}{self.CEND} in {self.CYELLOW}{round(time.time()*1000)-start}{self.CEND} milliseconds")
                
            PILOutput = downscale
                
            if reduce_palette =="enabled":
                print(f"### {self.CGREEN}[PixelArtDetector]{self.CEND} Reduce pallete max_colors: {self.CYELLOW}{reduce_palette_max_colors}{self.CEND}")
                # Start timer
                start = round(time.time()*1000)
                PILOutput, best_k = reducePalette(downscale, reduce_palette_max_colors)
                print(f"### {self.CGREEN}[PixelArtDetector]{self.CEND} Palette reduced to {self.CYELLOW}{best_k}{self.CEND} colors in {self.CYELLOW}{round(time.time()*1000)-start}{self.CEND} milliseconds")
                
            # resize
            if resize_w >= 128 and resize_h >= 128:
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

        return { "ui": { "images": results } }
        
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
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1
            
        workflowmetadata = str()
        promptstr = str()
        imgexif = output.getexif() #get the (empty) Exif data of the generated Picture
        
        if not args.disable_metadata and save_exif == "enabled":
            if prompt is not None:
                promptstr="".join(json.dumps(prompt)) #prepare prompt String
                imgexif[0x010f] ="Prompt:"+ promptstr #Add PromptString to EXIF position 0x010f (Exif.Image.Make)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    workflowmetadata += "".join(json.dumps(extra_pnginfo[x]))
            imgexif[0x010e] = "Workflow:"+ workflowmetadata #Add Workflowstring to EXIF position 0x010e (Exif.Image.ImageDescription)
            
        file = f"{filename}_{counter:05}_"
        
        if webp_mode =="lossless":
            boolloss = True
        if webp_mode =="lossy":
            boolloss = False

        output.save(os.path.join(full_output_folder, file + ".webp"), method=6 , exif=imgexif, lossless=boolloss , quality=compression) #Save as webp - options to be determined
        if save_jpg =="enabled":
            output.save(os.path.join(full_output_folder, file + ".jpeg"), exif=imgexif, quality=compression) #Save as jpeg
 
        print(f"### {self.CGREEN}[PixelArtDetector]{self.CEND} Saving file to {self.CYELLOW}{full_output_folder}{self.CEND} Filename: {self.CYELLOW}{file}{self.CEND}")
        
        return {
                "filename": file + ".webp",
                "subfolder": subfolder,
                "type": self.type
            }


            
NODE_CLASS_MAPPINGS = {
    "PixelArtDetectorSave": PixelArtDetectorSave,
    "PixelArtDetectorToImage": PixelArtDetectorToImage,
    "PixelArtDetectorConverter": PixelArtDetectorConverter,
    "PixelArtLoadPalettes": PixelArtLoadPalettes,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelArtDetectorSave": "PixelArt Detector (+Save)",
    "PixelArtDetectorToImage": "PixelArt Detector (Image->)",
    "PixelArtDetectorConverter": "PixelArt Palette Converter",
    "PixelArtLoadPalettes": "PixelArt Palette Loader"
}