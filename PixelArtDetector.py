"""

Custom node for SDXL in ComfyUI

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
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import scipy
from itertools import product
from comfy.cli_args import args

import os, json, time, folder_paths
from datetime import datetime

class PixelArtDetector:
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
    
    CATEGORY = "image"
    

    def process(self, images, reduce_palette, reduce_palette_max_colors, filename_prefix, webp_mode , compression, resize_w, resize_h, prompt=None, extra_pnginfo=None, save_jpg="disabled", save_exif="enabled"):
        # Tensor to PIL
        def tensor2pil(image):
            return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        
        results = list()
        for image in images:
            # Convert to PIL Image
            pilImage = tensor2pil(image)
                
            # Start timer
            start = round(time.time()*1000)
                
            # Find 1:1 pixel scale
            downscale = self.pixel_detect(pilImage)
                
            print(f"### {self.CGREEN}[PixelArtDetector]{self.CEND} Size detected and reduced from {self.CYELLOW}{pilImage.width}{self.CEND}x{self.CYELLOW}{pilImage.height}{self.CEND} to {self.CYELLOW}{downscale.width}{self.CEND}x{self.CYELLOW}{downscale.height}{self.CEND} in {self.CYELLOW}{round(time.time()*1000)-start}{self.CEND} milliseconds")
                
            PILOutput = downscale
                
            if reduce_palette =="enabled":
                print(f"### {self.CGREEN}[PixelArtDetector]{self.CEND} Reduce pallete max_colors: {self.CYELLOW}{reduce_palette_max_colors}{self.CEND}")
                # Start timer
                start = round(time.time()*1000)
                # Reduce color palette using elbow method
                best_k = self.determine_best_k(downscale, reduce_palette_max_colors)
                PILOutput = downscale.quantize(colors=best_k, method=1, kmeans=best_k, dither=0).convert('RGB')
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

    def determine_best_k(self, image: Image, max_k: int):
        # Convert the image to RGB mode
        image = image.convert("RGB")

        # Prepare arrays for distortion calculation
        pixels = np.array(image)
        pixel_indices = np.reshape(pixels, (-1, 3))

        # Calculate distortion for different values of k
        distortions = []
        for k in range(1, max_k + 1):
            quantized_image = image.quantize(colors=k, method=0, kmeans=k, dither=0)
            centroids = np.array(quantized_image.getpalette()[:k * 3]).reshape(-1, 3)
            
            # Calculate distortions
            distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
            min_distances = np.min(distances, axis=1)
            distortions.append(np.sum(min_distances ** 2))

        # Calculate the rate of change of distortions
        rate_of_change = np.diff(distortions) / np.array(distortions[:-1])
        
        # Find the elbow point (best k value)
        if len(rate_of_change) == 0:
            best_k = 2
        else:
            elbow_index = np.argmax(rate_of_change) + 1
            best_k = elbow_index + 2

        return best_k

    def kCentroid(self, image: Image, width: int, height: int, centroids: int):
        image = image.convert("RGB")

        # Create an empty array for the downscaled image
        downscaled = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate the scaling factors
        wFactor = image.width/width
        hFactor = image.height/height

        # Iterate over each tile in the downscaled image
        for x, y in product(range(width), range(height)):
            # Crop the tile from the original image
            tile = image.crop((x*wFactor, y*hFactor, (x*wFactor)+wFactor, (y*hFactor)+hFactor))

            # Quantize the colors of the tile using k-means clustering
            tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")

            # Get the color counts and find the most common color
            color_counts = tile.getcolors()
            most_common_color = max(color_counts, key=lambda x: x[0])[1]

            # Assign the most common color to the corresponding pixel in the downscaled image
            downscaled[y, x, :] = most_common_color

        return Image.fromarray(downscaled, mode='RGB')
    
    def pixel_detect(self, image: Image):
        # [Astropulse]
        # Thanks to https://github.com/paultron for optimizing my garbage code 
        # I swapped the axis so they accurately reflect the horizontal and vertical scaling factor for images with uneven ratios

        # Convert the image to a NumPy array
        npim = np.array(image)[..., :3]

        # Compute horizontal differences between pixels
        hdiff = np.sqrt(np.sum((npim[:, :-1, :] - npim[:, 1:, :])**2, axis=2))
        hsum = np.sum(hdiff, 0)

        # Compute vertical differences between pixels
        vdiff = np.sqrt(np.sum((npim[:-1, :, :] - npim[1:, :, :])**2, axis=2))
        vsum = np.sum(vdiff, 1)

        # Find peaks in the horizontal and vertical sums
        hpeaks, _ = scipy.signal.find_peaks(hsum, distance=1, height=0.0)
        vpeaks, _ = scipy.signal.find_peaks(vsum, distance=1, height=0.0)
        
        # Compute spacing between the peaks
        hspacing = np.diff(hpeaks)
        vspacing = np.diff(vpeaks)

        # Resize input image using kCentroid with the calculated horizontal and vertical factors
        return self.kCentroid(
            image,
            round(image.width/np.median(hspacing)),
            round(image.height/np.median(vspacing)),
            2
        )
        
    # this is just for testing
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
    "PixelArtDetector": PixelArtDetector
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelArtDetector": "PixelArt Detector"
}