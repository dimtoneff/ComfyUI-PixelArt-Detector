# ComfyUI PixelArt Detector
Generate, downscale and restore pixel art images with SDXL.
Save a picture as Webp (+optional JPEG) file in Comfy + Workflow loading.

## Description:

This adds a custom node to save a picture as a Webp File and also adds a script to Comfy to drag and drop generated webpfiles into the UI to load the workflow.

I've added a compression slider and a lossy/lossless option. The compression slider is a bit misleading.

In lossless mode, it only affects the "effort" taken to compress where 100 is the smallest possible size and 1 is the biggest possible size, it's a tradeoff for saving speed.

In lossy mode, that's the other way around, where 100 is the biggest possible size with the least compression and 1 is the smallest possible size with maximum compression.

Pixel Art manipulation code based on: https://github.com/Astropulse/pixeldetector

# Screenshot
![Example](./plugin.PNG)

## Installation: 

Use git clone https://github.com/dimtoneff/ComfyUI-PixelArt-Detector in your ComfyUI custom nodes directory

# Usage

LoRa: https://civitai.com/models/120096/pixel-art-xl

# Examples

![Example](./examples/Image_Upscaled_00012_.webp)
![Example](./examples/PixelArt_00021_.webp)
![Example](./examples/PixelArt_00024_.jpeg)

# Credits
Big thanks to https://github.com/Astropulse/pixeldetector for the main code.
Big thanks to https://github.com/Kaharos94/ComfyUI-Saveaswebp for the webp saving code.
Big thanks to https://github.com/paultron for numpy-ifying the downscale calculation and making it tons faster.
