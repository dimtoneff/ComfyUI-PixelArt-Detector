# ComfyUI PixelArt Detector
Generate, downscale and restore pixel art images with SDXL.

Save a picture as Webp (+optional JPEG) file in Comfy + Workflow loading.

## Description:
Pixel Art manipulation code based on: https://github.com/Astropulse/pixeldetector

This adds a custom node to save a picture as a Webp File and also adds a script to Comfy to drag and drop generated webpfiles into the UI to load the workflow.

This node also manipulates the pixel art image in ways that it should look pixel perfect (downscales, changes palette, upscales etc.).

There is a compression slider and a lossy/lossless option for webp. The compression slider is a bit misleading.

In lossless mode, it only affects the "effort" taken to compress where 100 is the smallest possible size and 1 is the biggest possible size, it's a tradeoff for saving speed.

In lossy mode, that's the other way around, where 100 is the biggest possible size with the least compression and 1 is the smallest possible size with maximum compression.

There is an option to save a JPEG alongside the webp file.

# Screenshot
![Example](./plugin.PNG)

## Installation: 

Use git clone https://github.com/dimtoneff/ComfyUI-PixelArt-Detector in your ComfyUI custom nodes directory

# Usage

Use LoRa: https://civitai.com/models/120096/pixel-art-xl

A ComfyUI workflow should be embedded in the examples. Just drag and drop the image in the ComfyUI window. This extension should be installed, so it can read WEBP or JPEG Workflows.

Set the resize inputs to 0 to disable upscaling.

# Examples

Normal image:

![Example](./examples/PixelArt_00024_.jpeg)

Reduced palette:

![Example](./examples/PixelArt_00021_.webp)

Upscaled:

![Example](./examples/Image_Upscaled_00012_.webp)

# Credits
Big thanks to https://github.com/Astropulse/pixeldetector for the main code.

Big thanks to https://github.com/Kaharos94/ComfyUI-Saveaswebp for the webp saving code.

Big thanks to https://github.com/paultron for numpy-ifying the downscale calculation and making it tons faster.
