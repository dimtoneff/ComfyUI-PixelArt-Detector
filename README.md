# ComfyUI PixelArt Detector
Generate, downscale and restore pixel art images with SDXL.

Save a picture as Webp (+optional JPEG) file in Comfy + Workflow loading.

## Description:
Pixel Art manipulation code based on: https://github.com/Astropulse/pixeldetector

This adds 2 custom nodes (Image & Save nodes): the "Image" node can manipulate the image and forward it to another node AND the "Save" node to manipulate the image and save it as a Webp (+JPEG) File. It also adds a script to Comfy to drag and drop generated webp|jpeg files into the UI to load the workflows.

The nodes are able to manipulate the pixel art image in ways that it should look pixel perfect (downscales, changes palette, upscales etc.).

Extra info about the "Save" Node:

There is a compression slider and a lossy/lossless option for webp. The compression slider is a bit misleading.

In lossless mode, it only affects the "effort" taken to compress where 100 is the smallest possible size and 1 is the biggest possible size, it's a tradeoff for saving speed.

In lossy mode, that's the other way around, where 100 is the biggest possible size with the least compression and 1 is the smallest possible size with maximum compression.

There is an option to save a JPEG alongside the webp file.

# Screenshot

Nodes:
![Example](./nodes.PNG)

Workflow view:
![Example](./plugin.PNG)

## Installation: 

Use git clone https://github.com/dimtoneff/ComfyUI-PixelArt-Detector in your ComfyUI custom nodes directory

# Usage

Use LoRa: https://civitai.com/models/120096/pixel-art-xl

Drag the workflow.json file in your ComfyUI

Set the resize inputs to 0 to disable upscaling in the "Save" node.

# Examples

Normal image:

![Example](./examples/PixelArtSave_00005_.webp)

![Example](./examples/PixelArt_00024_.jpeg)

Reduced palette:

![Example](./examples/Image_00005_.webp)

![Example](./examples/PixelArt_00021_.webp)

![Example](./examples/Image_Reduced_256_00011_.webp)

![Example](./examples/Image_Reduced_256_00004_.webp)

Upscaled:

![Example](./examples/Image_Upscaled_00004_.webp)

![Example](./examples/Image_Upscaled_00012_.webp)

# Credits
Big thanks to https://github.com/Astropulse/pixeldetector for the main code.

Big thanks to https://github.com/Kaharos94/ComfyUI-Saveaswebp for the webp saving code.

Big thanks to https://github.com/paultron for numpy-ifying the downscale calculation and making it tons faster.
