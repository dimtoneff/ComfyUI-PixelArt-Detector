# ComfyUI PixelArt Detector v1.0
Generate, downscale, change palletes and restore pixel art images with SDXL.

![](./examples/Image_00135_.webp) ![](./examples/Image_00157_.webp) ![](./examples/Image_00162_.webp) ![](./examples/Image_00165_.webp) ![](./examples/Image_00166_.webp)

Save a picture as Webp (+optional JPEG) file in Comfy + Workflow loading.

> [!IMPORTANT]
> If you have an older version of the nodes, delete the node and add it again. Location of the nodes: "Image/PixelArt". I've added some example workflow in the workflow.json. The example images might have outdated workflows with older node versions embedded inside.

## Description:
Pixel Art manipulation code based on: https://github.com/Astropulse/pixeldetector

This adds 4 custom nodes:
* __PixelArt Detector (+Save)__ - this node is All in One reduce palette, resize, saving image node
* __PixelArt Detector (Image->)__ - this node will downscale and reduce the palette and forward the image to another node
* __PixelArt Palette Converter__ - this node will change the palette of your input. There are a couple of embedded palettes. Use the Palette Loader for more
* __PixelArt Palette Loader__ - this node comes with a lot of custom palettes which can be an input to the PixelArt Palette Converter "paletteList" input

The plugin also adds a script to Comfy to drag and drop generated webp|jpeg files into the UI to load the workflows.

The nodes are able to manipulate the pixel art image in ways that it should look pixel perfect (downscales, changes palette, upscales etc.).

> [!IMPORTANT]
> You can disable the embedded resize function in the nodes by setting W & H to 0.

## Installation:

To use these nodes, simply open a terminal in ComfyUI/custom_nodes/ and run:

```
git clone https://github.com/dimtoneff/ComfyUI-PixelArt-Detector
```

Restart ComfyUI afterwards.

# Usage

Use LoRa: https://civitai.com/models/120096/pixel-art-xl

Drag the workflow.json file in your ComfyUI

Set the resize inputs to 0 to disable upscaling in the "Save" node.

Reduce palettes or completely exchange palettes on your images.

### Extra info about the "PixelArt Detector (+Save)" Node:

There is a compression slider and a lossy/lossless option for webp. The compression slider is a bit misleading.

In lossless mode, it only affects the "effort" taken to compress where 100 is the smallest possible size and 1 is the biggest possible size, it's a tradeoff for saving speed.

In lossy mode, that's the other way around, where 100 is the biggest possible size with the least compression and 1 is the smallest possible size with maximum compression.

There is an option to save a JPEG alongside the webp file.

### Extra info about the "PixelArt Palette Converter" Node:

The grid_size option is for the pixelize grid.Pixelate option. Size of 1 is pixel by pixel. Very slow. Increazing the size improves speed but kills quality. Experiment or not use that option.

### Extra info about the "PixelArt Palette Converter" and "PixelArt Palette Loader" Nodes:

Included palettes from: https://lospec.com/palette-list

> [!IMPORTANT]
> If you like some palette, download the 1px one and add it to the **"ComfyUI-PixelArt-Detector\palettes\1x"** directory.

Here are some examples:

![Example](./examples/Image_00169_.webp) ![Example](./examples/Image_00170_.webp)

![Example](./examples/Image_00171_.webp) ![Example](./examples/Image_00172_.webp)

![Example](./examples/Image_00048_.webp) ![Example](./examples/Image_00049_.webp)

![Example](./examples/Image_00050_.webp) ![Example](./examples/Image_00051_.webp)

![Example](./examples/Image_00053_.webp) ![Example](./examples/Image_00057_.webp)

# Screenshot

Nodes:
![Example](./nodes.PNG)

Workflow view:
![Example](./plugin.PNG)

# Examples

Normal image:

![Example](./examples/PixelArtSave_00005_.webp) ![Example](./examples/PixelArt_00024_.jpeg)

Reduced palette:

![Example](./examples/Image_00005_.webp) ![Example](./examples/PixelArt_00021_.webp)

![Example](./examples/Image_Reduced_256_00011_.webp) ![Example](./examples/Image_Reduced_256_00004_.webp)

Upscaled:

![Example](./examples/Image_Upscaled_00004_.webp)

![Example](./examples/Image_Upscaled_00012_.webp)

# Credits
Big thanks to https://github.com/Astropulse/pixeldetector for the main code.

Big thanks to https://github.com/Kaharos94/ComfyUI-Saveaswebp for the webp saving code.

Big thanks to https://github.com/paultron for numpy-ifying the downscale calculation and making it tons faster.

Big thanks to https://lospec.com/palette-list and the creators of the awesome palettes.
