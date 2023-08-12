import collections.abc
import numpy as np
import torch
import os, time, folder_paths, math
from pathlib import Path
from PIL import Image, ImageStat, ImageFont, ImageOps, ImageDraw
from collections import abc
from itertools import repeat, product
import scipy

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def scanFilesInDir(input_dir):
    return sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])

def getFont(size: int = 10, fontName: str = "Roboto-Regular.ttf"):
    nodes_path = folder_paths.get_folder_paths("custom_nodes")
    font_path = os.path.normpath(os.path.join(nodes_path[0], "ComfyUI-PixelArt-Detector/fonts/", fontName))
    return ImageFont.truetype(str(font_path), size=size)

def transformPalette(palette: list, output: str = "image"):
    match output:
        case "image":
            palIm = Image.new('P', (1,1))
            palIm.putpalette(palette)
            return palIm
        case "tuple":
            return paletteToTuples(palette, 3)
        case _: # default case
            return palette

def drawTextInImage(image: Image, text, fontSize: int = 26, fontColor = (255, 0, 0), strokeColor = "white"):
    # Create a draw object
    draw = ImageDraw.Draw(image)
    font = getFont(fontSize)
    # Get the width and height of the image
    width, height = image.size
    # Get the width and height of the text
    text_width, text_height = draw.textsize(text, font)
    # Calculate the position of the text
    x = 0 # left margin
    y = height - text_height # bottom margin
    # Draw the text on the image
    draw.text((x, y), text, font=font, fill=fontColor, stroke_width=2, stroke_fill=strokeColor)
    
def getPalettesPath():
    nodes_path = folder_paths.get_folder_paths("custom_nodes")
    full_pallete_path = os.path.normpath(os.path.join(nodes_path[0], "ComfyUI-PixelArt-Detector/palettes/"))
    return Path(full_pallete_path)

def getPaletteImage(palette_from_image):
    full_pallete_path = os.path.normpath(os.path.join(getPalettesPath(), palette_from_image))
    return Path(full_pallete_path)
    
def paletteToTuples(palette, n):        
    return list(zip(*[iter(palette)] * n))# zip the array with itself n times and convert it to list

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def reducePalette(image, reduce_palette_max_colors):
    # Reduce color palette using elbow method
    best_k = determine_best_k(image, reduce_palette_max_colors)
    return image.quantize(colors=best_k, method=1, kmeans=best_k, dither=0).convert('RGB'), best_k

#https://theartofdoingcs.com/blog/f/bit-me
def pixelate(image: Image, grid_size: int, palette: list):
    if len(palette) > 0:
        if not isinstance(palette[0], tuple):
            palette = paletteToTuples(palette, 3)
        
    pixel_image = Image.new('RGB', image.size)
        
    for i in range(0, image.size[0], grid_size):
        for j in range(0, image.size[1], grid_size):
            pixel_box = (i, j, i + grid_size, j + grid_size)
            current = image.crop(pixel_box)
                
            median_color = ImageStat.Stat(current).median
            median_color = tuple(median_color)
                
            closest_color = distance(median_color, palette)
            median_pixel = Image.new('RGB', (grid_size, grid_size), closest_color)
            pixel_image.paste(median_pixel, (i,j))
                
    return pixel_image

def distance(median_color, palette: list[tuple]):
    (r1,g1,b1) = median_color
        
    colors = {}
        
    for color in palette:
        (r2,g2,b2) = color
        distance = ((r2 - r1) **2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)            
        colors[distance] = color
        
    closest_distance = min(colors.keys())
    closest_color = colors[closest_distance]
        
    return closest_color

def determine_best_k(image: Image, max_k: int):
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

def kCentroid(image: Image, width: int, height: int, centroids: int):
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
    
def pixel_detect(image: Image):
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
    return kCentroid(
        image,
        round(image.width/np.median(hspacing)),
        round(image.height/np.median(vspacing)),
        2
    )

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    # Check if the image_tensor is a list of tensors
    if isinstance(image_tensor, list):
        # Initialize an empty list to store the converted images
        image_numpy = []
        # Loop through each tensor in the list
        for i in range(len(image_tensor)):
            # Recursively call the tensor2im function on each tensor and append the result to the list
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        # Return the list of converted images
        return image_numpy
    # If the image_tensor is not a list, convert it to a NumPy array on the CPU with float data type
    image_numpy = image_tensor.cpu().float().numpy()
    
    # Check if the normalize parameter is True
    if normalize:
        # This will scale the pixel values from [-1, 1] to [0, 255]
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        # This will scale the pixel values from [0, 1] to [0, 255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0     
         
    # Clip the pixel values to the range [0, 255] to avoid overflow or underflow
    image_numpy = np.clip(image_numpy, 0, 255)
    # Check if the array has one or more than three channels
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        # If so, select only the first channel and discard the rest
        # This will convert the array to grayscale
        image_numpy = image_numpy[:,:,0]
    # Return the array with the specified data type (default is unsigned 8-bit integer)
    return image_numpy.astype(imtype)

# From WAS Node Suite
def smart_grid_image(images: list, cols=6, size=(256,256), add_border=True, border_color=(255,255,255), border_width=3):
    cols = min(cols, len(images))
    # calculate row height
    max_width, max_height = size
    row_height = 0
    images_resized = []
    
    for img in images:            
        img_w, img_h = img.size
        aspect_ratio = img_w / img_h
        if aspect_ratio > 1: # landscape
            thumb_w = min(max_width, img_w-border_width)
            thumb_h = thumb_w / aspect_ratio
        else: # portrait
            thumb_h = min(max_height, img_h-border_width)
            thumb_w = thumb_h * aspect_ratio

        # pad the image to match the maximum size and center it within the cell
        pad_w = max_width - int(thumb_w)
        pad_h = max_height - int(thumb_h)
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        padding = (left, top, right, bottom)  # left, top, right, bottom
        img_resized = ImageOps.expand(img.resize((int(thumb_w), int(thumb_h))), padding)

        if add_border:
            img_resized_bordered = ImageOps.expand(img_resized, border=border_width//2, fill=border_color)
                
        images_resized.append(img_resized)
        row_height = max(row_height, img_resized.size[1])
    row_height = int(row_height)

    # calculate the number of rows
    total_images = len(images_resized)
    rows = math.ceil(total_images / cols)

    # create empty image to put thumbnails
    new_image = Image.new('RGB', (cols*size[0]+(cols-1)*border_width, rows*row_height+(rows-1)*border_width), border_color)

    for i, img in enumerate(images_resized):
        if add_border:
            border_img = ImageOps.expand(img, border=border_width//2, fill=border_color)
            x = (i % cols) * (size[0]+border_width)
            y = (i // cols) * (row_height+border_width)
            if border_img.size == (size[0], size[1]):
                new_image.paste(border_img, (x, y, x+size[0], y+size[1]))
            else:
                # Resize image to match size parameter
                border_img = border_img.resize((size[0], size[1]))
                new_image.paste(border_img, (x, y, x+size[0], y+size[1]))
        else:
            x = (i % cols) * (size[0]+border_width)
            y = (i // cols) * (row_height+border_width)
            if img.size == (size[0], size[1]):
                new_image.paste(img, (x, y, x+img.size[0], y+img.size[1]))
            else:
                # Resize image to match size parameter
                img = img.resize((size[0], size[1]))
                new_image.paste(img, (x, y, x+size[0], y+size[1]))
                
    new_image = ImageOps.expand(new_image, border=border_width, fill=border_color)

    return new_image
