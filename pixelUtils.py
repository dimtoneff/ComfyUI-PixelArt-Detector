import collections.abc
import numpy as np
import torch
import os, time, folder_paths
from pathlib import Path
from PIL import Image, ImageStat
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
