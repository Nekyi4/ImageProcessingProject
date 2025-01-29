from PIL import Image
import numpy as np
import sys
import time
import cmath
import cv2
from math import cos, sin, pi

### Image processing

def imageLoader(param):
    im = Image.open(param)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    image_arr = np.array(im, dtype=np.uint8)
    return image_arr

def imageLoaderG(param):
    im = Image.open(param)
    if im.mode != 'L':
        im = im.convert('L')
    image_arr = np.array(im, dtype=np.uint8)
    return image_arr

def saveImage(image_matrix, output_path):
    if image_matrix.dtype != np.uint8:
        image_matrix = np.clip(image_matrix, 0, 255).astype(np.uint8)
    new_image = Image.fromarray(image_matrix)
    new_image.save(output_path)
    print(f"Image saved at {output_path}")


def imageLoader1B(filepath):
    """
    Load an image and convert it into a binary matrix (1-bit representation).
    :param filepath: Path to the input image file
    :return: 2D list representing the binary image
    """
    try:
        # Open image and convert to grayscale
        img = Image.open(filepath).convert("L")  # L mode = 8-bit grayscale

        # Convert image to numpy array
        img_array = np.array(img)

        # Threshold to binary (1 for pixel values > 128, 0 otherwise)
        binary_image = (img_array > 128).astype(np.uint8)

        return binary_image

    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
def saveImage1B(image_matrix, output_path):
    """
    Save a binary matrix as a grayscale image (0 -> black, 255 -> white).
    :param image_matrix: 2D list representing the binary image
    :param output_path: Path to save the output image
    """
    try:
        height = len(image_matrix)
        width = len(image_matrix[0])

        # Create a new grayscale image
        new_image = Image.new("L", (width, height))
        for y in range(height):
            for x in range(width):
                new_image.putpixel((x, y), 255 if image_matrix[y][x] == 1 else 0)

        # Save the image
        new_image.save(output_path)
        print(f"Image saved at {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def brightnessChangerFlat(image_matrix, brightness_change, sign_negative):
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    lookup_table = {i: 0 for i in range(256)}
    
    for i in range(256):
        if sign_negative == False:
            # Increase brightness
            if (255 - i <= brightness_change):
                lookup_table[i] = 255 
            else:
                lookup_table[i] = i + brightness_change
        else:
            # Decrease brightness
            if (i <= brightness_change):
                lookup_table[i] = 0 
            else:
                lookup_table[i] = i - brightness_change

    if len(image_matrix.shape) == 2:
        # Grayscale image
        height, width = image_matrix.shape
        for i in range(height):
            for j in range(width):
                output_matrix[i, j] = lookup_table[image_matrix[i, j]]
    else:
        # RGB Image
        height, width, channels = image_matrix.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    output_matrix[i, j, k] = lookup_table[image_matrix[i, j, k]]
    
    return output_matrix

def brightnessChangerGamma(image_matrix, brightness_change):
    if(brightness_change < 0):
        print("Wrong brightness_changer!")
        return image_matrix
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    lookup_table = {i: 0 for i in range(256)}
    for i in range(1,256):
        if (255/i < brightness_change):
            lookup_table[i] = 255
        else:
            lookup_table[i] = i*brightness_change  
    if len(image_matrix.shape) == 2:  
        # Grayscale image
        height, width = image_matrix.shape
        for i in range(height):
            for j in range(width):
                output_matrix[i,j] = lookup_table[image_matrix[i,j]]
    else:
        # RGB Image
        height, width, channels = image_matrix.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                   output_matrix[i,j,k] = lookup_table[image_matrix[i,j,k]] 
    return output_matrix

def contrastChanger(image_matrix, contrast_factor):
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    lookup_table = {int(i): 0 for i in range(256)}
    for i in range(256):
        if(contrast_factor*(i-128)+128 > 255):
            lookup_table[i] = 255
        elif(contrast_factor * (i - 128) + 128 < 0):
            lookup_table[i] = 0
        else:
            lookup_table[i] = contrast_factor*(i-128) + 128
    if len(image_matrix.shape) == 2:  # Grayscale image
        height, width = image_matrix.shape
        for i in range(height):
            for j in range(width):
                output_matrix[i,j] = lookup_table[image_matrix[i,j]]
    else:
        # RGB Image
        height, width, channels = image_matrix.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    output_matrix[i,j,k] = lookup_table[image_matrix[i,j,k]]
    return output_matrix

def negative(image_matrix):
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    lookup_table = {int(i): 0 for i in range(256)}
    for i in range(256):
        lookup_table[i] = 255 - i
    if len(image_matrix.shape) == 2:  # Grayscale image
        height, width = image_matrix.shape
        for i in range(height):
            for j in range(width):
                output_matrix[i,j] = lookup_table[image_matrix[i,j]]
    else:
        # RGB Image
        height, width, channels = image_matrix.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    output_matrix[i,j,k] = lookup_table[image_matrix[i,j,k]]
    return output_matrix

def flipHorizontal(image_matrix):
    # Check if the image has 3 channels (RGB)
    if len(image_matrix.shape) == 3:
        height, width, channels = image_matrix.shape
        output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                output_matrix[i, j] = image_matrix[i, width - j - 1]
    # If the image is grayscale (single channel)
    elif len(image_matrix.shape) == 2:
        height, width = image_matrix.shape
        output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                output_matrix[i, j] = image_matrix[i, width - j - 1]
    else:
        raise ValueError("Input image must be either grayscale or RGB.")

    return output_matrix

def flipVertical(image_matrix):
    if len(image_matrix.shape) == 3:
        height, width, channels = image_matrix.shape
    else:
        height, width = image_matrix.shape
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
                output_matrix[i, j] = image_matrix[height-i-1, j]
    return output_matrix

def flipDiagonal(image_matrix):
    height, width, channels = image_matrix.shape
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
                output_matrix[i, j] = image_matrix[height-i-1, width - j - 1]
    return output_matrix

def shrinkImage(image_matrix, shrink_factor):
    if shrink_factor <= 0 or shrink_factor > 1:
        raise ValueError("Shrink factor must be between 0 and 1.")
    
    height, width = image_matrix.shape[:2]
    new_height = int(height * shrink_factor)
    new_width = int(width * shrink_factor)
    
    if len(image_matrix.shape) == 2:
        output_matrix = np.zeros((new_height, new_width), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                output_matrix[i, j] = image_matrix[int(i / shrink_factor), int(j / shrink_factor)]
    else:
        channels = image_matrix.shape[2]
        output_matrix = np.zeros((new_height, new_width, channels), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                output_matrix[i, j, :] = image_matrix[int(i / shrink_factor), int(j / shrink_factor), :]
    return output_matrix

def enlargeImage(image_matrix, enlarge_factor):
    if enlarge_factor<1:
        raise ValueError('Enlarge factor must be greater than 1')
    
    height, width = image_matrix.shape[:2]
    new_height = int(height * enlarge_factor)
    new_width = int(width * enlarge_factor)
    
    if len(image_matrix.shape) == 2:
        output_matrix = np.zeros((new_height, new_width), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                original_i = min(int(i / enlarge_factor), height - 1)
                original_j = min(int(j / enlarge_factor), width - 1)
                output_matrix[i, j, :] = image_matrix[original_i, original_j, :]
    elif len(image_matrix.shape) == 3:
        channels = image_matrix.shape[2]
        output_matrix = np.zeros((new_height, new_width, channels), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                original_i = min(int(i / enlarge_factor), height - 1)
                original_j = min(int(j / enlarge_factor), width - 1)
                output_matrix[i, j, :] = image_matrix[original_i, original_j, :] 
    else:
        raise ValueError("Unsupported image format.")

    return output_matrix

def alphatf(image_matrix, kernel_size, alpha):
    """
    Parameters:
    - image_matrix: Input image as a NumPy array (grayscale or RGB).
    - kernel_size: Size of the kernel (box) (must be odd).
    - alpha: Fraction of lowest and highest values to discard (0 <= alpha < 0.5).
    """
    if ((alpha < 0) or (alpha >= 0.5)):
        raise ValueError("Alpha must be between 0 and 0.5")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number")

    height, width, channels = image_matrix.shape
    filtered_image = image_matrix.copy().astype(np.uint8)

    # Number of pixels to remove on each end
    #d = max(1, int(alpha * kernel_size * kernel_size))
    d = int(alpha * kernel_size * kernel_size)
    border = int((kernel_size - 1) / 2) 

    for i in range(border, height - border):
        for j in range(border, width - border):
            for k in range(channels):
                window =  window = image_matrix[i - border:i + border+1, j - border:j + border+1, k].flatten()
                sorted_window = np.sort(window)
                # Check if d is less than the remaining number of pixels after trimming
                if d < len(sorted_window) // 2:
                    if(d>0):
                        trimmed_window = sorted_window[d:-d]
                    else:
                        trimmed_window = sorted_window
                    filtered_image[i, j, k] = np.mean(trimmed_window)
                else:
                    filtered_image[i, j, k] = np.mean(sorted_window)
                
    return filtered_image.astype(np.uint8)

def contra_harmonic_mean_filter(image_matrix, kernel, P):
    if kernel % 2 == 0:
        raise ValueError("Kernel size must be an odd number")
        
    height, width, channels = image_matrix.shape
    output_matrix = image_matrix.copy().astype(np.uint8)
    border = kernel // 2
    for row in range(border, height - border):
        for col in range(border, width - border):
            for chan in range(channels):   
                window = image_matrix[row - border:row + border + 1, col - border:col + border + 1, chan].flatten()
                mask = window != 0
                den = np.sum(window[mask] ** P) 
                num = np.sum(window[mask] ** (P + 1)) 
                if den != 0:
                    output_matrix[row, col, chan] = num / den
                else:
                    output_matrix[row, col, chan] = 0
    return output_matrix

def mse(original, processed):
    mse = np.mean((original - processed) ** 2)
    return mse

def pmse(original, processed):
    max_value = np.max(original)
    pmse = np.mean((original - processed) ** 2) / (max_value ** 2)
    return pmse

def snr(original, processed):
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - processed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def psnr(original, processed):
    max_value = np.max(original)
    msev = mse(original, processed)
    psnr = 10 * np.log10(max_value ** 2 / msev)
    return psnr

def md(original, processed):
    md = np.max(np.abs(original - processed))
    return md

### Task 2

def histogram(image_matrix, channel=0):
    lookup_table = {i: 0 for i in range(256)}
    if len(image_matrix.shape) == 2:  # Grayscale image
        for value in image_matrix.flatten():
            lookup_table[value] += 1
    else:  # Color image
        for value in image_matrix[:, :, channel].flatten():
            lookup_table[value] += 1
    return np.array([lookup_table[i] for i in range(256)])

def draw_histogram(hist_array):
    width = 256  
    height = 100
    output_matrix = np.zeros((height, width), dtype=np.uint8)
    max_value = np.max(hist_array)
    if max_value > 0:  
        normalized_hist = (hist_array * height / max_value).astype(int)
    else:
        normalized_hist = np.zeros_like(hist_array)
    for x in range(width):
        column_height = normalized_hist[x]
        output_matrix[height - column_height:height, x] = 255
    return output_matrix

def hrayleigh(hist, g_min, g_max, alpha, L=256):
    """
    Perform Rayleigh-based histogram equalization using a precomputed histogram.

    Parameters:
        hist: Precomputed histogram (1D array of size L, where L is the number of gray levels).
        g_min: Minimum brightness in the output image.
        g_max: Maximum brightness in the output image.
        alpha: Scaling factor for the Rayleigh distribution.
        L: Number of gray levels in the image (default is 256).

    Returns:
        The transformed image
    """
    N = np.sum(hist)
    cdf = np.cumsum(hist) / N
    output_matrix = np.zeros(L, dtype=float)
    
    # Apply the Rayleigh-based transformation for each gray level f
    for f in range(L):
        if cdf[f] > 0:
            # Avoid log(0) or negative values by clamping 1 - cdf[f] to a minimum value
            value = max(1 - cdf[f], 1e-10)
            output_matrix[f] = g_min + np.sqrt(-2 * alpha**2 * np.log(value))
    
    # Normalize output_matrix to range [g_min, g_max]
    output_matrix = np.clip(output_matrix, g_min, g_max)

    # Convert to 8-bit integer range [0, 255]
    #output_matrix = ((output_matrix - g_min) / (g_max - g_min) * 255).astype(np.uint8)

    return output_matrix

def u_slaplace(image_matrix, mask_number):
    """
    Universal convolution function that works for any given mask.
    
    Parameters:
        image: Input image to be filtered (2D array).
        mask: Convolution mask (kernel).
    
    Returns:
        Filtered image (2D array).
    """
    start_time = time.time()
    mask1 = [
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0] ]

    mask2 = [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1] ]

    mask3 = [
    [1, -2, 1],
    [-2, 4, -2],
    [1, -2, 1] ]

    if(mask_number == 1):
        mask = mask1
    elif(mask_number == 2):
        mask = mask2
    else:
        mask = mask3

    height = len(image_matrix)
    width = len(image_matrix[0])
    mask_size = len(mask)
    offset = mask_size // 2
    
    image_matrix = np.array(image_matrix, dtype=np.int32)
    output_image = np.zeros_like(image_matrix)
    
    # Iterate over each pixel in the image
    for p in range(offset, height - offset):
        for q in range(offset, width - offset):
            # Apply convolution on the region of interest
            conv_sum = 0
            for i in range(mask_size):
                for j in range(mask_size):
                    # Get the corresponding pixel from the image
                    conv_sum += image_matrix[p + i - offset][q + j - offset] * mask[i][j]
            
            output_image[p][q] = conv_sum

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    end_time = time.time() 
    print(f"Execution time for u_slaplace: {end_time - start_time:.4f} seconds")

    return output_image

def o_slaplace(image_matrix):
    """
    Optimized convolution function specifically for the Laplacian filter.
    
    Parameters:
        image_matrix: Input image to be filtered (2D array).
    
    Returns:
        Filtered image (2D array).
    """
    start_time = time.time()
    height = len(image_matrix)
    width = len(image_matrix[0])

    image_matrix = np.array(image_matrix, dtype=np.int32)
    
    output_image = np.zeros_like(image_matrix)

    # Perform convolution for each pixel (ignoring borders)
    for p in range(1, height - 1):
        for q in range(1, width - 1):
            # Apply the Laplacian filter (0, -1, 0; -1, 4, -1; 0, -1, 0)
            conv_sum = (
                -image_matrix[p-1][q]    # Top
                -image_matrix[p+1][q]    # Bottom
                -image_matrix[p][q-1]    # Left
                -image_matrix[p][q+1]    # Right
                + 5 * image_matrix[p][q] # Center
            )
            output_image[p][q] = conv_sum

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    end_time = time.time()
    print(f"Execution time for u_slaplace: {end_time - start_time:.4f} seconds")

    return output_image

def osobel(image_matrix):
    """
    Apply the Sobel operator for edge detection in a non-linear fashion using NumPy.
    
    Parameters:
        image_matrix: Input image matrix (2D array).
    
    Returns:
        Filtered image matrix (2D array).
    """
    image_matrix = np.array(image_matrix, dtype=np.uint16)
    height = len(image_matrix)
    width = len(image_matrix[0])
    output_image = np.zeros_like(image_matrix)
    for n in range(1, height - 1):
        for m in range(1, width - 1):
            # Extract the 3x3 neighborhood around the pixel x(n, m)
            A0 = image_matrix[n-1, m-1]
            A1 = image_matrix[n-1, m]
            A2 = image_matrix[n-1, m+1]
            A3 = image_matrix[n, m+1]
            A4 = image_matrix[n+1, m+1]
            A5 = image_matrix[n+1, m]
            A6 = image_matrix[n+1, m-1]
            A7 = image_matrix[n, m-1]
            x_nm = image_matrix[n, m]
            
            # Calculate Sobel gradients X and Y
            X = (A2 + 2 * A3 + A4) - (A0 + 2 * A7 + A6)
            Y = (A0 + 2 * A1 + A2) - (A6 + 2 * A5 + A4)
            
            # Compute the magnitude of the gradient g(n, m)
            g_nm = np.sqrt(X**2 + Y**2)
            
            output_image[n, m] = np.clip(g_nm, 0, 255) 

    return output_image.astype(np.uint8)

def mean(image_matrix, channel=0):
    """Calculate the mean brightness."""
    hist = histogram(image_matrix, channel)
    N = np.sum(hist)
    b = np.sum(np.arange(len(hist)) * hist) / N
    return b

def variance(image_matrix, channel=0):
    """Calculate the variance."""
    hist = histogram(image_matrix, channel)
    N = np.sum(hist)
    b = mean(image_matrix, channel)
    D_squared = np.sum((np.arange(len(hist)) - b)**2 * hist) / N
    return D_squared

def std_deviation(image_matrix, channel=0):
    """Calculate the standard deviation."""
    D_squared = variance(image_matrix, channel)
    sigma = np.sqrt(D_squared)
    return sigma

def variation_coefficient(image_matrix, channel=0):
    """Calculate the variation coefficient."""
    sigma = std_deviation(image_matrix, channel)
    b = mean(image_matrix, channel)
    bz = sigma / b
    return bz

def asymmetry_coefficient(image_matrix, channel=0):
    """Calculate the asymmetry coefficient."""
    hist = histogram(image_matrix, channel)
    N = np.sum(hist)
    b = mean(image_matrix, channel)
    sigma = std_deviation(image_matrix, channel)
    bs = (1 / N) * np.sum(((np.arange(len(hist)) - b)**3) * hist) / (sigma**3)
    return bs

def flattening_coefficient(image_matrix, channel=0):
    """Calculate the flattening coefficient."""
    hist = histogram(image_matrix, channel)
    N = np.sum(hist)
    b = mean(image_matrix, channel)
    sigma = std_deviation(image_matrix, channel)
    bk = ((1 / N) * np.sum(((np.arange(len(hist)) - b)**4) * hist)) / (sigma**4) - 3
    return bk

def entropy(image_matrix, channel=0):
    """Calculate the entropy."""
    hist = histogram(image_matrix, channel)
    N = np.sum(hist)
    probabilities = hist / N
    bg = -np.sum(probabilities * np.log2(probabilities + 1e-10)) 
    return bg


### Task3

def dilation(image, struct_element):
    """
    Perform morphological dilation on a binary 2D image (1-bit images).
    
    :param image: 2D list representing binary image (0 and 1)
    :param struct_element: 2D list representing binary structural element (0 and 1)
    :return: Dilated image as 2D list
    """
    height = len(image)  # Number of rows in the image
    width = len(image[0])  # Number of columns in the image
    se_rows = len(struct_element)  # Structural element rows
    se_cols = len(struct_element[0])  # Structural element columns

    # Calculate padding size for structural element
    pad_x = se_rows // 2
    pad_y = se_cols // 2

    # Step 1: Pad the original image with zeros to handle border pixels
    padded_image = [[0] * (width + 2 * pad_y) for _ in range(height + 2 * pad_x)]
    for i in range(height):
        for j in range(width):
            padded_image[i + pad_x][j + pad_y] = image[i][j]

    # Step 2: Initialize the output image
    output_image = [[0 for _ in range(width)] for _ in range(height)]

    # Step 3: Perform Dilation
    for i in range(height):  # Iterate over the image rows
        for j in range(width):  # Iterate over the image columns
            match_found = False
            for m in range(se_rows):  # Iterate over the structural element rows
                for n in range(se_cols):  # Iterate over the structural element columns
                    if struct_element[m][n] == 1:  # Check if SE element is 1
                        if padded_image[i + m][j + n] == 1:  # Check corresponding pixel in padded image
                            match_found = True
                            break
                if match_found:  # Stop searching once a match is found
                    break
            output_image[i][j] = 1 if match_found else 0  # Set output pixel

    output_image = np.array(output_image)
    return output_image

def erosion(image, struct_element):
    """
    Perform morphological erosion on a binary image.
    :param image: 2D binary image (0 and 1)
    :param struct_element: 2D binary structural element (0 and 1)
    :return: Eroded image as a 2D list
    """
    # Image dimensions
    height = len(image)
    width = len(image[0])
    
    # Structural element dimensions
    se_rows = len(struct_element)
    se_cols = len(struct_element[0])

    # Padding sizes
    pad_x = se_rows // 2
    pad_y = se_cols // 2

    # Step 1: Pad the image with zeros
    padded_image = [[0] * (width + 2 * pad_y) for _ in range(height + 2 * pad_x)]
    for i in range(height):
        for j in range(width):
            padded_image[i + pad_x][j + pad_y] = image[i][j]

    # Step 2: Initialize the output image
    output_image = [[0 for _ in range(width)] for _ in range(height)]

    # Step 3: Perform Erosion Operation
    for i in range(height):  # Traverse each pixel in the original image
        for j in range(width):
            match = True  # Assume match initially
            for m in range(se_rows):  # Traverse the structural element
                for n in range(se_cols):
                    if struct_element[m][n] == 1:  # Only consider structural element's 1s
                        if padded_image[i + m][j + n] != 1:  # Check if any mismatch occurs
                            match = False
                            break
                if not match:  # Exit early if no match
                    break
            if match:
                output_image[i][j] = 1  # Set output pixel to 1 if full match found

    output_image = np.array(output_image, dtype=np.uint8)
    return output_image


def opening(image, struct_element):
    """
    Perform morphological opening: erosion followed by dilation.
    :param image: 2D binary image
    :param struct_element: 2D structural element
    :return: Image after opening operation
    """
    eroded = erosion(image, struct_element)
    opened = dilation(eroded, struct_element)
    return opened

def closing(image, struct_element):
    """
    Perform morphological closing: dilation followed by erosion.
    :param image: 2D binary image
    :param struct_element: 2D structural element
    :return: Image after closing operation
    """
    dilated = dilation(image, struct_element)
    closed = erosion(dilated, struct_element)
    return closed

def hmt(image, struct_element):
    """
    Perform Hit-or-Miss Transformation (HMT) on a binary image.
    :param image: 2D binary image
    :param struct_element_B1: Foreground structural element (B1)
    :param struct_element_B2: Background structural element (B2, complement)
    :return: HMT-transformed image
    """
    struct_element_B1 = struct_element[0]
    struct_element_B2 = struct_element[1]
    eroded_B1 = erosion(image, struct_element_B1)
    eroded_B2 = erosion(1 - image, struct_element_B2)
    return eroded_B1 & eroded_B2

def successive_n_transform(image, struct_elements):
    """
    Apply the successive morphological transformations N(A, {B1, ..., Bn})
    :param image: 2D binary image
    :param struct_elements: List of structural elements [B1, B2, ..., Bn]
    :param n: Number of iterations
    :return: Transformed image
    """

    '''height = len(image)
    width = len(image[0])
    for i in range(1):
        for se in struct_elements:
            new_image = image.copy()
            new_image = hmt(new_image, se)
            for i in range(height):
                for j in range(width):
                    if(new_image[i][j] == 1):
                        image[i][j] = 0
    return image'''
    
    height = len(image)
    width = len(image[0])
    while True:  # Loop until convergence
        has_changed = False  # Track if any pixel value changes
        for se in struct_elements:
            # Perform hit-or-miss transformation for the current structuring element
            new_image = hmt(image, se)
            for i in range(height):
                for j in range(width):
                    if new_image[i][j] == 1:
                        # If a change is detected, update the pixel
                        if image[i][j] != 0:  # Check if the pixel changes
                            has_changed = True
                        image[i][j] = 0
        if not has_changed:  # Exit loop if no changes occurred in this iteration
            break
    return image
    

def structural_elements(param):
    # Define sample structural elements (as binary numpy arrays)
    B1 = np.array([[0, 0, 0],
               [0, 1, 1],
               [0, 0, 0]])

    B2 = np.array([[0, 0, 0],
                [0, 1, 0],
                [0, 1, 0]])

    B3 = np.array([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]])

    B4 = np.array([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]])

    B5 = np.array([[0, 0, 0],
                [0, 1, 1],
                [0, 1, 0]])

    B6 = np.array([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0]])

    B7 = np.array([[0, 0, 0],
                [1, 1, 1],
                [0, 0, 0]])

    B8 = np.array([[0, 0, 0],
                [1, 0, 1],
                [0, 0, 0]])

    B9 = np.array([[0, 0, 1],
                [1, 1, 0],
                [1, 0, 0]])

    B10 = np.array([[0, 1, 1],
                    [0, 1, 0],
                    [0, 0, 0]])

    # List of structural elements for N-transform
    structural_elements = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10]
    return structural_elements[param]

def structural_elements_XI(param):
    B1 = np.array([[1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]])

    B2 = np.array([[1, 1, 1],
                [0, 0, 0],
                [0, 0, 0]])

    B3 = np.array([[0, 0, 1],
                [0, 0, 1],
                [0, 0, 1]])

    B4 = np.array([[0, 0, 0],
                [0, 0, 0],
                [1, 1, 1]])

    BC = np.array([[0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]])
    
    structural_elements = [
                    [B1, BC], 
                    [B2, BC], 
                    [B3, BC], 
                    [B4, BC]]
    return structural_elements[param]

def structural_elements_XII():

    B1 = np.array([[0, 0, 0],
                [0, 1, 0],
                [1, 1, 1]])

    B1c = np.array([[1, 1, 1],
                [0, 0, 0],
                [0, 0, 0]])

    B2 = np.array([[0, 0, 0],
                [1, 1, 0],
                [1, 1, 0]])

    B2c = np.array([[0, 1, 1],
                [0, 0, 1],
                [0, 0, 0]])

    B3 = np.array([[1, 0, 0],
                [1, 1, 0],
                [1, 0, 0]])

    B3c = np.array([[0, 0, 1],
                [0, 0, 1],
                [0, 0, 1]])

    B4 = np.array([[1, 1, 0],
                [1, 1, 0],
                [0, 0, 0]])

    B4c = np.array([[0, 0, 1],
                [0, 0, 1],
                [1, 1, 1]])

    B5 = np.array([[1, 1, 1],
                [0, 1, 0],
                [0, 0, 0]])

    B5c = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 1]])

    B6 = np.array([[0, 1, 1],
                    [0, 1, 1],
                    [0, 0, 0]])

    B6c = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0]])

    B7 = np.array([[0, 0, 1],
                    [0, 1, 1],
                    [0, 0, 1]])

    B7c = np.array([[1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]])

    B8 = np.array([[0, 0, 0],
                    [0, 1, 1],
                    [0, 1, 1]])

    B8c = np.array([[1, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]])
    
    B9c  = np.array([[1, 1, 0],
                [0, 1, 0],
                [0, 1, 0]])
    
    B9 = np.array([[0, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1]])
    structural_elements = [
        [B1, B1c], 
        [B2, B2c], 
        [B3, B3c], 
        [B4, B4c], 
        [B5, B5c], 
        [B6, B6c], 
        [B7, B7c], 
        [B8, B8c],
        [B9, B9c]]
    return structural_elements

def region_growing(image, seed, threshold, criterion):
    """
    Perform region growing on an image starting from a seed point.

    Parameters:
        image (numpy.ndarray): Input image (H x W x C) where C is the number of channels (e.g., 3 for RGB).
        seed (tuple): A tuple (X, Y) representing the seed point.
        threshold (float): Threshold value for similarity criterion (0 for binary images).
        criterion (int): Distance criterion:
                        0 - Euclidean distance
                        1 - Manhattan distance
                        2 - Maximum absolute difference

    Returns:
        numpy.ndarray: Binary image (H x W) with the grown region marked as 255 (white).
    """
    image = image[:, :, 0]
    image = np.array(image, dtype=np.int16)
    seedX, seedY = seed
    # Initialize the result binary image
    result = np.zeros_like(image, dtype=np.uint8)

    # Seed pixel intensity
    seed_value = image[seedY, seedX]  # Cast seed value to int for arithmetic operations
    result[seedY, seedX] = 255  # Mark seed point as part of the region

    print(f"Seed Value at ({seedX}, {seedY}): {seed_value}")
    

    # Directions for 8-connectivity
    dx = [-1, 1, 0, 0, -1, 1, -1, 1]
    dy = [0, 0, -1, 1, -1, -1, 1, 1]

    # Queue for BFS
    pixel_queue = [(seedX, seedY)]
    
    while pixel_queue:
        x, y = pixel_queue.pop(0)

        # Check all 8 neighbors
        for i in range(8):
            nx = x + dx[i]
            ny = y + dy[i]

            # Ensure the neighbor is within bounds and not yet visited
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and result[ny, nx] == 0:
                neighbor_value = image[ny, nx]  # Pixel value in all channels (for color images)

                # Compute the distance based on the chosen criterion
                distance = 0
                if criterion == 0:  # Euclidean distance (for multi-channel)
                    distance = np.linalg.norm(neighbor_value - seed_value)
                elif criterion == 1:  # Manhattan distance (sum of absolute differences)
                    distance = np.sum(np.abs(neighbor_value - seed_value))
                elif criterion == 2:  # Maximum absolute difference
                    distance = np.max(np.abs(neighbor_value - seed_value))

                # Check if the distance is within the threshold
                if distance <= threshold:
                    result[ny, nx] = 255  # Add to the region
                    pixel_queue.append((nx, ny))  # Add to the queue

    return result

def seed_points():
    # Example of multiple seed points
    return [(50, 50), (100, 100), (320,450), (0,0)]  # List of seed points

# TASK 4

def precompute_twiddles(N):
    """Precompute twiddle factors for FFT"""
    # Create a 2D array for twiddles (rows x columns)
    twiddles = np.zeros((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            twiddles[k, n] = cmath.exp(-2j * np.pi * k * n / N)
    return twiddles

def dft_1d(signal, twiddles):
    """Compute the 1D DFT using precomputed twiddles, using matrix multiplication for efficiency."""
    return np.dot(twiddles, signal)

def fft_2d(image):
    """
    Perform a 2D FFT on the image (row-wise then column-wise).
    Args:
        image (ndarray): Input 2D image (real-valued).
    Returns:
        tuple: (Fourier transform result, magnitude spectrum, phase spectrum)
    """
    image = image.astype(np.float64)  # Convert to float64 for precision
    rows, cols = image.shape

    # Precompute twiddles
    row_twiddles = precompute_twiddles(cols)
    col_twiddles = precompute_twiddles(rows)

    # Apply DFT row-wise using vectorized approach
    row_transformed = np.array([dft_1d(row, row_twiddles) for row in image])

    # Apply DFT column-wise using vectorized approach
    fft_result = np.array([dft_1d(col, col_twiddles) for col in row_transformed.T]).T

    # Extract the DC component (top-left corner in unshifted result)
    dc_component = fft_result[0, 0]

    # Return the result shifted to center the low frequencies
    #fft_result = np.fft.fftshift(fft_result)
    fft_result = fftshift_manual(fft_result)

    # Compute magnitude and phase
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)

    return fft_result, magnitude, phase, dc_component

def fftshift_manual(fft_result):
    """
    Manually shift the zero-frequency component to the center of the spectrum.
    
    Args:
        fft_result (ndarray): The 2D Fourier transformed result (complex).
        
    Returns:
        ndarray: The shifted Fourier transform with DC component centered.
    """
    rows, cols = fft_result.shape
    mid_x, mid_y = cols // 2, rows // 2
    
    # Shift the quadrants of the 2D array 
    shifted_result = np.zeros_like(fft_result)
    
    # Top-left to bottom-right
    shifted_result[mid_y:, mid_x:] = fft_result[:mid_y, :mid_x]
    # Top-right to bottom-left
    shifted_result[mid_y:, :mid_x] = fft_result[:mid_y, mid_x:]
    # Bottom-left to top-right
    shifted_result[:mid_y, mid_x:] = fft_result[mid_y:, :mid_x]
    # Bottom-right to top-left
    shifted_result[:mid_y, :mid_x] = fft_result[mid_y:, mid_x:]
    
    return shifted_result


def ifft_2d(fft_data, dc_component):
    """
    Perform the 2D inverse FFT (IFFT) on the Fourier transformed image, restoring the DC component.

    Args:
        fft_data (ndarray): Fourier transformed image (complex).
        dc_component (complex): DC component to restore.

    Returns:
        ndarray: Reconstructed image in the spatial domain (real-valued).
    """
    # Shift the frequencies back to the original position
    #fft_data = np.fft.ifftshift(fft_data)
    fft_data = ifftshift_manual(fft_data)

    fft_data[0, 0] = dc_component

    rows, cols = fft_data.shape

    # Precompute correct IFFT twiddles (conjugate of FFT twiddles)
    row_twiddles = precompute_ifft_twiddles(cols)
    col_twiddles = precompute_ifft_twiddles(rows)

    # Apply IFFT row-wise first
    row_ifft = np.array([dft_1d(fft_data[row], row_twiddles) for row in range(rows)])

    # Apply IFFT column-wise second
    ifft_result = np.array([dft_1d(row_ifft[:, col], col_twiddles) for col in range(cols)]).T

    # Normalize by the total number of elements
    ifft_result /= (rows * cols)

    # Return the real part (since the final image should be real-valued)
    return np.real(ifft_result)

def ifftshift_manual(fft_data):
    """
    Manually shift the zero-frequency component back to the top-left corner.
    
    Args:
        fft_data (ndarray): The 2D Fourier transformed result (complex).
        
    Returns:
        ndarray: The shifted Fourier transform with the low frequencies moved back to the corners.
    """
    rows, cols = fft_data.shape
    mid_x, mid_y = cols // 2, rows // 2
    
    # Shift the quadrants of the 2D array
    shifted_result = np.zeros_like(fft_data)
    
    # Bottom-left to top-right
    shifted_result[:mid_y, :mid_x] = fft_data[mid_y:, mid_x:]
    # Bottom-right to top-left
    shifted_result[:mid_y, mid_x:] = fft_data[mid_y:, :mid_x]
    # Top-left to bottom-right
    shifted_result[mid_y:, :mid_x] = fft_data[:mid_y, mid_x:]
    # Top-right to bottom-left
    shifted_result[mid_y:, mid_x:] = fft_data[:mid_y, :mid_x]
    
    return shifted_result

def precompute_ifft_twiddles(N):
    """Precompute twiddle factors for IFFT, using the correct conjugate exponent."""
    twiddles = np.zeros((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            twiddles[k, n] = cmath.exp(2j * np.pi * k * n / N)  # Positive exponent for IFFT
    return twiddles

def low_pass_filter(image, radius):
    """
    Apply a low-pass filter in the frequency domain.
    Args:
        image (ndarray): Input image (real-valued).
        radius (int): Cutoff radius for the low-pass filter.
    Returns:
        ndarray: Filtered image (real-valued).
    """
    fft_result, magnitude, phase, dc_component = fft_2d(image)  # Get FFT result and shift it to center

    rows, cols = fft_result.shape

    # Center coordinates for the frequency domain
    center_x, center_y = cols // 2, rows // 2
    
    # Apply a mask to filter out high frequencies
    for x in range(cols):
        for y in range(rows):
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if dist > radius:
                fft_result[y, x] = 0  # Zero out high frequencies
    
    # Return the filtered image after inverse FFT
    return ifft_2d(fft_result, dc_component)

def high_pass_filter(image, radius):
    """
    Apply a high-pass filter in the frequency domain.
    Args:
        image (ndarray): Input image (real-valued).
        radius (int): Cutoff radius for the high-pass filter.
    Returns:
        ndarray: Filtered image (real-valued).
    """
    # Perform FFT on the image
    fft_result, _, _, dc_component = fft_2d(image)
    rows, cols = fft_result.shape

    # Center coordinates for the frequency domain
    center_x, center_y = cols // 2, rows // 2
    
    # Zero out frequencies inside the cutoff radius (low frequencies)
    for x in range(cols):
        for y in range(rows):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < radius:  # Zeroing out the low frequencies
                fft_result[y, x] = 0

    # Return the filtered image after inverse FFT
    return ifft_2d(fft_result, dc_component)

def band_pass_filter(image, low_radius, high_radius):
    """
    Apply a band-pass filter in the frequency domain.
    Args:
        image (ndarray): Input image (real-valued).
        low_radius (int): Lower cutoff radius.
        high_radius (int): Upper cutoff radius.
    Returns:
        ndarray: Filtered image (real-valued).
    """
    fft_result, _, _, dc_component = fft_2d(image)
    rows, cols = fft_result.shape

    # Zero out frequencies outside the band-pass range
    center_x, center_y = cols // 2, rows // 2
    for x in range(cols):
        for y in range(rows):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if distance < low_radius or distance > high_radius:
                fft_result[y, x] = 0

    return ifft_2d(fft_result, dc_component)

def band_cut(image, low, high):
    """
    Apply band-cut filter by removing frequencies in the specified band.
    
    Args:
        image (ndarray): Input image.
        low (float): Lower frequency cutoff.
        high (float): Upper frequency cutoff.
        
    Returns:
        ndarray: Image after band-cut filtering in the frequency domain.
    """
    # Perform FFT and get the frequency domain representation
    fft_data, _, _, dc_component = fft_2d(image)
    
    rows, cols = fft_data.shape
    center_x, center_y = cols // 2, rows // 2
    
    # Band-cut in frequency domain: Remove frequencies within the specified band
    for y in range(rows):
        for x in range(cols):
            distance = np.hypot(x - center_x, y - center_y)
            if low <= distance <= high:
                fft_data[y, x] = 0  # Set frequencies in the band to zero
    
    # Inverse FFT to transform back to the spatial domain
    return ifft_2d(fft_data, dc_component)

def high_pass_edge_detection(image, radius, mask):
    """
    Apply high-pass filter in the frequency domain with a mask and return the image after edge detection.
    
    Args:
        image (ndarray): Input image.
        radius (int): Radius for high-pass filtering (frequencies below this radius will be removed).
        mask (ndarray): Binary mask (1 for areas to keep, 0 for areas to filter).
        
    Returns:
        ndarray: Image after high-pass edge detection with mask applied.
    """
    # Perform FFT and get the frequency domain representation
    fft_data, _, _, dc_component = fft_2d(image)
    
    rows, cols = fft_data.shape
    center_x, center_y = cols // 2, rows // 2
    
    # High-pass filter: Set frequencies below the specified radius to zero
    for y in range(rows):
        for x in range(cols):
            distance = np.hypot(x - center_x, y - center_y)
            if distance <= radius:
                fft_data[y, x] = 0  # Remove low-frequency components
    
    # Apply the mask to the frequency domain
    fft_data = fft_data * mask
    
    # Inverse FFT to transform back to the spatial domain
    return ifft_2d(fft_data, dc_component)

def high_pass_edge_detection_mask_2(image, radius, mask_choice):
    """
    Apply high-pass filter in the frequency domain and return the image after edge detection,
    taking into account the specified angle direction for the mask.
    
    Args:
        image (ndarray): Input image.
        radius (int): Radius for high-pass filtering (frequencies below this radius will be removed).
        mask_choice (int): The mask choice for determining the angle range for edge detection.
        
    Returns:
        ndarray: Image after high-pass edge detection with respect to the angle direction.
    """
    # Perform FFT and get the frequency domain representation
    fft_data, _, _, dc_component = fft_2d(image)
    
    rows, cols = fft_data.shape
    center_x, center_y = cols // 2, rows // 2
    
    # Set angle range based on mask choice
    angle_range = set_angle_range_based_on_choice(mask_choice)
    
    # Apply high-pass filtering considering the angle direction
    for y in range(rows):
        for x in range(cols):
            # Calculate distance from the center (for low-pass cutoff)
            distance = np.hypot(x - center_x, y - center_y)
            
            # Calculate angle of the frequency point in degrees
            angle = np.degrees(np.arctan2(y - center_y, x - center_x)) % 180
            
            # Apply high-pass filter: remove low frequencies within the angle range
            if distance <= radius or not (angle_range[0] <= angle < angle_range[1]):
                fft_data[y, x] = 0  # Set low frequencies to zero

    # Inverse FFT to transform back to the spatial domain
    return ifft_2d(fft_data, dc_component)

def high_pass_edge_detection_mask(image, custom_mask):
    """
    Apply high-pass filter in the frequency domain using a custom mask for edge detection.
    
    Args:
        image (ndarray): Input image (grayscale).
        custom_mask (ndarray): Custom frequency domain mask (same size as the FFT of the image).
        
    Returns:
        ndarray: Image after applying the custom mask edge detection in frequency domain.
    """
    # Perform FFT and get the frequency domain representation
    fft_data, _, _, dc_component = fft_2d(image)

    # Ensure the custom mask is of the same size as the FFT result
    rows, cols = fft_data.shape
    if custom_mask.shape != (rows, cols):
        print(f"FFT Shape: {fft_data.shape}, Mask Shape: {custom_mask.shape}")
    # Apply the custom mask in the frequency domain by element-wise multiplication
    fft_data_filtered = fft_data * custom_mask
    
    # Inverse FFT to transform back to the spatial domain
    return ifft_2d(fft_data, dc_component)

def directional_high_pass_mask(image_shape, radius, edge_directions, preferred_angle_range=(0, 180)):
    """
    Create a high-pass mask that selectively keeps frequencies based on edge direction.
    
    Args:
        image_shape (tuple): Shape of the frequency domain image.
        radius (int): Cut-off radius for high-pass filtering.
        edge_directions (ndarray): Edge direction matrix (same size as image).
        preferred_angle_range (tuple): Range of angles to keep (e.g., (45, 135) for diagonal edges).
    
    Returns:
        ndarray: Directional high-pass filter mask.
    """
    rows, cols = image_shape
    center_x, center_y = cols // 2, rows // 2
    mask = np.ones((rows, cols), dtype=np.float64)
    
    for y in range(rows):
        for x in range(cols):
            distance = np.hypot(x - center_x, y - center_y)
            if distance <= radius:
                mask[y, x] = 0  # Remove low frequencies

            # Keep only edges within the specified angle range
            angle = edge_directions[y, x]
            if not (preferred_angle_range[0] <= angle <= preferred_angle_range[1]):
                mask[y, x] = 0  # Suppress unwanted directions
    
    return mask
    
def create_high_pass_mask(image_shape, radius):
    rows, cols = image_shape
    center_x, center_y = cols // 2, rows // 2
    mask = np.ones((rows, cols), dtype=np.float64)
    
    for y in range(rows):
        for x in range(cols):
            distance = np.hypot(x - center_x, y - center_y)
            if distance <= radius:
                mask[y, x] = 0  # Remove low frequencies, keep high frequencies
            
    return mask

def extract_edge_direction(mask, threshold=10):
    """
    Extract edge directions using Sobel gradients, removing weak edges based on a threshold.
    
    Args:
        mask (ndarray): Binary edge mask (white edges on black background).
        threshold (int): Gradient magnitude threshold to ignore weak edges.
    
    Returns:
        ndarray: Edge direction matrix (angle in degrees).
    """
    sobel_edges = osobel(mask)  # Apply Sobel operator to get edge magnitudes
    
    height, width = mask.shape
    grad_x = np.zeros_like(mask, dtype=np.float64)
    grad_y = np.zeros_like(mask, dtype=np.float64)
    
    # Compute gradients and directions
    for n in range(1, height - 1):
        for m in range(1, width - 1):
            A0 = mask[n-1, m-1]
            A1 = mask[n-1, m]
            A2 = mask[n-1, m+1]
            A3 = mask[n, m+1]
            A4 = mask[n+1, m+1]
            A5 = mask[n+1, m]
            A6 = mask[n+1, m-1]
            A7 = mask[n, m-1]
            
            # Compute Sobel gradients
            grad_x[n, m] = (A2 + 2 * A3 + A4) - (A0 + 2 * A7 + A6)
            grad_y[n, m] = (A0 + 2 * A1 + A2) - (A6 + 2 * A5 + A4)
    
    # Compute gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Filter out weak edges based on the threshold
    grad_x[grad_magnitude < threshold] = 0
    grad_y[grad_magnitude < threshold] = 0
    
    # Compute edge direction in degrees
    edge_direction = np.arctan2(grad_y, grad_x)  # Radians
    edge_direction_degrees = np.degrees(edge_direction)  # Convert to degrees

    return edge_direction_degrees

def determine_dominant_orientation(edge_directions):
    """
    Determine the dominant edge orientation based on the edge directions.
    
    Args:
        edge_directions (ndarray): Edge directions in degrees.
    
    Returns:
        float: Dominant angle in degrees.
    """
    # Flatten the array and remove any NaN values (where there's no edge)
    flattened = edge_directions.flatten()
    flattened = flattened[~np.isnan(flattened)]

    # Print how many times each unique value appears
    unique_values, counts = np.unique(flattened, return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"Angle {value:.2f} degrees appears {count} times.")
    
    # Compute a histogram of the edge directions (0-180 degrees)
    hist, bin_edges = np.histogram(flattened, bins=18, range=(0, 180))

    # Find the bin with the highest count (dominant orientation)
    dominant_bin = np.argmax(hist)
    dominant_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2  # Midpoint of the dominant bin
    
    return dominant_angle

def set_angle_range_based_on_orientation(dominant_angle):
    """
    Set the angle range based on the dominant edge orientation.
    
    Args:
        dominant_angle (float): Dominant angle in degrees.
    
    Returns:
        tuple: Angle range (start, end) for directional filtering.
    """
    # Define angle ranges based on the dominant orientation
    if 0 <= dominant_angle < 45 or 135 <= dominant_angle < 180:
        # Diagonal edges (0 to 45 degrees, or 135 to 180 degrees)
        return (0, 90)
    else:
        # Vertical/horizontal edges (45 to 135 degrees)
        return (90, 180)

def set_angle_range_based_on_choice(mask_choice):
    """
    Set the angle range based on the mask choice argument.
    
    Args:
        mask_choice (int): Choice of edge detection orientation.
    
    Returns:
        tuple: Angle range (start, end) for directional filtering.
    """
    if mask_choice == 1:
        # Horizontal edges: 0° to 45°
        return (0, 45)
    elif mask_choice == 2:
        # Vertical edges: 45° to 90°
        return (45, 90)
    elif mask_choice == 3:
        # Horizontal + Vertical edges: 0° to 90°
        return (0, 90)
    elif mask_choice == 4:
        # Horizontal + Vertical + Diagonal edges: 0° to 180°
        return (0, 180)
    else:
        raise ValueError("Invalid mask_choice. Please choose a value between 1 and 4.")

def phase_modulation(image, k, l):
    """
    Apply phase modulation in the frequency domain.
    
    Args:
        image (ndarray): Input image.
        k (int): Modulation factor for the horizontal direction.
        l (int): Modulation factor for the vertical direction.
        
    Returns:
        ndarray: Image after phase modulation.
    """
    # Perform FFT and get the frequency domain representation
    fft_data, _, phase, dc_component = fft_2d(image)
    
    rows, cols = fft_data.shape
    center_x, center_y = cols // 2, rows // 2
    
    # Apply phase modulation by shifting the phase spectrum
    for y in range(rows):
        for x in range(cols):
            phase_shift = -((x * k * 2 * np.pi) / cols) - ((y * l * 2 * np.pi) / rows) + (k + l) * np.pi
            fft_data[y, x] *= np.exp(1j * phase_shift)  # Apply the phase shift in frequency domain
    
    # Inverse FFT to get the image after phase modulation
    return ifft_2d(fft_data,dc_component)

def normalize_image(image):
    """Normalize the image values to be between 0 and 255 without using NumPy functions."""
    
    # Find the minimum and maximum values manually
    image_min = float('inf')
    image_max = float('-inf')
    
    # Traverse the image to find the min and max values
    for row in image:
        for pixel in row:
            if pixel < image_min:
                image_min = pixel
            if pixel > image_max:
                image_max = pixel
    
    # Initialize a NumPy array with the same shape as the input image
    normalized_image = np.empty(image.shape, dtype=np.uint8)
    
    # Normalize the image values and store them in the new array
    for i in range(image.shape[0]):  # For each row
        for j in range(image.shape[1]):  # For each column in the row
            pixel = image[i, j]
            normalized_pixel = 255.0 * (float(pixel) - float(image_min)) / float(image_max - image_min)
            # Clip values to ensure they stay between 0 and 255
            normalized_image[i, j] = min(max(int(normalized_pixel), 0), 255)
    
    return normalized_image

### CMD commands
def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <command> <image_path> [<parameters>] <output_path>")
        sys.exit(1)
    command = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[len(sys.argv)-1]

    # Load the image
    try:
        matrix = imageLoader(image_path)
    except FileNotFoundError:
        print(f"Error: File {image_path} not found.")
        sys.exit(1)
    
    # Handle commands
    if command == '--brightnessFlat':
        if len(sys.argv) != 5:
            print("Usage: python script.py --brightnessFlat <image_path> <brightness_value> <output_path>")
            sys.exit(1)
        try:
            brightness_value = int(sys.argv[3])
            sign_negative = brightness_value < 0  # Determine if brightness is increasing or decreasing
            modified_matrix = brightnessChangerFlat(matrix, abs(brightness_value), sign_negative)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error: Brightness value must be an integer.")
            sys.exit(1)

    elif command == '--brightnessGamma':
        if len(sys.argv) != 5:
            print("Usage: python script.py --brightnessGamma <image_path> <brightness_factor> <output_path>")
            sys.exit(1)
        try:
            brightness_factor = float(sys.argv[3])
            modified_matrix = brightnessChangerGamma(matrix, brightness_factor)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error: Brightness factor must be a valid float.")
            sys.exit(1)
    
    elif command == '--contrast':
        if len(sys.argv) != 5:
            print("Usage: python script.py --contrast <image_path> <contrast_factor> <output_path>")
            sys.exit(1)
        try:
            contrast_factor = float(sys.argv[3])
            modified_matrix = contrastChanger(matrix, contrast_factor)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error: Contrast factor must be a valid float.")
            sys.exit(1)
        
    elif command == '--negative':
        if len(sys.argv) != 4:
            print("Usage: python script.py --negative <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix = negative(matrix)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error processing the image.")
            sys.exit(1)

    elif command=='--hflip':
        if len(sys.argv) !=4:
            print("Usage: python script.py --hflip <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix=flipHorizontal(matrix)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command=='--vflip':
        if len(sys.argv) !=4:
            print("Usage: python script.py --vflip <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix=flipVertical(matrix)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command=='--dflip':
        if len(sys.argv) !=4:
            print("Usage: python script.py --dflip <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix=flipDiagonal(matrix)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    elif command == '--shrink':
        if len(sys.argv) != 5:
            print("Usage: python script.py --shrink <image_path> <shrink_factor> <output_path>")
            sys.exit(1)
        try:
            shrink_factor = float(sys.argv[3])
            output_path = sys.argv[4]
            modified_matrix = shrinkImage(matrix, shrink_factor)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--enlarge':
        if len(sys.argv) != 5:
            print("Usage: python script.py --enlarge <image_path> <enlarge_factor> <output_path>")
            sys.exit(1)
        try:
            enlarge_factor = float(sys.argv[3])
            output_path = sys.argv[4]
            modified_matrix = enlargeImage(matrix, enlarge_factor)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--alpha':
        if len(sys.argv) != 6:
                print("Usage: python script.py --alpha <image_path> <kernel_size> <alpha> <output_path>")
                sys.exit(1)
        try:
            kernel_size = int(sys.argv[3])
            alpha = float(sys.argv[4])
            if kernel_size % 2 == 0:
                print("Kernel size must be an odd number.")
                sys.exit(1)
            if ((alpha < 0) or (alpha >= 0.5)):
                print("Alpha must be between 0 and 0.5.")
                sys.exit(1)
            modified_matrix = alphatf(matrix, kernel_size, alpha)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error: Invalid kernel size or alpha value.")
            sys.exit(1)

    elif command=='--cmean':
        if len(sys.argv) != 6:
            print("Usage: python script.py --cmean <image_path> <kernel> <P> <output_path>")
            sys.exit(1)
        try:
            kernel = int(sys.argv[3])
            P = float(sys.argv[4])
        except ValueError: 
            print("Kernel must be an integer and P must be a float.")
            sys.exit(1)
        try:
            modified_matrix=contra_harmonic_mean_filter(matrix,kernel,P)
            saveImage(modified_matrix,output_path)
            sys.exit(0)
        except Exception as e:
            print(f"Error processing the image: {e}")
            sys.exit(1)
        
    elif command =='--mse':
        if len(sys.argv) != 4:
            print("Usage: python script.py --mse <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imageLoader(output_path)
            print(mse(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
    
    elif command =='--pmse':
        if len(sys.argv) != 4:
            print("Usage: python script.py --pmse <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imageLoader(output_path)
            print(pmse(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

    elif command =='--snr':
        if len(sys.argv) != 4:
            print("Usage: python script.py --snr <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imageLoader(output_path)
            print(snr(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
    
    elif command =='--psnr':
        if len(sys.argv) != 4:
            print("Usage: python script.py --psnr <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imageLoader(output_path)
            print(psnr(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

    elif command =='--md':
        if len(sys.argv) != 4:
            print("Usage: python script.py --md <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imageLoader(output_path)
            print(md(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

    elif command == '--test':
        if len(sys.argv) != 5:
            print("Usage: python script.py --md <image_path> <image2_path> <image3_path>")
            sys.exit(1)
        try:
            origin_path = sys.argv[3]
            matrix_o = imageLoader(origin_path)
            matrix_f = imageLoader(output_path)
            print(':Original:')
            print(mse(matrix, matrix_o))
            print(pmse(matrix, matrix_o))
            print(snr(matrix, matrix_o))
            print(psnr(matrix, matrix_o))
            print(md(matrix, matrix_o))
            print("-----")
            print("Filtered:")
            print(mse(matrix, matrix_f))
            print(pmse(matrix, matrix_f))
            print(snr(matrix, matrix_f))
            print(psnr(matrix, matrix_f))
            print(md(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)


    ### Task2

    elif command =='--histogram':
        if len(sys.argv) != 5:
            print("Usage: python script.py --histogram <image_path> <channel> <output_path>")
            sys.exit(1)
        try:
            channel = int(sys.argv[3])
            modified_matrix=histogram(matrix, channel)
            output_matrix = draw_histogram(modified_matrix)
            saveImage(output_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--hrayleigh':
        if len(sys.argv) != 7:
                print("Usage: python script.py --hrayleigh <image_path> <gmin> <gmax> <alpha> <output_path>")
                sys.exit(1)
        try:
            g_min = int(sys.argv[3])
            g_max = int(sys.argv[4])
            alpha = float(sys.argv[5])
            if g_min < 0 or g_max < 0:
                print("G values must be bigger than 0")
                sys.exit(1)
            histogram_matrix = histogram(matrix)
            modified_matrix = hrayleigh(histogram_matrix, g_min, g_max, alpha)
            output_image = np.array([modified_matrix[pixel] for pixel in matrix.flatten()]).reshape(matrix.shape)
            saveImage(output_image, output_path)
        except ValueError:
            print("Error: Invalid g values size or alpha value.")
            sys.exit(1)

    elif command =='--u_slaplace':
        if len(sys.argv) != 5:
            print("Usage: python script.py --u_slaplace <image_path> <mask_number> <output_path>")
            sys.exit(1)
        try:
            mask_number = int(sys.argv[3])
            if (mask_number < 1 or mask_number > 3):
                print("Mask number must be between 1 and 3")
                sys.exit(1)
            modified_matrix=u_slaplace(matrix,mask_number)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--o_slaplace':
        if len(sys.argv) != 4:
            print("Usage: python script.py --u_slaplace <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix=o_slaplace(matrix)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command =='--osobel':
        if len(sys.argv) != 4:
            print("Usage: python script.py --osobel <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix=osobel(matrix)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--mean':
        if len(sys.argv) != 4:
            print("Usage: python script.py --mean <image_path> <output_path>")
            sys.exit(1)
        try:
            output = mean(matrix)
            print(f"The mean is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--variance':
        if len(sys.argv) != 4:
            print("Usage: python script.py --variance <image_path> <output_path>")
            sys.exit(1)
        try:
            output = variance(matrix)
            print(f"The variance is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--std_deviation':
        if len(sys.argv) != 4:
            print("Usage: python script.py --std_deviation <image_path> <output_path>")
            sys.exit(1)
        try:
            output = std_deviation(matrix)
            print(f"The standard deviation is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--variation_coefficient':
        if len(sys.argv) != 4:
            print("Usage: python script.py --variation_coefficient <image_path> <output_path>")
            sys.exit(1)
        try:
            output = variation_coefficient(matrix)
            print(f"The variation coefficient is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--asymmetry_coefficient':
        if len(sys.argv) != 4:
            print("Usage: python script.py --asymmetry_coefficient <image_path> <output_path>")
            sys.exit(1)
        try:
            output = asymmetry_coefficient(matrix)
            print(f"The asymmetry coefficient is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--flattening_coefficient':
        if len(sys.argv) != 4:
            print("Usage: python script.py --flattening_coefficient <image_path> <output_path>")
            sys.exit(1)
        try:
            output = flattening_coefficient(matrix)
            print(f"The flattening coefficient is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--entropy':
        if len(sys.argv) != 4:
            print("Usage: python script.py --entropy <image_path> <output_path>")
            sys.exit(1)
        try:
            output = entropy(matrix)
            print(f"The entropy is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--test2':
        if len(sys.argv) != 4:
            print("Usage: python script.py --test2 <image_path> <output_path>")
            sys.exit(1)
        try:
            print("Calculating metrics...")
            mean_result = mean(matrix)
            print(f"The mean is: {mean_result}")
            
            variance_result = variance(matrix)
            print(f"The variance is: {variance_result}")
            
            std_dev_result = std_deviation(matrix)
            print(f"The standard deviation is: {std_dev_result}")
            
            variation_coeff_result = variation_coefficient(matrix)
            print(f"The variation coefficient is: {variation_coeff_result}")
            
            asymmetry_coeff_result = asymmetry_coefficient(matrix)
            print(f"The asymmetry coefficient is: {asymmetry_coeff_result}")
            
            flattening_coeff_result = flattening_coefficient(matrix)
            print(f"The flattening coefficient is: {flattening_coeff_result}")
            
            entropy_result = entropy(matrix)
            print(f"The entropy is: {entropy_result}")
            
            sys.exit(0)
        except ValueError:
            print("Error processing the image.")
            sys.exit(1)
    ## Task 3

    elif command =='--dilation':
        if len(sys.argv) != 5:
            print("Usage: python script.py --dilation <image_path> <struct_element> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            struct_number = int(sys.argv[3])
            modified_matrix = dilation(matrix, structural_elements(struct_number))
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--erosion':
        if len(sys.argv) != 5:
            print("Usage: python script.py --erosion <image_path> <struct_element> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            struct_number = int(sys.argv[3])
            modified_matrix = erosion(matrix, structural_elements(struct_number))
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--opening':
        if len(sys.argv) != 5:
            print("Usage: python script.py --opening <image_path> <struct_element> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            struct_number = int(sys.argv[3])
            modified_matrix = opening(matrix, structural_elements(struct_number))
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command =='--closing':
        if len(sys.argv) != 5:
            print("Usage: python script.py --closing <image_path> <struct_element> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            struct_number = int(sys.argv[3])
            modified_matrix = closing(matrix, structural_elements(struct_number))
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--hmt':
        if len(sys.argv) != 6:
            print("Usage: python script.py --hmt <image_path> <XI, or XII> <struct_element> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            xval = int(sys.argv[3])
            struct_B = int(sys.argv[4])
            if(xval == 11):
                modified_matrix = hmt(matrix, structural_elements_XI(struct_B))
            else:
                modified_matrix = hmt(matrix, structural_elements_XII()[struct_B])
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command =='--successive_n':
        if len(sys.argv) != 4:
            print("Usage: python script.py --successive_n <image_path> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            modified_matrix = successive_n_transform(matrix, structural_elements_XII())
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command =='--region_growing':
        if len(sys.argv) != 7:
            print("Usage: python script.py --region_growing <image_path> <seed_points> <threshold> <criterion> <output_path>")
            sys.exit(1)
        try:
            matrix = imageLoader(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            tseed_points = int(sys.argv[3])
            threshold = int(sys.argv[4])
            criterion = int(sys.argv[5])
            modified_matrix = region_growing(matrix, seed_points()[tseed_points], threshold, criterion)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    #TASK 4

    elif command =='--FFT':
        if len(sys.argv) != 4:
            print("Usage: python script.py --FFT <image_path> <output_path>")
            sys.exit(1)
        try:
            matrix = imageLoaderG(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
        try:
            print("Initial FFT")
            #matrix = decimate_spatial(matrix,2)
            fft_result, magnitude, phase, dc_component = fft_2d(matrix)
            print("Saving Magnitude and Phase")
            print("Saving Magnitude...")
            log_magnitude = np.log1p(magnitude)  # Log scale
            normalized_magnitude = (log_magnitude / log_magnitude.max()) * 255
            saveImage(normalized_magnitude, "magnitude.bmp")
            saveImage(fft_result, "fft_data.bmp")
            print("Saving Phase...")
            normalized_phase = ((phase + np.pi) / (2 * np.pi)) * 255
            saveImage(normalized_phase, "phase.bmp")
            print("Reconstructing image")
            reconstructed_image = ifft_2d(fft_result, dc_component)
            saveImage(reconstructed_image, output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command =='--low_pass':
        if len(sys.argv) != 5:
            print("Usage: python script.py --low_pass <image_path> <radius> <output_path>")
            sys.exit(1)
        try:
            matrix = imageLoaderG(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
        try:
            radius = int(sys.argv[3])
            print("Creating Filtered image")
            image_low_pass = low_pass_filter(matrix, radius)
            #image_low_pass = normalize_image(image_low_pass)
            saveImage(image_low_pass, output_path)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--high_pass':
        if len(sys.argv) != 5:
            print("Usage: python script.py --high_pass <image_path> <radius> <output_path>")
            sys.exit(1)
        try:
            matrix = imageLoaderG(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
        try:
            radius = int(sys.argv[3])
            print("Creating Filtered image")
            image_high_pass = high_pass_filter(matrix, radius)
            image_high_pass = normalize_image(image_high_pass)
            saveImage(image_high_pass, output_path)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--high_pass_edge':
        if len(sys.argv) != 6:
            print("Usage: python script.py --high_pass_edge <image_path> <radius> <mask> <output_path>")
            sys.exit(1)
        try:
            matrix = imageLoaderG(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
        try:
            radius = int(sys.argv[3])
            mask_choice = int(sys.argv[4])
            print("Creating Filtered image")
            mask_path = "F5mask1.bmp" if mask_choice == 1 else "F5mask2.bmp"
            matrix_mask = imageLoader1B(mask_path)
            image_low_pass = high_pass_edge_detection(matrix, radius, matrix_mask)
            image_low_pass = normalize_image(image_low_pass)
            saveImage(image_low_pass, output_path)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--high_pass_edge_mask':
        if len(sys.argv) != 6:
            print("Usage: python script.py --high_pass_edge_mask <image_path> <radius> <mask> <output_path>")
            sys.exit(1)
        try:
            matrix = imageLoaderG(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
        try:
            radius = int(sys.argv[3])
            mask_choice = int(sys.argv[4])
            print("Creating Filtered image")
            
            # Load mask image
            mask_path = "F5mask1.bmp" if mask_choice == 1 else "F5mask2.bmp"
            matrix_mask = imageLoader1B(mask_path)
            
            '''# Step 1: Extract edge directions from the mask
            edge_directions = extract_edge_direction(matrix_mask)

            # Step 2: Determine the dominant orientation
            dominant_angle = determine_dominant_orientation(edge_directions)

            # Step 3: Set the angle range for filtering based on the dominant orientation
            angle_range = set_angle_range_based_on_orientation(dominant_angle)

            # Print the chosen angle range for debugging
            print(f"Dominant Angle: {dominant_angle}°")
            print(f"Using Angle Range: {angle_range}")

            # Step 4: Create the directional high-pass filter
            mask = directional_high_pass_mask(matrix_mask.shape, radius, edge_directions, angle_range)
            
            # Apply high-pass edge detection
            image_filtered = high_pass_edge_detection_mask(matrix, mask)
            image_filtered_norm = normalize_image(image_filtered)
            '''
            image_low_pass = high_pass_edge_detection_mask_2(matrix, radius, mask_choice)
            image_low_pass = normalize_image(image_low_pass)
            saveImage(image_low_pass, output_path)
        except ValueError:
            print("Error processing the image.")
            sys.exit(1)



    elif command =='--band_pass':
        if len(sys.argv) != 6:
            print("Usage: python script.py --band_pass <image_path> <low_radius> <high_radius> <output_path>")
            sys.exit(1)
        try:
            matrix = imageLoaderG(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
        try:
            low_radius = int(sys.argv[3])
            high_radius = int(sys.argv[4])
            print("Creating Filtered image")
            image_low_pass = band_pass_filter(matrix, low_radius, high_radius)
            image_low_pass = normalize_image(image_low_pass)
            saveImage(image_low_pass, output_path)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command =='--band_cut':
        if len(sys.argv) != 6:
            print("Usage: python script.py ----band_cut <image_path> <low_radius> <high_radius> <output_path>")
            sys.exit(1)
        try:
            matrix = imageLoaderG(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
        try:
            low_radius = int(sys.argv[3])
            high_radius = int(sys.argv[4])
            print("Creating Filtered image")
            image_low_pass = band_cut(matrix, low_radius, high_radius)
            image_low_pass = normalize_image(image_low_pass)
            saveImage(image_low_pass, output_path)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--phase_modulation':
        if len(sys.argv) != 6:
            print("Usage: python script.py --phase_modulation <image_path> <k> <l> <output_path>")
            sys.exit(1)
        try:
            matrix = imageLoaderG(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
        try:
            k = int(sys.argv[3])
            l = int(sys.argv[4])
            print("Creating Filtered image")
            image_low_pass = phase_modulation(matrix, k, l)
            image_low_pass = normalize_image(image_low_pass)
            saveImage(image_low_pass, output_path)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    

    elif command =='--help':
        print("List of commands:")
        print("--brightnessFlat     | Usage: python script.py --brightnessFlat <image_path> <brightness_value> <output_path>")
        print('--brightnessGamma    | Usage: python script.py --brightnessGamma <image_path> <brightness_factor> <output_path>')
        print('--contrast           | Usage: python script.py --contrast <image_path> <contrast_factor> <output_path>')
        print('--negative           | Usage: python script.py --negative <image_path> <output_path>')
        print('--hflip              | Usage: python script.py --hflip <image_path> <output_path>')
        print('--vflip              | Usage: python script.py --vflip <image_path> <output_path>')
        print('--dflip              | Usage: python script.py --dflip <image_path> <output_path>')
        print('--shrink             | Usage: python script.py --shrink <image_path> <shrink_factor> <output_path>')
        print('--enlarge            | Usage: python script.py --enlarge <image_path> <enlarge_factor> <output_path>')
        print('--alpha              | Usage: python script.py --alpha <image_path> <kernel_size> <alpha> <output_path>')
        print('--cmean              | Usage: python script.py --cmean <image_path> <kernel> <P> <output_path>')
        print('--test               | Usage: python script.py --test <orignal_image_path> <noise_image_path> <filtered_image_path>')   
        print("--histogram          | Usage: python script.py --histogram <image_path> <channel> <output_path>")
        print("--hrayleigh          | Usage: python script.py --hrayleigh <image_path> <gmin> <gmax> <alpha> <output_path>")        
        print("--u_slaplace         | Usage: python script.py --u_slaplace <image_path> <mask_number> <output_path>")        
        print("--o_slaplace         | Usage: python script.py --o_slaplace <image_path> <output_path>")        
        print("--osobel             | Usage: python script.py --osobel <image_path> <output_path>")        
        print("--test2              | Usage: python script.py --test2 <image_path> <output_path>")
        print("--dilation           | Usage: python script.py --dilation <image_path> <struct_element> <output_path>")
        print("--erosion            | Usage: python script.py --erosion <image_path> <struct_element> <output_path>")            
        print("--opening            | Usage: python script.py --opening <image_path> <struct_element> <output_path>")            
        print("--closing            | Usage: python script.py --closing <image_path> <struct_element> <output_path>")            
        print("--hmt                | Usage: python script.py --hmt <image_path> <XI, or XII> <struct_element> <output_path>")            
        print("--successive_n       | Usage: python script.py --successive_n <image_path> <output_path>")            
        print("--region_growing     | Usage: python script.py --region_growing <image_path> <seed_points> <threshold> <criterion> <output_path>")
        print("--FFS                | Usage: python script.py --FFS <image_path> <output_path>")  
        print("--low_pass           | Usage: python script.py --low_pass <image_path> <radius> <output_path>") 
        print("--high_pass          | Usage: python script.py --high_pass <image_path> <radius> <output_path>")            
        print("--high_pass_edge     | Usage: python script.py --high_pass_edge <image_path> <radius> <output_path>")            
        print("--band_pass          | Usage: python script.py --band_pass <image_path> <low_radius> <high_radius> <output_path>")            
        print("--band_cut           | Usage: python script.py ----band_cut <image_path> <low_radius> <high_radius> <output_path>")      
        print("--phase_modulation   | Usage: python script.py --phase_modulation <image_path> <k> <l> <output_path>")
        print("--high_pass_edge_mask     | Usage: python script.py --high_pass_edge_mask <image_path> <radius> <mask> <output_path>")
                                               
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()

